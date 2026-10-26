"""llm_client: quota classification, parsing, and fallback configuration. No network/quota (the router is faked)."""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from pydantic import BaseModel

from src.llm import llm_client as lc  # before litellm: this module is what disables the cost map download

litellm = lc.litellm
ROOT = Path(__file__).resolve().parent.parent


class _Decision(BaseModel):
    is_duplicate: bool
    merged_name: str


class _FakeRouter:
    """`exc` = error from Gemini (alias dedup/general); `backup_exc` = error from Groq.
    `with_groq=False` imitates a configuration without GROQ_API_KEY/GROQ_MODEL."""

    def __init__(self, content=None, exc=None, backup_exc=None, with_groq=True):
        self.content, self.exc, self.backup_exc, self.with_groq = content, exc, backup_exc, with_groq
        self.calls = []

    def get_model_names(self):
        return ["dedup", "general"] + (["general_backup"] if self.with_groq else [])

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        is_backup = kwargs["model"] == "general_backup"
        exc = self.backup_exc if is_backup else self.exc
        if exc:
            raise exc
        msg = types.SimpleNamespace(content=self.content)
        model = "openai/gpt-oss-120b" if is_backup else "gemini-flash-lite-latest"
        return types.SimpleNamespace(model=model, choices=[types.SimpleNamespace(message=msg)])

    def aliases(self):
        return [c["model"] for c in self.calls]


def _rate_limit(msg):
    return litellm.RateLimitError(message=msg, llm_provider="gemini", model="x")


MSGS = [{"role": "user", "content": "q"}]


_NO_NETWORK_PROBE = """
import socket
attempts, orig = [], socket.socket.connect
def guarded(self, addr, *a):
    if isinstance(addr, tuple) and addr[0] in ("127.0.0.1", "::1"):
        return orig(self, addr, *a)
    attempts.append(addr)
    raise OSError("network blocked")
socket.socket.connect = guarded
import src.llm.llm_client as lc
lc.build_router(gemini_key="k", groq_key="g", groq_model="groq/openai/gpt-oss-20b")
print(len(attempts))
"""


def test_import_and_router_make_no_network_calls():
    # Subprocess without the env from conftest: proves llm_client itself disables the download.
    env = {k: v for k, v in os.environ.items() if k != "LITELLM_LOCAL_MODEL_COST_MAP"}
    out = subprocess.run([sys.executable, "-c", _NO_NETWORK_PROBE], env=env, cwd=ROOT,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "0"


GEMINI_DAILY = "429 Quota exceeded for metric: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
GEMINI_MINUTE = "429 Quota exceeded for metric: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
GROQ_DAILY = "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000"
GROQ_MINUTE = "Rate limit reached on requests per minute (RPM): Limit 30"
OK_JSON = '{"is_duplicate": true, "merged_name": "SGD"}'


@pytest.mark.parametrize("msg", [
    GEMINI_DAILY,
    "Rate limit reached for model `openai/gpt-oss-120b` on requests per day (RPD): Limit 1000",
    GROQ_DAILY,
])
def test_daily_markers_are_recognized(msg):
    assert lc.is_daily_quota_error(_rate_limit(msg))
    assert lc.LLMQuotaError is lc.DailyQuotaExceededError


@pytest.mark.parametrize("msg", [GEMINI_MINUTE, GROQ_MINUTE])
def test_per_minute_markers_are_not_daily(msg):
    assert not lc.is_daily_quota_error(_rate_limit(msg))


# --- When Groq is used: ONLY when Gemini's daily quota runs out, ONLY for the general alias ---

def test_gemini_daily_quota_on_general_switches_to_groq():
    r = _FakeRouter(content=OK_JSON, exc=_rate_limit(GEMINI_DAILY))
    out, model = lc.call_llm(MSGS, alias="general", schema=_Decision, router=r)
    assert out.merged_name == "SGD" and model == "openai/gpt-oss-120b"
    assert r.aliases() == ["general", "general_backup"]
    assert r.calls[1]["response_format"] == lc.response_format_for(_Decision)  # the schema goes to Groq too


def test_gemini_daily_quota_on_dedup_never_uses_groq():
    r = _FakeRouter(content=OK_JSON, exc=_rate_limit(GEMINI_DAILY))
    with pytest.raises(lc.DailyQuotaExceededError):
        lc.call_llm(MSGS, alias="dedup", router=r)
    assert r.aliases() == ["dedup"]


@pytest.mark.parametrize("exc", [
    _rate_limit(GEMINI_MINUTE),
    litellm.ServiceUnavailableError(message="503 model overloaded", llm_provider="gemini", model="x"),
    litellm.InternalServerError(message="500 internal", llm_provider="gemini", model="x"),
    litellm.Timeout(message="timeout", llm_provider="gemini", model="x"),
])
def test_transient_gemini_errors_do_not_use_groq(exc):
    # Re-raised as-is -> tenacity in LLMService retries on Gemini with a delay.
    r = _FakeRouter(content=OK_JSON, exc=exc)
    with pytest.raises(type(exc)):
        lc.call_llm(MSGS, alias="general", router=r)
    assert r.aliases() == ["general"]


def test_gemini_daily_without_groq_configured_raises_daily():
    r = _FakeRouter(content=OK_JSON, exc=_rate_limit(GEMINI_DAILY), with_groq=False)
    with pytest.raises(lc.DailyQuotaExceededError):
        lc.call_llm(MSGS, alias="general", router=r)
    assert r.aliases() == ["general"]


def test_both_daily_quotas_exhausted_raises_daily():
    r = _FakeRouter(exc=_rate_limit(GEMINI_DAILY), backup_exc=_rate_limit(GROQ_DAILY))
    with pytest.raises(lc.DailyQuotaExceededError):
        lc.call_llm(MSGS, alias="general", router=r)
    assert r.aliases() == ["general", "general_backup"]


def test_groq_per_minute_limit_is_reraised_for_retry():
    r = _FakeRouter(exc=_rate_limit(GEMINI_DAILY), backup_exc=_rate_limit(GROQ_MINUTE))
    with pytest.raises(litellm.RateLimitError) as ei:
        lc.call_llm(MSGS, alias="general", router=r)
    assert not isinstance(ei.value, lc.DailyQuotaExceededError)


# --- Parsing ---

def test_validate_false_returns_raw_text_but_still_sends_schema():
    r = _FakeRouter(content='{"is_duplicate": "bukan bool"')  # invalid AND truncated
    out, model = lc.call_llm(MSGS, schema=_Decision, router=r, validate=False)
    assert out == '{"is_duplicate": "bukan bool"' and model == "gemini-flash-lite-latest"
    assert r.calls[0]["response_format"] == lc.response_format_for(_Decision)


def test_schema_result_and_model_name():
    r = _FakeRouter(content=OK_JSON)
    out, model = lc.call_llm(MSGS, alias="dedup", schema=_Decision, router=r)
    assert out == _Decision(is_duplicate=True, merged_name="SGD")
    assert model == "gemini-flash-lite-latest"
    assert r.calls[0]["model"] == "dedup"
    assert r.calls[0]["response_format"] == lc.response_format_for(_Decision)


def test_parse_failure_is_not_retried_by_default():
    r = _FakeRouter(content="bukan json")
    with pytest.raises(ValueError):
        lc.call_llm(MSGS, schema=_Decision, router=r)
    assert len(r.calls) == 1  # every retry = a request + quota


@pytest.mark.parametrize("empty", [None, ""])
def test_empty_text_returns_empty_string(empty):
    out, _ = lc.call_llm(MSGS, router=_FakeRouter(content=empty))
    assert out == ""


def test_gemini_schema_keeps_optional_fields_optional():
    # Measured regression: a raw Pydantic class -> LiteLLM strict -> `description` becomes required and
    # Gemini names entities differently from the old google-genai path ("MedQuad dataset" -> "MedQuad").
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
    from src.config import ExtractedKnowledge
    cfg = VertexGeminiConfig().map_openai_params(
        non_default_params={"response_format": lc.response_format_for(ExtractedKnowledge)},
        optional_params={}, model="gemini-flash-lite-latest", drop_params=False)
    props = cfg["response_schema"]["properties"]
    assert props["entities"]["items"]["required"] == ["name", "type"]
    assert props["relationships"]["items"]["required"] == ["source_entity", "target_entity", "relation_type"]
    assert "$ref" not in json.dumps(cfg["response_schema"])  # $defs are unpacked for Gemini
    # Optional is sent like google-genai: a nullable string, not anyOf
    desc = props["entities"]["items"]["properties"]["description"]
    assert desc["type"] == "string" and desc["nullable"] is True and "anyOf" not in desc


# --- Router configuration ---

def test_router_without_groq_has_no_backup(monkeypatch):
    monkeypatch.setattr(lc, "GROQ_API_KEY", None)
    monkeypatch.setattr(lc, "GROQ_MODEL", None)
    r = lc.build_router(gemini_key="k")
    assert sorted(r.get_model_names()) == ["dedup", "general"]
    assert r.num_retries == 0


def test_router_with_groq_has_no_automatic_fallbacks_or_cooldowns():
    # The Router's automatic fallback would switch to Groq on EVERY error; there must be none.
    r = lc.build_router(gemini_key="k", groq_key="g", groq_model="groq/openai/gpt-oss-20b")
    assert sorted(r.get_model_names()) == ["dedup", "general", "general_backup"]
    assert not r.fallbacks
    assert r.num_retries == 0
    assert r.disable_cooldowns is True
