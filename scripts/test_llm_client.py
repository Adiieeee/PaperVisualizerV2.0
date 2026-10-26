"""
Manual test (uses the network + quota) for src/llm/llm_client.py. NOT part of pytest
(pytest.ini: testpaths = tests). Run: PYTHONUTF8=1 python -m scripts.test_llm_client
About 4 requests in total if Groq is not configured (3 Gemini + 1 fake key), about 6 if it is (+2 Groq).
"""
import logging

from pydantic import BaseModel

from src.config import ExtractedKnowledge, GEMINI_MODEL, GROQ_API_KEY, GROQ_MODEL
# llm_client must be imported before litellm: it is what disables the cost map download from GitHub.
from src.llm.llm_client import LLMQuotaError, build_router, call_llm

import litellm
from litellm.integrations.custom_logger import CustomLogger

logging.basicConfig(level=logging.INFO)
for noisy in ("LiteLLM", "LiteLLM Router", "httpx"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

GROQ_READY = bool(GROQ_API_KEY and GROQ_MODEL)


class DedupDecision(BaseModel):
    is_duplicate: bool
    merged_name: str | None = None


PROMPT = (
    "Apakah 'SGD' dan 'Stochastic Gradient Descent' merujuk entitas yang sama? "
    "Jawab dalam JSON dengan field is_duplicate (bool) dan merged_name (string atau null)."
)
MESSAGES = [{"role": "user", "content": PROMPT}]

EXTRACT_MESSAGES = [{"role": "user", "content": (
    "Extract entities and relationships strictly from the text below.\n"
    "<teks_paper>\nWe train ResNet-50 on ImageNet with Stochastic Gradient Descent and report "
    "Top-1 Error. Our method, proposed by Kaiming He, outperforms VGG-16.\n</teks_paper>"
)}]


class _RequestSpy(CustomLogger):
    """Record the generationConfig KEYS sent to Gemini (not the URL: the Gemini URL contains the key)."""

    def __init__(self):
        self.gen_config_keys = []

    def log_pre_api_call(self, model, messages, kwargs):
        body = (kwargs.get("additional_args") or {}).get("complete_input_dict") or {}
        gen = body.get("generationConfig") or body.get("generation_config")
        if isinstance(gen, dict):
            self.gen_config_keys.append(sorted(gen.keys()))


class _GeminiDailyQuotaRouter:
    """The real router, but calls to Gemini (alias dedup/general) are made to fail with a DAILY quota 429
    without a real request. Calls to Groq (general_backup) are still real."""

    def __init__(self, real):
        self.real = real

    def get_model_names(self):
        return self.real.get_model_names()

    def completion(self, model, **kwargs):
        if model != "general_backup":
            raise litellm.RateLimitError(
                message="429 Quota exceeded for metric: GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                llm_provider="gemini", model=GEMINI_MODEL)
        return self.real.completion(model=model, **kwargs)


def test_1_basic_general():
    out, model = call_llm(MESSAGES, alias="general")
    print("[1] general OK, model:", model)
    print(out[:200])


def test_2_schema_dedup():
    spy = _RequestSpy()
    litellm.callbacks = [spy]
    try:
        out, model = call_llm(MESSAGES, alias="dedup", schema=DedupDecision)
    finally:
        litellm.callbacks = []
    print("[2] dedup + schema OK, model:", model, "hasil:", out)
    print("    generationConfig keys yang dikirim:", spy.gen_config_keys)


def test_3_schema_fallback_groq():
    if not GROQ_READY:
        print("[3] DILEWATI: GROQ_API_KEY/GROQ_MODEL belum diisi di .env")
        return
    # Force the fallback to Groq without using up the Gemini quota. mock_testing_fallbacks is no longer used:
    # the fallback is no longer the Router's, but call_llm's (only when the daily quota runs out).
    router = _GeminiDailyQuotaRouter(build_router())
    out, model = call_llm(
        MESSAGES,
        alias="general",
        schema=DedupDecision,
        router=router,
    )
    print("[3] fallback OK, model:", model, "hasil:", out)


def test_4_dedup_tidak_fallback():
    # the dedup alias must not switch to Groq. Simulate a Gemini failure with a fake key.
    if not GROQ_READY:
        print("[4] catatan: Groq belum dikonfigurasi, jadi fallback memang tidak mungkin; "
              "tes ini lemah sampai .env diisi")
    router = build_router(gemini_key="key-palsu")
    try:
        call_llm(MESSAGES, alias="dedup", router=router)
        print("[4] GAGAL: dedup seharusnya error, bukan sukses")
    except LLMQuotaError:
        print("[4] OK: LLMQuotaError (tanpa fallback)")
    except Exception as e:
        print("[4] OK: error tanpa fallback ->", type(e).__name__)
    # And even when Gemini's DAILY quota runs out, dedup still stops (without a real request).
    try:
        call_llm(MESSAGES, alias="dedup", router=_GeminiDailyQuotaRouter(build_router()))
        print("[4b] GAGAL: dedup seharusnya LLMQuotaError, bukan sukses lewat Groq")
    except LLMQuotaError:
        print("[4b] OK: kuota harian habis pada dedup -> LLMQuotaError, Groq tidak dipanggil")


def test_5_schema_extraction():
    # The riskiest production schema: nested $defs, Literal enum, Optional with a default.
    out, model = call_llm(EXTRACT_MESSAGES, alias="general", schema=ExtractedKnowledge)
    print(f"[5a] Gemini ExtractedKnowledge OK, model: {model}, "
          f"{len(out.entities)} entity / {len(out.relationships)} relasi")
    print("     ", [(e.name, e.type) for e in out.entities])
    if not GROQ_READY:
        print("[5b] DILEWATI: GROQ_API_KEY/GROQ_MODEL belum diisi di .env")
        return
    out, model = call_llm(EXTRACT_MESSAGES, alias="general", schema=ExtractedKnowledge,
                          router=_GeminiDailyQuotaRouter(build_router()))
    print(f"[5b] Groq ExtractedKnowledge OK, model: {model}, "
          f"{len(out.entities)} entity / {len(out.relationships)} relasi")
    print("     ", [(e.name, e.type) for e in out.entities])


if __name__ == "__main__":
    print("GEMINI_MODEL:", GEMINI_MODEL)
    print("GROQ_MODEL:", GROQ_MODEL or "(kosong)")
    print("supports_response_schema(GEMINI_MODEL):", litellm.supports_response_schema(model=GEMINI_MODEL))
    if GROQ_MODEL:
        print("supports_response_schema(GROQ_MODEL):", litellm.supports_response_schema(model=GROQ_MODEL))
    for t in (test_1_basic_general, test_2_schema_dedup, test_3_schema_fallback_groq,
              test_4_dedup_tidak_fallback, test_5_schema_extraction):
        try:
            t()
        except Exception as e:  # one failing test must not hide the results of the other tests
            print(f"[{t.__name__}] ERROR {type(e).__name__}: {str(e)[:500]}")
