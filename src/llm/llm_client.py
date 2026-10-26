"""
The only path to the LLM providers: an in-process LiteLLM Router (no proxy, no cache).

Aliases:
- "dedup"   -> Gemini only, NO fallback. Merge decisions must come from a single model;
               when the quota runs out, the caller saves its checkpoint and stops.
- "general" -> Gemini; switches to Groq ONLY when Gemini's DAILY quota is exhausted (and only if
               GROQ_API_KEY and GROQ_MODEL are set). Groq tokens are conserved: gpt-oss reasons first.

The fallback deliberately does not use the Router's `fallbacks=`: that would switch to Groq on EVERY
failure (per-minute 429s, 503s, cooldowns). Here per-minute 429s and 5xx errors are re-raised as-is and
retried on Gemini by tenacity in LLMService. The Router also does not retry (num_retries=0) and has no
cooldowns: two layers of retries would multiply requests and burn the quota.
"""
import json
import logging
import os
from typing import Optional, Type

# Without this, `import litellm` downloads the model cost map from GitHub (pytest must run without network).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
from litellm import Router
from pydantic import BaseModel, ValidationError

from src.config import GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_MODEL

litellm.suppress_debug_info = True
# A build makes thousands of calls: the Gemini 3+ `temperature` DeprecationWarning (WARNING, every call) and
# the Router's error-body dump (ERROR, printed twice) would drown the output. The error still reaches the
# caller as an exception; switches to Groq are logged by the "llm" logger below.
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.CRITICAL)
log = logging.getLogger("llm")


class DailyQuotaExceededError(Exception):
    """Raised when the daily API quota is exhausted. Retrying is pointless."""
    pass


# An alias, not a new class: server.py and the tests already
# catch DailyQuotaExceededError.
LLMQuotaError = DailyQuotaExceededError

# Marker of a DAILY quota in a 429 message. Gemini: "GenerateRequestsPerDayPerProjectPerModel...".
# Groq: "... on requests per day (RPD)" / "tokens per day (TPD)". Anything else = a per-minute limit.
_DAILY_MARKERS = ("perday", "per day")

_DAILY_MESSAGE = (
    "\n\n[DAILY QUOTA EXHAUSTED]\n"
    "Your free-tier daily request limit has been reached (no fallback left for this call).\n"
    "Gemini quota resets at midnight (Pacific Time) / ~15:00 WIB.\n"
    "Progress has been saved. Run 'python main.py build' again after the reset.\n"
)


# Aliases allowed to switch to Groq when the daily quota runs out -> the alias of their Groq deployment.
FALLBACK_ALIASES = {"general": "general_backup"}
BACKUP_ALIAS = FALLBACK_ALIASES["general"]


def is_daily_quota_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _DAILY_MARKERS)


def build_router(
    gemini_key: Optional[str] = None,
    groq_key: Optional[str] = None,
    gemini_model: Optional[str] = None,
    groq_model: Optional[str] = None,
) -> Router:
    gemini_key = gemini_key or GEMINI_API_KEY
    groq_key = groq_key or GROQ_API_KEY
    gemini_model = gemini_model or GEMINI_MODEL
    groq_model = groq_model or GROQ_MODEL

    if not gemini_key:
        print("WARNING: GEMINI_API_KEY not found in environment variables.")

    model_list = [
        # dedup: Gemini only, no fallback
        {"model_name": "dedup", "litellm_params": {"model": gemini_model, "api_key": gemini_key}},
        # general: Gemini; BACKUP_ALIAS is called explicitly by call_llm when the daily quota runs out
        {"model_name": "general", "litellm_params": {"model": gemini_model, "api_key": gemini_key}},
    ]
    if groq_key and groq_model:
        model_list.append(
            {"model_name": BACKUP_ALIAS, "litellm_params": {"model": groq_model, "api_key": groq_key}}
        )
    else:
        log.info("Groq fallback disabled (GROQ_API_KEY/GROQ_MODEL not set).")

    return Router(model_list=model_list, num_retries=0, disable_cooldowns=True)


def _completion_daily_checked(router: Router, model_alias: str, kwargs: dict):
    try:
        return router.completion(model=model_alias, **kwargs)
    except litellm.RateLimitError as e:
        if is_daily_quota_error(e):
            raise DailyQuotaExceededError(_DAILY_MESSAGE) from e
        raise


def _complete(router: Router, alias: str, kwargs: dict):
    try:
        return _completion_daily_checked(router, alias, kwargs)
    except DailyQuotaExceededError:
        backup = FALLBACK_ALIASES.get(alias)
        if backup is None or backup not in router.get_model_names():
            raise  # dedup, or Groq is not configured
        log.warning("Kuota harian Gemini habis untuk alias=%s; pindah ke Groq (%s).", alias, backup)
        return _completion_daily_checked(router, backup, kwargs)


def _flatten_optional(node):
    """Pydantic `Optional[X]` = anyOf[X, null] -> X + nullable (the shape google-genai sends)."""
    if isinstance(node, list):
        return [_flatten_optional(n) for n in node]
    if not isinstance(node, dict):
        return node
    node = {k: _flatten_optional(v) for k, v in node.items()}
    variants = node.get("anyOf")
    if isinstance(variants, list) and len(variants) == 2 and {"type": "null"} in variants:
        (inner,) = [v for v in variants if v != {"type": "null"}]
        if "type" in inner:
            node.pop("anyOf")
            node.pop("default", None)
            node = {**inner, **node, "nullable": True}
    return node


def response_format_for(schema: Type[BaseModel]) -> dict:
    """NON-strict JSON schema of a Pydantic model, as close as possible to what google-genai sends.
    LiteLLM turns a Pydantic class passed directly into OpenAI-style strict mode: ALL fields become
    required (e.g. the Optional `description` on Entity/Relationship) -> Gemini's output changes and
    the required description tokens add up. Required fields stay exactly as in Pydantic."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": _flatten_optional(schema.model_json_schema()),
            "strict": False,
        },
    }


_router: Optional[Router] = None


def _get_router() -> Router:
    global _router
    if _router is None:
        _router = build_router()
    return _router


def call_llm(
    messages: list[dict],
    alias: str = "general",
    schema: Optional[Type[BaseModel]] = None,
    temperature: float = 0.0,
    max_parse_retries: int = 0,
    router: Optional[Router] = None,
    validate: bool = True,
    **extra,
):
    """Return (result, name_of_the_model_used).

    With a schema, the result is a Pydantic instance (ValueError if validation fails).
    Without one, a string ("" if the model returns no text, e.g. blocked by a safety filter).
    validate=False: the schema is still sent as response_format, but the result is raw text — used
    to retry/salvage part of an output that failed validation.
    Raises DailyQuotaExceededError (= LLMQuotaError) if the daily quota is exhausted and no
    fallback succeeds; per-minute 429s and other errors are re-raised as-is so the caller
    retries on Gemini (rather than switching to Groq).
    """
    router = router or _get_router()
    kwargs = {"messages": messages, "temperature": temperature, **extra}
    if schema is not None:
        kwargs["response_format"] = response_format_for(schema)

    last_err = None
    for attempt in range(max_parse_retries + 1):
        resp = _complete(router, alias, kwargs)

        used_model = resp.model
        log.info("alias=%s model=%s usage=%s", alias, used_model, getattr(resp, "usage", None))

        content = resp.choices[0].message.content or ""
        if schema is None or not validate:
            return content, used_model
        try:
            return schema.model_validate(json.loads(content)), used_model
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            log.warning("parse gagal (percobaan %d): %s", attempt + 1, e)

    raise ValueError(f"Output tidak lolos validasi schema: {last_err}")
