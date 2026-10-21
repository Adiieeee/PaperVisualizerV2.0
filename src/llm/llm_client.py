"""
Satu-satunya jalur ke penyedia LLM: LiteLLM Router di dalam proses (tanpa proxy, tanpa cache).

Alias:
- "dedup"   -> Gemini saja, TANPA fallback. Keputusan merge harus konsisten dari satu model;
               kalau kuota habis, caller simpan checkpoint lalu berhenti.
- "general" -> Gemini; pindah ke Groq HANYA saat kuota HARIAN Gemini habis (dan hanya bila
               GROQ_API_KEY dan GROQ_MODEL diisi). Token Groq dihemat: gpt-oss berpikir dulu.

Fallback sengaja tidak memakai `fallbacks=` milik Router: itu memindah ke Groq pada SETIAP
kegagalan (429 per menit, 503, cooldown). Di sini 429 per menit dan 5xx di-raise apa adanya dan
diretry ke Gemini oleh tenacity di LLMService. Router juga tidak meretry (num_retries=0) dan
tanpa cooldown — dua lapis retry akan melipatgandakan request dan membakar kuota.
"""
import json
import logging
import os
from typing import Optional, Type

# Tanpa ini `import litellm` mengunduh cost map model dari GitHub (pytest harus tanpa jaringan).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
from litellm import Router
from pydantic import BaseModel, ValidationError

from src.config import GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_MODEL

litellm.suppress_debug_info = True
# Build = ribuan panggilan: DeprecationWarning `temperature` Gemini 3+ (WARNING, tiap panggilan) dan
# dump body error Router (ERROR, dicetak dua kali) menenggelamkan output. Error-nya tetap sampai ke
# caller sebagai exception; perpindahan ke Groq dicatat oleh logger "llm" di bawah.
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.CRITICAL)
log = logging.getLogger("llm")


class DailyQuotaExceededError(Exception):
    """Raised when the daily API quota is exhausted. Retrying is pointless."""
    pass


# Nama dari LITELLM_INTEGRATION_PLAN.md. Alias, bukan kelas baru: server.py dan test sudah
# menangkap DailyQuotaExceededError.
LLMQuotaError = DailyQuotaExceededError

# Penanda kuota HARIAN di pesan 429. Gemini: "GenerateRequestsPerDayPerProjectPerModel...".
# Groq: "... on requests per day (RPD)" / "tokens per day (TPD)". Selain ini = limit per menit.
_DAILY_MARKERS = ("perday", "per day")

_DAILY_MESSAGE = (
    "\n\n[DAILY QUOTA EXHAUSTED]\n"
    "Your free-tier daily request limit has been reached (no fallback left for this call).\n"
    "Gemini quota resets at midnight (Pacific Time) / ~15:00 WIB.\n"
    "Progress has been saved. Run 'python main.py build' again after the reset.\n"
)


# Alias yang boleh pindah ke Groq saat kuota harian habis -> alias deployment Groq-nya.
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
        # dedup: Gemini saja, tanpa fallback
        {"model_name": "dedup", "litellm_params": {"model": gemini_model, "api_key": gemini_key}},
        # general: Gemini; BACKUP_ALIAS dipanggil eksplisit oleh call_llm saat kuota harian habis
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
            raise  # dedup, atau Groq tidak dikonfigurasi
        log.warning("Kuota harian Gemini habis untuk alias=%s; pindah ke Groq (%s).", alias, backup)
        return _completion_daily_checked(router, backup, kwargs)


def _flatten_optional(node):
    """`Optional[X]` Pydantic = anyOf[X, null] -> X + nullable (bentuk yang dikirim google-genai)."""
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
    """Skema JSON NON-strict dari model Pydantic, sedekat mungkin dengan yang dikirim google-genai.
    Kelas Pydantic yang diberikan langsung diubah LiteLLM ke mode strict ala OpenAI: SEMUA field
    jadi wajib (mis. `description` Optional di Entity/Relationship) -> output Gemini berubah dan
    token deskripsi wajib bertambah. Field wajib dibiarkan persis seperti di Pydantic."""
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
    **extra,
):
    """Return (hasil, nama_model_yang_dipakai).

    Kalau schema diisi, hasil adalah instance Pydantic (ValueError bila tidak lolos validasi).
    Kalau tidak, string ("" bila model tidak mengembalikan teks, mis. diblokir safety filter).
    Melempar DailyQuotaExceededError (= LLMQuotaError) kalau kuota harian habis dan tidak ada
    fallback yang berhasil; 429 per menit dan error lain di-raise apa adanya supaya caller
    meretry ke Gemini (bukan pindah ke Groq).
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
        if schema is None:
            return content, used_model
        try:
            return schema.model_validate(json.loads(content)), used_model
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            log.warning("parse gagal (percobaan %d): %s", attempt + 1, e)

    raise ValueError(f"Output tidak lolos validasi schema: {last_err}")
