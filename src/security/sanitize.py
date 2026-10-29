"""
Neutralize untrusted text before it enters an LLM prompt.

Sources of untrusted text in this project: user questions, PDF content (chunks), and LLM output over
PDF content (entity names, relation descriptions, community summaries — "second-order injection"). All of
it is treated as DATA. The functions here are used ONLY when assembling a prompt: text stored in
ChromaDB/the graph is NOT changed, because changing chunk text means changing its embedding (a rule of
the vector layer, see CLAUDE.md).

NOTE: invisible characters (zero-width etc.) are built with chr(), NOT written as literals
or escapes in the source — literals vanish without a trace when copied, and some editing tools decode
their escapes into literals. tests/test_sanitize.py keeps this source free of such characters.
"""
import re
from typing import List, Optional

# Zero-width space inserted into delimiter tags to break them.
_ZWSP = chr(0x200B)

# Delimiter tags used by the prompts in src/llm/prompts.py and the build-time prompts in llm_service.py.
# Add one here every time a prompt introduces a new delimiter tag.
PROTECTED_TAGS = [
    "konteks_graf",
    "konteks_teks",
    "ringkasan_komunitas",
    "konteks_eksternal",
    "pertanyaan_user",
    "teks_paper",
    "daftar_entitas",
]

# `<` / `>` include encoded forms (&lt; &#60; &#x3c;). `\s*` around `/` and the tag name covers the
# variants "< tag>", "</ tag>", and newlines. Attributes ("<tag x=1>") are caught too because some LLMs
# treat them as an opening tag.
_LT = r"(?:<|&lt;|&#0*60;|&#x0*3c;)"
_GT = r"(?:>|&gt;|&#0*62;|&#x0*3e;)"
_TAG_PATTERN = re.compile(
    _LT + r"\s*/?\s*(?:" + "|".join(PROTECTED_TAGS) + r")(?![\w-])(?:\s[^<>]*)?\s*" + _GT,
    re.IGNORECASE,
)


def neutralize_tags(text) -> str:
    """
    Break the prompt's delimiter tags that appear inside `text` (insert a zero-width space).
    Plain `<`/`>` ("< 2/3 of the total") are left alone so document quotes are not damaged.
    Idempotent: the result no longer matches the pattern, so it is safe to call repeatedly.
    """
    if text is None:
        return ""

    def _break(m: re.Match) -> str:
        s = m.group(0)
        return s[0] + _ZWSP + s[1:-1] + _ZWSP + s[-1]

    return _TAG_PATTERN.sub(_break, str(text))


# --- Character classes (built with chr(), see the note above) -------------------------------------

# Zero-width, bidi override/isolate, word-joiner, BOM: invisible, no reason to be in an entity name.
_INVISIBLE_CHARS = "".join(
    chr(c) for c in (*range(0x200B, 0x2010), *range(0x202A, 0x202F), 0x2060, 0xFEFF)
)
# Unicode line separators (LS/PS): treated like a newline.
_LINE_SEPARATORS = chr(0x2028) + chr(0x2029)

_INVISIBLE_RE = re.compile("[" + re.escape(_INVISIBLE_CHARS) + "]")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f" + re.escape(_LINE_SEPARATORS) + "]")
# For auditing chunk text: control + invisible characters, EXCEPT the tab/newline/CR that are normal in PDF text.
_HIDDEN_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f" + re.escape(_LINE_SEPARATORS + _INVISIBLE_CHARS) + "]"
)
_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"(?:https?://|ftp://|www\.)", re.IGNORECASE)

# --- Entity names (used by cleaner.py at build time; takes effect on the next rebuild) ------------

# Measured on the real graph: the longest Finding/Limitation name is 175 chars. This limit is deliberately loose —
# the goal is to drop clearly abnormal LLM output, not to trim legitimate statements.
MAX_ENTITY_NAME_LENGTH = 250


def sanitize_entity_name(name: str) -> Optional[str]:
    """
    Clean an entity name from LLM extraction. Returns the cleaned name, or None if the entity
    must be REJECTED: empty, too long, containing `<`/`>` (probably markup/a tag), or a URL.
    Control characters/newlines are replaced with a space (then double spaces are collapsed); invisible characters
    (zero-width, bidi) are removed so a name like "Res<ZWSP>Net" is still "ResNet".
    """
    cleaned = _WS_RE.sub(" ", _CONTROL_RE.sub(" ", _INVISIBLE_RE.sub("", name or ""))).strip()
    if not cleaned or len(cleaned) > MAX_ENTITY_NAME_LENGTH:
        return None
    if "<" in cleaned or ">" in cleaned or _URL_RE.search(cleaned):
        return None
    return cleaned


# --- Audit (read-only): used by `python corpus.py audit` ------------------------------------------

# Only IMPERATIVE phrases that replace instructions. The words "system prompt" alone are deliberately not counted: papers
# about LLMs discuss it in ordinary text (measured: 5 of the 189 chunks in the current corpus), that is not injection.
_INJECTION_HINT_RE = re.compile(
    r"(?:ignore|disregard|forget)\s+(?:(?:all|any|the|your)\s+)*(?:(?:previous|prior|above|earlier|preceding)\s+)?"
    r"(?:instructions?|rules|prompts?|guidelines)"
    r"|abaikan\s+(?:(?:semua|seluruh)\s+)?(?:instruksi|aturan|perintah)"
    r"|(?:reveal|show|print|repeat|tampilkan)\s+(?:(?:your|the)\s+)?system\s+prompt",
    re.IGNORECASE,
)


def audit_text(text: str, max_len: Optional[int] = None, *, names: bool = True) -> List[str]:
    """
    Return a list of reasons why a text looks suspicious (empty = looks normal).
      max_len : flag text longer than this (e.g. entity names).
      names   : True for entity names/descriptions (URLs and control characters are scored too). False for
                chunk/summary text: URLs are normal in papers, and control characters (e.g. U+0010) are only PDF font artifacts.
    """
    if not text:
        return []
    reasons = []
    if _TAG_PATTERN.search(text):
        reasons.append("mirip tag pembatas prompt")
    if _INJECTION_HINT_RE.search(text):
        reasons.append("frasa mirip instruksi/injeksi")
    if names and _URL_RE.search(text):
        reasons.append("berisi URL")
    if names and _HIDDEN_RE.search(text):
        reasons.append("karakter kontrol/tak terlihat")
    if max_len is not None and len(text) > max_len:
        reasons.append(f"lebih panjang dari {max_len} char ({len(text)})")
    return reasons
