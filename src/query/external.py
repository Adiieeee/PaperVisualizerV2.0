"""
External background for an entity the papers name but do not explain ("Apa itu MMLU?").

Sources: the English Wikipedia (one search request returns the intro extracts of the top pages), falling
back to arXiv abstracts for research terms Wikipedia has no page for (Semantic Scholar was dropped: without
an API key it answered every request with 429). Search results are often a different thing with the same
words ("Activation steering" -> "Steering wheel", "Gemini" -> the constellation), so a candidate is used
only when its title names the entity (`title_tier`) AND its text is close enough (bge-m3 cosine >=
EXTERNAL_MIN_SIMILARITY) to what the graph says about the entity; disambiguation pages are never used.

Only the entity NAME leaves the machine, never the user's question. Every network/parse error yields None:
a failed lookup must not fail the answer. Results are cached per corpus (EXTERNAL_CACHE_PATH) RAW; like any
document text they are neutralized only when the answer prompt is built (src/llm/prompts.py).
This module imports no torch: corpus.py uses its cache helpers.
"""
import json
import re
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Callable, List, Optional

from src import config
from src.security.integrity import atomic_write_json

WIKI_API = "https://en.wikipedia.org/w/api.php"
ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_MAX_RESPONSE_BYTES = 1_000_000  # a search answer is ~10-50 KB; anything far larger is not read
_CANDIDATES = 3
CACHE_VERSION = 1

SOURCE_LABELS = {"wikipedia": "Wikipedia", "arxiv": "arXiv"}


def http_get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": config.EXTERNAL_USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(_MAX_RESPONSE_BYTES)


def _clip(text: str, limit: int) -> str:
    """At most `limit` chars, cut at the last sentence end when there is one in the second half."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = cut.rfind(". ")
    return cut[:end + 1] if end >= limit // 2 else cut.rstrip() + "…"


def wikipedia_candidates(name: str, timeout: float, get: Callable = http_get) -> List[dict]:
    params = {
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
        "generator": "search", "gsrsearch": name, "gsrlimit": _CANDIDATES,
        "prop": "extracts|description|info|pageprops", "exintro": 1, "explaintext": 1, "exlimit": "max",
        "inprop": "url", "ppprop": "disambiguation",
    }
    data = json.loads(get(f"{WIKI_API}?{urllib.parse.urlencode(params)}", timeout))
    pages = sorted((data.get("query") or {}).get("pages") or [], key=lambda p: p.get("index", 99))
    out = []
    for p in pages:
        if "disambiguation" in (p.get("pageprops") or {}) or not p.get("extract"):
            continue
        out.append({
            "source": "wikipedia", "title": p.get("title", ""), "url": p.get("fullurl", ""),
            "revision": p.get("lastrevid"), "description": p.get("description") or "",
            "extract": _clip(p["extract"], config.EXTERNAL_MAX_CHARS),
        })
    return out


def arxiv_candidates(name: str, timeout: float, get: Callable = http_get) -> List[dict]:
    phrase = name.replace('"', " ")
    params = {"search_query": f'ti:"{phrase}"', "max_results": _CANDIDATES, "sortBy": "relevance"}
    root = ET.fromstring(get(f"{ARXIV_API}?{urllib.parse.urlencode(params)}", timeout))
    out = []
    for e in root.findall(f"{_ATOM}entry"):
        summary, title = e.findtext(f"{_ATOM}summary") or "", e.findtext(f"{_ATOM}title") or ""
        if not summary.strip():
            continue
        out.append({
            "source": "arxiv", "title": _clip(title, 300),
            "url": (e.findtext(f"{_ATOM}id") or "").strip().replace("http://", "https://", 1),
            "revision": (e.findtext(f"{_ATOM}published") or "")[:10], "description": "",
            "extract": _clip(summary, config.EXTERNAL_MAX_CHARS),
        })
    return out


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _base_title(c: dict) -> str:
    """Title without its qualifier: "Gemini (language model)" -> "Gemini", "TruthfulQA: Measuring ..." -> "TruthfulQA"."""
    if c["source"] == "arxiv":
        return c["title"].split(":", 1)[0]
    return re.sub(r"\s*\([^)]*\)\s*$", "", c["title"])


def title_tier(name: str, c: dict) -> int:
    """How well a candidate's title names the entity. 2: same name ("Gemini (language model)" for Gemini,
    "TruthfulQA: ..." for TruthfulQA); 1: the title contains every word of the name ("Accuracy and precision"),
    or the name is the title plus version-like words ("Claude 2" -> "Claude (AI)"); 0: another thing.
    Measured (2026-10-29): cosine alone ranked "Llama.cpp" above "Llama (language model)" and let
    "Steering wheel" through for "contrastive activation steering"."""
    name_t, base_t = _tokens(name), _tokens(_base_title(c))
    if not name_t or not base_t:
        return 0
    if base_t == name_t:
        return 2
    extra = name_t - base_t
    if name_t <= _tokens(c["title"]) or (base_t < name_t and all(any(ch.isdigit() for ch in w) or len(w) <= 6
                                                                  for w in extra)):
        return 1
    return 0


def candidate_text(c: dict) -> str:
    head = f"{c['title']} ({c['description']})" if c.get("description") else c["title"]
    return f"{head}: {c['extract']}"


def format_external(entry: dict) -> str:
    """One context line per entity: `[Wikipedia: <title>] <extract> (<url>)`."""
    return f"[{SOURCE_LABELS.get(entry['source'], entry['source'])}: {entry['title']}] {entry['extract']} ({entry['url']})"


# --- Cache (also used by corpus.py) ---------------------------------------------------------------

def load_cache(path: str) -> dict:
    """{entity name: entry}. A missing, corrupt or other-version file is an empty cache (it is only a cache)."""
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        return {}
    entries = payload.get("entries")
    return entries if isinstance(entries, dict) else {}


def save_cache(path: str, entries: dict) -> None:
    atomic_write_json(path, {"version": CACHE_VERSION, "entries": entries}, ensure_ascii=False, indent=1)


class ExternalContext:
    """Looks up, scores and caches the background text of an entity. Thread-safe (lookups run beside the
    graph traversal, two chats can run at once)."""

    def __init__(self, embed, cache_path: str = None, get: Callable = http_get, clock: Callable = time.time):
        self.embed = embed
        self.cache_path = cache_path or config.EXTERNAL_CACHE_PATH
        self.get = get
        self.clock = clock
        self._lock = threading.Lock()
        self._cache = None

    def _entries(self) -> dict:
        if self._cache is None:
            self._cache = load_cache(self.cache_path)
        return self._cache

    def _fresh(self, entry: dict) -> bool:
        days = config.EXTERNAL_TTL_DAYS if entry.get("status") == "ok" else config.EXTERNAL_NOT_FOUND_TTL_DAYS
        return self.clock() - float(entry.get("fetched_at", 0)) < days * 86400

    def _best(self, name: str, node_text: str, candidates: List[dict]) -> Optional[dict]:
        """The candidate with the best title match, then the highest cosine to the graph's text."""
        tiered = [(title_tier(name, c), c) for c in candidates]
        # arXiv: only the paper NAMED after the entity ("TruthfulQA: ..."). Papers that merely mention it
        # were measured to be about another sense ("negative prompting" -> diffusion models) or a niche variant.
        tiered = [(t, c) for t, c in tiered if t >= (2 if c["source"] == "arxiv" else 1)]
        if not tiered:
            return None
        vecs = self.embed.embed_batch([node_text] + [candidate_text(c) for _t, c in tiered])
        q = vecs[0]
        scored = [(t, sum(a * b for a, b in zip(q, v)), c) for v, (t, c) in zip(vecs[1:], tiered)]
        _tier, score, best = max(scored, key=lambda x: (x[0], x[1]))
        if score < config.EXTERNAL_MIN_SIMILARITY:
            print(f"[External] best candidate '{best['title']}' too far ({score:.3f})")
            return None
        return {**best, "score": round(score, 4)}

    def lookup(self, name: str, node_text: str, timeout: float = None) -> Optional[dict]:
        """Background entry for `name` ({source, title, url, extract, ...}) or None. `node_text` is what
        the graph says about the entity ("MMLU: A question-answering dataset."), used to pick the right page."""
        with self._lock:
            entry = self._entries().get(name)
        if entry and self._fresh(entry):
            return entry if entry.get("status") == "ok" else None

        deadline = time.monotonic() + (timeout or config.EXTERNAL_TIMEOUT)
        # "Direct Preference Optimization (DPO)": search and match without the bracketed acronym.
        query = re.sub(r"\s*\([^)]*\)", "", name).strip() or name
        found = None
        try:
            for fetch in (wikipedia_candidates, arxiv_candidates):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None  # out of time: not cached, the next question tries again
                found = self._best(query, node_text, fetch(query, remaining, self.get))
                if found:
                    break
        except Exception as e:  # network, HTTP 429/5xx, bad JSON/XML: transient, so not cached
            print(f"[External] lookup for '{name}' failed: {type(e).__name__}: {str(e)[:120]}")
            return None

        new = ({**found, "status": "ok"} if found else {"status": "not_found"})
        new["fetched_at"] = self.clock()
        with self._lock:
            self._entries()[name] = new
            try:
                save_cache(self.cache_path, self._cache)
            except OSError as e:
                print(f"[External] cache not saved: {e}")
        return new if found else None
