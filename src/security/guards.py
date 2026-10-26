"""
Request-level controls for the local server (see docs/SECURITY.md). Pure ASGI, not BaseHTTPMiddleware,
so the body size can also be counted for *chunked* bodies and every check runs BEFORE the body is parsed
by FastAPI/Pydantic (which runs before the handler — a check inside the handler can be bypassed by a
deliberately invalid payload).

Order for `POST /api/chat` (reject as early and as cheaply as possible):
    body size (header) -> Content-Type -> Origin -> rate limit -> body size (read) -> concurrency

Every rejection has the shape {"answer": "..."} so the UI shows it as-is.
"""
import base64
import hashlib
import logging
import math
import re
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, Iterable, List, Tuple

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse

logger = logging.getLogger("graphrag.security")


# ── Rate limit & concurrency ────────────────────────────────────────────────────────────────

class SlidingWindowLimiter:
    """Sliding window per key (IP), in-memory. Enough for a single local process (resets on restart)."""

    def __init__(self, max_requests: int, window_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.max_requests = max_requests
        self.window = window_seconds
        self._clock = clock
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> Tuple[bool, float]:
        """Record one hit if allowed. Returns (allowed, seconds_until_allowed_again)."""
        now = self._clock()
        with self._lock:
            if len(self._hits) > 1024:  # prevent unbounded growth from many unique keys
                self._hits = {k: d for k, d in self._hits.items() if d and now - d[-1] < self.window}
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False, max(self.window - (now - hits[0]), 0.0)
            hits.append(now)
            return True, 0.0


class ConcurrencyGate:
    """Hard limit on the number of concurrently running chats (each chat = 2 LLM calls that consume quota)."""

    def __init__(self, max_concurrent: int):
        self.max_concurrent = max_concurrent
        self._active = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._active >= self.max_concurrent:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    @property
    def active(self) -> int:
        return self._active


# ── Guard for POST /api/chat ──────────────────────────────────────────────────────────────────────

MSG_TOO_LARGE = "Permintaan terlalu besar."
MSG_BAD_TYPE = "Content-Type harus application/json."
MSG_BAD_ORIGIN = "Origin tidak diizinkan."
MSG_RATE_LIMITED = "Terlalu banyak permintaan. Silakan coba lagi sebentar lagi."
MSG_BUSY = "Server sedang memproses pertanyaan lain. Silakan coba lagi sebentar lagi."


class _BodyTooLarge(Exception):
    pass


def client_ip(scope) -> str:
    client = scope.get("client")
    return client[0] if client else "unknown"


class ChatGuard:
    def __init__(
        self,
        app,
        *,
        path: str = "/api/chat",
        max_body_bytes: int,
        allowed_origins: Iterable[str],
        limiter: SlidingWindowLimiter,
        gate: ConcurrencyGate,
    ):
        self.app = app
        self.path = path
        self.max_body_bytes = max_body_bytes
        self.allowed_origins = frozenset(allowed_origins)
        self.limiter = limiter
        self.gate = gate

    async def _reject(self, scope, receive, send, status: int, message: str, reason: str, extra_headers=None):
        logger.warning("chat rejected: %s (status=%s ip=%s)", reason, status, client_ip(scope))
        resp = JSONResponse(status_code=status, content={"answer": message}, headers=extra_headers)
        await resp(scope, receive, send)

    async def _read_body(self, receive) -> bytes:
        """Read the whole body with a size limit (also for chunked bodies without Content-Length)."""
        chunks: List[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                raise _BodyTooLarge  # client disconnected midway: treat as invalid
            total += len(message.get("body", b""))
            if total > self.max_body_bytes:
                raise _BodyTooLarge
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                return b"".join(chunks)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != self.path:
            return await self.app(scope, receive, send)

        headers = Headers(scope=scope)

        # 1. Declared body size
        cl = headers.get("content-length")
        if cl is not None and cl.isdigit() and int(cl) > self.max_body_bytes:
            return await self._reject(scope, receive, send, 413, MSG_TOO_LARGE, "content-length too large")

        # 2. Content-Type must be JSON. Without this, another site could send a "simple request" (no
        #    CORS preflight) that FastAPI would still parse; with it, the browser MUST preflight, and
        #    CORS is not enabled, so that preflight fails.
        ctype = headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return await self._reject(scope, receive, send, 415, MSG_BAD_TYPE, f"content-type={ctype!r}")

        # 3. Origin (if present) must be one of our local origins. "null" (file://, sandboxed iframe) is rejected.
        origin = headers.get("origin")
        if origin is not None and origin not in self.allowed_origins:
            return await self._reject(scope, receive, send, 403, MSG_BAD_ORIGIN, f"origin={origin!r}")

        # 4. Rate limit (also applies to payloads that would fail validation: that is why it lives in the middleware)
        allowed, retry_after = self.limiter.check(client_ip(scope))
        if not allowed:
            return await self._reject(
                scope, receive, send, 429, MSG_RATE_LIMITED, "rate limited",
                {"Retry-After": str(max(1, math.ceil(retry_after)))},
            )

        # 5. Read the body with a limit (closes chunked bodies without Content-Length), then replay it to the app.
        try:
            body = await self._read_body(receive)
        except _BodyTooLarge:
            return await self._reject(scope, receive, send, 413, MSG_TOO_LARGE, "streamed body too large")

        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()  # subsequent calls: pass through (e.g. http.disconnect)

        # 6. Concurrency limit
        if not self.gate.try_acquire():
            return await self._reject(scope, receive, send, 429, MSG_BUSY, "too many concurrent chats",
                                      {"Retry-After": "5"})
        try:
            await self.app(scope, replay, send)
        finally:
            self.gate.release()


# ── Security headers + CSP ──────────────────────────────────────────────────────────────────

_SCRIPT_RE = re.compile(r"<script\b([^>]*)>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL)
_HAS_SRC_RE = re.compile(r"\bsrc\s*=", re.IGNORECASE)


def inline_script_hashes(html: str) -> List[str]:
    """
    sha256 (base64) of the content of every inline <script> block. Used for a CSP `script-src` without
    'unsafe-inline'. Newlines are normalized to LF: the HTML parser does that before the browser computes the hash.
    """
    hashes = []
    for attrs, body in _SCRIPT_RE.findall(html):
        if _HAS_SRC_RE.search(attrs):
            continue  # external scripts are not allowed at all by our CSP
        text = body.replace("\r\n", "\n").replace("\r", "\n")
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        hashes.append(base64.b64encode(digest).decode("ascii"))
    return hashes


def build_index_csp(html: str) -> str:
    """CSP for the main page. style-src 'unsafe-inline' is kept (the panel uses style= attributes)."""
    hashes = " ".join(f"'sha256-{h}'" for h in inline_script_hashes(html)) or "'none'"
    return (
        "default-src 'none'; "
        f"script-src {hashes}; "
        "style-src 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self' data:; "
        "connect-src 'self'; "
        "base-uri 'none'; form-action 'none'; object-src 'none'; frame-ancestors 'none'"
    )


API_CSP = "default-src 'none'; frame-ancestors 'none'"


class SecurityHeaders:
    """Outermost: headers are attached to ALL responses, including 400/403/413/415/429 from other layers."""

    def __init__(self, app, *, csp_index: str = API_CSP, index_path: str = "/", no_store_prefix: str = "/api/"):
        self.app = app
        self.csp_index = csp_index
        self.index_path = index_path
        self.no_store_prefix = no_store_prefix

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                h = MutableHeaders(scope=message)
                h["X-Content-Type-Options"] = "nosniff"
                h["Referrer-Policy"] = "no-referrer"
                h["X-Frame-Options"] = "DENY"
                h["Cross-Origin-Resource-Policy"] = "same-origin"
                h["Content-Security-Policy"] = self.csp_index if path == self.index_path else API_CSP
                if path.startswith(self.no_store_prefix):
                    h["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, receive, send_with_headers)
