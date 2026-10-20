import asyncio
import base64
import hashlib

import pytest
from starlette.testclient import TestClient

import server
from src import config
from src.llm.gemini_client import DailyQuotaExceededError, EmptyResponseError
from src.security.guards import (
    ChatGuard,
    ConcurrencyGate,
    SlidingWindowLimiter,
    build_index_csp,
    inline_script_hashes,
)

BASE = "http://127.0.0.1:8000"
HTML = "<html><body><h1>hi</h1><script>var a = 1;</script></body></html>"


class FakeEngine:
    def __init__(self, exc=None):
        self.calls, self.exc = [], exc

    def query(self, q):
        self.calls.append(q)
        if self.exc:
            raise self.exc
        return f"jawaban untuk: {q}"


class SpyLimiter(SlidingWindowLimiter):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.checks = 0

    def check(self, key):
        self.checks += 1
        return super().check(key)


def make(engine=None, limiter=None, gate=None):
    engine = engine or FakeEngine()
    limiter = limiter or SpyLimiter(100, 60)
    app = server.create_app(engine, HTML, limiter=limiter, gate=gate or ConcurrencyGate(2))
    return TestClient(app, base_url=BASE), engine, limiter


def post(client, payload='{"query": "halo"}', headers=None):
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    return client.post("/api/chat", content=payload, headers=h)


# --- jalur normal ------------------------------------------------------------------------------

def test_normal_chat_ok_and_headers():
    c, engine, limiter = make()
    r = post(c)
    assert r.status_code == 200 and r.json() == {"answer": "jawaban untuk: halo"}
    assert engine.calls == ["halo"] and limiter.checks == 1
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]


def test_allowed_origins_pass():
    c, engine, _ = make()
    for origin in ("http://127.0.0.1:8000", "http://localhost:8000"):
        assert post(c, headers={"Origin": origin}).status_code == 200
    assert len(engine.calls) == 2


def test_index_serves_html_with_hash_csp():
    c, _, _ = make()
    r = c.get("/")
    assert r.status_code == 200 and "<h1>hi</h1>" in r.text
    assert r.headers["content-security-policy"] == build_index_csp(HTML)
    assert "'unsafe-inline'" not in r.headers["content-security-policy"].split("style-src")[0]  # script-src
    assert "script-src 'sha256-" in r.headers["content-security-policy"]


def test_docs_and_openapi_are_disabled():
    c, _, _ = make()
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert c.get(path).status_code == 404


# --- urutan penolakan; engine & limiter tidak tersentuh oleh penolakan murah -------------------------

def test_413_declared_size_before_anything_else():
    c, engine, limiter = make()
    big = '{"query": "' + "a" * 20000 + '"}'
    r = post(c, big, headers={"Origin": "https://evil.example"})  # Origin buruk pun kalah oleh 413
    assert r.status_code == 413 and "answer" in r.json()
    assert limiter.checks == 0 and engine.calls == []


@pytest.mark.parametrize("ctype", [None, "text/plain", "application/x-www-form-urlencoded", "multipart/form-data"])
def test_415_content_type_required(ctype):
    c, engine, limiter = make()
    headers = {"Content-Type": ctype} if ctype else {}
    r = c.post("/api/chat", content='{"query": "halo"}', headers=headers)
    assert r.status_code == 415 and r.json()["answer"]
    assert limiter.checks == 0 and engine.calls == []
    assert r.headers["x-content-type-options"] == "nosniff"  # header ikut terpasang di penolakan


def test_drive_by_from_other_site_never_reaches_engine():
    """Simulasi F3: situs lain menembak /api/chat. Tanpa Content-Type -> 415; dengan JSON -> Origin 403."""
    c, engine, limiter = make()
    origin = {"Origin": "http://evil.example"}
    no_ctype = c.post("/api/chat", content='{"query": "kuras kuota"}', headers=origin)
    assert no_ctype.status_code == 415
    with_json = post(c, '{"query": "kuras kuota"}', headers=origin)
    assert with_json.status_code == 403
    assert engine.calls == [] and limiter.checks == 0


@pytest.mark.parametrize("origin", ["null", "https://evil.example", "http://127.0.0.1:9999", "http://127.0.0.1.evil.com:8000"])
def test_403_bad_origin(origin):
    c, engine, limiter = make()
    r = post(c, headers={"Origin": origin})
    assert r.status_code == 403
    assert engine.calls == [] and limiter.checks == 0


def test_invalid_payload_still_consumes_rate_limit_and_never_calls_engine():
    """Bug playbook A1: cek di handler dilewati payload invalid. Di sini limiter jalan SEBELUM validasi."""
    c, engine, limiter = make()
    r = post(c, '{"query": "' + "x" * (config.MAX_CHAT_QUERY_LENGTH + 1) + '"}')
    assert r.status_code == 422 and limiter.checks == 1 and engine.calls == []
    assert r.json() == {"answer": server.MSG_INVALID}
    assert "xxxx" not in r.text  # tidak memantulkan input


@pytest.mark.parametrize("payload", ['{"query": ""}', '{"query": "   "}', '{}', 'bukan json', '{"query": 5}'])
def test_422_for_bad_payloads_with_uniform_message(payload):
    c, engine, limiter = make()
    r = post(c, payload)
    assert r.status_code == 422 and r.json() == {"answer": server.MSG_INVALID}
    assert limiter.checks == 1 and engine.calls == []


def test_429_rate_limit_with_retry_after():
    c, engine, _ = make(limiter=SlidingWindowLimiter(2, 60))
    assert post(c).status_code == 200 and post(c).status_code == 200
    r = post(c)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert len(engine.calls) == 2


def test_429_when_concurrency_gate_is_full():
    gate = ConcurrencyGate(1)
    assert gate.try_acquire()  # satu chat sedang berjalan
    c, engine, _ = make(gate=gate)
    r = post(c)
    assert r.status_code == 429 and engine.calls == []
    gate.release()
    assert post(c).status_code == 200


def test_gate_is_released_after_engine_error():
    gate = ConcurrencyGate(1)
    c, _, _ = make(engine=FakeEngine(exc=RuntimeError("boom")), gate=gate)
    assert post(c).status_code == 503
    assert gate.active == 0


# --- Host allow-list (anti DNS rebinding) ------------------------------------------------------------

def test_wrong_host_is_rejected_for_all_routes():
    c, engine, _ = make()
    assert c.get("/", headers={"Host": "evil.example"}).status_code == 400
    assert post(c, headers={"Host": "evil.example"}).status_code == 400
    assert engine.calls == []


def test_dns_rebinding_same_host_and_origin_is_still_rejected():
    """Kasus yang lolos dari cek 'Origin == Host request' milik playbook: keduanya milik penyerang."""
    c, engine, _ = make()
    r = post(c, headers={"Host": "evil.example:8000", "Origin": "http://evil.example:8000"})
    assert r.status_code == 400 and engine.calls == []
    assert r.headers["x-frame-options"] == "DENY"  # header keamanan juga di respons 400 dari TrustedHost


def test_localhost_host_is_accepted():
    c, _, _ = make()
    assert c.get("/", headers={"Host": "localhost:8000"}).status_code == 200


# --- error tidak bocor -----------------------------------------------------------------------------------

def test_internal_errors_are_generic_and_logged(caplog):
    secret = r"C:\Users\adian\.env GEMINI_API_KEY=AQ.SECRET traceback"
    c, _, _ = make(engine=FakeEngine(exc=RuntimeError(secret)))
    with caplog.at_level("ERROR"):
        r = post(c)
    assert r.status_code == 503 and r.json() == {"answer": server.MSG_ERROR}
    assert "SECRET" not in r.text and "adian" not in r.text
    assert any("chat failed" in rec.message for rec in caplog.records)  # detail hanya di log server


def test_quota_and_empty_response_have_fixed_messages():
    c, _, _ = make(engine=FakeEngine(exc=DailyQuotaExceededError("[DAILY QUOTA] detail internal")))
    r = post(c)
    assert r.status_code == 429 and r.json() == {"answer": server.MSG_QUOTA}
    c, _, _ = make(engine=FakeEngine(exc=EmptyResponseError("blocked")))
    r = post(c)
    assert r.status_code == 503 and r.json() == {"answer": server.MSG_EMPTY}


def test_log_line_never_contains_the_question(caplog):
    c, _, _ = make()
    with caplog.at_level("INFO", logger="graphrag.server"):
        post(c, '{"query": "pertanyaan-rahasia-pengguna"}')
    text = " ".join(rec.getMessage() for rec in caplog.records)
    assert "status=200" in text and "pertanyaan-rahasia-pengguna" not in text


# --- body chunked (tanpa Content-Length): dites langsung di level ASGI -------------------------------------

def _run_asgi(app, headers, chunks):
    sent, queue = [], list(chunks)

    async def receive():
        if queue:
            body = queue.pop(0)
            return {"type": "http.request", "body": body, "more_body": bool(queue)}
        await asyncio.sleep(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/api/chat", "headers": headers, "client": ("127.0.0.1", 5555)}
    asyncio.run(app(scope, receive, send))
    return sent


def _guard(inner, max_body=1000):
    return ChatGuard(
        inner, path="/api/chat", max_body_bytes=max_body, allowed_origins=["http://127.0.0.1:8000"],
        limiter=SlidingWindowLimiter(100, 60), gate=ConcurrencyGate(2),
    )


def test_chunked_body_over_limit_is_413_and_app_never_runs():
    ran = []

    async def inner(scope, receive, send):
        ran.append(True)

    headers = [(b"content-type", b"application/json"), (b"transfer-encoding", b"chunked")]  # tanpa content-length
    sent = _run_asgi(_guard(inner), headers, [b"a" * 600, b"b" * 600])
    assert sent[0]["status"] == 413 and ran == []


def test_chunked_body_within_limit_is_replayed_intact_to_app():
    got = []

    async def inner(scope, receive, send):
        body = b""
        while True:
            m = await receive()
            body += m.get("body", b"")
            if not m.get("more_body"):
                break
        got.append(body)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    headers = [(b"content-type", b"application/json"), (b"transfer-encoding", b"chunked")]
    sent = _run_asgi(_guard(inner), headers, [b'{"query":', b' "halo"}'])
    assert got == [b'{"query": "halo"}'] and sent[0]["status"] == 200


# --- unit: limiter / gate / CSP ---------------------------------------------------------------------------

def test_sliding_window_limiter_uses_window_and_keys():
    t = [0.0]
    lim = SlidingWindowLimiter(2, 10, clock=lambda: t[0])
    assert lim.check("a")[0] and lim.check("a")[0]
    ok, retry = lim.check("a")
    assert not ok and 0 < retry <= 10
    assert lim.check("b")[0]                # kunci lain tidak terpengaruh
    t[0] = 10.5
    assert lim.check("a")[0]                # jendela bergeser


def test_inline_script_hashes_skip_external_and_normalize_newlines():
    html = (
        '<script src="https://cdn.example/x.js"></script>'
        "<script>a();\r\nb();</script>"
        '<script type="text/javascript">c();</script>'
    )
    hashes = inline_script_hashes(html)
    expect = lambda s: base64.b64encode(hashlib.sha256(s.encode()).digest()).decode()
    assert hashes == [expect("a();\nb();"), expect("c();")]
    assert build_index_csp("<p>tanpa skrip</p>").count("script-src 'none'") == 1
