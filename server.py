"""
Web server for the GraphRAG Explorer.

Serves the interactive graph visualization at  /  and answers chatbot questions at
POST /api/chat, reusing the same query pipeline as `python main.py query`.

Run:
    python server.py
It loads the graph + embeddings + LLM once, then opens http://127.0.0.1:8000 in
your browser. Requires a completed build (data/processed/graph.graphml).

Note: each chat message makes 2 LLM calls (Gemini, or Groq once Gemini's daily quota is
gone), so it consumes daily free-tier quota — just like the CLI query. Budget stays $0 (local server, local embeddings).

Keamanan (lihat docs/SECURITY.md): server ini untuk pemakaian LOKAL. Host allow-list tetap, guard
request (ukuran/Content-Type/Origin/rate limit/concurrency), header keamanan + CSP berbasis hash,
error generik ke klien. Jangan di-expose ke internet.
"""
import logging
import os
import threading
import time
import webbrowser

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src import config
from src.llm.llm_service import DailyQuotaExceededError, EmptyResponseError
from src.security.integrity import ArtifactIntegrityError
from src.security.guards import (
    ChatGuard,
    ConcurrencyGate,
    SecurityHeaders,
    SlidingWindowLimiter,
    build_index_csp,
)

GRAPH_PATH = "data/processed/graph.graphml"
HOST, PORT = "127.0.0.1", 8000

logger = logging.getLogger("graphrag.server")

_NOT_BUILT_HTML = "<h1>Graph not built. Run 'python main.py build' first.</h1>"

MSG_INVALID = (
    f"Pertanyaan tidak valid. Pastikan tidak kosong dan maksimal {config.MAX_CHAT_QUERY_LENGTH} karakter."
)
MSG_QUOTA = "Kuota harian LLM habis. Coba lagi besok (kuota Gemini reset sekitar tengah malam waktu Pasifik)."
MSG_EMPTY = "Model tidak menghasilkan jawaban (kemungkinan diblokir filter keamanan). Coba ubah pertanyaan Anda."
MSG_ERROR = "Maaf, terjadi kesalahan saat memproses pertanyaan. Silakan coba lagi."


class ChatRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def _validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("empty")
        if len(v) > config.MAX_CHAT_QUERY_LENGTH:
            raise ValueError("too long")
        return v


def allowed_origins(port: int = PORT) -> list:
    return [f"http://{h}:{port}" for h in config.ALLOWED_HOSTS]


def create_app(engine, html: str = _NOT_BUILT_HTML, *, limiter=None, gate=None) -> FastAPI:
    """
    Bangun aplikasi. `engine` cukup punya `.query(str) -> str`. Dipisah dari main() supaya bisa dites
    tanpa memuat torch/graph. `limiter`/`gate` bisa disuntik oleh tes.
    """
    # docs/redoc/openapi dimatikan: tidak dipakai UI dan tidak perlu terlihat.
    app = FastAPI(title="GraphRAG Explorer", docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc):
        # Pesan tetap (tidak memantulkan input) dalam bentuk {"answer": ...} yang dipahami UI.
        return JSONResponse(status_code=422, content={"answer": MSG_INVALID})

    @app.get("/", response_class=HTMLResponse)
    def index():
        return html

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        start = time.monotonic()
        status, body = 200, None
        try:
            body = {"answer": engine.query(req.query)}
        except DailyQuotaExceededError:
            status, body = 429, {"answer": MSG_QUOTA}
        except EmptyResponseError:
            status, body = 503, {"answer": MSG_EMPTY}
        except Exception:  # never crash the endpoint; never leak internals to the client
            logger.exception("chat failed")  # detail + traceback hanya di log server
            status, body = 503, {"answer": MSG_ERROR}
        # Satu baris log per chat, TANPA teks pertanyaan/jawaban (privasi).
        logger.info("chat status=%s latency_ms=%d", status, (time.monotonic() - start) * 1000)
        return JSONResponse(status_code=status, content=body)

    # Urutan: yang ditambah TERAKHIR = lapisan terluar. SecurityHeaders paling luar supaya header
    # juga terpasang di penolakan 400/403/413/415/429 dari lapisan di dalamnya.
    app.add_middleware(
        ChatGuard,
        path="/api/chat",
        max_body_bytes=config.MAX_CHAT_BODY_BYTES,
        allowed_origins=allowed_origins(),
        limiter=limiter or SlidingWindowLimiter(config.CHAT_RATE_LIMIT_MAX, config.CHAT_RATE_LIMIT_WINDOW_SECONDS),
        gate=gate or ConcurrencyGate(config.CHAT_MAX_CONCURRENT),
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(config.ALLOWED_HOSTS))  # anti DNS rebinding
    app.add_middleware(SecurityHeaders, csp_index=build_index_csp(html))
    return app


def main():
    import networkx as nx
    import uvicorn

    # Root at WARNING (third-party INFO such as httpx request lines is noise); our own loggers at INFO.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("graphrag.server", "graphrag.security"):
        logging.getLogger(name).setLevel(logging.INFO)

    # Validasi konfigurasi hanya di server (evaluate.py mode retrieval tidak butuh key).
    if not config.GEMINI_API_KEY:
        print("[Error] GEMINI_API_KEY belum diisi. Salin .env.example menjadi .env dan isi key Anda.")
        return
    if not os.path.exists(GRAPH_PATH):
        print(f"[Error] {GRAPH_PATH} not found. Run 'python main.py build' first.")
        return

    # Import berat (torch, chromadb) ditunda ke sini supaya `import server` cepat dan bisa dites.
    from src.engine import GraphRAGEngine
    from src.graph.visualizer import GraphVisualizer

    print("Loading graph + embeddings + LLM (one-time startup)...")
    try:
        engine = GraphRAGEngine(GRAPH_PATH)
    except ArtifactIntegrityError as e:
        print(f"\n[Error] {e}")
        return
    html = GraphVisualizer(nx.read_graphml(GRAPH_PATH)).build_html()
    app = create_app(engine, html)
    print(f"Ready. Serving GraphRAG Explorer at http://{HOST}:{PORT}  (Ctrl+C to stop)")

    threading.Timer(1.2, lambda: webbrowser.open(f"http://{HOST}:{PORT}/")).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
