"""
Web server for the GraphRAG Explorer.

Serves the interactive graph visualization at  /  and answers chatbot questions at
POST /api/chat, reusing the same query pipeline as `python main.py query`.

Run:
    python server.py
It loads the graph + embeddings + LLM once, then opens http://127.0.0.1:8000 in
your browser. Requires a completed build (data/processed/graph.graphml).

Note: each chat message calls Gemini, so it consumes daily free-tier quota — just
like the CLI query. Budget stays $0 (local server, local embeddings).
"""
import os
import threading
import webbrowser

import networkx as nx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from src.graph.visualizer import GraphVisualizer
from src.engine import GraphRAGEngine

GRAPH_PATH = "data/processed/graph.graphml"
HOST, PORT = "127.0.0.1", 8000

app = FastAPI(title="GraphRAG Explorer")

# Loaded once in main() before the server starts (single process, no reload/workers,
# so these module globals are shared with the request handlers).
_engine: GraphRAGEngine | None = None
_html: str = "<h1>Graph not built. Run 'python main.py build' first.</h1>"


class ChatRequest(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
def index():
    return _html


@app.post("/api/chat")
def chat(req: ChatRequest):
    q = (req.query or "").strip()
    if not q:
        return {"answer": "Silakan ketik pertanyaan terlebih dahulu."}
    if _engine is None:
        return {"answer": "Engine belum siap, coba lagi sebentar."}
    try:
        return {"answer": _engine.query(q)}
    except Exception as e:  # never crash the endpoint on a bad query / quota error
        return {"answer": f"Maaf, terjadi kesalahan saat memproses pertanyaan: {e}"}


def main():
    global _engine, _html
    if not os.path.exists(GRAPH_PATH):
        print(f"[Error] {GRAPH_PATH} not found. Run 'python main.py build' first.")
        return

    print("Loading graph + embeddings + LLM (one-time startup)...")
    _engine = GraphRAGEngine(GRAPH_PATH)
    _html = GraphVisualizer(nx.read_graphml(GRAPH_PATH)).build_html()
    print(f"Ready. Serving GraphRAG Explorer at http://{HOST}:{PORT}  (Ctrl+C to stop)")

    threading.Timer(1.2, lambda: webbrowser.open(f"http://{HOST}:{PORT}/")).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
