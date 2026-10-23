"""
Kalibrasi CHUNK_MIN_SIMILARITY / COMMUNITY_MIN_SIMILARITY (ambang relevansi jalur vektor), TANPA kuota LLM.

Untuk setiap kasus eval yang memakai jalur vektor (thematic -> ringkasan komunitas; entity yang tidak
ter-resolve -> fallback chunk) dicetak cosine top-1 dan top-k. Kasus `expected_empty` adalah negatif
(semua hasil harus gugur); kasus yang tidak `expected_refuse` adalah positif (top-1 harus lolos).
Ambang yang aman berada di antara negatif tertinggi dan positif terendah. Pertanyaan riset yang tidak ada
di korpus (mis. ResNet) sengaja tidak dihitung negatif: itu urusan aturan prompt jawaban.

Usage:
    PYTHONUTF8=1 python -m scripts.calibrate_retrieval
"""
import contextlib
import io
import json

import networkx as nx

from evaluate import CASES_PATH, GRAPH_PATH
from src.config import CHUNK_MIN_SIMILARITY, COMMUNITY_MIN_SIMILARITY
from src.llm.embedding import EmbeddingClient
from src.query.resolver import QueryResolver, l2_to_cosine
from src.vectorstore.chromadb_store import ChromaStore


def main():
    with open(CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)
    embed, chroma = EmbeddingClient(), ChromaStore()
    resolver = QueryResolver(embed, chroma, nx.read_graphml(GRAPH_PATH))

    rows = {"chunk": [], "community": []}
    for c in cases:
        if c["intent"] == "thematic":
            path = "community"
        elif c["intent"] == "entity":
            with contextlib.redirect_stdout(io.StringIO()):  # resolver mencetak log per lapis
                if resolver.resolve_entities(c["q"], context_query=c["q"]):
                    continue  # ter-resolve ke node: jalur graph, ambang tidak berlaku
            path = "chunk"
        else:
            continue
        if c.get("expected_empty"):
            label = "NEG"
        elif not c.get("expected_refuse") and not c.get("e2e_only"):
            label = "POS"
        else:
            label = "---"  # refusal tanpa expected_empty: informatif saja
        emb = embed.embed_text(c["q"])
        res = chroma.search_chunks(emb, 5) if path == "chunk" else chroma.search_communities(emb, 3)
        scores = [l2_to_cosine(d) for d in res["distances"][0]]
        rows[path].append((label, scores[0], scores[-1], c["id"]))

    for path, configured in (("chunk", CHUNK_MIN_SIMILARITY), ("community", COMMUNITY_MIN_SIMILARITY)):
        print(f"\n== {path} (ambang sekarang {configured}) ==")
        for label, top, last, cid in sorted(rows[path], key=lambda r: -r[1]):
            print(f"  {label}  top={top:.3f}  last={last:.3f}  {cid}")
        pos = [r[1] for r in rows[path] if r[0] == "POS"]
        neg = [r[1] for r in rows[path] if r[0] == "NEG"]
        if pos and neg:
            lo, hi = max(neg), min(pos)
            verdict = f"celah {hi - lo:+.3f}, titik tengah {(lo + hi) / 2:.3f}" if hi > lo else "TUMPANG-TINDIH"
            print(f"  negatif tertinggi {lo:.3f} | positif terendah {hi:.3f} | {verdict}")


if __name__ == "__main__":
    main()
