"""
Calibrate the thresholds tied to the embedding model, WITHOUT LLM quota and without touching the real artifacts.

  (a) DEDUP_SIMILARITY_THRESHOLD — re-score every pair in similarity_debug.csv (the last build's output,
      with the LLM decisions) using the candidate model: how many candidates would be sent to the LLM
      and how many LLM-confirmed merges are still caught, per threshold.
  (b) RESOLVER_SIM_THRESHOLD + retrieval quality — a TEMPORARY Chroma (chunks from data/raw + community
      summaries from the current Chroma) is embedded with the candidate model, then evaluate.py's retrieval
      logic is run per threshold. Negative/security questions are checked: does the embedding layer map them wrongly.
  (c) Embedding time for all chunks and CPU vs GPU vector consistency.

Note: the graph used is the current graph (the result of dedup with the old model); the final result must still be
verified with `python evaluate.py` after a rebuild.

Usage:
    PYTHONUTF8=1 python -m scripts.calibrate_embedding --model BAAI/bge-m3 --revision <sha>
"""
import argparse
import contextlib
import csv
import io
import json
import os
import re
import time

import networkx as nx
import numpy as np

import evaluate
import src.query.resolver as R
from src.llm import embedding as E
from src.graph.analytics import GraphAnalytics
from src.ingestion.document import DocumentProcessor
from src.query.traversal import GraphTraversal
from src.vectorstore.chromadb_store import ChromaStore

DEBUG_CSV = "data/processed/similarity_debug.csv"
DEDUP_GRID = [round(0.70 + 0.025 * i, 3) for i in range(11)]      # 0.70 .. 0.95
# 0.55 .. 0.85, plus the configured value (so a run with the current model reproduces evaluate.py)
RESOLVER_GRID = sorted({round(0.55 + 0.05 * i, 2) for i in range(7)} | {R.RESOLVER_SIM_THRESHOLD})


def load_client(model, revision):
    E.EMBEDDING_MODEL_NAME, E.EMBEDDING_MODEL_REVISION = model, revision
    return E.EmbeddingClient()


# ── (a) dedup ───────────────────────────────────────────────────────────────────────────────

def calibrate_dedup(client, path=DEBUG_CSV):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = sorted({r["entity_baru"] for r in rows} | {r["entity_kanonik"] for r in rows})
    vecs = dict(zip(names, np.array(client.embed_batch(names))))
    groups = {"merge (LLM ya)": [], "ditolak LLM": [], "tak dikirim ke LLM": []}
    for r in rows:
        s = float(np.dot(vecs[r["entity_baru"]], vecs[r["entity_kanonik"]]))
        d = r["llm_decision"]
        key = "merge (LLM ya)" if "is_duplicate=True" in d else ("ditolak LLM" if d else "tak dikirim ke LLM")
        groups[key].append((s, float(r["similarity_score"]), r["entity_baru"], r["entity_kanonik"]))

    print(f"\n(a) DEDUP — {len(rows)} pasangan, {len(names)} nama unik")
    for g, items in groups.items():
        sc = np.array([x[0] for x in items])
        print(f"  {g:20} n={len(items):5}  min={sc.min():.3f}  p10={np.percentile(sc, 10):.3f}  "
              f"median={np.median(sc):.3f}  p90={np.percentile(sc, 90):.3f}  max={sc.max():.3f}")
    merges = groups["merge (LLM ya)"]
    print(f"\n  {'T':>6} {'kandidat ke LLM':>16} {'merge tertangkap':>17} {'ditolak LLM ikut':>17}")
    table = []
    for t in DEDUP_GRID:
        cand = sum(1 for items in groups.values() for x in items if x[0] >= t)
        kept = sum(1 for x in merges if x[0] >= t)
        rej = sum(1 for x in groups["ditolak LLM"] if x[0] >= t)
        table.append({"T": t, "candidates": cand, "merges_kept": kept, "rejected_kept": rej})
        print(f"  {t:6.3f} {cand:16} {kept:>12}/{len(merges):<4} {rej:17}")
    print("\n  5 merge terkonfirmasi dengan skor terendah (model kandidat | skor lama):")
    for s, old, a, b in sorted(merges)[:5]:
        print(f"    {s:.3f} | {old:.3f}  {a!r} ~ {b!r}")
    return {"groups": {g: len(v) for g, v in groups.items()}, "table": table}


# ── (b) resolver + retrieval ────────────────────────────────────────────────────────────────

def build_temp_store(client, out_dir):
    chunks = DocumentProcessor().process_directory("data/raw")
    live = ChromaStore().community_collection.get(include=["documents"])
    store = ChromaStore(persist_directory=os.path.join(out_dir, "chroma"))
    t0 = time.perf_counter()
    store.add_chunks(chunks, client.embed_batch([c["text"] for c in chunks]))
    secs = time.perf_counter() - t0
    summaries = [{"community_id": i, "summary": d} for i, d in zip(live["ids"], live["documents"])]
    if summaries:
        store.add_community_summaries(summaries, client.embed_batch([s["summary"] for s in summaries]))
    return store, chunks, secs


def best_embedding_match(resolver, query):
    """Nearest (node, sim) at layer 4, with the same probe as resolver._embedding_resolve."""
    if resolver._node_embs is None:
        resolver._node_names = list(resolver.graph.nodes())
        resolver._node_embs = np.array(resolver.embed.embed_batch(resolver._node_names))
    q = query.strip().strip("?.!,")
    tokens = [w for w in R._WORD_RE.findall(q.lower()) if w not in R._STOPWORDS]
    qv = np.array(resolver.embed.embed_text(" ".join(tokens) if tokens else q))
    sims = np.asarray(resolver._node_embs) @ qv
    i = int(sims.argmax())
    return resolver._node_names[i], float(sims[i])


def calibrate_resolver(client, store, graph):
    resolver = R.QueryResolver(client, store, graph)
    traversal, analytics = GraphTraversal(graph), GraphAnalytics(graph)
    cases = json.load(open(evaluate.CASES_PATH, encoding="utf-8"))
    graded = [c for c in cases if not c.get("expected_refuse") and not c.get("e2e_only")]
    negatives = [c for c in cases if c.get("expected_refuse") or c.get("e2e_only")]

    print(f"\n(b) RESOLVER + RETRIEVAL — {len(graded)} kasus dinilai")
    results = []
    for t in RESOLVER_GRID:
        R.RESOLVER_SIM_THRESHOLD = t
        passed, failed = 0, []
        for c in graded:
            with contextlib.redirect_stdout(io.StringIO()):  # resolver logs every resolution
                text, node = evaluate.retrieval_context(c, resolver, traversal, analytics)
            ok = evaluate.grade(c, text, node)[0]
            passed += ok
            if not ok:
                failed.append(c["id"])
        results.append({"T": t, "passed": passed, "failed": failed})
        print(f"  T={t:.2f}  {passed}/{len(graded)}  gagal: {failed}")

    # Negatives: only those NOT resolved by layers 0-3 (the embedding layer switched off) matter.
    R.RESOLVER_SIM_THRESHOLD = 1.01
    print("\n  Pertanyaan negatif/keamanan yang jatuh ke lapis embedding (node terdekat, sim):")
    neg = []
    for c in negatives:
        with contextlib.redirect_stdout(io.StringIO()):
            resolved = resolver.resolve_entities(c["q"], context_query=c["q"])
        if c["intent"] != "entity" or resolved:
            continue
        node, sim = best_embedding_match(resolver, c["q"])
        neg.append({"id": c["id"], "node": node, "sim": round(sim, 3)})
        print(f"    {c['id']:20} sim={sim:.3f} -> {node[:60]!r}")
    for t in RESOLVER_GRID:
        hits = [n["id"] for n in neg if n["sim"] >= t]
        print(f"  T={t:.2f}  salah cocok (negatif terpetakan ke node): {len(hits)} {hits}")
    return {"retrieval": results, "negatives": neg}


# ── (c) speed + device consistency ────────────────────────────────────────────────────────────────

def device_check(model, revision, chunks, n=32):
    from sentence_transformers import SentenceTransformer
    import torch
    texts = [c["text"] for c in chunks[:n]]
    out = {}
    for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        m = SentenceTransformer(model, device=dev, revision=revision)
        m.max_seq_length = min(E.EMBEDDING_MAX_SEQ_LENGTH, m.max_seq_length)
        t0 = time.perf_counter()
        out[dev] = m.encode(texts, normalize_embeddings=True, batch_size=E.EMBEDDING_BATCH_SIZE)
        print(f"\n(c) {dev}: {n} chunk dalam {time.perf_counter() - t0:.1f} dtk")
        del m
    if len(out) == 2:
        cos = (out["cuda"] * out["cpu"]).sum(1)
        print(f"    cosine GPU vs CPU: min={cos.min():.6f} mean={cos.mean():.6f}")
        return {"cosine_min": float(cos.min())}
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", default=None, help="folder output (default: data/processed/calibration/<model>)")
    args = ap.parse_args()
    out = args.out or os.path.join("data/processed/calibration", re.sub(r"[^\w.-]", "_", args.model))
    os.makedirs(out, exist_ok=True)

    client = load_client(args.model, args.revision)
    report = {"model": args.model, "revision": args.revision, "device": client.device}
    report["dedup"] = calibrate_dedup(client)
    store, chunks, secs = build_temp_store(client, out)
    print(f"\n  embedding {len(chunks)} chunk di {client.device}: {secs:.1f} dtk")
    report["chunk_embed_seconds"] = round(secs, 1)
    report["resolver"] = calibrate_resolver(client, store, nx.read_graphml(evaluate.GRAPH_PATH))
    report["device_check"] = device_check(args.model, args.revision, chunks)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nLaporan: {os.path.join(out, 'report.json')}")


if __name__ == "__main__":
    main()
