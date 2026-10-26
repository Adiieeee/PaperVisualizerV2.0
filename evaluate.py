"""
Evaluation harness for the GraphRAG pipeline.

Two modes:
  * retrieval (default) — measures whether the RIGHT information is retrieved for each question.
    Runs the resolver / traversal / analytics directly with the case's known intent, so it uses
    NO LLM quota (only local graph + embeddings + ChromaDB). This is the core GraphRAG quality.
  * --e2e — runs the full engine.query() (router + generation) and grades the final natural-language
    answer by keyword presence. Uses LLM quota (router + generation per question).

Usage:
    python evaluate.py            # free, retrieval-level accuracy
    python evaluate.py --e2e      # end-to-end (uses daily quota)
"""
import argparse
import json
from collections import defaultdict

import networkx as nx

from src.llm.embedding import EmbeddingClient
from src.vectorstore.chromadb_store import ChromaStore
from src.query.resolver import QueryResolver
from src.query.traversal import GraphTraversal
from src.query.graph_rerank import GraphFactReranker
from src.config import FACT_EMBEDDINGS_PATH
from src.graph.analytics import GraphAnalytics
from src.security.integrity import ArtifactIntegrityError, check_artifacts

GRAPH_PATH = "data/processed/graph.graphml"
CASES_PATH = "eval/eval_cases.json"

# Phrases that count as a correct "I don't know" (for negative / unanswerable cases).
REFUSAL = [
    "tidak tahu", "tidak ada informasi", "tidak terdapat", "tidak disebutkan",
    "tidak tersedia", "tidak dapat", "don't know", "no information", "not found",
]


def retrieval_context(case, resolver, traversal, analytics, reranker=None):
    """Reproduce the retrieval the engine would do for this intent — returns (text, resolved_node)."""
    intent = case["intent"]
    if intent == "counting":
        stats = analytics.get_counting_stats(case["target_type"])[:10]
        return " ; ".join(f"{n}: {d}" for n, d in stats), None
    if intent == "structural":
        bridges = analytics.get_bridge_entities()
        return " ; ".join(f"{n} ({c} papers)" for n, _ps, c in bridges), None
    if intent == "thematic":
        return " ".join(resolver.search_communities(case["q"])), None
    # entity — cerminkan engine: bisa >1 entitas (roadmap 1.2), budget konteks dibagi
    nodes = resolver.resolve_entities(case["q"], context_query=case["q"])
    if nodes:
        budget = 150 if len(nodes) == 1 else 125
        reranker = reranker or GraphFactReranker(resolver.embed, cache_path=FACT_EMBEDDINGS_PATH)
        text = "\n".join(
            traversal.format_traversal_context(
                reranker.select(traversal.traverse(n, max_hops=2, max_rels=None), case["q"], budget))
            for n in nodes
        )
        return text, nodes[0]
    return " ".join(h["text"] for h in resolver.search_chunks(case["q"])), None  # local vector fallback


def grade(case, text, node):
    text_l = (text or "").lower()
    facts_ok = all(kw.lower() in text_l for kw in case.get("expected_contains", []))
    exp_node = case.get("expected_node")
    if exp_node is None:
        node_ok = True
    else:
        node_ok = bool(node) and (exp_node.lower() in node.lower() or node.lower() in exp_node.lower())
    return facts_ok and node_ok, facts_ok, node_ok


def run_retrieval(cases, G):
    embed, chroma = EmbeddingClient(), ChromaStore()
    try:
        check_artifacts(GRAPH_PATH, G, chroma, embed)  # same startup check as the engine
    except ArtifactIntegrityError as e:
        raise SystemExit(f"[Integrity] {e}")
    resolver = QueryResolver(embed, chroma, G)
    traversal = GraphTraversal(G)
    analytics = GraphAnalytics(G)
    reranker = GraphFactReranker(embed, cache_path=FACT_EMBEDDINGS_PATH)  # fact embeddings cached across cases/runs

    # `e2e_only` (security cases: prompt-injection / format / language) are judged on the final LLM
    # answer, so they must stay OUT of the retrieval score — with no expected_contains they would
    # "pass" trivially and inflate the percentage.
    graded = [c for c in cases if not c.get("expected_refuse") and not c.get("e2e_only")]
    negatives = [c for c in cases if c.get("expected_refuse") or c.get("e2e_only")]

    by_cat = defaultdict(lambda: [0, 0])
    passed = 0
    print(f"{'id':22} {'cat':11} {'node':6} {'facts':6} result")
    print("-" * 60)
    for c in graded:
        text, node = retrieval_context(c, resolver, traversal, analytics, reranker)
        ok, facts_ok, node_ok = grade(c, text, node)
        by_cat[c["category"]][0] += int(ok)
        by_cat[c["category"]][1] += 1
        passed += int(ok)
        print(f"{c['id']:22} {c['category']:11} {'ok' if node_ok else 'MISS':6} "
              f"{'ok' if facts_ok else 'MISS':6} {'PASS' if ok else 'FAIL'}")
        if not ok:
            miss = [k for k in c.get('expected_contains', []) if k.lower() not in (text or '').lower()]
            print(f"    ↳ resolved={node!r} missing_facts={miss}")
    print("-" * 60)
    print(f"OVERALL (retrieval): {passed}/{len(graded)} = {passed/len(graded)*100:.0f}%")
    print("By category: " + " | ".join(f"{cat} {p}/{t}" for cat, (p, t) in sorted(by_cat.items())))
    # `expected_empty`: out-of-domain questions whose vector results must all fall below the relevance
    # threshold, so the engine refuses without an LLM call. Scored apart from the retrieval score above.
    empties = [c for c in cases if c.get("expected_empty")]
    if empties:
        ok_empty = 0
        print("\nGuardrail (context must be empty):")
        for c in empties:
            text, _node = retrieval_context(c, resolver, traversal, analytics, reranker)
            ok = not (text or "").strip()
            ok_empty += int(ok)
            print(f"  {c['id']:22} {'PASS' if ok else 'FAIL'}" + ("" if ok else f"  ({len(text)} chars)"))
        print(f"GUARDRAIL: {ok_empty}/{len(empties)}")
    if negatives:
        print(f"\n{len(negatives)} case(s) are graded only in --e2e (refusal / security checks): "
              + ", ".join(c['id'] for c in negatives))


def run_e2e(cases, graph_path):
    from src.engine import GraphRAGEngine
    from src.llm.llm_client import DailyQuotaExceededError
    engine = GraphRAGEngine(graph_path)
    passed, graded, errors, fallback = 0, 0, [], []
    for c in cases:
        engine.llm.last_model = None
        try:
            ans = engine.query(c["q"])
        except DailyQuotaExceededError:
            # No model left today: stop cleanly and still report what was graded so far.
            print(f"[{'QUOTA':14}] {c['id']}: daily quota exhausted and no fallback left; stopping the run.")
            break
        except Exception as e:
            # One bad case (empty answer, provider error after retries) must not discard the whole run.
            errors.append(c["id"])
            print(f"[{'ERROR':14}] {c['id']}: {c['q']}")
            print(f"    {type(e).__name__}: {str(e)[:180]}")
            continue
        al = ans.lower()
        # must_not_contain: injected/forbidden content (e.g. a leaked system-prompt marker, the answer
        # to a hijacked task) must NOT appear even when the rest of the answer looks fine.
        leaked = [k for k in c.get("must_not_contain", []) if k.lower() in al]
        if c.get("expected_refuse"):
            ok = any(p in al for p in REFUSAL) and not leaked  # must decline, not fabricate / obey
            label = "REFUSED-ok" if ok else ("LEAKED" if leaked else "HALLUCINATED?")
        else:
            ok = all(kw.lower() in al for kw in c.get("expected_contains", [])) and not leaked
            label = "PASS" if ok else ("LEAKED" if leaked else "FAIL")
        passed += int(ok)
        graded += 1
        # The last call of the case answered on the fallback provider: not comparable with a Gemini baseline.
        model = engine.llm.last_model
        on_fallback = bool(model) and "gemini" not in model.lower()
        if on_fallback:
            fallback.append(c["id"])
        print(f"[{label:14}] {c['id']}: {c['q']}" + (f"  (model: {model})" if on_fallback else ""))
        if not ok:
            print(f"    answer: {ans[:180]}...")
            if leaked:
                print(f"    forbidden content present: {leaked}")
    print()
    print(f"E2E OVERALL: {passed}/{len(cases)} = {passed/len(cases)*100:.0f}% "
          f"(includes router accuracy + generation + anti-hallucination)")
    if graded + len(errors) < len(cases):
        print(f"  Not run: {len(cases) - graded - len(errors)} case(s) (run stopped early).")
    if errors:
        print(f"  Errors (counted as failed): {len(errors)}: " + ", ".join(errors))
    if fallback:
        print(f"  Answered by the fallback model, not comparable with a Gemini baseline: {len(fallback)}: "
              + ", ".join(fallback))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", action="store_true", help="run full engine.query (uses LLM quota)")
    args = ap.parse_args()

    with open(CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)
    G = nx.read_graphml(GRAPH_PATH)
    print(f"Loaded {len(cases)} eval cases; graph = {G.number_of_nodes()} nodes.\n")

    if args.e2e:
        run_e2e(cases, GRAPH_PATH)
    else:
        run_retrieval(cases, G)


if __name__ == "__main__":
    main()
