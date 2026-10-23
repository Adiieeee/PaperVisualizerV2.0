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
from src.graph.analytics import GraphAnalytics
from src.security.integrity import ArtifactIntegrityError, check_artifacts

GRAPH_PATH = "data/processed/graph.graphml"
CASES_PATH = "eval/eval_cases.json"

# Phrases that count as a correct "I don't know" (for negative / unanswerable cases).
REFUSAL = [
    "tidak tahu", "tidak ada informasi", "tidak terdapat", "tidak disebutkan",
    "tidak tersedia", "tidak dapat", "don't know", "no information", "not found",
]


def retrieval_context(case, resolver, traversal, analytics):
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
        text = "\n".join(
            traversal.format_traversal_context(traversal.traverse(n, max_hops=2, max_rels=budget))
            for n in nodes
        )
        return text, nodes[0]
    return " ".join(resolver.search_chunks(case["q"])), None  # local vector fallback


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
        text, node = retrieval_context(c, resolver, traversal, analytics)
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
            text, _node = retrieval_context(c, resolver, traversal, analytics)
            ok = not (text or "").strip()
            ok_empty += int(ok)
            print(f"  {c['id']:22} {'PASS' if ok else 'FAIL'}" + ("" if ok else f"  ({len(text)} chars)"))
        print(f"GUARDRAIL: {ok_empty}/{len(empties)}")
    if negatives:
        print(f"\n{len(negatives)} case(s) are graded only in --e2e (refusal / security checks): "
              + ", ".join(c['id'] for c in negatives))


def run_e2e(cases, graph_path):
    from src.engine import GraphRAGEngine
    engine = GraphRAGEngine(graph_path)
    passed = 0
    for c in cases:
        ans = engine.query(c["q"])
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
        print(f"[{label:14}] {c['id']}: {c['q']}")
        if not ok:
            print(f"    answer: {ans[:180]}...")
            if leaked:
                print(f"    forbidden content present: {leaked}")
    print(f"\nE2E OVERALL: {passed}/{len(cases)} = {passed/len(cases)*100:.0f}% "
          f"(includes router accuracy + generation + anti-hallucination)")


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
