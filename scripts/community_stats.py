"""
Read-only community check, no LLM quota: how detect_communities splits the current graph.

    python -m scripts.community_stats [path/to/graph.graphml]

Run it after changing corpus: a single giant community (many unrelated models pulled together
by a very frequent entity) or very low coverage means the co-occurrence graph needs a look
before spending quota on summaries.
"""
import sys
from collections import Counter

import networkx as nx

from src.config import COMMUNITY_EXCLUDED_TYPES, COMMUNITY_MIN_SIZE
from src.graph.analytics import GraphAnalytics


def main(path: str = "data/processed/graph.graphml"):
    G = nx.read_graphml(path)
    analytics = GraphAnalytics(G)
    comms = analytics.detect_communities(min_size=1)
    eligible = sum(1 for _, d in G.nodes(data=True) if d.get("type") not in COMMUNITY_EXCLUDED_TYPES)
    kept = [c for c in comms if len(c["nodes"]) >= COMMUNITY_MIN_SIZE]
    covered = sum(len(c["nodes"]) for c in kept)

    print(f"Graph: {G.number_of_nodes()} nodes; {eligible} eligible (excluding {', '.join(COMMUNITY_EXCLUDED_TYPES)})")
    print(f"Communities >= {COMMUNITY_MIN_SIZE} nodes (summarized at build): {len(kept)}")
    print(f"Coverage: {covered}/{eligible} ({100 * covered / max(eligible, 1):.0f}%), outliers: {eligible - covered}")
    sizes = Counter(len(c["nodes"]) for c in comms)
    print("Size distribution (size: count):", ", ".join(f"{s}: {n}" for s, n in sorted(sizes.items())))

    print("\nLargest communities:")
    for c in kept[:10]:
        papers = {p.strip() for n in c["nodes"] for p in str(G.nodes[n].get("source_paper", "")).split(";") if p.strip()}
        types = Counter(G.nodes[n].get("type") for n in c["nodes"])
        names = ", ".join(c["nodes"][:8]) + (" ..." if len(c["nodes"]) > 8 else "")
        print(f"  [{c['community_id']}] {len(c['nodes'])} nodes, {len(papers)} papers, {dict(types)}")
        print(f"      {names}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
