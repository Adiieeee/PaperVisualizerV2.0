import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import networkx as nx

from src.llm.llm_service import LLMService
from src.llm.embedding import EmbeddingClient
from src.llm.prompts import KIND_GRAPH, KIND_TEXT, KIND_COMMUNITY, KIND_EXTERNAL
from src.vectorstore.chromadb_store import ChromaStore
from src.query.router import QueryRouter
from src.query.resolver import QueryResolver
from src.query.traversal import GraphTraversal
from src.query.graph_rerank import GraphFactReranker
from src.query.external import ExternalContext, format_external
from src.graph.analytics import GraphAnalytics
from src.security.integrity import check_artifacts
from src import config
from src.config import FACT_EMBEDDINGS_PATH, load_paper_metadata

# Fixed answer (no LLM call) when the vector search finds nothing above the relevance threshold.
# Bilingual because the question's language is not detected; its phrasing is recognized by evaluate.REFUSAL.
NO_CONTEXT_MESSAGE = (
    "Maaf, tidak ada informasi tentang hal itu di dokumen yang tersedia. / "
    "Sorry, there is no information about this in the provided documents."
)
# Fixed answer for the off_topic intent (router): no retrieval and no second LLM call.
OFF_TOPIC_MESSAGE = (
    "Maaf, saya hanya bisa menjawab pertanyaan tentang paper di korpus ini; tidak ada informasi tentang "
    "hal itu di dokumen. / Sorry, I can only answer questions about the papers in this corpus; the "
    "documents contain no information about this."
)


# Network lookups of external background run here, beside the graph traversal (2 = CHAT_MAX_CONCURRENT).
_EXTERNAL_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="external")


def format_chunk(i: int, hit: dict, paper_titles: dict) -> str:
    """`[i] (Sumber: <paper title>) <text>`; the title comes from paper_metadata.json, falling back to the PDF file name.
    The title is LLM output over a PDF: it is neutralized together with its context block in build_answer_prompt."""
    doc_id = hit.get("doc_id")
    if not doc_id:
        return f"[{i}] {hit['text']}"
    return f"[{i}] (Sumber: {paper_titles.get(doc_id) or doc_id}) {hit['text']}"


class GraphRAGEngine:
    def __init__(self, graph_path: str = "data/processed/graph.graphml"):
        self.llm = LLMService()
        self.embed = EmbeddingClient()
        self.chroma = ChromaStore()
        
        # Load Graph
        self.graph = nx.read_graphml(graph_path)
        print(f"Engine loaded graph with {self.graph.number_of_nodes()} nodes.")

        # Fail hard and loudly if graph / ChromaDB / embedding model no longer match what built them
        # (a swapped embedding model with the same dimension would NOT crash — just answer wrongly).
        check_artifacts(graph_path, self.graph, self.chroma, self.embed)

        self.router = QueryRouter(self.llm)
        self.resolver = QueryResolver(self.embed, self.chroma, self.graph)
        self.traversal = GraphTraversal(self.graph)
        self.reranker = GraphFactReranker(self.embed, cache_path=FACT_EMBEDDINGS_PATH)
        self.analytics = GraphAnalytics(self.graph)
        self.paper_titles = {doc: m.get("title") for doc, m in load_paper_metadata().items()}
        self.external = ExternalContext(self.embed) if config.EXTERNAL_CONTEXT_ENABLED else None
        self.last_trace = None

    def _start_external(self, routing, nodes) -> list:
        """Start background lookups for a definition question (router flag) about entities of the eligible
        types: [(name, node_text, cached entry | Future of candidates)]. Only the entity name goes out."""
        if not (routing.needs_background and getattr(self, "external", None)):
            return []
        jobs = []
        for n in nodes:
            data = self.graph.nodes[n]
            if data.get("type") not in config.EXTERNAL_ENTITY_TYPES:
                continue
            node_text = f"{n}: {data.get('description') or ''}"
            hit, entry = self.external.cached(n)
            jobs.append((n, node_text, entry if hit else _EXTERNAL_POOL.submit(self.external.fetch, n)))
            if len(jobs) == config.EXTERNAL_MAX_ENTITIES:
                break
        return jobs

    def _finish_external(self, jobs, started: float) -> list:
        """Entries found for the started lookups; one still running past EXTERNAL_TIMEOUT is skipped."""
        found = []
        for name, node_text, job in jobs:
            if hasattr(job, "result"):
                try:
                    candidates = job.result(timeout=max(0.0, config.EXTERNAL_TIMEOUT - (time.monotonic() - started)))
                except FutureTimeout:
                    print(f"[External] '{name}': no answer in time, answering without it")
                    continue
                job = None if candidates is None else self.external.choose(name, node_text, candidates)
            if job:
                found.append(job)
        return found

    def query(self, user_query: str) -> str:
        """
        End-to-end query processing pipeline.
        """
        print(f"\n[1] Routing query: '{user_query}'")
        routing = self.router.route(user_query)
        self.last_trace = {"intent": routing.intent, "needs_background": routing.needs_background, "external": []}
        if routing.intent == "off_topic":
            print("[2] Off-topic question: refused without retrieval")
            return OFF_TOPIC_MESSAGE

        # Context is collected per ORIGIN (graph / raw text / community summaries) so the answer prompt
        # can wrap each kind in its own tag and mark it as DATA
        # (src/llm/prompts.py). Each part is added to `context_parts` and its kind to `kind`.
        context_parts = []
        kind = KIND_GRAPH
        external = []  # background entries (Wikipedia/arXiv), sent as their own block

        if routing.intent == "counting":
            print(f"[2] Executing Graph Stats for '{routing.target_type}'")
            if not routing.target_type:
                return "Error: Could not determine which entity type to count."

            stats = self.analytics.get_usage_stats(routing.target_type)
            # Take top 10 to avoid context overflow. Both numbers are shown, so "most connected"
            # questions can still be answered from the connection counts.
            top_stats = stats[:10]
            context_parts.append(
                f"Top {routing.target_type}s by number of papers they appear in (then connections):")
            for node, n_papers, degree in top_stats:
                context_parts.append(f"- {node}: {n_papers} papers, {degree} connections")

        elif routing.intent == "thematic":
            print("[2] Executing Vector Search on Community Summaries")
            kind = KIND_COMMUNITY
            search_q = routing.search_query or user_query
            summaries = self.resolver.search_communities(search_q)
            if not summaries:
                print("[3] No community summary above the relevance threshold")
                return NO_CONTEXT_MESSAGE
            context_parts.append("Thematic Community Context:")
            for i, s in enumerate(summaries):
                context_parts.append(f"[{i+1}] {s}")

        elif routing.intent == "structural":
            print("[2] Executing Structural Query (cross-paper bridge entities)")
            bridges = self.analytics.get_bridge_entities()
            context_parts.append("Entities that bridge multiple papers (shared across papers):")
            if not bridges:
                context_parts.append("(None — entities in this corpus mostly belong to a single paper.)")
            for name, papers, count in bridges:
                context_parts.append(f"- {name}: appears in {count} papers ({'; '.join(p[:50] for p in papers)})")

        else: # entity
            print("[2] Executing Hybrid Entity Search")
            search_q = routing.search_query or user_query
            # Roadmap 1.2: a question can mention >1 entity ("SycEval dan ELEPHANT").
            nodes = self.resolver.resolve_entities(search_q, context_query=user_query)
            if not nodes and search_q != user_query:
                nodes = self.resolver.resolve_entities(user_query, context_query=user_query)

            if nodes:
                # Per-entity context budget (see config). Read at query time, not imported, so
                # `evaluate.py --budget` can override it for experiments.
                budget = config.ENTITY_BUDGET_SINGLE if len(nodes) == 1 else config.ENTITY_BUDGET_MULTI
                # Network lookups start first so they overlap the traversal (and the rate-limit pause before
                # the answer call); scoring them needs the embedding model, so it waits until after the rerank.
                started = time.monotonic()
                jobs = self._start_external(routing, nodes)
                print(f"[3] Traversing graph from {nodes} (multi-hop, budget {budget}/entitas)")
                context_parts.append("Graph Context:")
                for n in nodes:
                    # All 2-hop relations are fetched, then pruned & ordered by relevance to
                    # the original question (not search_query, which is only the entity name).
                    trav_result = self.traversal.traverse(n, max_hops=2, max_rels=None)
                    trav_result = self.reranker.select(trav_result, user_query, budget)
                    context_parts.append(self.traversal.format_traversal_context(trav_result))
                external = self._finish_external(jobs, started)
                if external:
                    print(f"[3b] External background: {[e['title'] for e in external]}")
            else:
                # A question naming only the corpus topic ("ways to reduce sycophancy?") is a broad one,
                # whatever the router said: answer it from the community summaries, searched with the
                # user's own question (the router's search_query is just the topic word).
                summaries = []
                if self.resolver.mentions_topic(user_query):
                    summaries = self.resolver.search_communities(user_query)
                if summaries:
                    print("[3] Broad question about the corpus topic: using community summaries")
                    kind = KIND_COMMUNITY
                    context_parts.append("Thematic Community Context:")
                    for i, s in enumerate(summaries):
                        context_parts.append(f"[{i+1}] {s}")
                else:
                    print("[3] Node not found, falling back to Chunk search")
                    kind = KIND_TEXT
                    chunks = self.resolver.search_chunks(search_q)
                    if not chunks:
                        print("[4] No chunk above the relevance threshold")
                        return NO_CONTEXT_MESSAGE
                    context_parts.append("Raw Text Chunks Context:")
                    for i, c in enumerate(chunks):
                        context_parts.append(format_chunk(i + 1, c, self.paper_titles))

        # Assemble Context
        assembled_context = "\n\n".join(context_parts)
        print("\n[4] Assembled Context Length:", len(assembled_context))

        # Generate Answer. generate_final_answer is the ONLY place context + question become a
        # prompt (neutralized + tagged as DATA) — do not interpolate context into prompts elsewhere.
        print("[5] Generating Final Answer...")
        blocks = [(kind, assembled_context)]
        if external:
            self.last_trace["external"] = [f"{e['source']}: {e['title']}" for e in external]
            blocks.append((KIND_EXTERNAL, "External Background (not from the papers):\n"
                           + "\n".join(format_external(e) for e in external)))
        final_answer = self.llm.generate_final_answer(user_query, blocks)

        return final_answer
