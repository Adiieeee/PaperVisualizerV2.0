import networkx as nx

from src.llm.llm_service import LLMService
from src.llm.embedding import EmbeddingClient
from src.llm.prompts import KIND_GRAPH, KIND_TEXT, KIND_COMMUNITY
from src.vectorstore.chromadb_store import ChromaStore
from src.query.router import QueryRouter
from src.query.resolver import QueryResolver
from src.query.traversal import GraphTraversal
from src.query.graph_rerank import GraphFactReranker
from src.graph.analytics import GraphAnalytics
from src.security.integrity import check_artifacts
from src.config import load_paper_metadata

# Jawaban tetap (tanpa panggilan LLM) bila pencarian vektor tidak menemukan apa pun di atas ambang
# relevansi. Dwibahasa karena bahasa pertanyaan tidak dideteksi; frasanya dikenali evaluate.REFUSAL.
NO_CONTEXT_MESSAGE = (
    "Maaf, tidak ada informasi tentang hal itu di dokumen yang tersedia. / "
    "Sorry, there is no information about this in the provided documents."
)


def format_chunk(i: int, hit: dict, paper_titles: dict) -> str:
    """`[i] (Sumber: <judul paper>) <teks>`; judul dari paper_metadata.json, fallback ke nama file PDF.
    Judul adalah keluaran LLM atas PDF: ikut dinetralkan bersama blok konteksnya di build_answer_prompt."""
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
        self.reranker = GraphFactReranker(self.embed)
        self.analytics = GraphAnalytics(self.graph)
        self.paper_titles = {doc: m.get("title") for doc, m in load_paper_metadata().items()}

    def query(self, user_query: str) -> str:
        """
        End-to-end query processing pipeline.
        """
        print(f"\n[1] Routing query: '{user_query}'")
        routing = self.router.route(user_query)

        # Konteks dikumpulkan per ASAL-USUL (graph / teks mentah / ringkasan komunitas) supaya prompt
        # jawaban bisa membungkus tiap jenis dengan tag sendiri dan menandainya sebagai DATA
        # (src/llm/prompts.py). Setiap bagian ditambahkan ke `context_parts` dan jenisnya di `kind`.
        context_parts = []
        kind = KIND_GRAPH

        if routing.intent == "counting":
            print(f"[2] Executing Graph Stats for '{routing.target_type}'")
            if not routing.target_type:
                return "Error: Could not determine which entity type to count."

            stats = self.analytics.get_counting_stats(routing.target_type)
            # Take top 10 to avoid context overflow
            top_stats = stats[:10]
            context_parts.append(f"Top {routing.target_type}s by number of connections:")
            for node, degree in top_stats:
                context_parts.append(f"- {node}: {degree} connections")

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
            # Roadmap 1.2: pertanyaan bisa menyebut >1 entitas ("SycEval dan ELEPHANT").
            nodes = self.resolver.resolve_entities(search_q, context_query=user_query)
            if not nodes and search_q != user_query:
                nodes = self.resolver.resolve_entities(user_query, context_query=user_query)

            if nodes:
                # Budget konteks per entitas. Terukur: membagi budget (150//n) justru memotong
                # sebelum hop-2 pada node hub, sehingga fakta seperti Author->Institution hilang.
                # 125/entitas sudah cukup dan konteks total jenuh (~31K char untuk 2 entitas)
                # karena dibatasi ukuran graph itu sendiri.
                budget = 150 if len(nodes) == 1 else 125
                print(f"[3] Traversing graph from {nodes} (multi-hop, budget {budget}/entitas)")
                context_parts.append("Graph Context:")
                for n in nodes:
                    # Semua relasi 2-hop diambil, lalu dipangkas & diurutkan menurut relevansi terhadap
                    # pertanyaan asli (bukan search_query yang hanya nama entitas).
                    trav_result = self.traversal.traverse(n, max_hops=2, max_rels=None)
                    trav_result = self.reranker.select(trav_result, user_query, budget)
                    context_parts.append(self.traversal.format_traversal_context(trav_result))
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
        final_answer = self.llm.generate_final_answer(user_query, [(kind, assembled_context)])

        return final_answer
