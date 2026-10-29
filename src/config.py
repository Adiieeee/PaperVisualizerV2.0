import os
import re
import json
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, PrivateAttr

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# LiteLLM (src/llm/llm_client.py): model names carry the provider prefix. Groq is optional — without a key
# or a model, the "general" alias runs without a fallback.
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini/gemini-flash-lite-latest"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL") or None

# --- Constants ---
# Embedding model (local; GPU when available, otherwise CPU)
# bge-m3: multilingual (English corpus, questions often in Indonesian), no trust_remote_code, MIT.
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
# Pinned model revision (commit): a weight change on the hub must not silently change the embedding space.
# This value = the snapshot that built the current corpus. Changing it (or the model above) means the corpus
# must be rebuilt — the manifest (src/security/integrity.py) refuses to start on a mismatch.
EMBEDDING_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_DIMENSION = 1024  # bge-m3
# "auto" = cuda if a CUDA build of torch is installed and a GPU is detected, otherwise cpu. Vectors stay fp32
# so a build (e.g. on GPU) and queries (e.g. the server on CPU) live in the same space.
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")  # auto | cpu | cuda
EMBEDDING_MAX_SEQ_LENGTH = 1024  # a 2000-character chunk is ~500 tokens; never exceeds the model's own limit
EMBEDDING_BATCH_SIZE = 16

# --- Security limits (see docs/SECURITY.md). All hard limits live in code, not in the prompt. ---
ANSWER_MAX_OUTPUT_TOKENS = 2048          # cap on the cost/latency of the final answer
LLM_MIN_INTERVAL = 6.0                   # seconds between the STARTS of two LLM calls (~10 req/min, all threads)
MAX_CHAT_QUERY_LENGTH = 500              # maximum question length in /api/chat
MAX_CHAT_BODY_BYTES = 10_000             # maximum body size in /api/chat (also for chunked bodies)
CHAT_RATE_LIMIT_MAX = 8                  # per IP per window; one chat takes >= 6 s (2 calls, starts 6 s apart)
CHAT_RATE_LIMIT_WINDOW_SECONDS = 60
CHAT_MAX_CONCURRENT = 2                  # chats allowed to run concurrently (each chat = 2 LLM calls)
ALLOWED_HOSTS = ("127.0.0.1", "localhost")  # fixed Host allow-list (anti DNS rebinding)
MAX_PDF_PAGES = 500                      # PDFs above this are skipped at ingest
MAX_PDF_BYTES = 50 * 1024 * 1024

# --- Thresholds TIED to the embedding model: recalibrate when EMBEDDING_MODEL_NAME changes ---
# (each model's cosine score distribution differs; see scripts/calibrate_embedding.py)
# Dedup: entity pairs >= the threshold are sent to the LLM for confirmation (this determines quota usage).
# bge-m3 (calibrated 2026-10-21): 221 candidates vs 267 before; correct merges used to be >= 0.76, most
# wrong merges (different model versions, hypernyms) < 0.75. It used to be 0.85 for bge-small-en-v1.5.
DEDUP_SIMILARITY_THRESHOLD = 0.75
# Resolver layer 4: the embedding-nearest node is used when its similarity is >= this threshold.
# bge-m3: correct Indonesian paraphrases >= 0.638, wrong matches observed <= 0.600 (thin margin; recheck
# with evaluate.py after a build). It used to be 0.72 for bge-small-en-v1.5.
RESOLVER_SIM_THRESHOLD = 0.62
# Corpus topic terms: title words found in >= this share of the paper titles (here only "sycophancy", 7/8).
# They occur in almost every question and in many node names, so the resolver does not match nodes made of
# them alone, and a broad question about the topic is answered from community summaries. Fixed before
# measuring (2026-10-26); do not tune against the eval. Below MIN_PAPERS no term is derived.
TOPIC_TERM_MIN_SHARE = 0.5
TOPIC_TERM_MIN_PAPERS = 3
# Vector paths (chunk fallback + thematic): results below the threshold are dropped; if all of them fall away, the engine
# refuses without an LLM call. bge-m3 (calibrated 2026-10-23, top-1 cosine): relevant fallbacks >= 0.563,
# clearly out-of-domain <= 0.466 (coffee, weather, translation); ResNet/ImageNet 0.529 deliberately passes (refused by the
# answer prompt's rules). Community summaries (in English): themes >= 0.557, but legitimate Indonesian paraphrases
# ("cara mengurangi sikap menjilat pada chatbot") score only 0.485-0.499 -> 0.50 rejects them (e2e
# 2026-10-26); out-of-domain <= 0.405 (ResNet), recipe 0.354. See scripts/calibrate_retrieval.py.
CHUNK_MIN_SIMILARITY = 0.50
COMMUNITY_MIN_SIMILARITY = 0.44

# --- Graph context budget per entity (relations kept by GraphFactReranker) ---
# Measured: splitting the budget (150//n) cut off hop-2 on hub nodes, so facts like
# Author->Institution were lost. 125/entity is enough for >1 entity. A server constant: the router cannot
# change it; `evaluate.py --budget N` overrides it ONLY for experiments.
ENTITY_BUDGET_SINGLE = 150
ENTITY_BUDGET_MULTI = 125

# --- Community (graph-only; does not touch the chunk vector store) ---
# Louvain runs on the co-occurrence graph: two entities are linked when they are mentioned in the same chunk.
# Paper is excluded (it becomes a hub, community = one paper), and so are Author/Institution (the first chunk
# contains all the authors + the abstract and would connect everything). Communities < MIN_SIZE are not
# summarized; the remaining nodes are treated as outliers.
COMMUNITY_MIN_SIZE = 3
COMMUNITY_EXCLUDED_TYPES = ("Paper", "Author", "Institution")
COMMUNITY_SEED = 42  # Louvain is random; a fixed seed = a reproducible build

# --- Post-extraction cleaning (graph quality; does NOT affect the vector store) ---
MIN_ENTITY_NAME_LENGTH = 2  # Entities whose normalized name is shorter than this are dropped.

# Generic terms that carry no graph value even if the LLM extracts them.
# Stored already-normalized (lowercase, punctuation-stripped) for O(1) matching.
GENERIC_ENTITY_BLOCKLIST = {
    "the model", "the approach", "the method", "the dataset",
    "the system", "the algorithm", "this model", "our model",
    "the proposed method", "our approach", "the proposed approach",
    "the framework", "our framework", "this framework",
    "the technique", "the network", "the architecture",
    "a model", "a method", "an approach",
    "model", "method", "approach", "technique", "network",
    "the paper", "this paper", "the authors", "our method",
}

# --- Section filter (ingestion) ---
# Matches a *lone* terminal-section heading (optionally numbered) on its own line.
# Everything from the first such heading in the latter half of a document is dropped
# before chunking, so References/Bibliography/Acknowledgements pollute neither the
# graph (citation authors, bibliography entries) nor the vector store.
SECTION_FILTER_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\s+)?(references|bibliography|acknowledge?ments?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Minimum position (fraction of the document) at which a terminal heading counts as valid.
# This guard keeps the word "References" appearing too early (e.g. in the table of contents)
# from cutting off the paper body. A VALUE OF 0.5 IS TOO STRICT for modern papers: those with a long
# appendix put References at ~29-47% of the document, so the filter never fires and
# the whole bibliography + appendix get chunked (wasting quota + citation noise).
SECTION_FILTER_MIN_POSITION = 0.20

# --- Paper metadata (B1 title injection + B2 model↔paper link) ---
# NOT hardcoded per corpus: auto-generated at build time (one LLM call per PDF extracts the
# title + the popular model names the paper's model is known by) and cached to this JSON.
# Papers rarely use their own popular name (the VGG paper never writes "VGG-16"), so we rely on
# the LLM's world knowledge for the model aliases — which generalizes to any paper corpus.
PAPER_METADATA_PATH = "data/processed/paper_metadata.json"
# Embedding cache of graph fact texts for rerank (query-time, derived; safe to delete at any time).
FACT_EMBEDDINGS_PATH = "data/processed/fact_embeddings.npz"


def load_paper_metadata() -> dict:
    """Load the auto-generated {doc_id: {title, models}} map, or {} if not built yet."""
    if not os.path.exists(PAPER_METADATA_PATH):
        return {}
    try:
        with open(PAPER_METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_paper_metadata(metadata: dict):
    from src.security.integrity import atomic_write_json  # lazy: integrity imports config
    atomic_write_json(PAPER_METADATA_PATH, metadata, indent=2)


def load_model_to_paper() -> dict:
    """popular model name (lowercased) -> canonical paper title (for paper-context resolution)."""
    meta = load_paper_metadata()
    return {m.lower(): d["title"] for d in meta.values() for m in d.get("models", []) if m}

# Types
ENTITY_TYPES = Literal[
    "Paper", "Author", "Institution", "Method", "Dataset", 
    "Metric", "Finding", "Limitation", "Task"
]

# Relations OFFERED to the LLM (part of the extraction schema).
LLM_RELATION_TYPES = Literal[
    "uses_method", "uses_dataset", "uses_metric", "has_limitation", 
    "compares", "finds", "proposes_method", "criticizes", 
    "builds_on", "written_by", "affiliated_with", "mentioned_in",
    "addresses_task", "introduces_dataset", "evaluated_on",
]
# Relations that may be STORED: + `related_to`, used only when salvaging LLM output that contains
# relations outside the list (the original label is kept in the description). Deliberately not offered to the LLM so
# it is not used as a shortcut.
RELATION_TYPES = Literal[LLM_RELATION_TYPES, "related_to"]


# --- Pydantic Schemas for LLM Extraction ---

class PaperMeta(BaseModel):
    title: str = Field(description="The exact, full title of THIS paper.")
    models: List[str] = Field(
        default_factory=list,
        description="Popular short names/acronyms the main model/method this paper introduces is "
                    "commonly known by (e.g. 'ResNet', 'VGG-16'). Empty if it introduces no named model.",
    )

class FamilyDecision(BaseModel):
    same_family: bool = Field(
        description="True hanya jika kedua nama merujuk pada keluarga produk/model yang SAMA."
    )
    canonical: str = Field(
        description="Nama induk yang sebaiknya dipakai (salah satu dari dua nama yang diberikan)."
    )
    reason: str = Field(description="Alasan singkat.")

class Entity(BaseModel):
    name: str = Field(description="The normalized name of the entity. Use canonical naming if possible.")
    type: ENTITY_TYPES = Field(description="The category of the entity.")
    description: Optional[str] = Field(None, description="A brief description of the entity based on the context.")
    _source_chunk_id: Optional[str] = PrivateAttr(default=None)

# Note: Pydantic class docstrings end up in the JSON schema sent to the LLM, so use comments.
# Relations as the LLM may write them (extraction schema).
class LLMRelationship(BaseModel):
    source_entity: str = Field(description="The name of the source entity.")
    target_entity: str = Field(description="The name of the target entity.")
    relation_type: LLM_RELATION_TYPES = Field(description="The type of relationship between source and target.")
    description: Optional[str] = Field(None, description="A brief explanation of how they are related in this context.")
    _source_chunk_id: Optional[str] = PrivateAttr(default=None)

# Stored relations (cleaner/dedup/graph): also accept the `related_to` produced by salvaging.
class Relationship(LLMRelationship):
    relation_type: RELATION_TYPES = Field(description="The type of relationship between source and target.")

# Schema SENT to the LLM during extraction (without `related_to`).
class LLMExtractedKnowledge(BaseModel):
    entities: List[Entity] = Field(default_factory=list, description="List of all distinct entities found in the text.")
    relationships: List[LLMRelationship] = Field(default_factory=list, description="List of relationships connecting the extracted entities.")

class ExtractedKnowledge(LLMExtractedKnowledge):
    relationships: List[Relationship] = Field(default_factory=list, description="List of relationships connecting the extracted entities.")

# --- Schema for Entity Linking / Dedup ---

class DedupDecision(BaseModel):
    is_duplicate: bool = Field(description="True if the candidate is the same entity as the master entity.")
    reason: str = Field(description="Brief reason for the decision.")
    merged_name: str = Field(description="The chosen canonical name if it is a duplicate. Otherwise, the original candidate name.")

# --- Schema for Query Routing ---

# off_topic: clearly not a question about research/papers -> the engine refuses without retrieval and without a second LLM call.
INTENT_TYPES = Literal["entity", "thematic", "counting", "structural", "off_topic"]

class QueryRouting(BaseModel):
    intent: INTENT_TYPES = Field(description="The classification of the user's query.")
    target_type: Optional[ENTITY_TYPES] = Field(None, description="Required for 'counting' intent. The type of entity being counted or ranked (e.g., 'Method').")
    search_query: Optional[str] = Field(None, description="The refined text query for 'entity' or 'thematic' search. Leave empty for counting if not applicable.")
