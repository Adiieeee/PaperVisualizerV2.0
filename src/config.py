import os
import re
import json
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, PrivateAttr

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# LiteLLM (src/llm/llm_client.py): nama model pakai prefix provider. Groq opsional — tanpa key
# atau model, alias "general" berjalan tanpa fallback.
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini/gemini-flash-lite-latest"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL") or None

# --- Constants ---
# Embedding Model (Local CPU)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Revisi (commit) model yang dipin: perubahan bobot di hub tidak boleh mengubah ruang embedding diam-diam.
# Nilai ini = snapshot yang membangun korpus saat ini. Mengubahnya (atau model di atas) berarti korpus
# harus dibangun ulang — manifest (src/security/integrity.py) akan menolak start bila tidak cocok.
EMBEDDING_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
EMBEDDING_DIMENSION = 384  # bge-small-en-v1.5

# --- Security limits (lihat SECURITY.md). Semua batas keras ada di kode, bukan di prompt. ---
ANSWER_MAX_OUTPUT_TOKENS = 2048          # cap biaya/latensi jawaban akhir
MAX_CHAT_QUERY_LENGTH = 500              # panjang pertanyaan maksimum di /api/chat
MAX_CHAT_BODY_BYTES = 10_000             # ukuran body maksimum di /api/chat (juga untuk body chunked)
CHAT_RATE_LIMIT_MAX = 8                  # per IP per jendela; 1 chat butuh >=12 dtk (2 panggilan + sleep 6)
CHAT_RATE_LIMIT_WINDOW_SECONDS = 60
CHAT_MAX_CONCURRENT = 2                  # chat yang boleh berjalan bersamaan (tiap chat = 2 panggilan Gemini)
ALLOWED_HOSTS = ("127.0.0.1", "localhost")  # allow-list Host tetap (anti DNS rebinding)
MAX_PDF_PAGES = 500                      # PDF di atas ini dilewati saat ingest
MAX_PDF_BYTES = 50 * 1024 * 1024

# Dedup
DEDUP_SIMILARITY_THRESHOLD = 0.85  # Optimized threshold after diagnostic analysis (catches genuine synonyms, reduces LLM noise)

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

# Posisi minimum (fraksi dokumen) sebuah heading terminal boleh dianggap sah.
# Guard ini mencegah kata "References" yang muncul terlalu awal (mis. di daftar isi)
# memotong badan paper. NILAI 0.5 TERLALU KETAT untuk paper modern: yang punya appendix
# panjang menaruh References di ~29-47% dokumen, sehingga filter tidak pernah aktif dan
# seluruh daftar pustaka + appendix ikut ter-chunk (boros kuota + noise sitasi).
SECTION_FILTER_MIN_POSITION = 0.20

# --- Paper metadata (B1 title injection + B2 model↔paper link) ---
# NOT hardcoded per corpus: auto-generated at build time (one LLM call per PDF extracts the
# title + the popular model names the paper's model is known by) and cached to this JSON.
# Papers rarely use their own popular name (the VGG paper never writes "VGG-16"), so we rely on
# the LLM's world knowledge for the model aliases — which generalizes to any paper corpus.
PAPER_METADATA_PATH = "data/processed/paper_metadata.json"


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

RELATION_TYPES = Literal[
    "uses_method", "uses_dataset", "uses_metric", "has_limitation", 
    "compares", "finds", "proposes_method", "criticizes", 
    "builds_on", "written_by", "affiliated_with", "mentioned_in"
]


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

class Relationship(BaseModel):
    source_entity: str = Field(description="The name of the source entity.")
    target_entity: str = Field(description="The name of the target entity.")
    relation_type: RELATION_TYPES = Field(description="The type of relationship between source and target.")
    description: Optional[str] = Field(None, description="A brief explanation of how they are related in this context.")
    _source_chunk_id: Optional[str] = PrivateAttr(default=None)

class ExtractedKnowledge(BaseModel):
    entities: List[Entity] = Field(default_factory=list, description="List of all distinct entities found in the text.")
    relationships: List[Relationship] = Field(default_factory=list, description="List of relationships connecting the extracted entities.")

# --- Schema for Entity Linking / Dedup ---

class DedupDecision(BaseModel):
    is_duplicate: bool = Field(description="True if the candidate is the same entity as the master entity.")
    reason: str = Field(description="Brief reason for the decision.")
    merged_name: str = Field(description="The chosen canonical name if it is a duplicate. Otherwise, the original candidate name.")

# --- Schema for Query Routing ---

INTENT_TYPES = Literal["entity", "thematic", "counting", "structural"]

class QueryRouting(BaseModel):
    intent: INTENT_TYPES = Field(description="The classification of the user's query.")
    target_type: Optional[ENTITY_TYPES] = Field(None, description="Required for 'counting' intent. The type of entity being counted or ranked (e.g., 'Method').")
    search_query: Optional[str] = Field(None, description="The refined text query for 'entity' or 'thematic' search. Leave empty for counting if not applicable.")
