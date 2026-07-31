import os
import re
import json
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, PrivateAttr

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Constants ---
# Embedding Model (Local CPU)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

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
    os.makedirs(os.path.dirname(PAPER_METADATA_PATH), exist_ok=True)
    with open(PAPER_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


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
