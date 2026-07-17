import os
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Constants ---
# Embedding Model (Local CPU)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Dedup
DEDUP_SIMILARITY_THRESHOLD = 0.5  # As per spec: ~0.5 for candidate filtering

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

class Entity(BaseModel):
    name: str = Field(description="The normalized name of the entity. Use canonical naming if possible.")
    type: ENTITY_TYPES = Field(description="The category of the entity.")
    description: Optional[str] = Field(None, description="A brief description of the entity based on the context.")

class Relationship(BaseModel):
    source_entity: str = Field(description="The name of the source entity.")
    target_entity: str = Field(description="The name of the target entity.")
    relation_type: RELATION_TYPES = Field(description="The type of relationship between source and target.")
    description: Optional[str] = Field(None, description="A brief explanation of how they are related in this context.")

class ExtractedKnowledge(BaseModel):
    entities: List[Entity] = Field(default_factory=list, description="List of all distinct entities found in the text.")
    relationships: List[Relationship] = Field(default_factory=list, description="List of relationships connecting the extracted entities.")

# --- Schema for Entity Linking / Dedup ---

class DedupDecision(BaseModel):
    is_duplicate: bool = Field(description="True if the candidate is the same entity as the master entity.")
    reason: str = Field(description="Brief reason for the decision.")
    merged_name: str = Field(description="The chosen canonical name if it is a duplicate. Otherwise, the original candidate name.")

# --- Schema for Query Routing ---

INTENT_TYPES = Literal["entity", "thematic", "counting"]

class QueryRouting(BaseModel):
    intent: INTENT_TYPES = Field(description="The classification of the user's query.")
    target_type: Optional[ENTITY_TYPES] = Field(None, description="Required for 'counting' intent. The type of entity being counted or ranked (e.g., 'Method').")
    search_query: Optional[str] = Field(None, description="The refined text query for 'entity' or 'thematic' search. Leave empty for counting if not applicable.")
