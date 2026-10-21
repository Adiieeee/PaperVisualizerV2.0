from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

from src.config import (
    EMBEDDING_BATCH_SIZE, EMBEDDING_DEVICE, EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_REVISION,
)


def resolve_device(setting: str = EMBEDDING_DEVICE) -> str:
    """'auto' -> 'cuda' bila torch versi CUDA melihat GPU, selain itu 'cpu'. 'cpu'/'cuda' dipakai apa adanya."""
    setting = (setting or "auto").strip().lower()
    if setting != "auto":
        return setting
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingClient:
    def __init__(self):
        # Local model pinned to an exact hub revision; GPU when available (see EMBEDDING_DEVICE)
        self.device = resolve_device()
        print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME} on {self.device}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device, revision=EMBEDDING_MODEL_REVISION)
        # Never above the model's own limit (bge-small: 512; bge-m3: 8192 -> capped to save memory/time)
        self.model.max_seq_length = min(EMBEDDING_MAX_SEQ_LENGTH, self.model.max_seq_length)

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single string.
        """
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of strings.
        """
        embeddings = self.model.encode(texts, normalize_embeddings=True, batch_size=EMBEDDING_BATCH_SIZE)
        return embeddings.tolist()

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two normalized vectors.
        """
        return float(np.dot(vec1, vec2))
