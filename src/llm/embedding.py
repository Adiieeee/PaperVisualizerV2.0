from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_REVISION

class EmbeddingClient:
    def __init__(self):
        # Load the local bge-small-en-v1.5 model on CPU, pinned to an exact hub revision
        print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu', revision=EMBEDDING_MODEL_REVISION)

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
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two normalized vectors.
        """
        return float(np.dot(vec1, vec2))
