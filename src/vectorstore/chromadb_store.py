import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict

class ChromaStore:
    def __init__(self, persist_directory: str = "data/processed/chroma_db"):
        os.makedirs(persist_directory, exist_ok=True)
        # Initialize local persistent Chroma client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Two collections as per spec
        self.chunks_collection = self.client.get_or_create_collection(name="chunks")
        self.community_collection = self.client.get_or_create_collection(name="community_summaries")

    def add_chunks(self, chunks: List[Dict[str, str]], embeddings: List[List[float]]):
        """
        Add raw document chunks to ChromaDB.
        """
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [{"doc_id": chunk["doc_id"]} for chunk in chunks]
        
        self.chunks_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Upserted {len(ids)} chunks to ChromaDB.")

    def add_community_summaries(self, summaries: List[Dict[str, str]], embeddings: List[List[float]]):
        """
        Add community summaries to ChromaDB.
        summaries is a list of dicts: {"community_id": "...", "summary": "..."}
        """
        ids = [str(s["community_id"]) for s in summaries]
        documents = [s["summary"] for s in summaries]
        
        self.community_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents
        )
        print(f"Upserted {len(ids)} community summaries to ChromaDB.")

    def search_chunks(self, query_embedding: List[float], n_results: int = 5):
        """
        Search for most relevant chunks.
        """
        results = self.chunks_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

    def search_communities(self, query_embedding: List[float], n_results: int = 3):
        """
        Search for most relevant community summaries.
        """
        results = self.community_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
