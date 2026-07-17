from typing import List, Dict
from src.llm.gemini_client import GeminiClient
from src.config import ExtractedKnowledge

class KnowledgeExtractor:
    def __init__(self, llm_client: GeminiClient):
        self.llm = llm_client

    def extract_from_chunks(self, chunks: List[Dict[str, str]]) -> List[ExtractedKnowledge]:
        """
        Process a list of text chunks and extract entities and relationships.
        """
        results = []
        for i, chunk in enumerate(chunks):
            print(f"Extracting knowledge from chunk {i+1}/{len(chunks)}...")
            knowledge = self.llm.extract_entities(chunk["text"])
            
            # Keep track of chunk provenance for grounding
            for entity in knowledge.entities:
                setattr(entity, "source_chunk_id", chunk["chunk_id"])
            for rel in knowledge.relationships:
                setattr(rel, "source_chunk_id", chunk["chunk_id"])
                
            results.append(knowledge)
            
        return results
