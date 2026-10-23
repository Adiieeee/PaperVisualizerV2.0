from typing import List, Dict
from src.llm.llm_service import LLMService
from src.config import ExtractedKnowledge

class KnowledgeExtractor:
    def __init__(self, llm_client: LLMService):
        self.llm = llm_client

    def extract_from_chunks(self, chunks: List[Dict[str, str]]) -> List[ExtractedKnowledge]:
        """
        Process a list of text chunks and extract entities and relationships.
        """
        results = []
        for i, chunk in enumerate(chunks):
            print(f"Extracting knowledge from chunk {i+1}/{len(chunks)}...")
            knowledge = self.llm.extract_entities(chunk["text"]).knowledge
            
            # Keep track of chunk provenance for grounding
            for entity in knowledge.entities:
                entity._source_chunk_id = chunk["chunk_id"]
            for rel in knowledge.relationships:
                rel._source_chunk_id = chunk["chunk_id"]
                
            results.append(knowledge)
            
        return results
