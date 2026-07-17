from typing import List, Dict, Optional
from src.llm.gemini_client import GeminiClient
from src.llm.embedding import EmbeddingClient
from src.config import DEDUP_SIMILARITY_THRESHOLD, Entity

class HybridDeduplicator:
    def __init__(self, llm_client: GeminiClient, embed_client: EmbeddingClient):
        self.llm = llm_client
        self.embed = embed_client
        
        # Master list structure: { entity_type: { canonical_name: { "entity": Entity, "embedding": List[float] } } }
        self.master_list: Dict[str, Dict[str, Dict]] = {}

    def resolve_entity(self, candidate: Entity) -> str:
        """
        Takes a candidate entity and returns its canonical name after hybrid deduplication.
        Updates the master list if it's a new entity.
        """
        ent_type = candidate.type
        cand_name = candidate.name
        
        if ent_type not in self.master_list:
            self.master_list[ent_type] = {}
            
        # 1. Exact match
        if cand_name in self.master_list[ent_type]:
            return cand_name
            
        # 2. Embedding candidate filter (Similarity search within the namespace)
        cand_emb = self.embed.embed_text(cand_name)
        
        candidates_to_check = []
        for canonical_name, data in self.master_list[ent_type].items():
            sim = self.embed.cosine_similarity(cand_emb, data["embedding"])
            if sim >= DEDUP_SIMILARITY_THRESHOLD:
                candidates_to_check.append(canonical_name)
                
        # 3. LLM Confirmation
        if candidates_to_check:
            print(f"Dedup candidate: '{cand_name}' vs {candidates_to_check}")
            decision = self.llm.resolve_dedup(cand_name, ent_type, candidates_to_check)
            
            if decision.is_duplicate and decision.merged_name in self.master_list[ent_type]:
                print(f" -> LLM merged '{cand_name}' into '{decision.merged_name}'")
                return decision.merged_name
            else:
                print(f" -> LLM decided '{cand_name}' is NEW.")

        # If not a duplicate, add to master list
        self.master_list[ent_type][cand_name] = {
            "entity": candidate,
            "embedding": cand_emb
        }
        return cand_name

    def process_extracted_knowledge(self, knowledge_list: List) -> List:
        """
        Iterates over a list of ExtractedKnowledge and deduplicates entities in-place.
        Returns the updated list with canonical names for entities and relationships.
        """
        for k in knowledge_list:
            name_mapping = {}
            
            # Dedup entities
            for ent in k.entities:
                canonical = self.resolve_entity(ent)
                name_mapping[ent.name] = canonical
                ent.name = canonical
                
            # Update relationship names
            for rel in k.relationships:
                if rel.source_entity in name_mapping:
                    rel.source_entity = name_mapping[rel.source_entity]
                if rel.target_entity in name_mapping:
                    rel.target_entity = name_mapping[rel.target_entity]
                    
        return knowledge_list
