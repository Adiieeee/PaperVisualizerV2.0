import google.generativeai as genai
from google.generativeai.types import generation_types
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted, InternalServerError
import json
import time

from src.config import GEMINI_API_KEY, ExtractedKnowledge, DedupDecision, QueryRouting

# Configure the API key
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")

class GeminiClient:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(self.model_name)

    # Retry strictly on 429 ResourceExhausted (Rate Limit) and 500s.
    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60), 
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type((ResourceExhausted, InternalServerError))
    )
    def extract_entities(self, text_chunk: str) -> ExtractedKnowledge:
        """
        Extract entities and relationships from a text chunk.
        Uses structured outputs to enforce the Pydantic schema.
        """
        prompt = f"""
        Extract all relevant academic entities and relationships from the following text chunk.
        Only use the specified node types and relationship types.
        
        Text:
        {text_chunk}
        """
        
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ExtractedKnowledge,
                temperature=0.1
            )
        )
        
        # Add a manual sleep to aggressively respect 15 RPM (1 request every 4 seconds)
        # Even with tenacity, it's better to pace ourselves.
        time.sleep(4.1) 
        
        try:
            return ExtractedKnowledge.parse_raw(response.text)
        except Exception as e:
            print(f"Failed to parse LLM output: {e}")
            return ExtractedKnowledge(entities=[], relationships=[])


    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60), 
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type((ResourceExhausted, InternalServerError))
    )
    def resolve_dedup(self, candidate_name: str, entity_type: str, master_list: list[str]) -> DedupDecision:
        """
        Determines if a candidate entity is the same as an existing entity in the master list.
        """
        prompt = f"""
        You are an expert in academic data deduplication.
        We have a new extracted entity: "{candidate_name}" of type "{entity_type}".
        Here is the current master list of existing canonical entities of the same type:
        {json.dumps(master_list)}
        
        Is the candidate entity the exact same concept/thing as any entity in the master list?
        Consider acronyms, casing, and slight variations.
        If it is the same, indicate it is a duplicate and provide the canonical merged_name from the master list.
        If it is new, indicate it is NOT a duplicate, and provide the original candidate name as the merged_name.
        """
        
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=DedupDecision,
                temperature=0.0
            )
        )
        
        time.sleep(4.1)
        
        try:
            return DedupDecision.parse_raw(response.text)
        except Exception as e:
            return DedupDecision(is_duplicate=False, reason="parse error", merged_name=candidate_name)


    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60), 
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((ResourceExhausted, InternalServerError))
    )
    def summarize_community(self, community_text: str) -> str:
        """
        Summarize a community of entities for thematic search.
        """
        prompt = f"""
        Summarize the following group of related academic entities. 
        What is the common theme, approach, or finding they represent?
        
        Entities:
        {community_text}
        """
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3
            )
        )
        time.sleep(4.1)
        return response.text


    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60), 
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((ResourceExhausted, InternalServerError))
    )
    def route_query(self, query: str) -> QueryRouting:
        """
        Route the user query to the appropriate pipeline.
        """
        prompt = f"""
        Analyze the following user query about academic papers.
        Classify it into one of three intents:
        1. 'counting': The user wants to count or rank entities (e.g., "what method is most used?", "how many datasets"). 
           You MUST also provide the 'target_type' (e.g. Method, Dataset, Paper) they want to count.
        2. 'thematic': The user is asking a broad, narrative, or trend question (e.g., "what are the trends in object detection?").
        3. 'entity': The user is asking about a specific entity or local connections (e.g., "what does ResNet use?", "who wrote YOLO?").
        
        Query: {query}
        """
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=QueryRouting,
                temperature=0.0
            )
        )
        
        time.sleep(4.1)
        try:
            return QueryRouting.parse_raw(response.text)
        except Exception as e:
            # Fallback
            return QueryRouting(intent="entity", search_query=query)


    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60), 
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((ResourceExhausted, InternalServerError))
    )
    def generate_final_answer(self, query: str, context: str) -> str:
        """
        Generate final grounded answer based on assembled context.
        """
        prompt = f"""
        You are an AI assistant answering questions about academic papers.
        Answer the user's question using ONLY the provided context.
        If the answer is not contained in the context, explicitly state that you don't know based on the provided documents.
        Do not make up information.
        
        Context:
        {context}
        
        Question:
        {query}
        """
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0
            )
        )
        time.sleep(4.1)
        return response.text
