from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception
import json
import time

# ──────────────────────────────────────────
# Custom exception for daily quota exhaustion
# ──────────────────────────────────────────
class DailyQuotaExceededError(Exception):
    """Raised when the daily API quota is exhausted. Retrying is pointless."""
    pass

def _is_retryable(exc: BaseException) -> bool:
    """
    Returns True only for transient rate limits (per-minute).
    Returns False (stops retry immediately) for daily quota exhaustion.
    """
    if isinstance(exc, DailyQuotaExceededError):
        return False
    return True

def _raise_if_daily_limit(exc: Exception):
    """
    Inspects a 429 error. If it's a daily quota violation, re-raises as
    DailyQuotaExceededError so tenacity won't bother retrying.
    """
    msg = str(exc)
    if "GenerateRequestsPerDayPerProjectPerModel" in msg:
        raise DailyQuotaExceededError(
            "\n\n[DAILY QUOTA EXHAUSTED]\n"
            "Your free-tier daily request limit for this model has been reached.\n"
            "Quota resets at midnight (Pacific Time) / ~15:00 WIB.\n"
            "Progress has been saved. Run 'python main.py build' again tomorrow.\n"
        ) from exc

def print_retry(retry_state):
    exc = retry_state.outcome.exception()
    print(f"\n[Warning] API call failed. Retrying... (Attempt {retry_state.attempt_number})")
    if exc:
        print(f"Error: {exc}")

from src.config import GEMINI_API_KEY, ExtractedKnowledge, DedupDecision, QueryRouting, PaperMeta

class GeminiClient:
    def __init__(self, model_name: str = "gemini-flash-lite-latest"):
        self.model_name = model_name
        if not GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY not found in environment variables.")
        
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    @retry(
        wait=wait_exponential(multiplier=2, min=10, max=90),
        stop=stop_after_attempt(6),
        retry=retry_if_exception(_is_retryable),
        before_sleep=print_retry
    )
    def extract_entities(self, text_chunk: str, paper_title: str = None) -> ExtractedKnowledge:
        """
        Extract entities and relationships from a text chunk.
        Uses structured outputs to enforce the Pydantic schema.
        `paper_title` (B1): the canonical title of the paper this chunk belongs to, so the LLM
        anchors this paper's authors / proposed method to a consistent Paper entity.
        """
        paper_line = (
            f'\n        THIS TEXT IS FROM THE PAPER TITLED: "{paper_title}".\n'
            if paper_title else ""
        )
        prompt = f"""
        You are an academic paper analyzer. Extract entities and relationships strictly from the text chunk below.
        {paper_line}
        STRICT RULES:
        1. Only extract entities and relationships EXPLICITLY stated in this text. Do NOT infer, guess, or add outside knowledge.
        2. Prefer the full canonical name over an acronym when the text defines both (write "Batch Normalization" not "BN", "Stochastic Gradient Descent" not "SGD"). Keep established proper names as-is ("ResNet", "VGG", "ImageNet", "GoogLeNet" are correct — never invent an expansion for them). If an acronym is never defined here and you cannot confidently expand it, keep it exactly as written.
        3. Preserve specific model/variant names exactly (e.g. "ResNet-50", "VGG-16", "CIFAR-100") — do NOT collapse them to a generic base name.
        4. Do NOT extract vague generic terms: "the model", "our approach", "the method", "the system", "the framework", "the algorithm", "this paper", "the proposed method", "the network".
        5. For Finding: extract only MAJOR results or claims as concise statements (max 15 words). Skip minor observations.
        6. For Limitation: only extract weaknesses EXPLICITLY stated by the authors.
        7. For Author: only extract the actual authors of THIS paper (listed near the title). Do NOT extract authors from in-text citations (e.g. skip "Simonyan et al. (2014)") and do NOT extract initials from acknowledgements.
        8. Do NOT extract anything from a references, bibliography, or acknowledgements section.
        9. If a paper title is given above, use it EXACTLY (verbatim) as the name of the Paper entity
           for THIS document. Connect this paper's own authors to it with `written_by`, and the main
           method/model it proposes with `proposes_method`, whenever they appear in this text.

        ENTITY TYPE GUIDE:
        - Paper: the academic paper itself
        - Author: a person who wrote this paper
        - Institution: a university, company, or lab (affiliation)
        - Method: a technique, model, architecture, layer, or algorithm
        - Dataset: a named dataset (e.g. "ImageNet", "CIFAR-10")
        - Metric: an evaluation measure (e.g. "Top-1 Error", "Accuracy")
        - Finding: a major empirical result or claim
        - Limitation: a weakness explicitly stated by the authors
        - Task: a problem the work addresses (e.g. "Image Classification", "Object Detection")

        RELATIONSHIP TYPES (use exactly these): uses_method, uses_dataset, uses_metric, has_limitation, compares, finds, proposes_method, criticizes, builds_on, written_by, affiliated_with, mentioned_in
        
        Text:
        {text_chunk}
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractedKnowledge,
                    temperature=0.1
                )
            )
        except Exception as e:
            _raise_if_daily_limit(e)
            raise
        
        # Respect rate limits: ~10 requests/minute = 6 seconds/request
        time.sleep(6.0)
        
        try:
            return ExtractedKnowledge.model_validate_json(response.text)
        except Exception as e:
            print(f"Failed to parse LLM output: {e}")
            return ExtractedKnowledge(entities=[], relationships=[])


    @retry(
        wait=wait_exponential(multiplier=2, min=10, max=90),
        stop=stop_after_attempt(6),
        retry=retry_if_exception(_is_retryable),
        before_sleep=print_retry
    )
    def extract_paper_metadata(self, first_page_text: str) -> PaperMeta:
        """
        One call per PDF: extract the paper's title + the popular short names its model is known by.
        The popular name (e.g. 'ResNet') is usually NOT in the paper's own text, so we ask the LLM
        to use its world knowledge — this generalizes to any paper instead of a hardcoded map.
        """
        prompt = f"""
        The text below is the first page of an academic paper.
        Return:
        - "title": the exact, full title of THIS paper.
        - "models": the popular short name(s) or acronym(s) that the main model/architecture this
          paper introduces is commonly known by in the research community (e.g. "ResNet", "VGG-16",
          "GoogLeNet", "DenseNet"). Use widely-recognized names even if the paper's own text only
          uses a longer formal description. If it introduces no well-known named model, return [].

        Text:
        {first_page_text}
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PaperMeta,
                    temperature=0.0,
                ),
            )
        except Exception as e:
            _raise_if_daily_limit(e)
            raise

        time.sleep(6.0)
        try:
            return PaperMeta.model_validate_json(response.text)
        except Exception:
            return PaperMeta(title="", models=[])

    @retry(
        wait=wait_exponential(multiplier=2, min=10, max=90),
        stop=stop_after_attempt(6),
        retry=retry_if_exception(_is_retryable),
        before_sleep=print_retry
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
        
        Decide whether the new entity is the SAME real-world entity as one in the list above.
        Merge ONLY true synonyms/acronyms/casing variants that mean 100% the same thing
        (e.g. "ConvNet" == "Convolutional Neural Network", "BN" == "Batch Normalization").

        CRITICAL 1 - DO NOT merge entities that differ by a number, version, or size, even if they
        share a base name. "ResNet-44" vs "ResNet-56", "CIFAR-10" vs "CIFAR-100", "Top-1 Error" vs
        "Top-5 Error", "VGG-16" vs "VGG-19" are DISTINCT entities. Keep them separate.

        CRITICAL 2 - DO NOT map a specific entity onto a broader/more-general one (no hypernymy).
        e.g. do NOT merge "Top-1 Error" into "Classification Error", nor "ResNet-50" into "ResNet".

        If it is a true duplicate, set is_duplicate=true and merged_name to the matching canonical
        name from the list. Otherwise set is_duplicate=false and merged_name to the original
        candidate name "{candidate_name}".
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DedupDecision,
                    temperature=0.0
                )
            )
        except Exception as e:
            _raise_if_daily_limit(e)
            raise
        
        time.sleep(6.0)
        
        try:
            return DedupDecision.model_validate_json(response.text)
        except Exception as e:
            return DedupDecision(is_duplicate=False, reason="parse error", merged_name=candidate_name)


    @retry(
        wait=wait_exponential(multiplier=2, min=10, max=90),
        stop=stop_after_attempt(5),
        retry=retry_if_exception(_is_retryable),
        before_sleep=print_retry
    )
    def summarize_community(self, community_text: str) -> str:
        """Summarize a community of entities for thematic search."""
        prompt = f"""
        Summarize the following group of related academic entities. 
        What is the common theme, approach, or finding they represent?
        
        Entities:
        {community_text}
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3)
            )
        except Exception as e:
            _raise_if_daily_limit(e)
            raise

        time.sleep(6.0)
        return response.text


    @retry(
        wait=wait_exponential(multiplier=2, min=10, max=90),
        stop=stop_after_attempt(5),
        retry=retry_if_exception(_is_retryable),
        before_sleep=print_retry
    )
    def route_query(self, query: str) -> QueryRouting:
        """Route the user query to the appropriate pipeline."""
        prompt = f"""
        Analyze the following user query about academic papers.
        Classify it into one of three intents:
        1. 'counting': The user wants to count or rank entities (e.g., "what method is most used?", "how many datasets"). 
           You MUST also provide the 'target_type' (e.g. Method, Dataset, Paper) they want to count.
        2. 'thematic': The user is asking a broad, narrative, or trend question (e.g., "what are the trends in object detection?").
        3. 'entity': The user is asking about a specific entity or local connections (e.g., "what does ResNet use?", "who wrote YOLO?").
        4. 'structural': The user asks about connections ACROSS papers / the whole graph structure —
           shared or bridging entities (e.g., "what method/dataset appears in multiple papers?",
           "apa yang menjadi jembatan antar paper?", "what is common across the papers?").

        For 'entity': set 'search_query' to ONLY the specific entity/proper-noun name mentioned
        (e.g. "ResNet", "VGG", "ImageNet") — strip question words, verbs, and category words like
        "architecture", "method", "dataset". For 'thematic': set 'search_query' to the topic phrase.

        Query: {query}
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=QueryRouting,
                    temperature=0.0
                )
            )
        except Exception as e:
            _raise_if_daily_limit(e)
            raise

        time.sleep(6.0)
        try:
            return QueryRouting.model_validate_json(response.text)
        except Exception as e:
            return QueryRouting(intent="entity", search_query=query)


    @retry(
        wait=wait_exponential(multiplier=2, min=10, max=90),
        stop=stop_after_attempt(5),
        retry=retry_if_exception(_is_retryable),
        before_sleep=print_retry
    )
    def generate_final_answer(self, query: str, context: str) -> str:
        """Generate final grounded answer based on assembled context."""
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
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
        except Exception as e:
            _raise_if_daily_limit(e)
            raise

        time.sleep(6.0)
        return response.text
