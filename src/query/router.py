from src.llm.llm_service import LLMService
from src.config import QueryRouting

class QueryRouter:
    def __init__(self, llm_client: LLMService):
        self.llm = llm_client

    def route(self, user_query: str) -> QueryRouting:
        """
        Takes a natural language query and uses the LLM to classify intent
        and extract necessary arguments (like target_type for counting).
        """
        routing_info = self.llm.route_query(user_query)
        print(f"Query routed as: Intent='{routing_info.intent}', Target='{routing_info.target_type}'")
        return routing_info
