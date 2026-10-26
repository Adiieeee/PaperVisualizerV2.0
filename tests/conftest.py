"""
Test configuration. All tests run WITHOUT network and WITHOUT LLM quota (the LLM is mocked).
Run: python -m pytest tests -q
"""
import os

# Fake key: just enough for `src.config` to be importable. `load_dotenv()` does not override existing env vars,
# so the real key in .env is never used by the tests.
os.environ["GEMINI_API_KEY"] = "test-key-not-real"
os.environ["GROQ_API_KEY"] = ""  # an empty string also stops load_dotenv from loading the real Groq key
# Must be set before `import litellm` anywhere; without it litellm downloads the cost map from GitHub.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
