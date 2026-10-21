"""
Konfigurasi tes. Semua tes berjalan TANPA jaringan dan TANPA kuota LLM (LLM di-mock).
Jalankan: python -m pytest tests -q
"""
import os

# Key palsu: cukup supaya `src.config` bisa di-import. `load_dotenv()` tidak menimpa env yang sudah ada,
# jadi key asli di .env tidak pernah dipakai oleh tes.
os.environ["GEMINI_API_KEY"] = "test-key-not-real"
os.environ["GROQ_API_KEY"] = ""  # string kosong juga mencegah load_dotenv memuat key Groq asli
# Harus diset sebelum `import litellm` di mana pun; tanpanya litellm mengunduh cost map dari GitHub.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
