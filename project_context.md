# GraphRAG Paper Explorer — Konteks Proyek Lengkap

> Dokumen ini dibuat untuk diserahkan ke agent lain agar dapat langsung memahami dan bekerja pada proyek ini tanpa perlu membaca ulang seluruh codebase dari awal.

---

## 1. Tujuan Proyek

Membangun asisten AI akademik yang menjawab pertanyaan berdasarkan **Knowledge Graph** yang dibangun dari kumpulan paper PDF (corpus saat ini: paper computer vision — ResNet, DenseNet, VGG, GoogLeNet, dst).

Sistem menggabungkan:
- **Graph-based retrieval** (NetworkX + Louvain community detection)
- **Vector-based retrieval** (ChromaDB + lokal embedding)
- **LLM generation** (Gemini AI Studio, free tier)

**Hard constraint: Budget $0.** Tidak ada kartu kredit, tidak ada cloud berbayar. Seluruh embedding berjalan lokal.

---

## 2. Struktur Direktori

```
PaperVisualizerV2.0-GraphRAG/
├── main.py                          # CLI entry point: build + query
├── requirements.txt
├── .env                             # GEMINI_API_KEY
├── GraphRAG_Paper_Explorer_Spec.md  # Spesifikasi arsitektur asli
├── data/
│   ├── raw/                         # PDF paper input (taruh di sini)
│   └── processed/
│       ├── checkpoint_extraction.json  # Progress ekstraksi LLM (per chunk)
│       ├── checkpoint_dedup.json       # Progress dedup (per knowledge item)
│       ├── graph.graphml               # Graph final hasil build
│       ├── chroma_db/                  # Vector store lokal
│       ├── similarity_debug.csv        # Log semua cosine similarity (diagnostik)
│       └── graph_visualization.html   # Output visualisasi (generate manual)
└── src/
    ├── config.py                    # Schema Pydantic, konstanta, tipe
    ├── engine.py                    # Orchestrator query pipeline
    ├── ingestion/
    │   └── document.py              # PyMuPDF PDF parser + chunker
    ├── extraction/
    │   ├── extractor.py             # LLM entity extraction per chunk
    │   └── deduplicator.py          # Hybrid dedup (norm → embedding → LLM)
    ├── graph/
    │   ├── builder.py               # NetworkX graph builder
    │   ├── analytics.py             # Louvain community detection + degree stats
    │   └── visualizer.py            # Pyvis HTML visualization generator
    ├── llm/
    │   ├── gemini_client.py         # Semua LLM calls (extract/dedup/summarize/route/generate)
    │   └── embedding.py             # Local embedding via sentence-transformers
    ├── query/
    │   ├── router.py                # LLM query intent classifier
    │   ├── resolver.py              # Hybrid entity resolver (exact → fuzzy → vector)
    │   └── traversal.py             # 1-hop graph traversal dengan Paper-blocking
    └── vectorstore/
        └── chromadb_store.py        # Dua koleksi ChromaDB: chunks + community_summaries
```

---

## 3. Tech Stack

| Komponen | Library | Catatan |
|---|---|---|
| PDF parsing | PyMuPDF (fitz) | |
| LLM | `google-genai` → Gemini Flash Lite (free tier) | Model: `gemini-flash-lite-latest` |
| Structured output | Pydantic v2 | Enforce schema JSON dari LLM |
| Embedding | `sentence-transformers`, model `BAAI/bge-small-en-v1.5` | Lokal, CPU |
| Vector store | ChromaDB (PersistentClient) | Lokal embedded |
| Graph | NetworkX | In-memory, simpan ke GraphML |
| Community detection | `networkx.algorithms.community.louvain_communities` | Bukan `python-louvain` eksterna |
| Visualisasi | Pyvis (vis.js) | Output HTML interaktif |
| Fuzzy matching | `difflib.SequenceMatcher` | Fallback query resolver |
| Rate limit handling | `tenacity` | Retry exponential backoff, distinguish transient vs daily quota |

---

## 4. Schema Data (src/config.py)

### Entity Types (9, fixed)
```
Paper, Author, Institution, Method, Dataset, Metric, Finding, Limitation, Task
```

### Relationship Types (12, fixed)
```
uses_method, uses_dataset, uses_metric, has_limitation,
compares, finds, proposes_method, criticizes,
builds_on, written_by, affiliated_with, mentioned_in
```

### Pydantic Models
```python
class Entity(BaseModel):
    name: str
    type: ENTITY_TYPES
    description: Optional[str]

class Relationship(BaseModel):
    source_entity: str
    target_entity: str
    relation_type: RELATION_TYPES
    description: Optional[str]

class ExtractedKnowledge(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

class DedupDecision(BaseModel):
    is_duplicate: bool
    reason: str
    merged_name: str

class QueryRouting(BaseModel):
    intent: Literal["entity", "thematic", "counting"]
    target_type: Optional[ENTITY_TYPES]
    search_query: Optional[str]
```

---

## 5. Pipeline Build (`python main.py build`)

### Tahap 1: Ingestion
- `DocumentProcessor` memproses semua PDF di `data/raw/`
- Output: list of chunks `[{"chunk_id", "doc_id", "text"}]`
- Checkpoint per-chunk di `checkpoint_extraction.json`

### Tahap 2: Extraction
- `KnowledgeExtractor` memanggil `GeminiClient.extract_entities(chunk_text)`
- LLM mengembalikan `ExtractedKnowledge` (entities + relationships) per chunk
- Structured output via Pydantic schema, temperature=0.1
- Rate: `time.sleep(6.0)` setelah setiap call (respek free tier ~10 req/menit)

### Tahap 3: Entity Dedup — Hybrid 4-Lapis

**Kelas:** `HybridDeduplicator` di `src/extraction/deduplicator.py`

**Konfigurasi:**
```python
DEDUP_SIMILARITY_THRESHOLD = 0.85  # Di src/config.py
```

**Alur per entity (`resolve_entity`):**

1. **Fast-lookup cache** — `name_mapping_global` (O(1), sudah pernah diproses sebelumnya)
2. **Exact match** — `master_list[type][name]`
3. **Normalisasi** — lower + strip punctuation + `KNOWN_ACRONYMS` lookup:
   ```python
   KNOWN_ACRONYMS = {
       "sgd": "Stochastic gradient descent",
       "bn": "Batch Normalization",
       "lstm": "Long Short-Term Memory",
       "relu": "Rectified linear unit",
       "fcn": "Fully convolutional network"
   }
   ```
   > **PERHATIAN**: Dictionary ini manual, domain-specific. Jika ada paper baru dengan akronim baru, harus ditambah manual — kalau tidak, akronim akan jatuh ke zona noise cosine ~0.5-0.6 dan tidak tertangkap.

4. **Embedding similarity** — cosine similarity vs semua entity bertipe sama di `master_list`. Hanya kandidat dengan skor ≥ 0.85 yang lanjut ke LLM.

5. **LLM final decision** — `GeminiClient.resolve_dedup()`, temperature=0.0. LLM memutuskan is_duplicate: true/false dan canonical name.

**Namespace isolation**: Entity hanya dibandingkan dengan entity bertipe sama.

**Checkpoint**: Disimpan ke `checkpoint_dedup.json` setelah setiap knowledge item selesai. File dihapus hanya jika build selesai sukses penuh.

**Relasi juga diupdate**: Setelah merge, `source_entity` dan `target_entity` di semua relasi diupdate ke `name_mapping_global` (bukan hanya chunk-local mapping).

**Diagnostik**: Semua perbandingan similarity dicatat ke `similarity_debug.csv` (bukan hanya yang lolos threshold), termasuk kolom `llm_decision`.

### Tahap 4: Graph Building

**Kelas:** `GraphBuilder` di `src/graph/builder.py`

- Entity → node dengan atribut `type`, `description`, `source_chunk_id` (semua di-coerce ke string karena GraphML tidak support `None`)
- Relationship → edge dengan atribut `relation`, `description`, `source_chunk_id`
- Disimpan ke `data/processed/graph.graphml`

### Tahap 5: Community Detection + Summarization

**Kelas:** `GraphAnalytics` di `src/graph/analytics.py`

- Node `Paper` **di-exclude** dari clustering (mencegah paper jadi artificial hub)
- Louvain dari `networkx.algorithms.community`
- Setiap community dirangkum LLM: `GeminiClient.summarize_community()`
- Hasil di-embed dan disimpan ke ChromaDB collection `community_summaries`

### Tahap 6: Chunk Embedding + ChromaDB

- Semua chunk di-embed secara batch
- Disimpan ke ChromaDB collection `chunks`

---

## 6. Pipeline Query (`python main.py query --query "..."`)

### Kelas Orchestrator: `GraphRAGEngine` (src/engine.py)

**Langkah:**

1. **Query Routing** (`QueryRouter` → `GeminiClient.route_query()`)
   - Mengklasifikasikan intent: `"entity"`, `"thematic"`, atau `"counting"`
   - Untuk counting: juga mengekstrak `target_type`

2. **Eksekusi berdasarkan intent:**

   **`counting`**: Direct graph stats via `GraphAnalytics.get_counting_stats(target_type)`. Tidak memanggil LLM atau vector search.

   **`thematic`**: Vector similarity ke `community_summaries` ChromaDB collection.

   **`entity`** (default):
   - `QueryResolver.resolve_entity()`: exact match → fuzzy match (SequenceMatcher threshold 0.8) → return `None` jika tidak ditemukan
   - Jika ditemukan: `GraphTraversal.traverse_1_hop()` — mengambil semua 1-hop neighbors beserta edge data
   - Jika tidak ditemukan: fallback ke vector search di `chunks` ChromaDB collection

3. **Context Assembly** — gabungkan semua hasil jadi string

4. **LLM Generation** (`GeminiClient.generate_final_answer()`, temperature=0.0) — grounded ke konteks, menolak menjawab jika info tidak tersedia

### Paper-blocking di Traversal
Node `Paper` tetap muncul sebagai neighbor (provenance), tapi traversal tidak dilanjutkan melalui mereka (relevan untuk multi-hop ke depan).

---

## 7. Cara Menjalankan

### Build (jalankan sekali saat ada paper baru)
```bash
python main.py build
```
Otomatis resume dari checkpoint jika build sebelumnya terpotong (misal karena quota habis).

### Query (chatbot CLI)
```bash
python main.py query --query "What methods does ResNet use?"
python main.py query --query "What are the trends in the corpus?"
python main.py query --query "Which method is most commonly used?"
```

### Visualisasi Graph (generate manual)
```bash
python -c "
import networkx as nx
from src.graph.visualizer import GraphVisualizer
G = nx.read_graphml('data/processed/graph.graphml')
viz = GraphVisualizer(G)
viz.generate_html('data/processed/graph_visualization.html')
"
```
Buka file `data/processed/graph_visualization.html` di browser. Node berwarna per tipe, bisa di-drag dan hover.

---

## 8. LLM Client Details (src/llm/gemini_client.py)

- Model: `gemini-flash-lite-latest` (free tier, 500 req/day)
- Rate limiting: manual `time.sleep(6.0)` setelah setiap call
- Error handling:
  - Transient 429 → retry dengan exponential backoff (tenacity)
  - Daily quota exhausted → raise `DailyQuotaExceededError` (tidak di-retry, checkpoint sudah tersimpan)
- Semua calls menggunakan structured output (`response_mime_type="application/json"`, Pydantic schema)

---

## 9. Known Limitations

1. **Threshold 0.85 bergantung pada KNOWN_ACRONYMS**: Akronim domain yang tidak ada di dictionary akan lolos dari dedup dan berdiri sebagai node duplikat terpisah di graph.

2. **1-hop traversal ceiling**: Pertanyaan multi-hop ("metode apa yang mengatasi kelemahan arsitektur X yang dipakai paper Y?") tidak didukung oleh local search.

3. **Fuzzy match resolver lemah**: `SequenceMatcher` dengan threshold 0.8 tidak menangkap query yang makna sama tapi leksikalnya jauh berbeda. Spec menyebutkan vector similarity sebagai jalur resolusi entity juga, tapi implementasi saat ini hanya exact + fuzzy (tidak ada vector search ke node graph).

4. **Community clustering dieksekusi tanpa validasi manual**: Hasil Louvain bisa jadi tidak bermakna secara tematik kalau graph-nya sparse atau Paper-blocking gagal.

5. **Garbage in, garbage out**: Seluruh sistem bergantung pada akurasi LLM di tahap ekstraksi. Tidak ada mekanisme koreksi downstream.

6. **Tidak ada web UI**: Chatbot dan visualisasi hanya tersedia via CLI + file HTML statis.

---

## 10. File yang Dihasilkan Setelah Build Selesai

| File | Isi |
|---|---|
| `data/processed/graph.graphml` | Graph final (NetworkX GraphML) |
| `data/processed/chroma_db/` | Vector store lokal (chunks + community summaries) |
| `data/processed/similarity_debug.csv` | Log diagnostik semua cosine similarity comparison |
| `data/processed/graph_visualization.html` | Visualisasi interaktif (generate manual) |

---

## 11. Catatan Pengembangan Saat Ini

- Build pipeline sudah selesai dijalankan (127/127 knowledge items processed)
- `graph.graphml` (228 KB) dan `chroma_db` sudah tersedia
- `checkpoint_extraction.json` sudah terhapus (build selesai sukses)
- `checkpoint_dedup.json` mungkin masih ada atau sudah terhapus tergantung apakah cleanup berjalan — cek `data/processed/`
- Visualisasi HTML belum di-generate (perlu jalankan perintah manual di atas)
- Komponen yang **belum dievaluasi** kualitasnya secara end-to-end: query pipeline (apakah jawaban akurat), community summaries (apakah tematik atau hanya per-paper), edge case dangling edges di graph
