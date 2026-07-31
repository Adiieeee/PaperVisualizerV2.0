# Technical Reference: GraphRAG Pipeline

Dokumen ini menjelaskan secara lengkap dan detail seluruh mekanisme teknis yang digunakan di proyek PaperVisualizer-GraphRAG, mulai dari ekstraksi informasi dari PDF, penanganan duplikasi entitas, aturan pembangunan graf, hingga cara kerja chatbot.

---

## 1. Ekstraksi Informasi dari Paper (LLM Extraction)

### 1.1 Alur Pre-extraction

Sebelum teks dikirim ke LLM, ada dua tahap persiapan:

**Section Filter (`app/section_filter.py`)**
Teks PDF difilter menggunakan regex pattern untuk membuang bagian yang tidak produktif. Pattern yang dibuang (didefinisikan di `config.py`):
```
- References / Bibliography
- Appendix / Appendices
- Acknowledgements
- Notes
- Supplementary Material
```
Tujuannya: mengurangi noise (nama penulis dari sitasi, kode supplementary) dan menghemat API call.

**Chunking (`app/chunker.py`)**
Teks hasil filter dipotong menjadi chunk berukuran ~3500 karakter dengan overlap ~500 karakter. Overlap bertujuan mengurangi kehilangan informasi di batas antar chunk (cross-chunk entity loss).

---

### 1.2 Prompt Ekstraksi (Isi Lengkap)

Setiap chunk dikirim ke Gemini API dengan prompt berikut:

```
You are an academic paper analyzer. Extract entities and relationships from the text below.

PAPER METADATA:
Title: {paper_title}
Year: {paper_year}
Venue: {paper_venue}

TEXT CHUNK:
{text_chunk}

STRICT RULES:
1. Only extract entities EXPLICITLY mentioned in this text. Do NOT infer.
2. Always use the FULL NAME, never acronyms. Write "Retrieval-Augmented Generation" not "RAG",
   "Large Language Model" not "LLM", "Bidirectional Encoder Representations from Transformers"
   not "BERT", etc. If the text only uses an acronym without defining it, skip that entity.
3. Do NOT extract generic terms: "the model", "our approach", "the method", "the system",
   "the framework", "the algorithm", "this paper", "the proposed method".
4. For Finding: extract only MAJOR findings as concise statements (max 15 words).
   Skip minor observations.
5. For Limitation: only extract limitations EXPLICITLY stated by the authors.
6. Do NOT extract anything from a references or bibliography section.
7. For Author: Extract ALL authors listed under the paper title.
   ONLY extract the actual authors of the current paper.
8. Do NOT extract authors from in-text citations (e.g., skip "Smith et al. (2020)").
   Do NOT extract author initials (e.g., "D.J.", "M.C.") from Author Contributions sections.
9. Output MUST be valid JSON only. No markdown, no explanation, no code blocks.

ENTITY TYPES (use exactly): Paper | Author | Institution | Method | Dataset | Metric | Finding | Limitation

RELATIONSHIP TYPES (use exactly): written_by | affiliated_with | uses_method | proposes_method |
uses_dataset | uses_metric | finds | has_limitation | criticizes | builds_on | compares

OUTPUT FORMAT (JSON only):
{
  "entities": [
    {"name": "full name here", "type": "EntityType"}
  ],
  "relationships": [
    {"source": "entity name", "target": "entity name", "type": "relationship_type",
     "description": "brief description"}
  ]
}
```

### 1.3 Penjelasan Setiap Aturan Prompt

| Rule | Tujuan |
|------|--------|
| Rule 1 (explicit only) | Mencegah LLM mengarang entitas/relasi yang tidak ada di teks |
| Rule 2 (full name) | Mengurangi variasi ejaan sejak awal → mengurangi beban deduplikasi di tahap berikutnya |
| Rule 3 (no generic) | Mencegah node tidak bermakna seperti "the model" masuk ke graf |
| Rule 4 (finding max 15 kata) | Menjaga agar node Finding tetap ringkas dan bisa ditampilkan di visualisasi |
| Rule 5 (limitation explicit) | Mencegah LLM mengarang kelemahan yang tidak dinyatakan penulis |
| Rule 6 (no references) | Section filter sudah membuang bagian ini, tapi instruksi ini sebagai jaring pengaman |
| Rule 7-8 (author) | Mencegah nama penulis dari sitasi (bukan penulis paper ini) masuk ke graf |
| Rule 9 (JSON only) | Memastikan output bisa di-parse secara programatik |

### 1.4 Validasi dan Pembersihan Output (`_validate_and_clean`)

Setelah LLM mengembalikan JSON, dilakukan validasi:
- Tipe entitas dicek terhadap `VALID_ENTITY_TYPES` (8 tipe). Tipe tidak dikenal dibuang.
- Tipe relasi dicek terhadap `VALID_RELATIONSHIP_TYPES` (11 tipe). Tipe tidak dikenal dibuang.
- Nama entitas kurang dari 2 karakter dibuang.
- Nama yang masuk `GENERIC_ENTITY_BLOCKLIST` dibuang (contoh: "the model", "our approach", "model", "method").
- Self-loop (source == target) dibuang.

### 1.5 Masalah yang Diketahui di Tahap Ekstraksi

**Cross-chunk Entity Loss**
Satu relasi bisa terpotong di batas antar chunk. Mitigasi: overlap antar chunk, tapi tidak sempurna.

**Hallucination & Inferensi**
LLM kadang mengekstrak relasi yang tersirat tapi tidak eksplisit. Rule 1 upaya pencegahan, tapi LLM tidak selalu patuh.

**Section Pollution**
Jika format PDF tidak standar, section filter bisa gagal membuang References/Acknowledgements. Noise dari bagian ini akan masuk ke graf.

**Granularity Tidak Konsisten**
LLM mengekstrak "ResNet-50" (spesifik) dari satu chunk dan "ResNet" (generik) dari chunk lain. Keduanya berbeda secara string tapi mungkin merujuk hal yang sama.

**Missing Context**
Setiap chunk dikirim secara independen. LLM di chunk ke-5 tidak tahu bahwa "the proposed method" merujuk ke "Dense Block" yang didefinisikan di chunk ke-1.

### 1.6 Caching

Setiap paper yang berhasil diekstrak disimpan sebagai JSON di `output/extractions/<nama_file>.json`. Pada re-run, jika file JSON sudah ada, ekstraksi di-skip dan langsung dimuat dari cache. Ini menghemat API call secara signifikan.

---

## 2. Penanganan Entity Duplication

Setelah semua paper diekstrak, entitas dari berbagai chunk dan paper digabungkan. Di sinilah duplikasi menjadi masalah utama: "DenseNet", "Dense Convolutional Network", "densenet" semuanya merujuk hal yang sama.

### 2.1 Lapis 1 — Normalisasi Kasar (Tanpa API)

**Lokasi:** `graph_builder.py`, sebelum daftar entitas dikirim ke LLM Entity Linking.

Fungsi `_normalize()` melakukan:
```python
def _normalize(name: str) -> str:
    return name.lower().strip().strip(".,;:!?\"'()[]{}")
```

Semua variasi casing dan whitespace dari nama yang sama digabung di sini. Contoh: `"ImageNet"`, `"imagenet"`, `"IMAGENET"` → hanya satu yang dikirim ke tahap berikutnya.

Implementasi menggunakan `seen_normalized` set:
```python
raw_names = []
seen_normalized = set()
for ent in raw_entities:
    t = ent.get("type", "Unknown")
    name = ent.get("name", "").strip()
    if name and t in ["Method", "Dataset", "Metric"]:
        norm = _normalize(name)
        if norm not in seen_normalized:
            seen_normalized.add(norm)
            raw_names.append(name)  # Casing asli dipertahankan untuk display
```

Dampak terukur: mengurangi entitas dari 381 menjadi 336 (hemat 45 entity / 11.81% API call).

### 2.2 Lapis 2 — LLM Entity Linking (`entity_resolver.py`)

Daftar entitas unik (setelah lapis 1) dikirim ke Gemini API secara batch (20 item per chunk). Prompt lengkap:

```
You are an expert academic knowledge graph curator.
Your task is Entity Linking (deduplication).
You are given a "Master List of Seen Entities" and a "New Entities Chunk".
For each entity in the "New Entities Chunk", check if it is a duplicate, synonym,
or acronym of any entity ALREADY IN the "Master List of Seen Entities".
If it is a duplicate, output a JSON mapping: {"New Entity": "Matched Entity from Master List"}.
If it is NOT a duplicate of anything in the Master List, DO NOT include it in the JSON output.
DO NOT group entities that are distinct versions, variants, or opposite concepts
(e.g., 'ResNet-44' vs 'ResNet-56' are distinct).
CRITICAL 1: JANGAN gabungkan entitas yang mengandung angka, metrik, atau nomor versi yang berbeda
(misal: 'Top-1' vs 'Top-5', 'CIFAR-10' vs 'CIFAR-100', 'ResNet-44' vs 'ResNet-56') meskipun
mereka berada dalam kategori yang sama. Mereka adalah entitas yang BERBEDA secara semantik.
CRITICAL 2: JANGAN petakan entitas spesifik ke kategori yang lebih umum (Hypernymy). Contoh:
JANGAN petakan 'Top-1 Error' atau 'Top-5 Error' ke 'Classification error'. HANYA gabungkan
jika keduanya adalah sinonim yang 100% sama maknanya (seperti 'ConvNet' dan 'Convolutional
Neural Network').

Master List of Seen Entities:
{master_list}

New Entities Chunk:
{entity_chunk}

Output ONLY valid JSON. Example: {"DenseNet": "Dense Convolutional Network"}
```

**Mekanisme incremental:**
- Entitas diproses per chunk 20 item.
- Setelah setiap chunk, entitas yang baru diproses ditambahkan ke Master List.
- Entitas yang terdeteksi sebagai duplikat oleh LLM ditambahkan ke kamus alias, bukan ke Master List.
- Hasilnya di-cache ke `entity_aliases.json` (berisi `aliases` dan `checked_entities`).

**Proteksi pada prompt (hasil iterasi perbaikan):**
- `CRITICAL 1`: Melarang penggabungan entitas dengan angka/versi berbeda. Ditambahkan setelah ditemukan kasus "Top-1 Error" dan "Top-5 Error" digabung.
- `CRITICAL 2`: Melarang hypernymy (pemetaan spesifik ke umum). Ditambahkan karena LLM menemukan celah dari CRITICAL 1 — ia tidak memetakan Top-1 ke Top-5 secara horizontal, tapi keduanya dipetakan ke atas ke "Classification error".

**Cycle Detection:**
Saat menelusuri rantai alias (A → B → C), ada potensi infinite loop jika LLM menghasilkan pemetaan sirkular (A → B → A). Diatasi dengan:
```python
final_target = matched_ent
visited = set()
while final_target in aliases and final_target not in visited:
    visited.add(final_target)
    final_target = aliases[final_target]
```

### 2.3 Lapis 3 — Fallback Fuzzy Match & Substring Match

Di dalam `_deduplicate_entities()` di `graph_builder.py`, untuk entitas yang tidak ter-handle oleh lapis 1 atau 2:

**Untuk tipe Author:**
- Substring check: jika satu nama terkandung di dalam nama lain (misal "K. He" dan "Kaiming He"), keduanya digabung. Nama yang lebih panjang dipertahankan sebagai canonical.
- Syarat minimum: kedua nama harus lebih dari 4 karakter (mencegah false positive pada nama pendek).

**Untuk semua tipe:**
- Exact match setelah normalisasi.
- Entity type diproses terpisah (tidak pernah menggabungkan Method dengan Author meskipun namanya mirip).

### 2.4 Kamus Akronim Statis (`config.py`)

Sebagai safety net jika LLM tetap mengembalikan akronim meskipun sudah dilarang di prompt, ada kamus `COMMON_ACRONYMS` yang berisi ~30 akronim umum ML/AI:
```python
COMMON_ACRONYMS = {
    "rag":   "Retrieval-Augmented Generation",
    "cnn":   "Convolutional Neural Network",
    "lstm":  "Long Short-Term Memory",
    ...
}
```

Digunakan oleh `_expand_acronym()` dan `_resolve()` saat mencocokkan nama entitas mentah ke nama canonical.

---

## 3. Aturan Pembangunan Graf

### 3.1 Pre-processing Author

Sebelum entitas Author masuk ke graf:
- Hapus " et al." dan tahun kutipan (misal ", 2024")
- Split nama gabungan: "He & Zhang" → dua entitas terpisah "He" dan "Zhang"
- Relasi yang merujuk ke nama gabungan otomatis diduplikasi ke masing-masing nama yang di-split

### 3.2 No Orphan Nodes

Setiap node yang tidak memiliki jalur (path) ke Paper sumbernya secara paksa dihubungkan menggunakan relasi `mentioned_in`. Pengecekan menggunakan `nx.has_path()`.

Implementasi di `_connect_isolated_nodes()`.

### 3.3 No Self-Loop

Relasi dari entitas ke dirinya sendiri otomatis di-skip. Ini bisa terjadi karena dua entitas yang awalnya berbeda nama ternyata ter-merge ke canonical yang sama setelah deduplikasi.

### 3.4 No Duplicate Edges

Pasangan edge (A, B, tipe_relasi) hanya boleh ada satu kali. Pengecekan dilakukan secara undirected — (A→B, `uses_method`) dan (B→A, `uses_method`) dianggap sama.

Implementasi menggunakan `_canonical_pair()` yang mengurutkan nama secara alfabet:
```python
def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (min(a, b), max(a, b))
```

### 3.5 Edge Hanya untuk Node yang Ada

Relasi yang merujuk ke entitas (source atau target) yang tidak ada di daftar node setelah deduplikasi otomatis di-skip. Dicek via `source not in graph.nodes or target not in graph.nodes`.

### 3.6 Validasi Tipe Entitas dan Relasi

Hanya 8 tipe entitas dan 11 tipe relasi yang diterima (didefinisikan di `config.py`). Entitas/relasi dengan tipe di luar daftar dibuang di tahap `_validate_and_clean()`.

### 3.7 Blocklist Entitas Generik

Nama-nama generik difilter keluar meskipun LLM mengekstraknya. Daftar di `config.py`:
```python
GENERIC_ENTITY_BLOCKLIST = {
    "the model", "the approach", "the method", "the dataset",
    "the system", "the algorithm", "this model", "our model",
    "the proposed method", "our approach", "the proposed approach",
    "the framework", "our framework", "this framework",
    "the technique", "the network", "the architecture",
    "a model", "a method", "an approach",
    "model", "method", "approach", "technique",
}
```

### 3.8 Minimum Name Length

Entitas dengan nama kurang dari 2 karakter otomatis dibuang.

### 3.9 Same-Author Edge Antar Paper

Jika dua paper memiliki Author yang sama (terdeteksi dari relasi `written_by`), edge `same_author` otomatis ditarik antar kedua paper tersebut. Implementasi di `_add_same_author_edges()`.

---

## 4. Visualisasi

**Ukuran Node:** Ditentukan oleh jumlah relasi (degree). Node dengan banyak koneksi ditampilkan lebih besar.

**Warna Node:** Setiap tipe entitas punya warna berbeda (didefinisikan di `config.py`):
| Tipe | Warna |
|------|-------|
| Paper | #4A90D9 (Biru) |
| Author | #E74C3C (Merah) |
| Institution | #34495E (Abu gelap) |
| Method | #2ECC71 (Hijau) |
| Dataset | #F39C12 (Oranye) |
| Metric | #00BCD4 (Cyan) |
| Finding | #9B59B6 (Ungu) |
| Limitation | #95A5A6 (Abu terang) |

**Tooltip:** Hover di atas node menampilkan nama entitas. Hover di atas edge menampilkan tipe relasi dan deskripsi.

**Aset di-embed inline:** Menggunakan `cdn_resources="in_line"` pada PyVis agar semua JavaScript/CSS ter-embed langsung di HTML (mencegah error 404 pada file eksternal).

---

## 5. Chatbot / Query Pipeline

### 5.1 Query Entity Extraction

Pertanyaan user dikirim ke LLM dengan prompt:
```
You are an expert at natural language understanding.
Extract all named entities (methods, datasets, metrics, authors, concepts, findings)
mentioned in the following user query.
CRITICAL RULE: DO NOT include entity type words (like "Paper", "Dataset", "Method",
"Metode", "Metric", "Author", "Finding", "Limitation") in the extracted string.
Only extract the specific proper noun or name.

Query: "{query}"
Example Output: ["Entity1", "Entity2"]
```

Setelah LLM mengembalikan hasil, dilakukan post-processing mekanis (jaring pengaman): strip kata-kata kategori ("dataset", "paper", "method", dll.) dari setiap entitas yang diekstrak.

### 5.2 Query Resolver (`query_resolver.py`)

Entitas yang diekstrak dicocokkan dengan node di graf melalui 4 lapisan pencocokan berurutan:

1. **Exact match ke alias keys:** Bandingkan lowercase query entity dengan semua key di `entity_aliases.json`. Jika cocok, ambil canonical value-nya.
2. **Exact match ke canonical nodes:** Bandingkan dengan daftar `checked_entities` dan values dari alias dict.
3. **Exact match ke graph nodes:** Fallback untuk tipe yang tidak masuk proses dedup (Paper, Author, Finding, Limitation).
4. **Fuzzy match:** Menggunakan `SequenceMatcher` dari Python stdlib. Bandingkan terhadap alias keys, canonical nodes, dan graph nodes. Threshold > 0.70. Semua perbandingan dilakukan dalam lowercase.

### 5.3 Graph Traversal (`graph_traversal.py`)

Dari setiap node yang berhasil di-resolve (entry point), diambil semua tetangga langsung (1-hop) menggunakan `graph.neighbors()`.

Node bertipe `Paper` ditandai sebagai `is_paper_blocked` — mereka tidak dilintasi lebih jauh (jika traversal diperluas ke multi-hop di masa depan). Ini mencegah ledakan konteks ke topik lain yang tidak relevan.

### 5.4 Serialisasi Konteks

Sub-graf 1-hop diubah menjadi teks naratif berbahasa Indonesia menggunakan pemetaan frase:
```python
phrase_map = {
    "compares": "dibandingkan dengan",
    "uses_dataset": "menggunakan dataset",
    "uses_method": "menggunakan metode/komponen",
    "uses_metric": "dievaluasi menggunakan metrik",
    "proposes_method": "diusulkan dalam",
    "has_limitation": "memiliki batasan/kelemahan berupa",
    "finds": "memiliki temuan atau karakteristik:",
    "written_by": "ditulis oleh author",
    "mentioned_in": "secara eksplisit dibahas dalam"
}
```

Contoh output:
```
=== KONTEKS UNTUK: DenseNet ===

KATEGORI RELASI: HAS_LIMITATION
- [Entry Point] DenseNet memiliki batasan/kelemahan berupa (has_limitation) 'Overfitting on small datasets'.

KATEGORI RELASI: USES_DATASET
- [Entry Point] DenseNet menggunakan dataset (uses_dataset) 'CIFAR-10'.
```

Jika ada multiple entry points (multi-entity query), konteks dari masing-masing entry point digabungkan.

### 5.5 LLM Generation (Grounded Answer)

Konteks gabungan disuntikkan ke prompt chatbot:
```
Anda adalah asisten akademik yang bertugas menjawab pertanyaan berdasarkan
graf pengetahuan ilmiah (GraphRAG).
Gunakan HANYA informasi yang disediakan dalam blok "KONTEKS GRAPH" di bawah ini.
Jika informasi yang ditanyakan TIDAK ADA di dalam konteks, jawab dengan jujur
"Saya tidak tahu berdasarkan dokumen yang ada" atau
"Informasi tidak tersedia di dalam konteks".
JANGAN mengarang jawaban (no hallucination).
Jawablah dalam bahasa Indonesia dengan jelas dan profesional.

PERTANYAAN USER:
{query}

KONTEKS GRAPH:
{context}
```

Temperature di-set ke 0.0 untuk meminimalisir halusinasi.

---

## 6. Error Handling & Rate Limiting

**Retry dengan Exponential Backoff:**
Setiap API call (extraction dan entity linking) memiliki mekanisme retry hingga 3-5 kali. Jika mendapat 429 (rate limit), sistem tidur dengan durasi yang meningkat per percobaan (`API_RETRY_DELAY_SECONDS * attempt`).

**Fatal Error Detection:**
Jika error mengandung keyword "API_KEY_INVALID" atau "PERMISSION_DENIED", sistem langsung berhenti total (bukan retry). Mencegah pembakaran retry sia-sia.

**Rate Limit Protection:**
Setiap API call diikuti `time.sleep(API_DELAY_SECONDS)` (default 3 detik) untuk menghindari 15 RPM limit pada Gemini free tier.

---

## 7. Konfigurasi (`config.py`)

| Parameter | Nilai Default | Keterangan |
|-----------|--------------|------------|
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Model LLM untuk semua API call |
| `API_DELAY_SECONDS` | 3 | Jeda antar API call |
| `API_MAX_RETRIES` | 3 | Maksimum retry per call |
| `API_RETRY_DELAY_SECONDS` | 10 | Delay awal saat retry |
| `CHUNK_SIZE` | 3500 | Ukuran chunk (karakter) |
| `CHUNK_OVERLAP` | 500 | Overlap antar chunk (karakter) |
| `ENTITY_SIMILARITY_THRESHOLD` | 0.85 | Threshold fuzzy match di dedup |
