# GraphRAG Paper Explorer — Spesifikasi Arsitektur

Dokumen ini adalah spesifikasi teknis untuk membangun asisten AI akademik yang menjawab pertanyaan berdasarkan Knowledge Graph yang dibangun dari kumpulan paper PDF (studi kasus: paper computer vision seperti ResNet, DenseNet, VGG, GoogLeNet). Sistem menggabungkan retrieval berbasis struktur graph dengan retrieval berbasis vector search (hybrid), dan menyertakan mekanisme untuk menjawab pertanyaan tematik maupun agregatif.

---

## 1. Constraint Wajib

- **Budget: $0.** Tidak ada kartu kredit tersedia sama sekali.
- Semua service cloud yang dipakai harus bisa signup tanpa kartu kredit. **Google AI Studio** (bukan Vertex AI) dipakai untuk seluruh kebutuhan LLM.
- Semua vector storage harus **lokal** — tidak ada vector DB cloud (Pinecone/Weaviate/Qdrant Cloud/Milvus Cloud). Skala corpus (puluhan paper) tidak butuh infrastruktur seberat itu, dan ini menghindari risiko signup yang minta kartu kredit.
- Semua embedding dijalankan **lokal**, bukan lewat API — supaya tidak ada rate limit atau ketergantungan jaringan di jalur yang paling sering dieksekusi (setiap chunk saat build, setiap pertanyaan saat runtime).

---

## 2. Keputusan Arsitektur & Alasannya

### 2.1 Entity Dedup: Hybrid — embedding sebagai filter, LLM sebagai keputusan final

Nama entity yang sama sering ditulis berbeda antar paper (akronim vs kepanjangan, variasi casing, dsb). Pendekatan embedding similarity murni (cosine similarity dengan satu threshold tunggal) **tidak boleh dipakai sebagai keputusan final**, karena embedding model general-purpose mengukur kemiripan topikal, bukan identitas faktual. Ini menciptakan dua kegagalan berlawanan yang sulit diatasi dengan satu threshold:
- **Over-merge**: dua entity yang topiknya sangat dekat tapi secara faktual berbeda (misal dua varian dataset dengan skala berbeda) bisa punya similarity tinggi padahal harus tetap dipisah.
- **Under-merge**: dua nama untuk entity yang sama tapi bentuk permukaannya jauh berbeda (akronim vs kepanjangan penuh) bisa punya similarity rendah padahal harus digabung.

**Arsitektur yang dipakai**: embedding jadi **candidate filter** dengan threshold rendah (mulai ~0.5, di-tune berdasarkan hasil) untuk mempersempit pasangan entity yang perlu dibandingkan lebih lanjut. Keputusan final tetap dilakukan oleh **LLM entity linking** — nama entity dikirim per micro-batch (±20 item) bersama Master List entity kanonikal yang sudah terbentuk, LLM memutuskan apakah entity baru merujuk ke entity yang sudah ada atau membentuk grup baru. Pendekatan ini memangkas jumlah panggilan LLM dibanding membandingkan semua pasangan tanpa filter, tanpa mengorbankan akurasi keputusan.

Dedup dilakukan dalam **namespace per tipe** — entity hanya dibandingkan dengan entity bertipe sama (Method dengan Method, Dataset dengan Dataset, dst), supaya node dengan nama mirip tapi tipe berbeda (misal judul paper vs nama method) tidak pernah tergabung tidak sengaja.

### 2.2 Embedding Model: Lokal, bukan API

Gunakan `bge-small-en-v1.5` via `sentence-transformers` — jalan di CPU, gratis tanpa syarat, tidak perlu signup. Dipakai untuk tiga kebutuhan: candidate filter dedup (2.1), chunk embedding untuk retrieval (2.4), dan query embedding saat runtime.

### 2.3 Vector Store: ChromaDB

ChromaDB dipilih karena sudah termasuk metadata storage, filtering, dan persistence dalam satu package embedded, jalan lokal tanpa server terpisah. Gunakan dua collection:
1. `chunks` — potongan teks mentah dari paper, untuk retrieval/grounding tambahan
2. `community_summaries` — ringkasan tematik dari cluster graph (lihat 2.5)

### 2.4 Chunk-to-Node Linking

Selain diekstrak jadi entity, teks chunk mentah juga di-embed dan disimpan di ChromaDB. Ini jadi fallback: kalau graph traversal tidak memberi informasi cukup detail untuk menjawab pertanyaan, sistem bisa menarik teks asli dari chunk yang relevan.

### 2.5 Community Detection: dua mekanisme terpisah untuk dua jenis pertanyaan agregatif

Pertanyaan yang butuh pandangan menyeluruh atas graph (bukan cuma satu entry point) terbagi jadi dua jenis yang butuh solusi berbeda — jangan digabung jadi satu mekanisme:

| Jenis pertanyaan | Contoh | Mekanisme |
|---|---|---|
| Tematik/naratif | "Apa tren riset di corpus ini?", "Pendekatan apa saja yang mengatasi vanishing gradient?" | Community detection (Louvain) mengelompokkan node berdasarkan kepadatan koneksi, tiap cluster diringkas LLM jadi community report, report di-embed dan dicari lewat vector similarity ke query |
| Counting/ranking literal | "Method apa yang paling banyak dipakai?", "Dataset apa yang paling sering muncul?" | **Graph statistics langsung** (`graph.degree()`, hitung frekuensi tipe edge) di NetworkX — tanpa LLM, tanpa vector search sama sekali |

**Peringatan desain**: node `Paper` berpotensi jadi *structural hub* — satu Paper bisa terhubung ke puluhan entity hanya karena semuanya disebut di paper yang sama, bukan karena entity-entity itu berhubungan secara logis. Kalau node `Paper` ikut dimasukkan apa adanya ke proses clustering, hasil community berisiko cuma jadi "pengelompokan per paper", bukan cluster tematik lintas-paper yang diinginkan. **Exclude atau turunkan bobot node `Paper` sebelum menjalankan community detection**, dan validasi hasil clustering-nya sebelum dianggap benar.

Algoritma: **Louvain** (`python-louvain`), jalan langsung di atas `networkx.Graph` tanpa perlu konversi ke library graph lain — cukup untuk skala corpus puluhan paper.

### 2.6 Query Resolution: Hybrid, bukan string matching murni

Fuzzy string matching (`SequenceMatcher`) rapuh untuk kasus di mana pertanyaan pakai istilah yang maknanya sama tapi bentuknya beda jauh secara leksikal (misal nama panjang challenge vs kode singkatnya). Query resolution memakai dua jalur sekaligus:
1. **Vector similarity** — query di-embed, dicari node/entity terdekat di ChromaDB
2. **Entity extraction + string match** — LLM ekstrak entity dari pertanyaan, dicocokkan exact match dulu ke alias dictionary, fuzzy match sebagai fallback terakhir

### 2.7 Graph Traversal: 1-hop dengan Paper-blocking

Dari entry point yang ditemukan, ambil seluruh 1-hop neighbor beserta jenis relasinya. Kalau neighbor bertipe `Paper`, tetap ditampilkan sebagai informasi provenance tapi **tidak dilanjutkan traversal**-nya — mencegah context explosion karena node Paper cenderung berfungsi sebagai hub struktural, bukan hub tematik.

---

## 3. Skema Entity & Relationship

### Node types (9, fixed)

`Paper`, `Author`, `Institution`, `Method`, `Dataset`, `Metric`, `Finding`, `Limitation`, `Task`.

`Task` memisahkan "task riset" (image classification, object detection, semantic segmentation) dari `Dataset`, karena satu dataset kadang dipakai untuk beberapa task berbeda — kalau digabung ke satu tipe, pertanyaan seperti "method apa yang dirancang untuk object detection" jadi sulit dijawab langsung.

> Tipe entity sengaja dibuat **fixed**, bukan dinamis mengikuti konteks tiap paper. Tipe dinamis membuat isolasi namespace saat dedup dan aturan Paper-blocking saat traversal tidak bisa dijalankan dengan andal, karena keduanya butuh tahu persis daftar tipe yang mungkin muncul. Kalau nanti ditemukan kebutuhan tipe baru, tambahkan lewat keputusan eksplisit berdasarkan pertanyaan konkret yang tidak terjawab — bukan biarkan LLM menciptakan tipe bebas per chunk.

### Edge types

`uses_method`, `uses_dataset`, `uses_metric`, `has_limitation`, `compares`, `finds`, `proposes_method`, `criticizes`, `builds_on`, `written_by`, `affiliated_with`, `mentioned_in`.

---

## 4. Pipeline

### 4.1 Build-time

```
PDF papers
    │
    ▼
[1] Ingestion & Chunking (PyMuPDF)
    │
    ▼
[2] LLM Extraction (Gemini AI Studio, Pydantic schema, 9 tipe entity)
    │
    ▼
[3] Entity Dedup — Hybrid
    ├─ embedding candidate filter (bge-small-en-v1.5, threshold rendah ~0.5)
    └─ LLM entity linking untuk keputusan final (micro-batch + Master List)
    │
    ▼
[4] Graph Building (NetworkX, isolasi namespace per tipe)
    │
    ├──► [5a] Visualisasi (Pyvis)
    │
    ├──► [5b] Chunk embedding → ChromaDB collection `chunks`
    │
    └──► [5c] Community Detection
            ├─ exclude/turunkan bobot node Paper sebelum clustering
            ├─ Louvain (python-louvain)
            ├─ LLM summary per community
            └─ embed summary → ChromaDB collection `community_summaries`
```

### 4.2 Query-time

```
Pertanyaan user
    │
    ▼
[6] Query Router — klasifikasi jenis pertanyaan (LLM classification sederhana)
    │
    ├─ ENTITY-SPECIFIC (local search)
    │     │
    │     ▼
    │  Query Resolution — Hybrid (vector similarity + entity extraction/match)
    │     │
    │     ▼
    │  Graph Traversal (1-hop, Paper-blocking)
    │     │
    │     ▼
    │  Serialization + chunk retrieval sebagai fallback (ChromaDB `chunks`)
    │
    ├─ TEMATIK/NARATIF (global search)
    │     │
    │     ▼
    │  Vector similarity ke community summary (ChromaDB `community_summaries`)
    │
    └─ COUNTING/RANKING (graph stats)
          │
          ▼
       Query langsung ke NetworkX (graph.degree(), frekuensi edge type)
       — tanpa LLM, tanpa vector search
    │
    ▼
[7] Context Assembly — gabungkan hasil dari jalur yang relevan
    │
    ▼
[8] LLM Generation (grounding ketat ke konteks, Temperature = 0, menolak menjawab kalau info tidak tersedia)
    │
    ▼
Jawaban ke user
```

---

## 5. Tech Stack

| Komponen | Teknologi | Catatan |
|---|---|---|
| PDF parsing | PyMuPDF (fitz) | |
| LLM (extraction, entity linking, summary, generation, query classification) | Gemini API — Google AI Studio, free tier | Bukan Vertex AI — tidak butuh kartu kredit |
| Structured output | Pydantic | Memaksa skema JSON konsisten dari LLM |
| Embedding | `bge-small-en-v1.5` via `sentence-transformers` | Lokal, CPU, gratis tanpa syarat |
| Vector store | ChromaDB | Lokal, embedded, collection `chunks` + `community_summaries` |
| Graph storage | NetworkX | In-memory, tanpa database server |
| Community detection | `python-louvain` | Langsung di atas `networkx.Graph` |
| Visualisasi | Pyvis (vis.js) | Physics-based interactive HTML |
| Fuzzy matching | `difflib.SequenceMatcher` | Fallback lapis kedua di query resolver, bukan mekanisme utama |

---

## 6. Known Limitations (risiko yang perlu diantisipasi, bukan dianggap sudah selesai)

1. **Fragmentasi entity yang berkaitan secara hierarkis** — variasi dataset yang berkaitan (misal versi berbeda dari dataset yang sama) berpotensi tetap tercatat sebagai node terpisah, karena LLM entity linking tidak selalu menganggapnya sebagai duplikat mutlak. Hybrid dedup mengurangi risiko ini tapi tidak menjamin hilang total.
2. **1-hop traversal ceiling** — pertanyaan yang butuh reasoning berlapis (multi-hop, misal "metode apa yang dirancang untuk mengatasi kelemahan arsitektur X") tidak bisa dijawab lewat local search. Community search menangani sebagian kasus yang butuh konteks lebih luas, tapi bukan pengganti true multi-hop traversal.
3. **Garbage in, garbage out** — seluruh sistem downstream bergantung penuh pada akurasi tagging tipe entity di tahap ekstraksi awal. Kesalahan atribusi tipe tidak punya mekanisme koreksi otomatis di tahap berikutnya.
4. **Intersection spesifik dua entity** (misal "finding khusus untuk kombinasi Method X + Dataset Y") — skema relasi saat ini tidak menangkap granularitas kombinasi seperti ini. Sistem harus tetap transparan menyatakan keterbatasan, bukan menggabungkan informasi secara keliru.
5. **Community clustering bisa gagal** kalau node Paper tidak ditangani dengan benar (lihat 2.5) — validasi hasil clustering sebelum dipakai sebagai sumber jawaban.
