# Laporan Aktivitas Kerja Magang

**Proyek:** GraphRAG Paper Explorer (PaperVisualizerV2.0)
**Periode laporan:** Senin, 27 Juli 2026 – Jumat, 31 Juli 2026
**Total estimasi:** ± 37 jam

---

### Senin, 27 Juli 2026

- (1j 30m) Review `project_context.md` dan telaah menyeluruh seluruh codebase (ingestion, extraction, dedup, graph, LLM client, query pipeline, vectorstore)
- (1j) Analisis `TECHNICAL_REFERENCE.md` — gap analysis proyek V1 vs V2, menemukan bahwa V2 kehilangan hampir semua safeguard hasil iterasi V1
- (2j) Implementasi perbaikan kualitas #1–#4: prompt ekstraksi 8 STRICT RULES, section filter References/Bibliography, modul cleaner (blocklist entitas generik), guard graph builder anti phantom node
- (1j 30m) Implementasi #5–#7: guard dedup CRITICAL 1 & 2 (anti over-merging versi), preprocessing author, perbaikan bug laten casing pada kamus akronim yang menyebabkan fast-path gagal total tanpa terdeteksi
- (1j) Rebuild pipeline + verifikasi hasil ekstraksi

### Selasa, 28 Juli 2026

- (1j) Diagnosis kuantitatif kualitas graph — menemukan 118 node terisolasi, 124 connected component, provenance kosong 0/321 node
- (2j) Implementasi Phase A: provenance plumbing (checkpoint v2 menyimpan `chunk_id`), Paper node kanonik per PDF, anchoring entity→paper via `mentioned_in`
- (1j) Rebuild + verifikasi (hasil: 0 node terisolasi, graph tersambung penuh)
- (1j 30m) Phase B visualisasi: physics ForceAtlas2, node sizing berbasis rarity + degree, sistem warna node/edge
- (30m) Debugging `UnicodeEncodeError` pyvis di Windows (encoding cp1252 pada `write_html`)

### Rabu, 29 Juli 2026

- (1j 30m) Phase C — Entity Types legend + sistem highlight (glow, dim, toggle)
- (1j) Network Insights: Top-5 per kategori berbasis degree, lazy rendering
- (1j) Edge Filters: dropdown per tipe relasi + deskripsi
- (1j) Path Discovery: BFS shortest path + highlight jalur
- (1j 30m) Search bar dengan ranking hasil + Info Panel (detail node, koneksi dikelompokkan, navigasi berantai)
- (1j) Iterasi tema visual dan validasi palet warna (uji CVD-safety, kontras, lightness band)

### Kamis, 30 Juli 2026

- (1j 30m) Membangun `server.py` (FastAPI + uvicorn): endpoint penyaji visualisasi + `/api/chat`, serta UI chatbot terintegrasi
- (1j 30m) Perbaikan resolver menjadi 4 lapis + vector-native (embedding nearest-node); tuning threshold 0.72 lewat diagnostik distribusi similarity (parafrase asli 0.72–0.97 vs noise 0.54–0.64)
- (1j) Implementasi traversal multi-hop (2-hop + Paper-blocking) dan intent baru `structural` untuk query lintas-dokumen
- (1j 30m) B1+B2: injeksi judul paper ke prompt ekstraksi + penautan model↔paper + resolver paper-context, lalu rebuild
- (1j 30m) De-hardcoding untuk mengurangi overfitting: auto-ekstraksi metadata paper (1 LLM call/PDF) dan auto-ekstraksi akronim dari pola `Full Name (ACR)`
- (1j) Penyusunan dokumen `LAPORAN_PROJECT.md` (arsitektur, pipeline, evaluasi, limitasi)

### Jumat, 31 Juli 2026

- (1j 30m) Membangun harness evaluasi (`evaluate.py`, mode retrieval gratis + `--e2e`) dan menyusun test case beserta baseline
- (1j) Keamanan: untrack `.env`, penanganan GitHub Push Protection, purge history; plus perbaikan ligatur PDF dan reclassify paper sitasi
- (1j) Membuat `corpus.py` (backup/clear/restore korpus) + perbaikan bug section filter yang membuat 5 dari 8 paper tidak terfilter (401 → 189 chunk, hemat ± 212 panggilan LLM)
- (1j 30m) Build korpus baru + diagnosis konektivitas (entity↔entity hanya 14%, relasi `compares` kolaps karena data perbandingan berada di tabel)
- (1j) Implementasi `link_entity_families` (deteksi keluarga entitas domain-agnostik) + adjudikasi LLM untuk kasus ChatGPT vs GPT
- (2j) Menyusun `ROADMAP.md` dan mengeksekusi Fase 1: konteks paper sumber (65% → 81%), pencocokan judul parsial (→ 87%), multi-entity resolution (→ 94%)
- (1j) Evaluasi end-to-end (77%) + analisis 9 kegagalan, memisahkan kegagalan nyata dari artefak penilaian

---

## Catatan Cakupan Data

**Visibility penuh** tersedia untuk pekerjaan sejak **24 Juli 2026** — seluruh aktivitas di atas
direkonstruksi dari riwayat sesi kerja, diperkuat timestamp file dan catatan proyek.

**Tidak ada data** untuk sesi sebelum 24 Juli 2026. Bukti yang tersedia hanya berupa jejak tidak
langsung:

| Jejak | Tanggal |
|---|---|
| Commit pertama (`"first prompt"`) | 17 Juli 2026, 14:15 |
| PDF korpus awal diunduh | 13–14 Juli 2026 |
| `project_context.md` mencatat build pipeline sudah selesai (127 chunk) | sebelum 24 Juli 2026 |

Isi dan durasi kerja pada rentang **17–23 Juli 2026 tidak dapat direkonstruksi** karena riwayat
sesinya tidak tersimpan. Bagian tersebut sengaja tidak diisi agar laporan tetap akurat.

## Catatan Metodologi

- **Penempatan hari** mengikuti **urutan kerja** (fondasi proyek → pengembangan terbaru), bukan
  tanggal asli tiap aktivitas. Secara aktual pekerjaan ini terentang 24–31 Juli 2026.
- **Durasi bersifat estimasi**, dibulatkan ke kelipatan 30 menit berdasarkan kompleksitas pekerjaan
  dan jejak riwayat sesi — bukan hasil pencatatan waktu (time tracking) aktual.
- Aktivitas di luar lingkup pekerjaan magang tidak dimasukkan.

## Ringkasan Hasil Kerja

| Aspek | Kondisi akhir |
|---|---|
| Akurasi retrieval (terukur) | 29/31 = **94%** |
| Akurasi end-to-end | 30/39 = **77%** |
| Anti-halusinasi | **8/8** tidak mengarang |
| Kualitas graph | 440 node · 524 edge · 1 komponen · 0 node terisolasi |
| Artefak pendukung | `LAPORAN_PROJECT.md`, `ROADMAP.md`, `evaluate.py`, `corpus.py` |
