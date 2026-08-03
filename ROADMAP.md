# Roadmap Perbaikan — GraphRAG Paper Explorer

Dokumen pelacak. **Update checkbox + kolom "Hasil" setiap kali satu item selesai**, supaya tidak ada
yang tertinggal saat pindah fokus.

---

## Baseline saat ini (korpus sycophancy, 8 paper)

Diukur `2026-07-31` dengan `python evaluate.py` (retrieval, gratis, tanpa LLM):

```
OVERALL (retrieval): 20/31 = 65%
counting 5/5 | family 3/3 | paper-attr 9/10 | structural 2/2
relational 1/6 | resolution 0/2 | adversarial 0/3
+ 8 kasus negatif (anti-halusinasi) — hanya dinilai di --e2e
```

**Statistik graph:** 440 node · 524 edge · 1 komponen · 0 terisolasi · entity↔entity 24%

**Target realistis setelah Fase 1:** ≥ 85% retrieval.

### Progres

| Tanggal | Item | Skor retrieval |
|---|---|---|
| 2026-07-31 | _baseline_ | 20/31 = 65% |
| 2026-07-31 | ✅ 1.1 konteks Paper sumber | **25/31 = 81%** |
| 2026-07-31 | ✅ 1.5 judul paper parsial | **27/31 = 87%** 🎯 target ≥85% tercapai |
| 2026-07-31 | ✅ 1.2 multi-entity | **29/31 = 94%** ✅ **FASE 1 SELESAI** |

**Skor akhir Fase 1:** `adversarial 3/3 · counting 5/5 · family 3/3 · paper-attr 10/10 · relational 6/6 · structural 2/2 · resolution 0/2`

**Sisa kegagalan (2):** `resolve-paraphrase`, `resolve-paraphrase2` → **1.4 (sengaja diturunkan
jadi limitasi terdokumentasi)** — parafrase Indonesia, gagal *graceful* (jatuh ke chunk search).

---

## 🚦 ALUR KERJA — 3 BATCH, LALU SELESAI (disepakati 2026-07-31)

Roadmap sempat membengkak jadi 10 item terbuka karena tiap pengukuran memunculkan temuan baru
tanpa ada yang dipangkas. Disepakati dipersempit jadi **6 item dalam 3 batch**, dengan garis
finis yang jelas.

### ✅ BATCH A — SELESAI (2026-08-03)
- [x] **3.6** Struktur konteks generation
- [x] **3.5** Prompt router: query "versi/varian apa saja"
- [x] Verifikasi: `evaluate.py` (94%, tanpa regresi) + `--e2e` (**77% → 90%**)
- 🆕 Ditemukan **3.7** (resolver singkatan gabungan) — prioritas rendah, tidak memblokir Batch B

### ✅ BATCH B — SELESAI (2026-08-03), TANPA REBUILD
- [x] **2.1** ❌ Dibatalkan — investigasi ulang: **tidak ada bug**, "duplikat" yang dicurigai ternyata dua entitas berbeda secara sah (Paper vs Dataset)
- [x] **2.2** ❌ Dibatalkan — premis keliru: data komparatif ada di **grafik/chart**, bukan tabel teks; `find_tables()` tidak relevan; solusi asli (vision LLM) dievaluasi & ditolak (biaya besar, di luar cakupan)
- **Hasil:** tidak ada rebuild yang diperlukan. Konektivitas 24% & `compares` rendah **diterima sebagai limitasi terdokumentasi**, bukan bug yang perlu diperbaiki

### ⏸️ BATCH C — penutup, non-teknis
- [ ] **3.2** Perbarui `LAPORAN_PROJECT.md` dengan angka korpus aktif
- [x] ~~**3.3** Bereskan `git push`~~ ✅ **SELESAI** — terverifikasi `origin/main` bersih dari `.env`, key asli tak pernah ter-push

> **Definisi SELESAI:** setelah Batch C, proyek layak diserahkan. Item lain hanya dikerjakan
> kalau muncul **kebutuhan nyata**, bukan karena ada di daftar.

### ❌ DICORET (agar daftar berhenti bertambah)

| Item | Alasan |
|---|---|
| **2.3** Opsi B appendix | Biaya besar (+63% kuota, mungkin 2 hari), manfaat belum terbukti dibutuhkan |
| **2.4** Tuning community/thematic | **Tidak ada kasus eval yang gagal karenanya** — perbaikan tanpa masalah terukur |
| **2.5** Model ekstraksi lebih kuat | Belum terbukti `flash-lite` adalah bottleneck. Spekulatif |
| **3.4** Warna Finding vs Paper | Kosmetik, tidak mengganggu pemakaian |

> ⚠️ **Ekspektasi realistis untuk 3.6:** kegagalannya sebagian ada di kemampuan LLM merangkai
> fakta dari konteks panjang. Kita bisa memperbaiki **struktur** konteks agar lebih mudah dicerna,
> tapi tidak bisa memaksa `flash-lite` menalar lebih baik. Target: **berkurang**, bukan hilang.

---

## Urutan prioritas — ARSIP (revisi 2026-07-31, setelah 1.1)

> ⚠️ **Usang per 2026-08-03.** Prioritas #1 di tabel ini (2.2 ekstraksi tabel) **dibatalkan**
> setelah investigasi menemukan premisnya keliru — data komparatif ada di grafik/chart, bukan
> tabel teks (detail di 2.2). Disimpan sebagai arsip keputusan, bukan rencana aktif.

| # | Item | Kenapa |
|---|---|---|
| ~~1~~ | ~~2.2 Ekstraksi tabel~~ | ❌ Dibatalkan — lihat 2.2 |
| 2 | 1.5 Judul parsial | Pola pemakaian nyata, murah — tapi **gerbangi ketat** |
| 3 | 1.2 Multi-entity | Relevan untuk domain ini (banyak perbandingan) — awasi regresi konteks |
| ❌ | 1.3 3-hop | **Dibatalkan** — motivasinya hilang setelah 1.1 |
| 🔻 | 1.4 Multilingual | **Diturunkan** — risiko membatalkan threshold, imbalan kecil |

> **Pergeseran penting saat itu:** setelah 1.1, nilai terbesar tidak lagi ada di mengejar sisa
> kasus eval, melainkan di **kualitas data** (Fase 2). Ternyata Fase 2 sendiri tidak punya
> pekerjaan yang bisa dieksekusi — lihat Batch B.

> Cara mengukur ulang (selalu jalankan setelah tiap perubahan):
> ```powershell
> python evaluate.py            # gratis, tanpa LLM
> python evaluate.py --e2e      # pakai kuota — hanya di akhir tiap fase
> ```

---

## FASE 1 — Query-time (TANPA rebuild, tanpa kuota LLM)

Dikerjakan berurutan; ukur ulang setelah tiap item.

### [x] 1.1 Sertakan konteks Paper sumber  ⭐ SELESAI (2026-07-31)

| | |
|---|---|
| **Masalah** | Nama benchmark ada 2× di graph: sebagai `Paper` **dan** sebagai `Dataset`/`Method`. Resolver memilih node non-Paper yang kosong, padahal `uses_dataset`/`has_limitation` menempel di node Paper. |
| **Bukti** | `resolved='SYCEval'` (Dataset) padahal fakta ada di `SycEval: Evaluating LLM Sycophancy` (Paper) |
| **Perbaikan** | ✅ Longgarkan Paper-blocking **hanya untuk paper sumber** (paper yang bertetangga langsung dgn node awal non-Paper). Paper di hop >1 tetap diblokir agar konteks tidak meledak. Param `through_source_paper=True` |
| **File** | `src/query/traversal.py` (`traverse`) |
| **Menyembuhkan** | ✅ `ent-syceval-data`, `ent-elephant-data`, `ent-syceval-limit`, `ent-elephant-limit` · ⚠️ `ent-towards-data` TIDAK sembuh (sebab lain → lihat 1.5) · 🎁 bonus: `adv-3hop` ikut lolos |
| **Risiko** | Konteks membesar → perlu jaga `max_rels` |
| **Hasil** | **20/31 (65%) → 25/31 (81%)**. relational 1/6 → **5/6**, adversarial 0/3 → 1/3 |

### [x] 1.2 Multi-entity resolution  ✅ SELESAI (2026-07-31)

| | |
|---|---|
| **Masalah** | `resolve_entity()` mengembalikan **satu** node. Pertanyaan "A dan B" hanya terjawab separuh. |
| **Bukti** | `adv-multi-entity`: dapat Stanford, **Carnegie Mellon terlewat** |
| **Perbaikan** | ✅ `resolver.resolve_entities()` — ambil semua match verbatim, pilih greedy dari terpanjang sambil **membuang rentang yang tumpang tindih** (`ResNet` tidak ikut saat `ResNet-50` sudah dipilih). Kalau <2 entitas, jatuh balik ke `resolve_entity()` sehingga perilaku lama tak berubah. Paper-context ikut dipetakan per entitas. Maks 3 entitas |
| **File** | `src/query/resolver.py`, `src/engine.py`, `evaluate.py` |
| **Menyembuhkan** | ✅ `adv-multi-entity`, `adv-multi-limit` (2 kasus) |
| **🔑 Temuan penting** | Membagi budget (`150 // n`) justru **MERUSAK**: pada node hub, relasi hop-1 menghabiskan jatah sebelum hop-2 → fakta `Author→Institution` hilang. Dua perbaikan: (a) **jatah budget per-hop** di `traverse()` (hop-1 dibatasi 60%, sisanya untuk hop-2); (b) budget **125/entitas** (bukan dibagi). Terukur: konteks jenuh di ~31K char karena dibatasi ukuran graph — jadi kekhawatiran "meledak" tidak terbukti |
| **Hasil** | **27/31 (87%) → 29/31 (94%)**. adversarial 1/3 → **3/3**, tanpa regresi |

### [~] 1.3 Traversal 3-hop selektif — ❌ **DIBATALKAN (2026-07-31)**

| | |
|---|---|
| **Alasan dibatalkan** | Motivasinya **hilang setelah 1.1**. Dua kasus yang dulu memotivasi item ini: `adv-3hop` **sudah lolos** berkat ekspansi paper sumber; `paper-towards-inst` ternyata masalah **resolusi**, bukan kedalaman hop (dipindah ke 1.5). |
| **Sisa nilai** | Nol kasus yang disembuhkan |
| **Risiko tetap** | 🔴 Tinggi — terukur konteks membengkak ke ~130 relasi dan menabrak `max_rels`, bisa **memperburuk** jawaban |
| **Keputusan** | Mengerjakannya = menambah risiko tanpa imbalan terukur. Buka lagi **hanya jika** muncul kegagalan nyata yang benar-benar butuh 3 hop |

### [x] 1.5 Pencocokan judul Paper secara parsial  ✅ SELESAI (2026-07-31)

| | |
|---|---|
| **Masalah** | Judul paper panjang jarang disebut utuh oleh user. Lapis 2 (nama termuat verbatim) melewatkannya, lalu node generik pendek yang menang. |
| **Bukti** | Query *"Dataset apa yang dipakai **paper Towards Understanding Sycophancy**?"* → resolve ke `'Sycophancy'` (node **Metric**, 10 char). Judul asli `TOWARDS UNDERSTANDING SYCOPHANCY IN LANGUAGE MODELS` berbagi **3/6 kata** dengan query tapi tidak termuat verbatim |
| **Perbaikan** | ✅ Lapis **1.5** di resolver (setelah exact, sebelum verbatim): `_match_paper_title()` menghitung tumpang-tindih kata judul. **Digerbangi dua syarat**: minimal **2 kata judul** bermakna cocok DAN rasio **≥ 0.55**; stopword judul dibuang |
| **File** | `src/query/resolver.py` |
| **Menyembuhkan** | ✅ `ent-towards-data`, `paper-towards-inst` (2 kasus) |
| **Risiko** | Ambang terlalu rendah → paper acak menang. **Terverifikasi tidak ada regresi**: query 1-kata seperti "metrik sycophancy" tertahan gerbang min-2-kata |
| **Hasil** | **25/31 (81%) → 27/31 (87%)**. relational 5/6 → **6/6**, paper-attr 9/10 → **10/10**, kategori lain tetap |

### [~] 1.4 Parafrase non-Inggris — 🔻 **DITURUNKAN: dokumentasikan sebagai limitasi**

| | |
|---|---|
| **Masalah** | Embedding `bge-small-**en**-v1.5` hanya untuk Inggris; parafrase Indonesia meleset |
| **Bukti** | "benchmark sikap menjilat pada model medis" → `420-prompt benchmark`, bukan `EchoBench` |
| **Kenapa diturunkan** | 🔴 **Mengganti model embedding membatalkan SEMUA threshold yang sudah di-tuning.** Dedup pakai `0.85` dan resolver pakai `0.72` — keduanya diturunkan dari distribusi similarity `bge-small-en`. Model lain punya distribusi berbeda → kedua angka langsung tidak valid → berpotensi merusak **dedup di seluruh graph**, bukan cuma 2 kasus resolusi |
| **Imbalan** | Hanya 2 kasus, dan kegagalannya *graceful* (jatuh ke chunk search, tetap ada jawaban) |
| **Keputusan** | Tulis sebagai **limitasi yang diketahui** di `LAPORAN_PROJECT.md` (sudah ada di §11.2). Query dalam bahasa Inggris bekerja normal |
| **Kalau tetap dikerjakan** | Urutan wajib: ganti model → **tuning ulang kedua threshold lewat diagnostik** (seperti waktu menetapkan 0.72) → rebuild → ukur seluruh suite. Ini pekerjaan besar, bukan tempelan |

---

## FASE 2 — Butuh rebuild (**BATCH JADI SATU** agar kuota tidak terbuang)

⚠️ **Aturan:** jangan rebuild per item. Kumpulkan semua, lalu **satu** `python main.py build`.

### [~] 2.1 Dedup casing yang lolos  ❌ **DIBATALKAN (2026-08-03) — diagnosis awal keliru**

| | |
|---|---|
| **Investigasi ulang** | Scan penuh graph: **0 grup duplikat casing** dalam namespace tipe yang sama. Log `similarity_debug.csv` konfirmasi `SYCEval` (Dataset) dibandingkan ke Dataset lain dan **benar** tidak digabung (similarity 0,51–0,64, di bawah threshold 0,85) |
| **Kenapa dulu terlihat seperti bug** | `SYCEval` (node Dataset) disalahartikan sebagai duplikat dari `SycEval: Evaluating LLM Sycophancy` (node Paper) — padahal keduanya **memang harus terpisah**: satu paper, satu benchmark/dataset yang paper itu perkenalkan. Namespace-per-tipe bekerja benar di sini |
| **Temuan sungguhan (beda dari rencana awal)** | Satu-satunya near-duplicate string yang nyata: `'Online Information Bias'` (**Task**) vs `'online information bias'` (**Method**) — sama persis kecuali kapitalisasi, tapi **beda tipe**. LLM tidak konsisten mengklasifikasikan konsep yang sama di chunk berbeda |
| **Kenapa tidak diperbaiki** | Memperbaikinya butuh menggabungkan **lintas tipe entity** — bertentangan dengan prinsip namespace isolation yang justru mencegah salah gabung (Author↔Method, dst). Dampak sangat kecil: 1 pasang node, **tidak ada kasus eval yang gagal karenanya** |
| **Keputusan** | Dibatalkan. Tidak ada bug dedup casing yang perlu diperbaiki |

### [~] 2.2 Ekstraksi tabel untuk memulihkan `compares`  ❌ **DIBATALKAN (2026-08-03) — premis keliru**

| | |
|---|---|
| **Masalah** | `compares` = **2** dari 482 edge (korpus lama: 31). Konektivitas entity↔entity 24% vs 48% korpus lama |
| **Dugaan awal** | Perbandingan model ada di **tabel PDF**, dan PyMuPDF mengekstraknya jadi teks berantakan (bubur kolom). Rencana: `page.find_tables()` → serialisasi jadi teks terstruktur |
| **Investigasi** | Dilacak balik ke sumber aslinya: baris kacau `"Overall \| Emotional validation \| ... \| Claude \| Gemini \| GPT-4o..."` **BUKAN** dari tabel — itu label sumbu **grafik batang (bar chart)** matplotlib di halaman terakhir paper ELEPHANT. Data perbandingan tersimpan sebagai **tinggi batang vektor**, bukan teks — tidak ada representasi tekstual angkanya di PDF sama sekali |
| **Verifikasi** | Scan `find_tables()` di **seluruh 8 paper**: hanya 6 kandidat tabel data, semuanya matriks korelasi (angka tanpa keterangan model) atau ekstraksi rusak (banyak sel `None`). **Tidak ada** satu pun tabel "Model \| Metrik" yang bisa memulihkan `compares` |
| **Kesimpulan** | `find_tables()` tidak relevan — masalahnya bukan tabel yang berantakan, tapi data yang memang **tidak berbentuk teks** (grafik). Solusi nyata butuh vision LLM membaca chart (proyek baru, biaya kuota besar, akurasi tak pasti, di luar cakupan "satu rebuild") — dievaluasi dan **ditolak**, lihat opsi 2 di diskusi |
| **Keputusan** | **Diterima sebagai limitasi arsitektural**: ekstraksi berbasis teks tidak menjangkau data yang disajikan sebagai grafik visual. Dicatat di `LAPORAN_PROJECT.md` §11.2 (lihat 3.2) |

### [~] 2.3 Opsi B — pertahankan appendix  ❌ **DICORET** (biaya besar, manfaat belum terbukti)

| | |
|---|---|
| **Masalah** | Filter memotong dari `References` ke bawah → appendix ikut terbuang (berisi metode mitigasi: DPO, Truthfulness ITI, dsb.) |
| **Perbaikan** | Buang **hanya blok daftar pustaka**, sambung kembali appendix. Prasyarat: deteksi heading appendix diperkuat (gaya `G.1`, `C.1`, bukan hanya `Appendix`/`A.1`) |
| **Biaya** | 189 → ~308 chunk (+63% panggilan LLM) → kemungkinan perlu 2 hari kuota |
| **Referensi** | `LAPORAN_PROJECT.md` §11.3 poin 4 |

### [~] 2.4 Tuning community / thematic  ❌ **DICORET** (tidak ada kasus eval yang gagal karenanya)

| | |
|---|---|
| **Masalah** | Louvain menghasilkan ratusan community terfragmentasi; jalur `thematic` jadi jalur terlemah |
| **Perbaikan** | Tuning resolusi Louvain, atau ringkas hanya community besar |
| **File** | `src/graph/analytics.py`, `main.py` |

### [~] 2.5 Model ekstraksi lebih kuat  ❌ **DICORET** (belum terbukti flash-lite bottleneck)

| | |
|---|---|
| **Trade-off** | Kualitas ekstraksi naik, tapi kuota harian lebih ketat → build bisa perlu 2 hari |
| **Keputusan** | Tunda sampai terbukti `flash-lite` adalah bottleneck |

---

## FASE 3 — Kualitas & rigor (kapan saja)

### [x] 3.1 Jalankan `--e2e` ✅ SELESAI (2026-07-31)

**Hasil: 30/39 = 77%** (retrieval 94% → e2e 77%). Selisihnya mengungkap **dua permukaan gagal
yang tidak terlihat di eval retrieval**: router dan generation.

| Aspek | Hasil |
|---|---|
| **Anti-halusinasi** | **8/8 tidak mengarang** (7 menolak eksplisit + 1 menjawab pertanyaan terkait tanpa fabrikasi) |
| **Router** | ~35/39 ≈ **90%** (4 keliru) |
| Kasus positif | 23 PASS |

**Rincian 9 non-PASS:**

| Sebab | Kasus | Catatan |
|---|---|---|
| **Router salah intent** (3) | `fam-gpt` (→counting, harusnya entity), `resolve-paraphrase`×2 (→thematic) | `fam-gemini` juga salah rute tapi kebetulan lolos |
| **Generation menjatuhkan fakta yang SUDAH ada di konteks** (4) | `ent-elephant-data`, `paper-syceval-find`, `adv-multi-limit`, `adv-3hop` | Retrieval lolos, tapi LLM tidak memunculkan fakta spesifiknya di jawaban |
| **Ekspektasi test case salah** (1) | `count-dataset` | LLM menjawab peringkat #1 (ELEPHANT) dengan benar; test case menuntut #2 (hh-rlhf) |
| **Artefak penilai** (1) | `adv-negation` | Ditandai `HALLUCINATED?` padahal LLM **tidak mengarang** — ia menafsir ulang pertanyaan negasi dan menyebut dataset yang dipakai |

**Kesimpulan:** tidak ada halusinasi. Kegagalan nyata terpusat di router (3) dan generation (4).
Panjang konteks **bukan** penyebab (31K char ada yang lolos & ada yang gagal).

### [x] 3.5 Router: query "versi/varian apa saja" salah dirute ke `counting`  ✅ SELESAI (2026-08-03)

| | |
|---|---|
| **Bukti** | `fam-gpt` "Versi GPT apa saja yang diuji?" → router pilih `counting/Method` → dapat ranking degree, bukan varian GPT → LLM jujur bilang "tidak tahu". `fam-gemini` juga salah rute (kebetulan lolos) |
| **Perbaikan** | ✅ Ditambah contoh eksplisit di prompt router (`route_query`): pertanyaan "versi/varian apa saja dari X" = `entity`, bukan `counting`, meski bentuknya mendaftar banyak item |
| **File** | `src/llm/gemini_client.py` (`route_query`) |
| **Menyembuhkan** | ✅ `fam-gpt`, `fam-gemini` — keduanya kini `entity` dan PASS |

### [x] 3.6 Generation menjatuhkan fakta spesifik dari konteks  ✅ SELESAI (2026-08-03) ⭐

| | |
|---|---|
| **Bukti** | 4 kasus: retrieval **berhasil** membawa faktanya ke konteks, tapi jawaban akhir tidak memuatnya (mis. `paper-syceval-find` kehilangan angka "58.19%") |
| **Perbaikan** | ✅ (a) `format_traversal_context()` ditulis ulang: relasi hop-1 dikelompokkan **per tipe relasi** (bukan daftar datar yang mengulang nama entitas tiap baris), relasi hop-2 disajikan sebagai rantai eksplisit `A → relasi → B`; (b) prompt `generate_final_answer` ditambah 5 aturan: kutip angka persis, jangan parafrase-hilangkan angka, jawab lengkap semua item bukan cuma yang pertama, ikuti rantai relasi untuk multi-hop, jawab dalam bahasa pertanyaan |
| **File** | `src/query/traversal.py`, `src/llm/gemini_client.py` |
| **Dampak konteks** | Kasus uji `paper-syceval-find`: **11.934 → 6.060 char (−49%)** |
| **Menyembuhkan** | ✅ `ent-elephant-data`, `paper-syceval-find`, `adv-multi-limit` + bonus `adv-multi-entity` · 🎁 `adv-negation` berubah dari `HALLUCINATED?` jadi `REFUSED-ok` (kejujuran ikut membaik) · ⚠️ `adv-3hop` TIDAK sembuh — bukan soal generation, ternyata **resolver** salah pilih node ("hh-rlhf" → nyantol ke `RLHF` bukan `hh-rlhf dataset`), lihat 3.7 |

**Hasil Batch A (verifikasi `--e2e`, 2026-08-03): 30/39 (77%) → 35/39 (90%)**. Retrieval tetap 94%
(sesuai ekspektasi — perubahan ini di router+generation, bukan retrieval). Anti-halusinasi tetap
solid, malah 1 kasus (`adv-negation`) naik dari abu-abu (`HALLUCINATED?`) jadi bersih (`REFUSED-ok`).

**Sisa 4 kegagalan di 90%:** `count-dataset` (ekspektasi test case salah, LLM sudah benar) ·
`resolve-paraphrase`×2 (item 1.4, sengaja diturunkan) · `adv-3hop` (temuan baru → **3.7**)

### [~] 3.7 Resolver salah pilih node untuk singkatan gabungan  ⏸️ DITUNDA (2026-08-03)

| | |
|---|---|
| **Bukti** | Query "...memakai **dataset hh-rlhf**?" (urutan kata: dataset → hh-rlhf) vs nama node sebenarnya **"hh-rlhf dataset"** (urutan terbalik) → tidak cocok verbatim berurutan → gagal. Sementara node `RLHF` kebetulan cocok karena regex `\w` menganggap tanda hubung sebagai pemisah kata, sehingga `rlhf` di dalam `hh-rlhf` lolos sebagai "kata utuh" tersendiri |
| **Akar masalah** | **Terkonfirmasi** lewat pengujian regex langsung: `"RLHF"` ditemukan verbatim di query = `True`; `"hh-rlhf dataset"` = `False` |
| **Kenapa ditunda** | (a) Kasus sempit — butuh nama node multi-kata dgn urutan beda + ada node lain yg kebetulan jadi substring lewat tanda hubung; (b) kegagalan **graceful** ("tidak tahu", bukan mengarang) — bukan risiko kejujuran; (c) perbaikan cara tanda hubung diperlakukan berisiko regresi pada entitas ber-tanda-hubung lain (`GPT-4o`, `ResNet-50`) yang sudah lolos di kasus lain — perlu pengujian hati-hati, bukan tempelan cepat |
| **File** | `src/query/resolver.py` |
| **Prioritas** | Rendah — buka kembali kalau pola serupa muncul di kasus nyata lain |

### [ ] 3.2 Perbarui `LAPORAN_PROJECT.md`  — 🅲 BATCH C
Angka di laporan masih milik korpus CNN lama (258 node, 94%). Perlu disesuaikan ke korpus aktif —
atau ditulis sebagai dua studi kasus (bagus untuk menunjukkan sistem diuji di 2 domain berbeda).

### [x] 3.3 Bereskan git push  ✅ SELESAI (2026-07-31, dikerjakan user)

Terverifikasi: `origin/main` hanya berisi 2 commit (`4783a70`, `482c2b3`); commit dengan API key
asli (`99db0cc`) **tidak ikut ter-push** dan `.env` tidak ada di tree remote. Key asli tidak pernah
sampai ke GitHub — push protection + purge history berhasil. Batch C tinggal item 3.2.

<details><summary>catatan lama</summary>

`.env` masih ada di commit yang belum ter-push → GitHub Push Protection memblokir.
Solusi: purge `.env` dari history commit yang belum ter-push (lihat instruksi `git filter-branch`).

</details>

### [~] 3.4 Warna Finding vs Paper  ❌ **DICORET** (kosmetik)
Terukur ΔE 9,4 (`#c084fc` vs `#818cf8`) — mirip. Belum diubah karena belum mengganggu.

---

## Catatan yang tidak boleh hilang

0. ⚠️ **Waspadai overfitting ke eval set sendiri.** 31 kasus ini disusun oleh asisten, dari fakta
   yang ditarik dari graph yang sama. Kalau tiap kegagalan dijawab dengan aturan khusus sampai
   semuanya hijau, yang terukur bukan lagi "sistem membaik" melainkan "sistem cocok dengan tes
   buatan sendiri". Gejalanya: item 1.5 pada dasarnya menambah aturan demi 2 kasus tertentu.
   **81% dengan mode kegagalan yang dipahami lebih jujur daripada 100% hasil tambal-sulam.**
   Penyeimbang yang sudah ada: 8 kasus negatif + kasus adversarial, dan tiap perubahan diukur
   di **seluruh** suite sehingga regresi ketahuan.
1. **Selalu ukur sebelum & sesudah.** Setiap item punya kolom "Hasil" — isi dengan skor
   `evaluate.py` sebelum → sesudah. Kalau tidak naik, pertimbangkan rollback.
2. **Jangan optimasi tanpa bukti.** Semua item di sini berasal dari kegagalan yang **terukur**,
   bukan firasat.
3. **Batch semua item Fase 2 jadi satu rebuild.** Kuota harian terbatas.
4. **Restart `server.py`** setiap kali graph atau kode berubah (engine di-cache saat startup).
5. **Backup sebelum eksperimen besar:** `python corpus.py backup <nama>`.
