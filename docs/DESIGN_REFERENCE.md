# Design Reference: GraphRAG Visualizer UI

Dokumen ini mendokumentasikan seluruh aspek desain visual dan interaksi dari halaman visualisasi `graph.html`, yang di-generate oleh `app/visualizer.py`. Fokusnya murni pada **apa yang dilihat dan digunakan user** — bukan pipeline backend.

---

## 1. Technology Stack (UI Layer)

### 1.1 Rendering Engine

| Komponen | Teknologi | Peran |
|----------|-----------|-------|
| Graph rendering | **vis.js** (via PyVis) | Menggambar node, edge, dan mengelola physics simulation |
| HTML generation | **PyVis** (`Network.generate_html()`) | Menghasilkan boilerplate HTML + vis.js setup |
| Custom UI | Vanilla **HTML/CSS/JS** (injected) | Panel, search, info, chatbot — semua di-inject ke output PyVis |
| Server | **FastAPI** (`server.py`) | Menyajikan `graph.html` dan endpoint chatbot `/api/chat` |

### 1.2 Arsitektur File

Seluruh UI termuat dalam **satu file** `output/graph.html` (~1.2 MB). Tidak ada file CSS/JS terpisah. Semua aset (vis.js library, custom CSS, custom JS) di-embed inline menggunakan `cdn_resources="in_line"` pada PyVis.

**Mengapa satu file?**
- Portabel: file bisa di-share, dibuka langsung di browser tanpa server (kecuali fitur chatbot).
- Offline-ready: tidak ada dependensi CDN saat membuka file.

### 1.3 Injection Mechanism

`visualizer.py` menggunakan pattern sederhana:
1. PyVis generate HTML standar via `net.generate_html()`
2. Fungsi `_inject_ui()` menyisipkan blok `<style>`, HTML elements, dan `<script>` tepat sebelum `</body>`
3. Custom JS mengakses variabel global vis.js: `network`, `nodes`, `edges`

---

## 2. Layout & Spatial Design

### 2.1 Viewport

```
┌─────────────────────────────────────────────────────────┐
│ [Search Bar]                              [Right Panel] │
│                                           ┌───────────┐ │
│                                           │ Entity    │ │
│                                           │ Types     │ │
│   [Info Panel]     CANVAS (vis.js)        │ (Legend)  │ │
│   (slide-in)                              ├───────────┤ │
│                                           │ Network   │ │
│                                           │ Insights  │ │
│                                           ├───────────┤ │
│                                           │ Edge      │ │
│                                           │ Filters   │ │
│                                           ├───────────┤ │
│                                           │ Path      │ │
│                                           │ Discovery │ │
│                                           └───────────┘ │
│ [Chatbot]                          [Fit/Center Button]  │
│ (bottom-left)                                           │
└─────────────────────────────────────────────────────────┘
```

- **Canvas**: Mengisi seluruh viewport (`100vh × 100%`), di bawah semua overlay.
- **Search Bar**: Fixed, top-left (`top: 20px; left: 20px`), width 320px.
- **Right Panel**: Fixed, top-right (`top: 16px; right: 16px`), width 220px, scrollable.
- **Info Panel**: Fixed, left side, slide-in animation (`left: -400px` → `left: 20px`), width 320px, muncul di bawah search bar.
- **Chatbot**: Fixed, bottom-left (`bottom: 20px; left: 20px`), width 340px.
- **Fit Button**: Tombol bawaan vis.js (navigasi), posisi bottom-right.

### 2.2 Z-Index Hierarchy

| Layer | z-index | Element |
|-------|---------|---------|
| Top | 10000 | Legend tooltip (global) |
| High | 9999 | Search bar, Right panel, Chatbot |
| Mid | 9998 | Info panel |
| Base | (default) | vis.js canvas |

---

## 3. Color System

### 3.1 Background & Surface

| Element | Color | Kode |
|---------|-------|------|
| Canvas (graph area) | Sangat gelap navy | `#0f0f1a` |
| Panel background | Semi-transparent dark | `rgba(15,15,26,0.92)` dengan `backdrop-filter: blur(10px)` |
| Panel border | Subtle white | `rgba(255,255,255,0.12)` |
| Section divider | Faint line | `rgba(255,255,255,0.1)` |

### 3.2 Node Colors (didefinisikan di `config.py`)

| Tipe Entity | Hex | Visual |
|-------------|-----|--------|
| Paper | `#4A90D9` | 🔵 Biru |
| Author | `#E74C3C` | 🔴 Merah |
| Institution | `#34495E` | ⬛ Abu gelap |
| Method | `#2ECC71` | 🟢 Hijau |
| Dataset | `#F39C12` | 🟠 Oranye |
| Metric | `#00BCD4` | 🔷 Cyan |
| Finding | `#9B59B6` | 🟣 Ungu |
| Limitation | `#95A5A6` | 🩶 Abu terang |

Border setiap node digelapkan 30% dari warna asli menggunakan `_darken_hex(color, factor=0.7)`.

### 3.3 Edge Colors (didefinisikan di `visualizer.py`)

| Relasi | Hex | Warna |
|--------|-----|-------|
| `written_by` | `#E74C3C` | Merah |
| `affiliated_with` | `#E74C3C` | Merah |
| `uses_method` | `#2ECC71` | Hijau |
| `proposes_method` | `#27AE60` | Hijau tua |
| `uses_dataset` | `#F39C12` | Oranye |
| `uses_metric` | `#E67E22` | Oranye tua |
| `finds` | `#9B59B6` | Ungu |
| `has_limitation` | `#95A5A6` | Abu |
| `criticizes` | `#E74C3C` | Merah |
| `builds_on` | `#3498DB` | Biru |
| `compares` | `#1ABC9C` | Teal |
| `same_author` | `#BDC3C7` | Abu terang |
| (default) | `#555577` | Ungu gelap |

### 3.4 Accent & Text Colors

| Peran | Hex |
|-------|-----|
| Accent utama (heading, link, active state) | `#a0c4ff` (light blue) |
| Text primary | `#e0e0e0` |
| Text muted | `#888888` |
| Count badge | `#888888` |
| Analytics score | `#ffd700` (gold) |
| Error message | `#E74C3C` |
| Success message | `#2ECC71` |
| Warning message | `#F39C12` |

---

## 4. Node Rendering

### 4.1 Ukuran (Size)

Ukuran node ditentukan oleh dua faktor:

1. **Base size** (berdasarkan raritas tipe): Semakin sedikit jumlah node dari tipe tersebut, semakin besar base size-nya.
   ```
   base_size = 60 - (type_count / max_type_count) × 40
   ```
   Contoh: Jika Paper hanya ada 3 node tapi Method ada 50, maka Paper akan lebih besar secara default.

2. **Degree bonus**: Setiap koneksi menambah 1.5px.
   ```
   final_size = base_size + (degree × 1.5)
   ```

3. **Cap**: Ukuran maksimum dibatasi di 250px untuk mencegah masalah physics engine pada hub nodes.

### 4.2 Shape & Style

- Shape: `dot` (lingkaran)
- Shadow: enabled, `rgba(0,0,0,0.6)`, size 8px, offset (2,2)
- Border width: 1.5px
- Font: size 30, color `#e0e0e0`, stroke 4px `#0f0f1a` (outline agar terbaca di dark bg)
- Label: di-truncate ke 28 karakter menggunakan `_truncate()`

### 4.3 Tooltip (Hover)

Tooltip plain text (bukan HTML) berisi:
```
Node Name
Type: [EntityType]
Connections: [degree]
Paper: [source_paper, max 70 char]
```

---

## 5. Edge Rendering

| Property | Value |
|----------|-------|
| Width | 3.5px |
| Smooth type | `continuous` (curved) |
| Shadow | disabled (performa) |
| Label | Kosong (hanya tooltip saat hover) |
| Highlight color | `#ffffff` |
| Hover color | `#ffffff` |

### Tooltip Edge

```
[Relation Type (Title Case)]
[Description, max 120 char]
```

---

## 6. Physics Engine

### 6.1 Solver: ForceAtlas2Based

| Parameter | Value | Keterangan |
|-----------|-------|------------|
| `gravitationalConstant` | -1500 | Gaya tolak antar node (negatif = tolak) |
| `centralGravity` | 0.005 | Tarikan ke pusat (sangat lemah, agar menyebar) |
| `springLength` | 500 | Panjang ideal edge |
| `springConstant` | 0.05 | Kekakuan pegas edge |
| `damping` | 0.8 | Peredam gerakan (tinggi = cepat stabil) |
| `avoidOverlap` | 1.0 | Cegah node bertumpuk |

### 6.2 Stabilization

| Parameter | Value |
|-----------|-------|
| `enabled` | true |
| `iterations` | 400 |
| `updateInterval` | 25 |
| `fit` | true (zoom ke seluruh graph setelah stabil) |

### 6.3 Interaction Options

| Option | Value |
|--------|-------|
| `hover` | true |
| `tooltipDelay` | 80ms |
| `hideEdgesOnDrag` | false |
| `hideNodesOnDrag` | false |
| `navigationButtons` | true (hanya Fit yang ditampilkan) |
| `keyboard` | enabled |

**Navigation buttons**: Tombol zoom in/out dan directional (up/down/left/right) di-hide via CSS. Hanya tombol **Fit** (zoom extends / center) yang ditampilkan, dengan styling custom (background putih, rounded, shadow).

---

## 7. Interactive Features (Right Panel)

Right panel menggunakan pattern `<details>/<summary>` HTML untuk collapsible sections. Setiap section bisa dibuka/tutup secara independen.

### 7.1 Entity Types (Legend)

**Default state**: Open.

Menampilkan daftar tipe entitas dengan:
- Dot berwarna sesuai `NODE_COLORS`
- Nama tipe + jumlah node dalam tanda kurung
- **Clickable**: Klik untuk toggle highlight semua node bertipe tersebut

**Highlight behavior** (`highlightType()`):
1. Node bertipe yang dipilih: **glow** (border putih, shadow berwarna, font putih)
2. Node yang terhubung langsung (1st-degree neighbor): **normal** (tetap jelas)
3. Node lainnya: **sangat redup** (opacity 0.15)
4. Edge terhubung: opacity 1.0; edge lain: opacity 0.02
5. Klik lagi tipe yang sama → reset semua (toggle off)

**Active state visual**: Background `rgba(160,196,255,0.2)`, border-left 3px `#a0c4ff`, shadow inset.

**Legend Tooltip** (ikon `?`):
- Hover pada ikon `?` di samping "Legend" menampilkan tooltip fixed-position yang menjelaskan setiap tipe entitas dalam bahasa Indonesia.

**Stats Bar**: Di bawah legend, menampilkan total node dan edge (contoh: "245 nodes • 382 edges").

**Hint Text**: "Hover untuk detail · Drag & scroll to navigate"

### 7.2 Network Insights (Analytics)

**Default state**: Closed. Data di-generate **lazy** (hanya saat section pertama kali dibuka).

Menampilkan **Top 5** node berdasarkan degree (jumlah koneksi) untuk 4 kategori:
1. **Top Method** (warna hijau)
2. **Top Dataset** (warna oranye)
3. **Top Metric** (warna cyan)
4. **Top Author** (warna merah)

Setiap item menampilkan:
- Nama node (truncated, max-width 140px)
- Badge score: ⭐ + jumlah degree (gold badge, background `rgba(255,215,0,0.1)`)

**Clickable**: Klik item → tutup section insights, lalu execute search selection (focus + highlight node tersebut + buka info panel).

### 7.3 Edge Filters

**Default state**: Closed.

Komponen:
- **Dropdown select** (`<select>`) berisi semua tipe relasi yang ada di graph, diurutkan berdasarkan jumlah (descending). Format: `Relation Name (count)`.
- Opsi pertama: `-- Tampilkan Semua --` (value `all`).
- **Description box**: Muncul di bawah dropdown saat tipe relasi dipilih. Menjelaskan arti relasi tersebut dalam format `[From → To]` + deskripsi bahasa Indonesia.

**Filter behavior** (`edgeSelect change event`):
1. Edge yang cocok dengan tipe terpilih: opacity 1.0
2. Edge lainnya: opacity 0.03 (hampir invisible)
3. Node yang terhubung ke edge aktif: warna normal
4. Node lainnya: redup (opacity 0.15)
5. Pilih "Tampilkan Semua" → reset ke normal

**Edge descriptions** (hardcoded di JS):

| Edge Type | Deskripsi |
|-----------|-----------|
| `written_by` | **[Paper → Author]** Menunjukkan penulis dari sebuah paper. |
| `affiliated_with` | **[Author → Institution]** Menunjukkan institusi tempat penulis bernaung. |
| `uses_method` | **[Paper → Method]** Menunjukkan bahwa paper tersebut menggunakan metode atau algoritma ini. |
| `proposes_method` | **[Paper → Method]** Menunjukkan bahwa paper tersebut menciptakan metode ini perdana. |
| `uses_dataset` | **[Paper → Dataset]** Menunjukkan bahwa paper mengevaluasi modelnya di dataset ini. |
| `uses_metric` | **[Paper → Metric]** Menunjukkan bahwa paper menggunakan metrik ini untuk tolak ukur. |
| `finds` | **[Paper → Finding]** Menunjukkan hasil temuan, kesimpulan, atau klaim dari paper. |
| `has_limitation` | **[Method/Dataset → Limitation]** Menunjukkan batasan atau kelemahan (kegagalan) yang diakui. |
| `criticizes` | **[Paper → Method/Metric]** Menunjukkan paper memberikan kritik tajam atau menemukan kecacatan fatal. |
| `builds_on` | **[Method → Method]** Menunjukkan metode ini adalah perbaikan/pengembangan dari metode lain. |
| `compares` | **[Method/Paper → Method]** Menunjukkan perbandingan performa atau efisiensi (A vs B). |
| `same_author` | **[Paper → Paper]** Menunjukkan dua paper ini memiliki penulis yang sama (Cross-research). |

### 7.4 Path Discovery

**Default state**: Closed. Datalist (autocomplete) di-populate **lazy** saat pertama kali section dibuka.

Komponen:
- **Node A input**: Text input dengan `<datalist>` autocomplete berisi semua node di graph.
- **Node B input**: Sama dengan Node A.
- **Button "Temukan Jalur"**: Menjalankan BFS shortest path.
- **Result div**: Menampilkan hasil.

**Algorithm**: BFS (Breadth-First Search), bukan Dijkstra (meskipun button ID-nya `mh-run-dijkstra`). Graph undirected, semua edge weight = 1.

**Path highlight behavior** (`highlightPathInGraph()`):
1. Node di jalur: border putih, glow putih (shadow size 20)
2. Node lainnya: opacity 0.1
3. Edge di jalur: putih, width 4px, opacity 1.0
4. Edge lainnya: opacity 0.05, width 1px
5. Camera: fit ke nodes di jalur dengan animasi 800ms

**Error states**:
- Kedua input kosong → pesan merah "Pilih kedua node terlebih dahulu."
- Node tidak valid → pesan merah "Node tidak valid. Pastikan dari dropdown."
- Tidak ada jalur → pesan kuning "Tidak ada jalur yang menghubungkan kedua node ini."
- Jalur ditemukan → pesan hijau "Jalur ditemukan! (N lompatan)"

---

## 8. Search Feature

### 8.1 Search Bar

- Position: Top-left, fixed.
- Placeholder: "🔍 Search Node"
- Styling: Glassmorphism (semi-transparent dark bg, blur, subtle border)
- Focus state: border `#a0c4ff`, glow shadow, slightly darker bg

### 8.2 Search Dropdown

Muncul di bawah search bar saat user mengetik. Max 350px height, scrollable.

**Matching**: Case-insensitive substring match pada `node.id`. Tidak ada fuzzy search — hanya `includes()`.

**Sorting priority**:
1. Exact match → muncul pertama
2. Starts with query → muncul berikutnya
3. Contains query → diurutkan alfabetis

**Limit**: Maksimum 50 hasil.

**Item format**: Nama node (kiri) + badge tipe entity (kanan, styled `#a0c4ff`).

### 8.3 Selection Behavior (`executeSearchSelection()`)

Saat user memilih node dari dropdown atau menekan Enter:
1. Dropdown tertutup
2. Search input diisi nama node
3. Clear button (×) muncul
4. Node di-highlight (sama seperti highlight type, tapi untuk 1 node)
5. Camera zoom + focus ke node tersebut (scale 0.25, animasi 1 detik)
6. **Info Panel** terbuka dengan detail node

### 8.4 Clear Search

Klik tombol × atau klik area kosong di canvas:
- Reset highlight
- Tutup info panel
- Camera kembali fit ke seluruh graph (animasi 800ms)

---

## 9. Info Panel (Node Detail)

### 9.1 Trigger

Info panel terbuka saat:
- User memilih node dari **search dropdown**
- User **double-click** node di canvas
- User **klik node** di canvas (click event juga meng-update search bar)

### 9.2 Animation

Slide-in dari kiri: `left: -400px` → `left: 20px` dengan `cubic-bezier(0.25, 0.8, 0.25, 1)`, durasi 300ms.

### 9.3 Content Structure

```
┌──────────────────────────┐
│ [Type Badge]           × │ ← Close button
│ Node Full Name           │ ← h3, white
│ Total Koneksi: N relasi  │ ← small, muted
├──────────────────────────┤
│ Source Paper             │ ← Hanya untuk non-Paper nodes
│ [paper title tag]        │
├──────────────────────────┤
│ 🔵 Ditemukan di Paper (N)│ ← Cross-paper connections
│ [paper1] [paper2] ...    │ ← Clickable tags
├──────────────────────────┤
│ 🟢 Terkoneksi dgn Method │ ← Grouped by entity type
│ [method1] [method2]      │ ← Clickable tags
├──────────────────────────┤
│ 🟠 Terkoneksi dgn Dataset│
│ [dataset1] [dataset2]    │
└──────────────────────────┘
```

### 9.4 Connection Grouping Logic

Koneksi node dikelompokkan sebagai berikut:
1. **"Ditemukan di Paper"**: Jika node bukan Paper, dan neighbor-nya bertipe Paper → masuk grup khusus "Ditemukan di Paper" dengan dot biru.
2. **"Terkoneksi dengan [Type]"**: Semua neighbor lainnya dikelompokkan berdasarkan entity type mereka. Dot warna sesuai node color tipe tersebut.

### 9.5 Navigation

Setiap tag koneksi (`.conn-tag`) bersifat **clickable**. Klik tag → `executeSearchSelection()` dipanggil untuk node tersebut, sehingga info panel langsung berpindah ke node yang diklik. Ini memungkinkan **navigasi berantai antar node** tanpa harus kembali ke search.

### 9.6 Max Height

Info panel memiliki `max-height` dinamis:
- Saat chatbot terbuka: `calc(100vh - 480px)` (memberi ruang untuk chatbot di bawah)
- Saat chatbot tertutup: `calc(100vh - 100px)`

Content di dalam panel scrollable (overflow-y: auto).

---

## 10. Chatbot UI

> **Note**: User request menyatakan fitur chatbot tidak termasuk scope design ini. Bagian ini didokumentasikan hanya untuk kelengkapan.

### 10.1 Layout

- Position: Fixed, bottom-left.
- Collapsible: Header clickable untuk toggle body (▼ ↔ ▲).
- Saat collapse, hanya header yang terlihat.

### 10.2 Styling

- Header: Background `rgba(255,255,255,0.05)`, green dot indicator, title "GraphRAG Chat".
- Messages area: 350px height, scrollable, flex column.
- User bubble: Blue tinted (`rgba(74,144,217,0.2)` bg, blue border).
- Bot bubble: Dark tinted (`rgba(255,255,255,0.05)` bg), align left.
- Input: Dark background, 1px border, placeholder "Ketik pertanyaan..."

### 10.3 Endpoint

POST `/api/chat` dengan body `{ "query": "..." }`. Response: `{ "answer": "..." }`.

---

## 11. Orphan Node Handling (Design Impact)

### 11.1 Apa itu Orphan Node?

Node yang tidak memiliki jalur (path) ke Paper sumbernya di graph. Ini bisa terjadi karena:
- Relasi yang menghubungkannya ke Paper di-skip (gagal validasi).
- Entitas ditemukan di chunk yang tidak mengandung relasi ke entitas lain.

### 11.2 Solusi: `_connect_isolated_nodes()`

Di `graph_builder.py`, setiap node non-Paper dicek apakah ada path ke source paper-nya menggunakan `nx.has_path()`. Jika tidak ada, edge `mentioned_in` dipaksa dibuat.

**Dampak visual**: Tidak ada node yang "melayang sendirian" di graph. Semua node terhubung ke setidaknya satu Paper node, menjaga integritas visual cluster.

### 11.3 Relasi `mentioned_in`

| Property | Value |
|----------|-------|
| Edge type | `mentioned_in` |
| Label | "mentioned in" |
| Tooltip | "Entity found in this paper" |
| Warna | Default edge color (`#555577`) |

Relasi ini **tidak ada di daftar `_EDGE_COLORS`** sehingga menggunakan warna default. Ini bisa dianggap sebagai minor design gap (tidak tampil di Edge Filter descriptions).

---

## 12. Global Reset Mechanism

### 12.1 Reset via Click

**Klik area kosong** di canvas (tidak mengenai node/edge):
- Jika ada highlight aktif (legend filter ATAU search selection) → reset semua
- Clear search input, sembunyikan clear button
- Tutup info panel

### 12.2 Reset via Double-Click

**Double-click area kosong** di canvas:
- Reset Edge Filter dropdown ke "Tampilkan Semua" (trigger change event)
- Trigger Clear Search button click (reset search + highlight)
- Call `resetHighlight()` (reset legend)

Ini adalah **master reset** — membersihkan semua filter dan highlight sekaligus.

---

## 13. State Management

Tidak ada framework state management. State di-track menggunakan variabel global:

| Variable | Type | Purpose |
|----------|------|---------|
| `currentHighlightedType` | `string \| null` | Tipe entity yang sedang di-highlight via legend |
| `currentSearchNodeId` | `string \| null` | ID node yang sedang di-select via search |
| `analyticsGenerated` | `boolean` | Apakah analytics sudah di-compute (lazy) |
| `datalistsPopulated` | `boolean` | Apakah datalist path discovery sudah di-populate (lazy) |

**Mutual exclusion**: Legend highlight dan search selection saling membatalkan. Mengaktifkan satu akan me-reset yang lain.

---

## 14. Loading Screen

PyVis menyertakan loading bar bawaan saat physics stabilization berjalan. Styling custom diterapkan:

- Background: `#0f0f1a` (sesuai canvas)
- Border: Rounded 20px, semi-transparent dark, box-shadow
- Progress bar color: `#a0c4ff`
- Text color: `#a0c4ff`
- Posisi: Centered vertically via `transform: translateY(-50%)`

---

## 15. Responsive Behavior

**Tidak ada responsive breakpoint yang didefinisikan.** UI dirancang untuk viewport desktop lebar (>1024px).

Pada layar sempit (<768px), beberapa masalah yang mungkin terjadi:
- Right panel dan search bar bisa overlap
- Info panel dan chatbot bisa overlap
- Node label terlalu besar relatif terhadap viewport

> **Catatan**: Ini acceptable karena target user adalah researcher yang menggunakan laptop/desktop.

---

## 16. Known Design Issues & Improvement Opportunities

### 16.1 `mentioned_in` tidak ada di Edge Filter

Relasi `mentioned_in` (yang dibuat otomatis untuk orphan nodes) tidak memiliki entry di `edgeDescriptions` dan tidak muncul di Edge Filter dropdown jika ada. Ini karena relasi ini ditambahkan di `graph_builder.py` tapi tidak didaftarkan di `_EDGE_COLORS` di `visualizer.py`.

### 16.2 Node Font Size Tidak Adaptive

Font size node di-set ke 30px fixed. Pada graph yang sangat padat (>200 node), label bisa saling overlap. Saat ini dimitigasi dengan `font.size: 1` pada saat `add_node()` (label hampir tidak terlihat kecuali zoom in), tapi `set_options()` override ke size 30.

### 16.3 No Dark/Light Mode Toggle

UI hanya mendukung dark mode. Tidak ada opsi untuk beralih ke light mode.

### 16.4 Search Hanya Substring Match

Search menggunakan `includes()` — tidak ada fuzzy/semantic search. User harus mengetik substring persis dari nama node.

### 16.5 Path Discovery Label Tidak Akurat

Button berlabel "Temukan Jalur" (Path Discovery) menggunakan **BFS** (unweighted shortest path), bukan Dijkstra. Secara fungsional benar karena graph unweighted, tapi penamaan variabel (`mh-run-dijkstra`) tidak akurat.

### 16.6 Analytics Tidak Mencakup Semua Tipe

Network Insights hanya menampilkan Top 5 untuk Method, Dataset, Metric, dan Author. Tipe Paper, Institution, Finding, dan Limitation tidak ditampilkan.

### 16.7 Chatbot Overlap dengan Info Panel

Saat chatbot dibuka dan info panel juga terbuka, keduanya bisa saling overlap di layar kecil. Mitigasi partial: max-height info panel dikurangi saat chatbot terbuka.
