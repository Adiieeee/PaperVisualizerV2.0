# Design Reference: GraphRAG Visualizer UI

This document describes the visual design and interaction behavior of the graph visualizer page. It covers **what the
user sees and uses**, not the backend pipeline. The code is the source of truth: everything here is taken from
`src/graph/visualizer.py`. When you change the UI, update this document in the same commit.

UI text is in Indonesian (with some English labels); it is quoted here verbatim.

---

## 1. Technology and file layout

### 1.1 Stack

| Component | Technology | Role |
|---|---|---|
| Graph rendering | **vis.js** (via pyvis 0.3.x) | Draws nodes and edges, runs the physics simulation |
| HTML generation | **pyvis** (`Network.generate_html()`) | Produces the base HTML and the vis.js setup |
| Custom UI | Vanilla **HTML/CSS/JS** | Search, right panel, info panel, chat; injected into the pyvis output |
| Server | **FastAPI** (`server.py`) | Serves the page at `/` and the chat endpoint `POST /api/chat` |

### 1.2 One self-contained page

`GraphVisualizer.build_html()` returns a single HTML string (~1.1 MB) with every asset inline (`cdn_resources="in_line"`).
It is used in two places:

- **`server.py`** builds it once at startup and serves it at `/` with a strict CSP (see §13).
- **`python main.py build`** writes a static snapshot to `data/processed/graph_visualization.html`. It opens directly in a
  browser; everything works except the chat, which shows a "cannot reach the server" message.

pyvis adds Bootstrap tags from a CDN even in inline mode. They are stripped (`_BOOTSTRAP_TAG_RE`), so the page has **no
external runtime dependency**; the build prints a warning if any external `src`/`href` remains. The few Bootstrap styles
the page relied on are replaced at the top of `_PANEL_CSS`. Write the HTML as UTF-8 yourself: pyvis's `write_html` uses
the OS codepage and crashes on Windows.

### 1.3 Injection mechanism

1. pyvis renders the base page (`net.generate_html()`).
2. The Bootstrap CDN tags are removed.
3. `_ui_injection()` inserts, right before `</body>`: a `<style>` block (`_PANEL_CSS`), the panel markup (`_PANEL_HTML`),
   and one `<script>` holding `const GR_DATA = {...}` followed by `_PANEL_JS`.
4. `_PANEL_JS` waits until the vis.js globals `network`, `nodes`, `edges` exist, then wires everything up.

`GR_DATA` carries the precomputed data: `TYPE_COLORS`, `TYPE_COUNTS`, `NODE_META` (type, fill, border, description per
node), `TYPE_DESCRIPTIONS`, `EDGE_COUNTS`, `EDGE_DESCRIPTIONS`, `INSIGHTS`, `FAMILIES`, `TOTAL_NODES`, `TOTAL_EDGES`. It is
serialized with `_json_for_script()`, which escapes `<`, `>`, `&`, U+2028 and U+2029 so graph-derived text cannot break out
of the `<script>` block.

---

## 2. Layout

### 2.1 Viewport

```
┌───────────────────────────────────────────────────────────┐
│ [Search bar]                               [Right panel]  │
│ [Info panel]                               ┌────────────┐ │
│ (fades in on                               │ Entity     │ │
│  node selection)    CANVAS (vis.js)        │ Types      │ │
│                                            ├────────────┤ │
│                                            │ Network    │ │
│                                            │ Insights   │ │
│                                            ├────────────┤ │
│                                            │ Edge       │ │
│                                            │ Filters    │ │
│                                            ├────────────┤ │
│                                            │ Path       │ │
│                                            │ Discovery  │ │
│                                            └────────────┘ │
│ [Chat] (bottom-left, collapsed)            [Fit button]   │
└───────────────────────────────────────────────────────────┘
```

| Element | Position | Size |
|---|---|---|
| Canvas | Full viewport (`100vh × 100%`), below all overlays | — |
| Search bar `#gr-search` | Fixed, `top: 20px; left: 20px` | 330 px wide |
| Right panel `#gr-panel` | Fixed, `top: 16px; right: 16px`, scrollable | 230 px wide, `max-height: calc(100vh - 32px)` |
| Info panel `#gr-info` | Fixed, `top: 78px; left: 20px` (under the search bar), scrollable | 330 px wide, `max-height: calc(100vh - 100px)` |
| Chat `#gr-chat` | Fixed, `bottom: 20px; left: 20px` | 340 px wide, body 400 px tall |
| Legend tooltip `#gr-tooltip` | Fixed, `top: 16px; right: 256px` (left of the panel) | 300 px wide |
| Fit button | vis.js navigation button, bottom-right | — |

### 2.2 Z-index

| z-index | Elements |
|---|---|
| 10000 | Legend tooltip |
| 9999 | Search bar, right panel, chat |
| 9998 | Info panel |
| default | vis.js canvas |

---

## 3. Color system

Dark theme only. Panels use a glass look: translucent dark background with `backdrop-filter: blur(20px) saturate(180%)`,
a 1 px faint border, 12–14 px radius and a large soft shadow.

### 3.1 Surfaces and CSS tokens

| Token / element | Value |
|---|---|
| Canvas background | `#080c1a` |
| `--bg-panel` | `rgba(15,17,32,0.88)` (info panel: 0.92) |
| `--border` | `rgba(255,255,255,0.08)` |
| `--bg-hover` | `rgba(255,255,255,0.05)` |
| `--accent` (headings, active states, focus) | `#818cf8` (indigo) |
| `--accent-dim` / `--accent-glow` | `rgba(129,140,248,0.18)` / `rgba(129,140,248,0.35)` |
| `--text-primary` | `#e2e8f0` |
| `--text-dim` (descriptions, bot messages) | `#94a3b8` |
| `--text-muted` (counts, hints) | `#64748b` |
| Insight score badge | `#fbbf24` on `rgba(251,191,36,0.12)` |
| Messages: error / warning / success | `#f87171` / `#fbbf24` / `#34d399` |

Fonts: `'Inter', 'Segoe UI', system-ui, sans-serif` (not loaded; falls back to system fonts); code in chat uses
`'Cascadia Code', 'Fira Code', monospace`.

### 3.2 Node colors (`GraphVisualizer.NODE_COLORS`)

| Entity type | Hex | Color |
|---|---|---|
| Paper | `#818cf8` | Indigo-violet |
| Author | `#f87171` | Coral red |
| Institution | `#a89684` | Warm taupe (chosen so it is distinguishable from Paper and Metric) |
| Method | `#4ade80` | Green |
| Dataset | `#fbbf24` | Amber |
| Metric | `#22d3ee` | Cyan |
| Finding | `#c084fc` | Lavender |
| Limitation | `#fb923c` | Orange |
| Task | `#f472b6` | Pink |
| (unknown type) | `#9E9E9E` | Grey |

Node borders are the fill darkened to 70% (`_darken_hex(color, 0.7)`).

### 3.3 Edge colors (`GraphVisualizer.EDGE_COLORS`)

| Relation | Hex | Note |
|---|---|---|
| `written_by`, `affiliated_with`, `criticizes` | `#f87171` | Author red |
| `uses_method` | `#4ade80` | Method green |
| `proposes_method` | `#22c55e` | Darker green |
| `uses_dataset` | `#fbbf24` | Dataset amber |
| `introduces_dataset` | `#eab308` | Darker amber |
| `evaluated_on` | `#fcd34d` | Light amber |
| `uses_metric` | `#f59e0b` | Orange-amber |
| `finds` | `#c084fc` | Finding lavender |
| `has_limitation` | `#fb923c` | Limitation orange |
| `builds_on` | `#38bdf8` | Sky blue |
| `compares` | `#22d3ee` | Cyan |
| `addresses_task` | `#f472b6` | Task pink |
| `related_to` | `#94a3b8` | Slate-400, generic link kept when salvaging LLM output |
| `variant_of` | `#64748b` | Slate-500, synthetic (build-time) |
| `mentioned_in` | `#475569` | Slate-600, synthetic provenance edge, deliberately recessive |
| (default) | `#475569` | `DEFAULT_EDGE_COLOR` |

Adding an entity or relation type means adding it here, in `TYPE_DESCRIPTIONS`/`EDGE_DESCRIPTIONS`, and in `config.py`
(see CLAUDE.md).

---

## 4. Nodes

### 4.1 Size

1. **Base size by type rarity**: the fewer nodes of a type, the bigger its base.
   `base = 60 - (type_count / max_type_count) × 40`
2. **Degree bonus**: +1.5 px per connection (all edges, including `mentioned_in`).
3. **Cap**: 250 px, to keep hub nodes from destabilizing the physics.

`size = min(base + degree × 1.5, 250)`

### 4.2 Style

- Shape `dot`, border width 2, shadow `rgba(0,0,0,0.8)` size 12 offset (0, 4).
- Label font size 28, color `#e2e8f0`, stroke 5 px `#080c1a` (readable on the dark canvas).
- Labels are truncated to 28 characters (`_truncate`).

### 4.3 Hover tooltip (plain text)

```
<node name>
Type: <entity type>
Connections: <degree>
Paper: <source paper(s), max 70 chars>   ← only when the node has source_paper
```

`source_paper` is set by the graph builder on non-Paper nodes (papers joined with `"; "`).

---

## 5. Edges

| Property | Value |
|---|---|
| Width | 2.5 px at load; 3.5 px once any highlight/filter has been applied or reset (see §14) |
| Smoothing | `continuous` |
| Shadow | Off (performance) |
| Label | None (`font.size: 0`); details only in the tooltip |
| Highlight / hover color | `#ffffff` at load, `#e0e0e0` after JS updates |

Edge tooltip: the relation in Title Case (`uses_dataset` → `Uses Dataset`), plus the description truncated to 120
characters when present. Every edge also carries a custom `rel` property so the Edge Filter can match it.

---

## 6. Physics and interaction

### 6.1 Solver: ForceAtlas2Based

| Parameter | Value | Meaning |
|---|---|---|
| `gravitationalConstant` | -1500 | Repulsion between nodes |
| `centralGravity` | 0.005 | Very weak pull to the center, so the graph spreads out |
| `springLength` | 500 | Ideal edge length |
| `springConstant` | 0.05 | Edge stiffness |
| `damping` | 0.8 | High damping, settles quickly |
| `avoidOverlap` | 1.0 | Prevents node overlap |

Stabilization: enabled, 400 iterations, `updateInterval` 25, `fit: true` (zooms to the whole graph when done). The
loading bar during stabilization is pyvis's default, without custom styling.

### 6.2 Interaction options

| Option | Value |
|---|---|
| `hover` | true |
| `tooltipDelay` | 80 ms |
| `hideEdgesOnDrag` / `hideNodesOnDrag` | false |
| `navigationButtons` | true, but only **Fit** (`vis-zoomExtends`) is shown; zoom and arrow buttons are hidden via CSS |
| `keyboard` | enabled (arrow keys are stopped from reaching vis.js while typing in the chat) |

The Fit button is restyled to match the panels (dark glass, indigo border, accent label).

---

## 7. Highlight styles

Three node states are shared by every feature:

| State | Look |
|---|---|
| **Glow** (target) | Size ×1.4, no border, fill color as border, white label, halo `shadow` in the node's own color, size 26 |
| **Normal** (direct neighbor, or reset) | Base size and colors, border 1.5, label `#e0e0e0`, soft shadow `rgba(0,0,0,0.18)` |
| **Dim** (everything else) | Opacity 0.22, faded label, no shadow |

Edges touching a target stay at opacity 1.0; all others drop to 0.02.

Highlight sources are **mutually exclusive**: legend type, single node (search / insights / info tag / double-click),
path, and edge filter. Activating one resets the others.

---

## 8. Right panel

Four collapsible `<details>/<summary>` sections; each opens and closes independently. The open section's summary gets a
3 px accent left border and an accent gradient; the chevron rotates when closed.

### 8.1 Entity Types (legend) — open by default

- One row per type present in the graph: colored dot (with glow), type name, node count.
- **Click a row** → highlight all nodes of that type (glow), their neighbors normal, the rest dim. The active row gets
  `--accent-dim` background and an accent left border. Clicking the same row again resets.
- **`?` icon** → on hover, shows the legend tooltip: each type with its Indonesian description (`TYPE_DESCRIPTIONS`).
- Stats bar: `"<N> nodes • <M> edges"`.
- Hint: `"Hover untuk detail · Drag & scroll to navigate"`.

### 8.2 Network Insights — closed, built lazily on first open

- **Top 5 per category**: Method, Dataset, Metric, Author. Ranked by degree over real relations only (`mentioned_in` and
  `variant_of` are excluded, consistent with `analytics.get_counting_stats`; keep the two in sync). Each item shows the
  name (ellipsis at 145 px) and a gold badge `⭐ <degree>`. A category with no nodes shows `"Tidak ada data."`.
- **Top Entity Families**: families linked by `variant_of` with at least 2 variants, top 6, ranked by number of Papers
  reached (parent + variants), then by variant count. Badge `<n>v · <p>p`. The group's dot uses the first family's type
  color. The group **hides itself** when the corpus has no versioned entities.
- **Click an item** → closes the section and selects the node (same as a search selection, §9.3).

Note: this panel ranks by degree; the chatbot's counting answers rank by number of distinct Papers first
(`get_usage_stats`), so the two can differ.

### 8.3 Edge Filters — closed by default

- A `<select>` listing every relation present in the graph, sorted by count (descending), formatted
  `Relation Name (count)`. First option: `-- Tampilkan Semua --` (value `all`).
- Selecting a relation shows a description box: `[From → To]` plus an Indonesian description (`EDGE_DESCRIPTIONS`,
  trusted HTML from Python constants). An unknown relation falls back to its escaped name.
- Filter effect: matching edges opacity 1.0, others 0.03; nodes touching a matching edge normal, others dim. Choosing
  `all` resets.

### 8.4 Path Discovery — closed by default, node list filled lazily on first open

- Two text inputs (`Node A`, `Node B`) sharing a `<datalist>` of every node id, and a button `Temukan Jalur`. Enter in
  either input also runs the search.
- Algorithm: **BFS** shortest path on the undirected, unweighted graph (all edges, including synthetic ones).
- Result highlight: path nodes keep their size with a `#e0e0e0` halo (size 20) and white labels; other nodes opacity 0.15.
  Path edges `#e0e0e0`, width 4; other edges width 1, opacity 0.05. The camera fits the path (800 ms, easeInOutQuad).
- Messages:
  - either input empty → red `"Pilih kedua node terlebih dahulu."`
  - unknown node → red `"Node tidak valid. Pastikan dari dropdown."`
  - no path → amber `"Tidak ada jalur yang menghubungkan kedua node ini."`
  - found → green `"Jalur ditemukan! (N lompatan)"`

---

## 9. Search

### 9.1 Search bar

Fixed top-left, glass style, placeholder `"🔍 Search Node"`. On focus the box gets an accent border and a 3 px accent
glow ring. A clear button `×` appears while the input has text.

### 9.2 Dropdown

- Appears under the bar while typing; max height 360 px, scrollable.
- Matching: case-insensitive **substring** match on the node id (no fuzzy matching).
- Ranking: exact match, then prefix match, then contains; ties sorted alphabetically. At most 50 results.
- Each row: node name (left, ellipsis) and its type (right, small uppercase accent text).
- No match: `"Tidak ada node cocok."`
- Keyboard: ↑/↓ move the active row, Enter selects the active row (or the first one), Escape closes the dropdown.
  Clicking outside the search box closes it.

### 9.3 Selecting a node (`executeSearchSelection`)

The single "select a node" action, used by search, Network Insights and info-panel tags:

1. Closes the Network Insights section and the dropdown.
2. Fills the search input with the node name and shows the clear button.
3. Highlights the node (glow, neighbors normal, rest dim) and opens the info panel.
4. Focuses the camera on it (scale 0.25, 1000 ms, easeInOutQuad).

### 9.4 Clearing

The `×` button resets every highlight and filter, empties the input, closes the info panel, and fits the whole graph
(800 ms).

---

## 10. Info panel (node detail)

### 10.1 When it opens

On every single-node highlight: a search selection, a Network Insights item, a tag inside the info panel, or a
**double-click on a node**. A single click on a node does not open it (vis.js only selects the node).

### 10.2 Animation

Fades and slides in: `opacity 0 → 1`, `translateX(-24px) → 0`, 280 ms, `cubic-bezier(0.25, 0.8, 0.25, 1)`. While closed it
ignores pointer events.

### 10.3 Content

```
┌──────────────────────────────┐
│ [TYPE BADGE]               × │ ← badge in the node's color; × resets everything
│ Full node name               │
│ Description (if any)         │
│ Total Koneksi: N relasi      │
├──────────────────────────────┤
│ ● Ditemukan di Paper (N)     │ ← non-Paper nodes only: neighbors of type Paper
│ [paper] [paper] ...          │
├──────────────────────────────┤
│ ● Terkoneksi dengan Method (N)│ ← other neighbors grouped by type, types sorted A–Z
│ [method] [method] ...        │
└──────────────────────────────┘
```

Every tag is clickable and selects that node (§9.3), so the user can walk the graph from panel to panel. Long names are
truncated with an ellipsis; the full name is in the tag's `title`.

---

## 11. Chat

- Fixed bottom-left, **collapsed by default**; clicking the header (`● GraphRAG Chat`, green status dot) toggles it.
- Welcome line: `"Tanya apa saja tentang paper di graph ini — metode, dataset, author, atau tren penelitian."`
- Input: auto-growing textarea (up to 160 px), placeholder `"Ketik pertanyaan..."`. Enter sends, Shift+Enter adds a new
  line. The send button is disabled while a request is in flight; only one request at a time.
- While waiting, a bot bubble shows `"mengetik…"` in italics.
- Bubbles: user right-aligned with accent tint; bot left-aligned, dim text. User text is inserted with `textContent`; bot
  text goes through `renderMarkdown` (§13).
- Endpoint: `POST /api/chat` with `{"query": "..."}`. Every response, including errors (422, 429 quota, 503), has the
  shape `{"answer": "..."}` and is shown as a bot message.
- If the server cannot be reached (e.g. the static snapshot opened from disk): `"⚠️ Tidak bisa menghubungi server.
  Jalankan `python server.py`, lalu buka http://127.0.0.1:8000"`.

---

## 12. Reset behavior

- **Click on empty canvas** while any highlight, path, search selection or edge filter is active → reset everything
  (highlights, edge filter, search input, info panel). The camera does not move.
- **`×` in the search bar** → the same reset plus a fit to the whole graph.
- **`×` in the info panel** → the same reset, without moving the camera.

State is kept in plain variables inside `_PANEL_JS`: `currentHighlightedType`, `currentSearchNodeId`, `currentPathActive`,
`currentEdgeFilter`, plus the lazy-build flags `insightsBuilt` and `pathDatalistPopulated`.

---

## 13. Security constraints on the UI

These are binding rules (details in `docs/SECURITY.md`); `tests/test_frontend.py` and `tests/test_frontend_browser.py`
cover them.

- Node names, descriptions and relation names come from LLM output over untrusted PDFs. Every graph-derived value that
  goes into `innerHTML` must pass through `esc()`; attribute values that carry node ids use `encodeURIComponent`.
- `renderMarkdown` must **escape all HTML first**, then add whitelisted tags only (`h1`–`h3`, `strong`, `em`, `code`,
  `ul`/`li`, `p`). No links, images, attributes or raw HTML.
- The page served by `server.py` has a hash-based CSP (`script-src` lists the hashes of the inline scripts). A new inline
  `<script>` block is fine (hashes are recomputed); inline event attributes (`onclick=`) and external scripts are not.
  `style=` attributes are allowed (`style-src 'unsafe-inline'`).
- Never reintroduce a CDN or other external resource.

---

## 14. Known issues and possible improvements

- **Edge width and highlight color change after the first interaction**: edges are created at 2.5 px with a white
  highlight, but every highlight/reset rewrites them at 3.5 px with `#e0e0e0`.
- **Stale Paper dot in the info panel**: the "Ditemukan di Paper" group dot is hard-coded `#3B52B5`, while the Paper type
  color is `#818cf8`.
- **Invalid fallback border** in `metaOf()` (`#a99e8f666`) for nodes missing from `NODE_META`; harmless in practice since
  every node is in it.
- **Fixed label size** (28): labels overlap in dense areas until you zoom in.
- **Dark mode only**, no light theme or toggle.
- **Substring search only**, no fuzzy or semantic search.
- **Network Insights covers four types** (Method, Dataset, Metric, Author) plus families; Paper, Institution, Finding,
  Limitation and Task are not ranked.
- **No responsive layout**: designed for desktop widths (> 1024 px). On narrow screens the search bar, right panel, info
  panel and chat can overlap, and the info panel's max height does not account for the open chat.
- **Mixed language** in UI labels (English section titles, Indonesian messages).
