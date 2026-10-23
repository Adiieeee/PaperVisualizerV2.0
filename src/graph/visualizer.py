import json
import re
from collections import Counter
from pyvis.network import Network
import networkx as nx

# pyvis menyisipkan Bootstrap dari CDN (dengan SRI) walau cdn_resources="in_line". Panel kita tidak
# memakainya, dan itu satu-satunya dependensi eksternal saat runtime — melepasnya membuat halaman
# benar-benar offline (DESIGN_REFERENCE §1.2) dan memungkinkan CSP tanpa origin eksternal.
_BOOTSTRAP_TAG_RE = re.compile(
    r"<(?:link|script)\b[^>]*cdn\.jsdelivr\.net/npm/bootstrap[^>]*>(?:\s*</script\s*>)?",
    re.IGNORECASE,
)
_EXTERNAL_REF_RE = re.compile(r"""<(?:script|link|img|iframe)\b[^>]*\b(?:src|href)\s*=\s*["']https?://""", re.IGNORECASE)

# Karakter yang tidak boleh muncul mentah di dalam <script>. Ditulis sebagai escape JSON (\\uXXXX)
# yang dibangun via chr(): "<" / "<!--" bisa memecah blok skrip, U+2028/2029 memecah literal JS lama.
_SCRIPT_UNSAFE_RE = re.compile("[<>&" + chr(0x2028) + chr(0x2029) + "]")


def _json_for_script(obj) -> str:
    """JSON yang aman ditempel di dalam <script>: <, >, &, U+2028, U+2029 -> escape JSON (\\uXXXX)."""
    raw = json.dumps(obj, ensure_ascii=False)
    return _SCRIPT_UNSAFE_RE.sub(lambda m: f"{chr(92)}u{ord(m.group(0)):04x}", raw)


def _darken_hex(hex_color: str, factor: float = 0.7) -> str:
    """Return a darker shade of a #RRGGBB color (used for node borders)."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _truncate(s: str, n: int = 28) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ── Injected right-panel UI (DESIGN_REFERENCE §7). Kept as module constants so the
#    braces don't fight with Python f-strings. Data is passed in via a GR_DATA JSON blob. ──
_PANEL_CSS = """
* { box-sizing: border-box; }
/* Pengganti bagian Bootstrap yang dulu memberi tampilan ini (CDN dilepas: DESIGN_REFERENCE §1.2
   "offline-ready", canvas mengisi seluruh viewport). Judul pyvis selalu kosong -> disembunyikan. */
html, body { margin: 0; }
body { line-height: 1.5; }
body > center { display: none; }
.card { width: 100%; }
:root {
  --accent: #818cf8;
  --accent-dim: rgba(129,140,248,0.18);
  --accent-glow: rgba(129,140,248,0.35);
  --bg-panel: rgba(15,17,32,0.88);
  --bg-hover: rgba(255,255,255,0.05);
  --border: rgba(255,255,255,0.08);
  --text-primary: #e2e8f0;
  --text-muted: #64748b;
  --text-dim: #94a3b8;
}
#gr-panel {
  position: fixed; top: 16px; right: 16px; width: 230px;
  max-height: calc(100vh - 32px); overflow-y: auto;
  background: var(--bg-panel); backdrop-filter: blur(20px) saturate(180%);
  border: 1px solid var(--border); border-radius: 14px;
  color: var(--text-primary); font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  font-size: 13px; z-index: 9999;
  box-shadow: 0 24px 60px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.04) inset;
  scrollbar-width: thin; scrollbar-color: rgba(129,140,248,0.3) transparent;
}
#gr-panel::-webkit-scrollbar { width: 4px; }
#gr-panel::-webkit-scrollbar-thumb { background: rgba(129,140,248,0.3); border-radius: 4px; }
#gr-panel details { border-bottom: 1px solid var(--border); }
#gr-panel details:last-child { border-bottom: none; }
#gr-panel summary {
  cursor: pointer; padding: 12px 14px; font-weight: 600; color: var(--accent);
  list-style: none; display: flex; align-items: center; gap: 7px; user-select: none;
  letter-spacing: 0.01em;
}
#gr-panel summary::-webkit-details-marker { display: none; }
#gr-panel summary::after {
  content: "\\25be"; margin-left: auto; color: var(--accent); transition: transform .25s ease;
}
#gr-panel details:not([open]) summary::after { transform: rotate(-90deg); }
#gr-panel summary:hover { background: var(--bg-hover); }
#gr-panel summary { border-left: 3px solid transparent; padding-left: 11px; }
#gr-panel details[open] > summary {
  border-left-color: var(--accent);
  background: linear-gradient(90deg, var(--accent-dim), transparent);
}
.gr-help {
  display: inline-flex; align-items: center; justify-content: center;
  width: 16px; height: 16px; border-radius: 50%; font-size: 10px; font-weight: 700;
  border: 1px solid rgba(129,140,248,0.45); color: var(--accent); cursor: help;
  transition: background .15s;
}
.gr-help:hover { background: var(--accent-dim); }
#gr-legend-list { padding: 2px 0 8px; }
.gr-legend-item {
  display: flex; align-items: center; gap: 9px; padding: 7px 14px;
  cursor: pointer; border-left: 3px solid transparent; transition: all .15s;
}
.gr-legend-item:hover { background: var(--bg-hover); }
.gr-legend-item.active {
  background: var(--accent-dim); border-left: 3px solid var(--accent);
}
.gr-dot { width: 10px; height: 10px; border-radius: 50%; flex: 0 0 auto; box-shadow: 0 0 6px currentColor; }
.gr-lname { flex: 1; color: var(--text-primary); }
.gr-count { color: var(--text-muted); font-size: 11px; }
.gr-statsbar {
  padding: 8px 14px; color: var(--text-muted); font-size: 11px;
  border-top: 1px solid var(--border); letter-spacing: 0.02em;
}
.gr-hint { padding: 3px 14px 12px; color: var(--text-muted); font-size: 11px; }
#gr-tooltip {
  display: none; position: fixed; top: 16px; right: 256px; width: 300px;
  background: rgba(10,12,28,0.95); backdrop-filter: blur(20px);
  border: 1px solid var(--border); border-radius: 12px;
  padding: 14px 16px; color: var(--text-primary); font-size: 12px; line-height: 1.6;
  z-index: 10000; box-shadow: 0 24px 60px rgba(0,0,0,0.7);
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
.gr-tip-row { display: flex; align-items: baseline; gap: 9px; margin-bottom: 7px; }
.gr-tip-row:last-child { margin-bottom: 0; }
.gr-tip-row .gr-dot { margin-top: 4px; flex-shrink: 0; }
.gr-insight-group { padding: 0 14px 8px; }
.gr-insight-head {
  font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em;
  margin: 10px 0 5px; display: flex; align-items: center; gap: 7px; color: var(--text-muted);
}
.gr-insight-item {
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
  padding: 5px 0; cursor: pointer; transition: opacity .15s;
}
.gr-insight-item:hover { opacity: 0.8; }
.gr-insight-item:hover .gr-insight-name { color: var(--accent); }
.gr-insight-name { max-width: 145px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-primary); }
.gr-badge {
  color: #fbbf24; background: rgba(251,191,36,0.12); border-radius: 6px;
  padding: 2px 8px; font-size: 10px; font-weight: 700; flex: 0 0 auto; letter-spacing: 0.02em;
}
.gr-insight-empty { color: var(--text-muted); font-size: 11px; padding: 2px 0 4px; }
.gr-edge-wrap { padding: 8px 14px 14px; }
#gr-edge-select {
  width: 100%; background: rgba(255,255,255,0.04); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 8px;
  padding: 7px 10px; font-size: 12px; cursor: pointer; transition: border-color .15s;
  font-family: inherit;
}
#gr-edge-select:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-glow); }
#gr-edge-select option { background: #0f1120; color: var(--text-primary); }
#gr-edge-desc {
  display: none; margin-top: 9px; padding: 9px 11px; font-size: 11px; line-height: 1.6;
  background: rgba(129,140,248,0.07); border-left: 2px solid var(--accent); border-radius: 6px;
  color: var(--text-dim);
}
#gr-edge-desc b { color: var(--accent); }
.gr-path-wrap { padding: 8px 14px 14px; display: flex; flex-direction: column; gap: 9px; }
.gr-path-input {
  width: 100%; background: rgba(255,255,255,0.04); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 8px; padding: 7px 10px; font-size: 12px;
  font-family: inherit; transition: border-color .15s;
}
.gr-path-input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-glow); }
.gr-path-btn {
  width: 100%; background: linear-gradient(135deg, rgba(129,140,248,0.25), rgba(167,139,250,0.2));
  color: var(--accent); cursor: pointer;
  border: 1px solid rgba(129,140,248,0.4); border-radius: 8px;
  padding: 8px; font-size: 12px; font-weight: 600; font-family: inherit;
  transition: all .2s; letter-spacing: 0.02em;
}
.gr-path-btn:hover {
  background: linear-gradient(135deg, rgba(129,140,248,0.4), rgba(167,139,250,0.3));
  box-shadow: 0 0 16px rgba(129,140,248,0.3);
}
.gr-path-msg { font-size: 11px; line-height: 1.5; padding: 8px 10px; border-radius: 6px; }
.gr-path-msg.err { color: #f87171; background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.2); }
.gr-path-msg.warn { color: #fbbf24; background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.2); }
.gr-path-msg.ok { color: #34d399; background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.2); }
#gr-search {
  position: fixed; top: 20px; left: 20px; width: 330px; z-index: 9999;
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
#gr-search-box {
  display: flex; align-items: center;
  background: rgba(15,17,32,0.88); backdrop-filter: blur(20px) saturate(180%);
  border: 1px solid var(--border); border-radius: 12px;
  box-shadow: 0 16px 48px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.03) inset;
  transition: all .2s;
}
#gr-search-box:focus-within {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-glow), 0 16px 48px rgba(0,0,0,0.5);
}
#gr-search-input {
  flex: 1; min-width: 0; background: transparent; border: none; outline: none;
  color: var(--text-primary); font-size: 14px; padding: 12px 14px;
  font-family: inherit;
}
#gr-search-input::placeholder { color: var(--text-muted); }
#gr-search-clear {
  display: none; cursor: pointer; color: var(--text-muted); font-size: 18px; line-height: 1;
  padding: 0 14px 0 4px; user-select: none; transition: color .15s;
}
#gr-search-clear:hover { color: var(--text-primary); }
#gr-search-dropdown {
  display: none; margin-top: 8px; max-height: 360px; overflow-y: auto;
  background: rgba(12,14,28,0.96); backdrop-filter: blur(20px);
  border: 1px solid var(--border); border-radius: 12px;
  box-shadow: 0 24px 60px rgba(0,0,0,0.7);
  scrollbar-width: thin; scrollbar-color: rgba(129,140,248,0.3) transparent;
}
#gr-search-dropdown::-webkit-scrollbar { width: 4px; }
#gr-search-dropdown::-webkit-scrollbar-thumb { background: rgba(129,140,248,0.3); border-radius: 4px; }
.gr-search-item {
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
  padding: 10px 14px; cursor: pointer; font-size: 13px; color: var(--text-primary);
  transition: background .12s;
}
.gr-search-item:hover, .gr-search-item.active { background: var(--accent-dim); }
.gr-si-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.gr-si-type { color: var(--accent); font-size: 10px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; flex: 0 0 auto; }
.gr-search-empty { padding: 12px 14px; color: var(--text-muted); font-size: 12px; }
#gr-info {
  position: fixed; top: 78px; left: 20px; width: 330px;
  max-height: calc(100vh - 100px); overflow-y: auto;
  background: rgba(15,17,32,0.92); backdrop-filter: blur(20px) saturate(180%);
  border: 1px solid var(--border); border-radius: 14px;
  color: var(--text-primary); font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  font-size: 13px; z-index: 9998;
  box-shadow: 0 24px 60px rgba(0,0,0,0.65), 0 0 0 1px rgba(255,255,255,0.04) inset;
  opacity: 0; transform: translateX(-24px); pointer-events: none;
  transition: opacity .28s cubic-bezier(0.25,0.8,0.25,1), transform .28s cubic-bezier(0.25,0.8,0.25,1);
  scrollbar-width: thin; scrollbar-color: rgba(129,140,248,0.3) transparent;
}
#gr-info::-webkit-scrollbar { width: 4px; }
#gr-info::-webkit-scrollbar-thumb { background: rgba(129,140,248,0.3); border-radius: 4px; }
#gr-info.open { opacity: 1; transform: translateX(0); pointer-events: auto; }
.gr-info-head { display: flex; align-items: center; justify-content: space-between; padding: 14px 16px 0; }
.gr-info-badge {
  color: #fff; font-size: 10px; font-weight: 700; padding: 3px 10px; border-radius: 20px;
  text-transform: uppercase; letter-spacing: .06em;
}
.gr-info-close {
  cursor: pointer; color: var(--text-muted); font-size: 18px; line-height: 1;
  user-select: none; width: 26px; height: 26px; display: flex; align-items: center;
  justify-content: center; border-radius: 50%; transition: all .15s;
}
.gr-info-close:hover { color: var(--text-primary); background: rgba(255,255,255,0.08); }
.gr-info-name { padding: 8px 16px 2px; font-size: 15px; font-weight: 700; color: #f1f5f9; word-break: break-word; line-height: 1.4; }
.gr-info-desc { padding: 4px 16px 10px; font-size: 12px; line-height: 1.6; color: var(--text-dim); }
.gr-info-sub { padding: 0 16px 10px; font-size: 11px; color: var(--text-muted); border-bottom: 1px solid var(--border); }
.gr-info-group { padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.04); }
.gr-info-group:last-child { border-bottom: none; }
.gr-info-glabel {
  display: flex; align-items: center; gap: 7px; font-size: 10px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .07em; color: var(--text-muted); margin-bottom: 9px;
}
.gr-info-tags { display: flex; flex-wrap: wrap; gap: 6px; }
.gr-info-tag {
  max-width: 100%; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 6px; padding: 4px 9px; font-size: 12px; cursor: pointer; color: var(--text-dim);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; transition: all .15s;
}
.gr-info-tag:hover { background: var(--accent-dim); border-color: var(--accent); color: #e0e7ff; }
#gr-chat {
  position: fixed; bottom: 20px; left: 20px; width: 340px; z-index: 9999;
  background: var(--bg-panel); backdrop-filter: blur(20px) saturate(180%);
  border: 1px solid var(--border); border-radius: 14px; overflow: hidden;
  box-shadow: 0 24px 60px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.04) inset;
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; color: var(--text-primary);
  display: flex; flex-direction: column;
}
#gr-chat-header {
  display: flex; align-items: center; gap: 8px; padding: 12px 14px; cursor: pointer;
  background: rgba(255,255,255,0.04); user-select: none;
}
.gr-chat-dot { width: 8px; height: 8px; border-radius: 50%; background: #34d399; box-shadow: 0 0 8px #34d399; flex: 0 0 auto; }
.gr-chat-title { font-weight: 600; font-size: 13px; flex: 1; color: var(--accent); letter-spacing: 0.01em; }
#gr-chat-toggle { color: var(--text-muted); font-size: 11px; transition: transform .25s; }
#gr-chat.collapsed #gr-chat-toggle { transform: rotate(180deg); }
#gr-chat-body { display: flex; flex-direction: column; height: 400px; }
#gr-chat.collapsed #gr-chat-body { display: none; }
#gr-chat-messages {
  flex: 1; overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 10px;
  scrollbar-width: thin; scrollbar-color: rgba(129,140,248,0.3) transparent;
}
#gr-chat-messages::-webkit-scrollbar { width: 4px; }
#gr-chat-messages::-webkit-scrollbar-thumb { background: rgba(129,140,248,0.3); border-radius: 4px; }
.gr-msg { max-width: 85%; padding: 9px 12px; border-radius: 12px; font-size: 12.5px; line-height: 1.55; word-wrap: break-word; }
.gr-msg.user { align-self: flex-end; background: var(--accent-dim); border: 1px solid rgba(129,140,248,0.3); color: var(--text-primary); border-bottom-right-radius: 4px; }
.gr-msg.bot { align-self: flex-start; background: rgba(255,255,255,0.05); color: var(--text-dim); border-bottom-left-radius: 4px; }
.gr-msg.bot.loading { color: var(--text-muted); font-style: italic; }
.gr-msg.welcome { align-self: stretch; max-width: 100%; background: transparent; color: var(--text-muted); font-size: 11.5px; text-align: center; line-height: 1.5; }
#gr-chat-input-row { display: flex; gap: 8px; padding: 12px 14px; border-top: 1px solid var(--border); align-items: flex-end; }
#gr-chat-input {
  flex: 1; min-width: 0; background: rgba(255,255,255,0.04); border: 1px solid var(--border); border-radius: 10px;
  padding: 9px 12px; color: var(--text-primary); font-size: 12.5px; font-family: inherit; outline: none;
  resize: none; overflow-y: hidden; min-height: 38px; max-height: 160px; line-height: 1.5;
  display: block;
}
#gr-chat-input:focus { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-glow); }
#gr-chat-input::placeholder { color: var(--text-muted); }
/* Bot message markdown styles */
.gr-msg.bot p { margin: 0 0 6px; }
.gr-msg.bot p:last-child { margin-bottom: 0; }
.gr-msg.bot ul, .gr-msg.bot ol { margin: 4px 0 6px; padding-left: 18px; }
.gr-msg.bot li { margin-bottom: 3px; }
.gr-msg.bot strong { color: #e2e8f0; font-weight: 600; }
.gr-msg.bot code { background: rgba(255,255,255,0.1); padding: 1px 5px; border-radius: 4px; font-size: 11.5px; font-family: 'Cascadia Code', 'Fira Code', monospace; }
.gr-msg.bot h1, .gr-msg.bot h2, .gr-msg.bot h3 { margin: 6px 0 4px; font-size: 13px; color: var(--accent); font-weight: 700; }
#gr-chat-send {
  flex: 0 0 auto; width: 40px; height: 38px; background: linear-gradient(135deg, rgba(129,140,248,0.4), rgba(167,139,250,0.3));
  border: 1px solid rgba(129,140,248,0.4); border-radius: 10px; color: var(--accent); cursor: pointer;
  font-size: 14px; transition: all .2s;
}
#gr-chat-send:hover { box-shadow: 0 0 16px rgba(129,140,248,0.3); }
#gr-chat-send:disabled { opacity: 0.5; cursor: not-allowed; }
/* Hide all vis.js nav buttons except the fit/center one */
.vis-button.vis-zoomIn,
.vis-button.vis-zoomOut,
.vis-button.vis-up,
.vis-button.vis-down,
.vis-button.vis-left,
.vis-button.vis-right { display: none !important; }
.vis-button.vis-zoomExtends {
  background: rgba(15,17,32,0.88) !important;
  border: 1px solid rgba(129,140,248,0.3) !important;
  border-radius: 8px !important;
  backdrop-filter: blur(10px) !important;
  box-shadow: 0 4px 16px rgba(0,0,0,0.4) !important;
}
.vis-button.vis-zoomExtends:hover { background: rgba(129,140,248,0.2) !important; border-color: var(--accent) !important; }
.vis-button.vis-zoomExtends .vis-label { color: var(--accent) !important; font-weight: 600 !important; }
"""

_PANEL_HTML = """
<div id="gr-search">
  <div id="gr-search-box">
    <input id="gr-search-input" type="text" placeholder="\U0001F50D Search Node" autocomplete="off">
    <span id="gr-search-clear">&times;</span>
  </div>
  <div id="gr-search-dropdown"></div>
</div>
<div id="gr-panel">
  <details open>
    <summary><span class="gr-sum-title">Entity Types</span><span class="gr-help" id="gr-help">?</span></summary>
    <div id="gr-legend-list"></div>
    <div class="gr-statsbar" id="gr-statsbar"></div>
    <div class="gr-hint">Hover untuk detail &middot; Drag &amp; scroll to navigate</div>
  </details>
  <details id="gr-insights-details">
    <summary><span class="gr-sum-title">Network Insights</span></summary>
    <div id="gr-insights-list"></div>
  </details>
  <details id="gr-edges-details">
    <summary><span class="gr-sum-title">Edge Filters</span></summary>
    <div class="gr-edge-wrap">
      <select id="gr-edge-select"></select>
      <div id="gr-edge-desc"></div>
    </div>
  </details>
  <details id="gr-path-details">
    <summary><span class="gr-sum-title">Path Discovery</span></summary>
    <div class="gr-path-wrap">
      <input id="gr-path-a" class="gr-path-input" list="gr-path-nodes" placeholder="Node A" autocomplete="off">
      <input id="gr-path-b" class="gr-path-input" list="gr-path-nodes" placeholder="Node B" autocomplete="off">
      <datalist id="gr-path-nodes"></datalist>
      <button id="gr-path-run" class="gr-path-btn">Temukan Jalur</button>
      <div id="gr-path-result" class="gr-path-msg" style="display:none"></div>
    </div>
  </details>
</div>
<div id="gr-info"></div>
<div id="gr-tooltip"></div>
<div id="gr-chat" class="collapsed">
  <div id="gr-chat-header">
    <span class="gr-chat-dot"></span>
    <span class="gr-chat-title">GraphRAG Chat</span>
    <span id="gr-chat-toggle">&#9650;</span>
  </div>
  <div id="gr-chat-body">
    <div id="gr-chat-messages"></div>
    <div id="gr-chat-input-row">
      <textarea id="gr-chat-input" placeholder="Ketik pertanyaan..." rows="1" autocomplete="off"></textarea>
      <button id="gr-chat-send">&#10148;</button>
    </div>
  </div>
</div>
"""

_PANEL_JS = """
(function initLegend() {
  if (typeof network === "undefined" || typeof nodes === "undefined" || typeof edges === "undefined") {
    return setTimeout(initLegend, 100);
  }
  var TYPE_COLORS = GR_DATA.TYPE_COLORS,
      TYPE_COUNTS = GR_DATA.TYPE_COUNTS,
      NODE_META = GR_DATA.NODE_META,
      TYPE_DESCRIPTIONS = GR_DATA.TYPE_DESCRIPTIONS;

  // Capture each edge's base color and each node's base size once, so we can
  // dim/restore and enlarge/restore without losing the originals.
  var _edgeColor = {};
  edges.get().forEach(function (e) {
    _edgeColor[e.id] = (e.color && e.color.color) ? e.color.color : (e.color || "#555577");
  });
  var _nodeSize = {};
  nodes.get().forEach(function (n) { _nodeSize[n.id] = n.size || 20; });

  // Adjacency list + edge lookup by node-pair, for Path Discovery BFS (undirected).
  var _adj = {};
  var _edgeBetween = {};
  edges.get().forEach(function (e) {
    (_adj[e.from] = _adj[e.from] || []).push(e.to);
    (_adj[e.to] = _adj[e.to] || []).push(e.from);
    var key = (e.from < e.to) ? (e.from + "\\u0000" + e.to) : (e.to + "\\u0000" + e.from);
    _edgeBetween[key] = e.id;
  });

  var currentHighlightedType = null;
  var currentSearchNodeId = null;
  var currentPathActive = false;

  function metaOf(id) { return NODE_META[id] || { t: "", bg: "#9E9E9E", br: "#a99e8f666" }; }

  // Lighten a #RRGGBB toward white by fraction f (for a bright glow halo).
  function lighten(hex, f) {
    hex = String(hex).replace("#", "");
    if (hex.length !== 6) return "#e0e0e0";
    var r = parseInt(hex.substr(0, 2), 16), g = parseInt(hex.substr(2, 2), 16), b = parseInt(hex.substr(4, 2), 16);
    r = Math.round(r + (255 - r) * f); g = Math.round(g + (255 - g) * f); b = Math.round(b + (255 - b) * f);
    return "rgb(" + r + "," + g + "," + b + ")";
  }

  function normalNode(id) {
    var m = metaOf(id);
    return { id: id, opacity: 1, borderWidth: 1.5, size: _nodeSize[id],
             color: { background: m.bg, border: m.br },
             font: { color: "#e0e0e0" },
             shadow: { enabled: true, color: "rgba(0,0,0,0.18)", size: 8, x: 2, y: 2 } };
  }

  // Glowing node: enlarged, brighter fill, a tight & bright halo, and NO border ring —
  // brightness alone marks it (per feedback). A tight blur reads as a real glow; a huge
  // blur just spreads the light too thin to see.
  function glowNode(id) {
    var m = metaOf(id);
    return { id: id, opacity: 1, borderWidth: 0, size: (_nodeSize[id] || 20) * 1.4,
             color: { background: m.bg, border: m.bg },
             font: { color: "#ffffff" },
             shadow: { enabled: true, color: m.bg, size: 26, x: 0, y: 0 } };
  }

  function dimNode(id) {
    var m = metaOf(id);
    return { id: id, opacity: 0.22, borderWidth: 1.5, size: _nodeSize[id],
             color: { background: m.bg, border: m.br },
             font: { color: "rgba(224,224,224,0.18)" },
             shadow: { enabled: false } };
  }

  // Apply glow-to-targets / normal-to-neighbors / dim-to-rest given a set of target ids.
  function applyHighlight(targets) {
    var all = nodes.get();
    var neighbors = {};
    Object.keys(targets).forEach(function (id) {
      network.getConnectedNodes(id).forEach(function (nb) { neighbors[nb] = true; });
    });
    nodes.update(all.map(function (n) {
      if (targets[n.id]) return glowNode(n.id);
      if (neighbors[n.id]) return normalNode(n.id);
      return dimNode(n.id);
    }));
    edges.update(edges.get().map(function (e) {
      var on = targets[e.from] || targets[e.to];
      return { id: e.id, width: 3.5, color: { color: _edgeColor[e.id], opacity: on ? 1.0 : 0.02,
                                  highlight: "#e0e0e0", hover: "#e0e0e0" } };
    }));
  }

  function resetVars() {
    currentHighlightedType = null; currentSearchNodeId = null; currentPathActive = false; setActive(null);
    closeInfoPanel();
    try { network.unselectAll(); } catch (e) {}  // clear vis.js default selection ring
  }

  // --- Legend list ---
  var list = document.getElementById("gr-legend-list");
  Object.keys(TYPE_COLORS).forEach(function (t) {
    var cnt = TYPE_COUNTS[t] || 0;
    if (cnt === 0) return;
    var item = document.createElement("div");
    item.className = "gr-legend-item";
    item.setAttribute("data-type", t);
    item.innerHTML =
      '<span class="gr-dot" style="background:' + TYPE_COLORS[t] + '"></span>' +
      '<span class="gr-lname">' + esc(t) + '</span>' +
      '<span class="gr-count">' + cnt + '</span>';
    item.addEventListener("click", function () { highlightType(t); });
    list.appendChild(item);
  });

  document.getElementById("gr-statsbar").textContent =
    GR_DATA.TOTAL_NODES + " nodes \\u2022 " + GR_DATA.TOTAL_EDGES + " edges";

  function setActive(t) {
    document.querySelectorAll(".gr-legend-item").forEach(function (el) {
      el.classList.toggle("active", el.getAttribute("data-type") === t);
    });
  }

  window.highlightType = function (t) {
    if (currentHighlightedType === t) { resetHighlight(); return; }
    resetVars();
    resetEdgeFilterUI();          // legend highlight and edge filter are mutually exclusive
    currentHighlightedType = t;
    setActive(t);
    var targets = {};
    nodes.get().forEach(function (n) { if (metaOf(n.id).t === t) targets[n.id] = true; });
    applyHighlight(targets);
  };

  // Highlight a single node (glow it, dim the rest) and open its info panel.
  window.highlightNode = function (id) {
    resetVars();
    resetEdgeFilterUI();
    currentSearchNodeId = id;
    var targets = {}; targets[id] = true;
    applyHighlight(targets);
    openInfoPanel(id);
  };

  window.resetHighlight = function () {
    resetVars();
    resetEdgeFilterUI();
    try { network.unselectAll(); } catch (e) {}
    nodes.update(nodes.get().map(function (n) { return normalNode(n.id); }));
    edges.update(edges.get().map(function (e) {
      return { id: e.id, width: 3.5, color: { color: _edgeColor[e.id], opacity: 1.0,
                                  highlight: "#e0e0e0", hover: "#e0e0e0" } };
    }));
  };

  // Click empty canvas -> reset any active highlight or filter.
  network.on("click", function (params) {
    if (params.nodes.length === 0 && params.edges.length === 0 &&
        (currentHighlightedType || currentSearchNodeId || currentPathActive || currentEdgeFilter !== "all")) {
      resetAll();
    }
  });

  // Double-click a node -> highlight it (note #1: consistent highlight trigger).
  network.on("doubleClick", function (params) {
    if (params.nodes.length) { highlightNode(params.nodes[0]); }
  });

  // --- Edge Filters (§7.3): dropdown dims all edges except the chosen relation type ---
  var currentEdgeFilter = "all";

  function resetEdgeFilterUI() {
    currentEdgeFilter = "all";
    var sel = document.getElementById("gr-edge-select");
    if (sel) sel.value = "all";
    var box = document.getElementById("gr-edge-desc");
    if (box) { box.style.display = "none"; box.innerHTML = ""; }
  }

  function updateEdgeDesc(rel) {
    var box = document.getElementById("gr-edge-desc");
    if (!box) return;
    if (rel === "all") { box.style.display = "none"; box.innerHTML = ""; return; }
    // EDGE_DESCRIPTIONS = HTML tepercaya dari konstanta Python; fallback `rel` berasal dari file
    // graph (data), jadi WAJIB di-escape.
    box.innerHTML = (GR_DATA.EDGE_DESCRIPTIONS && GR_DATA.EDGE_DESCRIPTIONS[rel]) || esc(rel);
    box.style.display = "block";
  }

  window.filterEdges = function (rel) {
    resetVars();                 // edge filter and legend/node highlight are mutually exclusive
    currentEdgeFilter = rel;
    updateEdgeDesc(rel);
    if (rel === "all") {
      nodes.update(nodes.get().map(function (n) { return normalNode(n.id); }));
      edges.update(edges.get().map(function (e) {
        return { id: e.id, width: 3.5, color: { color: _edgeColor[e.id], opacity: 1.0,
                                    highlight: "#e0e0e0", hover: "#e0e0e0" } };
      }));
      return;
    }
    var activeNodes = {};
    edges.update(edges.get().map(function (e) {
      var on = (e.rel === rel);
      if (on) { activeNodes[e.from] = true; activeNodes[e.to] = true; }
      return { id: e.id, width: 3.5, color: { color: _edgeColor[e.id], opacity: on ? 1.0 : 0.03,
                                  highlight: "#e0e0e0", hover: "#e0e0e0" } };
    }));
    nodes.update(nodes.get().map(function (n) {
      return activeNodes[n.id] ? normalNode(n.id) : dimNode(n.id);
    }));
  };

  (function initEdgeFilter() {
    var sel = document.getElementById("gr-edge-select");
    if (!sel) return;
    var counts = GR_DATA.EDGE_COUNTS || {};
    var rels = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; });
    var html = '<option value="all">-- Tampilkan Semua --</option>';
    rels.forEach(function (rel) {
      var label = rel.replace(/_/g, " ").replace(/\\b\\w/g, function (c) { return c.toUpperCase(); });
      html += '<option value="' + esc(rel) + '">' + esc(label) + " (" + Number(counts[rel]) + ")</option>";
    });
    sel.innerHTML = html;
    sel.addEventListener("change", function () { filterEdges(sel.value); });
  })();

  // --- "?" legend tooltip ---
  var help = document.getElementById("gr-help");
  var tip = document.getElementById("gr-tooltip");
  if (help && tip) {
    tip.innerHTML = Object.keys(TYPE_COLORS).filter(function (t) {
      return (TYPE_COUNTS[t] || 0) > 0;
    }).map(function (t) {
      return '<div class="gr-tip-row"><span class="gr-dot" style="background:' + TYPE_COLORS[t] +
             '"></span><span><b>' + esc(t) + '</b> &mdash; ' + (TYPE_DESCRIPTIONS[t] || "") + '</span></div>';
    }).join("");
    help.addEventListener("mouseenter", function () { tip.style.display = "block"; });
    help.addEventListener("mouseleave", function () { tip.style.display = "none"; });
    help.addEventListener("click", function (e) { e.preventDefault(); e.stopPropagation(); });
  }

  // --- Network Insights (§7.2): top-5 by degree per category, DOM built lazily on open ---
  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // Canonical "select a node" action, shared by Search (§8.3) and Network Insights (§7.2):
  // reflect it in the search bar, glow it, zoom to it. (§9 Info Panel will hook in here.)
  window.executeSearchSelection = function (id) {
    var det = document.getElementById("gr-insights-details");
    if (det) det.removeAttribute("open");
    if (searchInput) searchInput.value = id;
    if (searchClear) searchClear.style.display = "flex";
    hideSearchDropdown();
    highlightNode(id);
    network.focus(id, { scale: 0.25, animation: { duration: 1000, easingFunction: "easeInOutQuad" } });
  };

  function buildInsights() {
    var container = document.getElementById("gr-insights-list");
    var INS = GR_DATA.INSIGHTS || {};
    ["Method", "Dataset", "Metric", "Author"].forEach(function (t) {
      var items = INS[t] || [];
      var grp = document.createElement("div");
      grp.className = "gr-insight-group";
      var head = '<div class="gr-insight-head"><span class="gr-dot" style="background:' +
                 (TYPE_COLORS[t] || "#999999") + '"></span>Top ' + t + "</div>";
      var body = "";
      if (!items.length) {
        body = '<div class="gr-insight-empty">Tidak ada data.</div>';
      } else {
        items.forEach(function (it) {
          body += '<div class="gr-insight-item" data-id="' + encodeURIComponent(it.name) + '">' +
                  '<span class="gr-insight-name" title="' + esc(it.name) + '">' + esc(it.name) + "</span>" +
                  '<span class="gr-badge">\\u2b50 ' + it.score + "</span></div>";
        });
      }
      grp.innerHTML = head + body;
      container.appendChild(grp);
    });

    // Top Entity Families — varian ber-versi yang ditautkan lewat `variant_of`.
    // Bisa Method / Dataset / Metric / Task tergantung korpus; section ini MENYEMBUNYIKAN
    // DIRI kalau korpus tidak punya entitas berversi sama sekali.
    var FAM = GR_DATA.FAMILIES || [];
    if (FAM.length) {
      var fg = document.createElement("div");
      fg.className = "gr-insight-group";
      var fh = '<div class="gr-insight-head"><span class="gr-dot" style="background:' +
               (TYPE_COLORS[FAM[0].type] || "#999999") + '"></span>Top Entity Families</div>';
      var fb = "";
      FAM.forEach(function (f) {
        fb += '<div class="gr-insight-item" data-id="' + encodeURIComponent(f.name) + '">' +
              '<span class="gr-insight-name" title="' + esc(f.name) + ' (' + esc(f.type) +
              ') \\u2014 ' + f.variants + ' varian, ' + f.papers + ' paper">' +
              esc(f.name) + '</span>' +
              '<span class="gr-badge">' + f.variants + 'v \\u00b7 ' + f.papers + 'p</span></div>';
      });
      fg.innerHTML = fh + fb;
      container.appendChild(fg);
    }

    container.querySelectorAll(".gr-insight-item").forEach(function (el) {
      el.addEventListener("click", function () {
        executeSearchSelection(decodeURIComponent(el.getAttribute("data-id")));
      });
    });
  }

  var insDetails = document.getElementById("gr-insights-details");
  var insightsBuilt = false;
  if (insDetails) {
    insDetails.addEventListener("toggle", function () {
      if (insDetails.open && !insightsBuilt) { buildInsights(); insightsBuilt = true; }
    });
  }

  // --- Path Discovery (§7.4): BFS shortest path on the undirected, unweighted graph ---
  function bfsPath(a, b) {
    if (a === b) return [a];
    var visited = {}; visited[a] = true;
    var prev = {};
    var queue = [a];
    while (queue.length) {
      var cur = queue.shift();
      var nbrs = _adj[cur] || [];
      for (var i = 0; i < nbrs.length; i++) {
        var nb = nbrs[i];
        if (!visited[nb]) {
          visited[nb] = true; prev[nb] = cur;
          if (nb === b) {
            var path = [b], p = b;
            while (p !== a) { p = prev[p]; path.unshift(p); }
            return path;
          }
          queue.push(nb);
        }
      }
    }
    return null;
  }

  window.highlightPath = function (path) {
    resetVars();
    resetEdgeFilterUI();
    currentPathActive = true;
    var inPath = {}; path.forEach(function (id) { inPath[id] = true; });
    var pathEdges = {};
    for (var i = 0; i < path.length - 1; i++) {
      var a = path[i], b = path[i + 1];
      var key = (a < b) ? (a + "\\u0000" + b) : (b + "\\u0000" + a);
      var eid = _edgeBetween[key];
      if (eid !== undefined) pathEdges[eid] = true;
    }
    nodes.update(nodes.get().map(function (n) {
      var m = metaOf(n.id);
      if (inPath[n.id]) {
        return { id: n.id, opacity: 1, borderWidth: 0, size: _nodeSize[n.id],
                 color: { background: m.bg, border: m.bg },
                 font: { color: "#ffffff" },
                 shadow: { enabled: true, color: "#e0e0e0", size: 20, x: 0, y: 0 } };
      }
      return { id: n.id, opacity: 0.15, borderWidth: 1.5, size: _nodeSize[n.id],
               color: { background: m.bg, border: m.br },
               font: { color: "rgba(224,224,224,0.1)" }, shadow: { enabled: false } };
    }));
    edges.update(edges.get().map(function (e) {
      if (pathEdges[e.id]) {
        return { id: e.id, width: 4, color: { color: "#e0e0e0", opacity: 1.0, highlight: "#e0e0e0", hover: "#e0e0e0" } };
      }
      return { id: e.id, width: 1, color: { color: _edgeColor[e.id], opacity: 0.05, highlight: "#e0e0e0", hover: "#e0e0e0" } };
    }));
    network.fit({ nodes: path, animation: { duration: 800, easingFunction: "easeInOutQuad" } });
  };

  function runPathDiscovery() {
    var a = (document.getElementById("gr-path-a").value || "").trim();
    var b = (document.getElementById("gr-path-b").value || "").trim();
    var res = document.getElementById("gr-path-result");
    function msg(text, cls) { res.className = "gr-path-msg " + cls; res.textContent = text; res.style.display = "block"; }
    if (!a || !b) { msg("Pilih kedua node terlebih dahulu.", "err"); return; }
    if (nodes.get(a) == null || nodes.get(b) == null) { msg("Node tidak valid. Pastikan dari dropdown.", "err"); return; }
    var path = bfsPath(a, b);
    if (!path) { msg("Tidak ada jalur yang menghubungkan kedua node ini.", "warn"); return; }
    msg("Jalur ditemukan! (" + (path.length - 1) + " lompatan)", "ok");
    highlightPath(path);
  }

  var pathDatalistPopulated = false;
  function populatePathDatalist() {
    var dl = document.getElementById("gr-path-nodes");
    if (!dl) return;
    var html = "";
    nodes.get().forEach(function (n) { html += '<option value="' + esc(n.id) + '"></option>'; });
    dl.innerHTML = html;
  }
  var pathDetails = document.getElementById("gr-path-details");
  if (pathDetails) {
    pathDetails.addEventListener("toggle", function () {
      if (pathDetails.open && !pathDatalistPopulated) { populatePathDatalist(); pathDatalistPopulated = true; }
    });
  }
  var pathRunBtn = document.getElementById("gr-path-run");
  if (pathRunBtn) pathRunBtn.addEventListener("click", runPathDiscovery);
  ["gr-path-a", "gr-path-b"].forEach(function (idn) {
    var el = document.getElementById(idn);
    if (el) el.addEventListener("keydown", function (e) { if (e.key === "Enter") runPathDiscovery(); });
  });

  // --- Search (§8): case-insensitive substring match on node ids, ranked, top 50 ---
  var searchInput = document.getElementById("gr-search-input");
  var searchDropdown = document.getElementById("gr-search-dropdown");
  var searchClear = document.getElementById("gr-search-clear");
  var _allNodeIds = nodes.getIds();
  var searchResults = [];
  var searchActiveIndex = -1;

  function hideSearchDropdown() {
    if (searchDropdown) { searchDropdown.style.display = "none"; searchDropdown.innerHTML = ""; }
    searchResults = []; searchActiveIndex = -1;
  }

  function renderSearch(raw) {
    var q = (raw || "").trim().toLowerCase();
    if (!q) { hideSearchDropdown(); return; }
    var matches = [];
    for (var i = 0; i < _allNodeIds.length; i++) {
      var id = String(_allNodeIds[i]);
      var lc = id.toLowerCase();
      var idx = lc.indexOf(q);
      if (idx === -1) continue;
      matches.push({ id: id, rank: (lc === q) ? 0 : (idx === 0 ? 1 : 2) });
    }
    matches.sort(function (a, b) {
      if (a.rank !== b.rank) return a.rank - b.rank;
      return a.id.toLowerCase() < b.id.toLowerCase() ? -1 : 1;
    });
    matches = matches.slice(0, 50);
    searchResults = matches; searchActiveIndex = -1;
    if (!matches.length) {
      searchDropdown.innerHTML = '<div class="gr-search-empty">Tidak ada node cocok.</div>';
      searchDropdown.style.display = "block";
      return;
    }
    searchDropdown.innerHTML = matches.map(function (m, i) {
      return '<div class="gr-search-item" data-idx="' + i + '">' +
             '<span class="gr-si-name">' + esc(m.id) + '</span>' +
             '<span class="gr-si-type">' + esc(metaOf(m.id).t || "") + '</span></div>';
    }).join("");
    searchDropdown.style.display = "block";
    searchDropdown.querySelectorAll(".gr-search-item").forEach(function (el) {
      el.addEventListener("click", function () {
        var r = searchResults[parseInt(el.getAttribute("data-idx"), 10)];
        if (r) executeSearchSelection(r.id);
      });
    });
  }

  function moveSearch(delta) {
    if (!searchResults.length) return;
    searchActiveIndex = (searchActiveIndex + delta + searchResults.length) % searchResults.length;
    var items = searchDropdown.querySelectorAll(".gr-search-item");
    items.forEach(function (el, i) { el.classList.toggle("active", i === searchActiveIndex); });
    if (items[searchActiveIndex]) items[searchActiveIndex].scrollIntoView({ block: "nearest" });
  }

  function clearSearch() {
    resetAll();
    network.fit({ animation: { duration: 800, easingFunction: "easeInOutQuad" } });
  }

  if (searchInput) {
    searchInput.addEventListener("input", function () {
      if (searchClear) searchClear.style.display = searchInput.value ? "flex" : "none";
      renderSearch(searchInput.value);
    });
    searchInput.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown") { e.preventDefault(); moveSearch(1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); moveSearch(-1); }
      else if (e.key === "Enter") {
        e.preventDefault();
        var pick = (searchActiveIndex >= 0) ? searchResults[searchActiveIndex] : searchResults[0];
        if (pick) executeSearchSelection(pick.id);
      } else if (e.key === "Escape") { hideSearchDropdown(); }
    });
  }
  if (searchClear) searchClear.addEventListener("click", clearSearch);
  document.addEventListener("click", function (e) {
    var s = document.getElementById("gr-search");
    if (s && !s.contains(e.target)) hideSearchDropdown();
  });

  // --- Info Panel (§9): node detail (right side), opens on any single-node highlight ---
  function infoTag(id) {
    return '<span class="gr-info-tag" data-id="' + encodeURIComponent(id) +
           '" title="' + esc(id) + '">' + esc(id) + '</span>';
  }

  function closeInfoPanel() {
    var panel = document.getElementById("gr-info");
    if (panel) panel.classList.remove("open");
  }

  function openInfoPanel(id) {
    var panel = document.getElementById("gr-info");
    if (!panel) return;
    var m = metaOf(id);
    var neighbors = network.getConnectedNodes(id);
    var isPaper = (m.t === "Paper");
    var papers = [];   // Paper-type neighbors -> "Ditemukan di Paper"
    var groups = {};   // other neighbors grouped by type
    neighbors.forEach(function (nb) {
      var t = metaOf(nb).t || "Unknown";
      if (t === "Paper" && !isPaper) papers.push(nb);
      else (groups[t] = groups[t] || []).push(nb);
    });

    var html =
      '<div class="gr-info-head">' +
        '<span class="gr-info-badge" style="background:' + (m.bg || "#999999") + '">' + esc(m.t || "Node") + '</span>' +
        '<span class="gr-info-close" id="gr-info-close">&times;</span>' +
      '</div>' +
      '<div class="gr-info-name">' + esc(id) + '</div>' +
      (m.d ? '<div class="gr-info-desc">' + esc(m.d) + '</div>' : '') +
      '<div class="gr-info-sub">Total Koneksi: ' + neighbors.length + ' relasi</div>';

    if (papers.length) {
      html += '<div class="gr-info-group"><div class="gr-info-glabel">' +
              '<span class="gr-dot" style="background:#3B52B5"></span>Ditemukan di Paper (' + papers.length + ')</div>' +
              '<div class="gr-info-tags">' + papers.map(infoTag).join("") + '</div></div>';
    }
    Object.keys(groups).sort().forEach(function (t) {
      var color = GR_DATA.TYPE_COLORS[t] || "#999999";
      html += '<div class="gr-info-group"><div class="gr-info-glabel">' +
              '<span class="gr-dot" style="background:' + color + '"></span>Terkoneksi dengan ' + esc(t) + ' (' + groups[t].length + ')</div>' +
              '<div class="gr-info-tags">' + groups[t].map(infoTag).join("") + '</div></div>';
    });

    panel.innerHTML = html;
    panel.classList.add("open");
    var closeBtn = document.getElementById("gr-info-close");
    if (closeBtn) closeBtn.addEventListener("click", resetAll);
    panel.querySelectorAll(".gr-info-tag").forEach(function (el) {
      el.addEventListener("click", function () {
        executeSearchSelection(decodeURIComponent(el.getAttribute("data-id")));
      });
    });
  }

  function resetAll() {
    resetHighlight();
    if (searchInput) searchInput.value = "";
    if (searchClear) searchClear.style.display = "none";
    hideSearchDropdown();
  }

  // --- Chatbot (§10): posts to /api/chat (needs server.py; degrades gracefully on file://) ---
  var chatEl = document.getElementById("gr-chat");
  var chatHeader = document.getElementById("gr-chat-header");
  var chatMsgs = document.getElementById("gr-chat-messages");
  var chatInput = document.getElementById("gr-chat-input");
  var chatSend = document.getElementById("gr-chat-send");
  var chatBusy = false;

  if (chatHeader && chatEl) {
    chatHeader.addEventListener("click", function () { chatEl.classList.toggle("collapsed"); });
  }

  // KEAMANAN: jawaban LLM berasal dari dokumen tak tepercaya. Urutan di bawah TIDAK BOLEH diubah:
  // escape SEMUA HTML dulu, baru tambahkan tag whitelist (tanpa link/gambar/atribut). Tag yang
  // dihasilkan hanya membungkus teks yang sudah ter-escape, jadi tidak ada jalan menuju XSS.
  // Jangan tambahkan fitur (link, gambar, HTML mentah) di sini tanpa sanitizer dan tes di tests/test_frontend.py.
  function renderMarkdown(raw) {
    var html = String(raw).replace(/[&<>"']/g, function(m) {
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m];
    });
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    html = html.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');
    html = html.replace(/\\*([^\\*\\n]+)\\*/g, '<em>$1</em>');
    html = html.replace(/_([^\\_\\n]+)_/g, '<em>$1</em>');
    html = html.replace(/`([^`\\n]+)`/g, '<code>$1</code>');
    html = html.replace(/^[\\*\\-] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/((<li>[^]*?<\\/li>\\n?)+)/g, '<ul>$1</ul>');
    html = html.replace(/^\\d+\\. (.+)$/gm, '<li>$1</li>');
    return html.split('\\n').map(function(line) {
      var t = line.trim();
      if (!t || /^<(h[1-6]|ul|ol|li|p)[> ]/.test(t)) return line;
      return '<p>' + line + '</p>';
    }).join('');
  }

  function chatAdd(text, cls) {
    var d = document.createElement("div");
    d.className = "gr-msg " + cls;
    if (cls.indexOf("user") >= 0) {
      d.textContent = text;
    } else {
      d.innerHTML = renderMarkdown(text);
    }
    chatMsgs.appendChild(d);
    chatMsgs.scrollTop = chatMsgs.scrollHeight;
    return d;
  }

  if (chatMsgs) {
    chatAdd("Tanya apa saja tentang paper di graph ini \u2014 metode, dataset, author, atau tren penelitian.", "welcome");
  }

  function chatSendMsg() {
    if (chatBusy || !chatInput) return;
    var q = chatInput.value.trim();
    if (!q) return;
    chatAdd(q, "user");
    chatInput.value = "";
    chatBusy = true;
    if (chatSend) chatSend.disabled = true;
    var loading = chatAdd("mengetik\\u2026", "bot loading");
    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q })
    }).then(function (r) { return r.json(); }).then(function (data) {
      loading.remove();
      chatAdd((data && data.answer) ? data.answer : "(tidak ada jawaban)", "bot");
    }).catch(function () {
      loading.remove();
      chatAdd("\\u26a0\\ufe0f Tidak bisa menghubungi server. Jalankan `python server.py`, lalu buka http://127.0.0.1:8000", "bot");
    }).finally(function () {
      chatBusy = false;
      if (chatSend) chatSend.disabled = false;
    });
  }

  if (chatSend) chatSend.addEventListener("click", chatSendMsg);
  if (chatInput) {
    // Auto-grow textarea
    chatInput.addEventListener("input", function () {
      chatInput.style.height = "auto";
      chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + "px";
    });
    chatInput.addEventListener("keydown", function (e) {
      // Prevent arrow keys from reaching vis.js while typing in chat
      if (e.key === "ArrowLeft" || e.key === "ArrowRight" ||
          e.key === "ArrowUp"   || e.key === "ArrowDown") {
        e.stopPropagation();
      }
      // Send on Enter (without Shift)
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); chatSendMsg(); }
    });
  }
})();
"""


class GraphVisualizer:
    # Node colors — vivid palette optimised for deep-navy dark background.
    NODE_COLORS = {
        "Paper": "#818cf8",       # Indigo-violet
        "Author": "#f87171",      # Soft coral-red
        "Institution": "#a89684", # Warm taupe — netral, tidak bentrok dgn Paper/Metric
                                  # (sky blue lama: ΔE 6.7 vs Metric cyan = nyaris tak terbedakan)
        "Method": "#4ade80",      # Bright green
        "Dataset": "#fbbf24",     # Amber gold
        "Metric": "#22d3ee",      # Cyan
        "Finding": "#c084fc",     # Lavender purple
        "Limitation": "#fb923c",  # Orange
        "Task": "#f472b6",        # Hot pink
    }

    # Edge colors — matched to node palette, slightly dimmed for legibility.
    EDGE_COLORS = {
        "written_by": "#f87171",
        "affiliated_with": "#f87171",
        "uses_method": "#4ade80",
        "proposes_method": "#22c55e",
        "uses_dataset": "#fbbf24",
        "uses_metric": "#f59e0b",
        "finds": "#c084fc",
        "has_limitation": "#fb923c",
        "criticizes": "#f87171",
        "builds_on": "#38bdf8",
        "compares": "#22d3ee",
        "addresses_task": "#f472b6",      # Task pink
        "introduces_dataset": "#eab308",  # dataset amber, a shade darker than uses_dataset
        "evaluated_on": "#fcd34d",
        "related_to": "#94a3b8",          # slate-400: generic link kept when salvaging LLM output
        "mentioned_in": "#475569",  # slate-600: recessive provenance edge on dark bg
        "variant_of": "#64748b",    # slate-500: tautan varian -> keluarga (build-time)
    }
    DEFAULT_EDGE_COLOR = "#475569"

    # Indonesian type descriptions for the legend "?" tooltip (DESIGN_REFERENCE §7.1).
    TYPE_DESCRIPTIONS = {
        "Paper": "Makalah penelitian akademik (sumber utama).",
        "Author": "Penulis dari sebuah paper.",
        "Institution": "Institusi, universitas, atau lab afiliasi penulis.",
        "Method": "Teknik, model, arsitektur, layer, atau algoritma.",
        "Dataset": "Kumpulan data untuk melatih atau menguji model.",
        "Metric": "Ukuran evaluasi performa (mis. akurasi, Top-1 error).",
        "Finding": "Temuan, hasil, atau klaim utama dari paper.",
        "Limitation": "Batasan atau kelemahan yang diakui penulis.",
        "Task": "Masalah yang ingin diselesaikan (mis. klasifikasi citra).",
    }

    # Indonesian edge descriptions for the Edge Filters box (DESIGN_REFERENCE §7.3),
    # adapted to our relation types (config.RELATION_TYPES + build-time variant_of).
    EDGE_DESCRIPTIONS = {
        "written_by": "<b>[Paper &rarr; Author]</b> Penulis dari sebuah paper.",
        "affiliated_with": "<b>[Author &rarr; Institution]</b> Institusi tempat penulis bernaung.",
        "uses_method": "<b>[Paper &rarr; Method]</b> Paper memakai metode/algoritma ini.",
        "proposes_method": "<b>[Paper &rarr; Method]</b> Paper mengusulkan/menciptakan metode ini.",
        "uses_dataset": "<b>[Paper &rarr; Dataset]</b> Paper mengevaluasi modelnya pada dataset ini.",
        "uses_metric": "<b>[Paper &rarr; Metric]</b> Paper memakai metrik ini sebagai tolak ukur.",
        "finds": "<b>[Paper &rarr; Finding]</b> Temuan, kesimpulan, atau klaim dari paper.",
        "has_limitation": "<b>[Method/Dataset &rarr; Limitation]</b> Batasan atau kelemahan yang diakui.",
        "criticizes": "<b>[Paper &rarr; Method/Metric]</b> Paper mengkritik atau menemukan kelemahan.",
        "builds_on": "<b>[Method &rarr; Method]</b> Metode ini pengembangan dari metode lain.",
        "compares": "<b>[Method/Paper &rarr; Method]</b> Perbandingan performa/efisiensi (A vs B).",
        "addresses_task": "<b>[Paper/Method &rarr; Task]</b> Masalah yang ingin diselesaikan oleh paper/metode ini.",
        "introduces_dataset": "<b>[Paper &rarr; Dataset]</b> Paper memperkenalkan/membuat dataset ini.",
        "evaluated_on": "<b>[Method &rarr; Dataset]</b> Metode/model ini diuji pada dataset tersebut.",
        "related_to": "<b>[Entity &rarr; Entity]</b> Hubungan lain yang tidak masuk daftar relasi (label asli di deskripsi).",
        "mentioned_in": "<b>[Entity &rarr; Paper]</b> Entity ini disebut/muncul di paper tersebut.",
        "variant_of": "<b>[Varian &rarr; Keluarga]</b> Versi/ukuran spesifik dari satu keluarga model atau dataset (mis. Llama-3-8B &rarr; Llama).",
    }

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def _node_size(self, node_type: str, type_counts: Counter, max_type_count: int, degree: int) -> float:
        """DESIGN_REFERENCE §4.1: rarer types get a bigger base, +1.5px/connection, capped 250."""
        tc = type_counts.get(node_type, 1)
        base = 60 - (tc / max_type_count) * 40
        return min(base + degree * 1.5, 250)

    def _insights(self, top_n: int = 5) -> dict:
        """Top-N nodes by real-relationship degree (mentioned_in excluded, consistent with
        get_counting_stats) per category (DESIGN_REFERENCE §7.2)."""
        result = {}
        for t in ("Method", "Dataset", "Metric", "Author"):
            scored = []
            for n, d in self.graph.nodes(data=True):
                if d.get("type") != t:
                    continue
                deg = sum(
                    1 for _, _, ed in self.graph.edges(n, data=True)
                    if ed.get("relation") not in ("mentioned_in", "variant_of")
                )
                scored.append((n, deg))
            scored.sort(key=lambda x: x[1], reverse=True)
            result[t] = [{"name": n, "score": s} for n, s in scored[:top_n]]
        return result

    def _entity_families(self, top_n: int = 6) -> list:
        """
        Keluarga entitas (varian ber-versi yang ditautkan lewat `variant_of`), diurutkan
        berdasarkan jangkauan lintas-paper. Domain-agnostik: bisa berupa Method, Dataset,
        Metric, atau Task — tergantung korpus. Kosong bila korpus tidak memakai penomoran versi,
        dan panel akan menyembunyikan diri.
        """
        papers = {n for n, d in self.graph.nodes(data=True) if d.get("type") == "Paper"}
        members = {}
        for u, v, d in self.graph.edges(data=True):
            if d.get("relation") != "variant_of":
                continue
            du = sum(1 for _, _, e in self.graph.edges(u, data=True) if e.get("relation") == "variant_of")
            dv = sum(1 for _, _, e in self.graph.edges(v, data=True) if e.get("relation") == "variant_of")
            parent, child = (u, v) if du >= dv else (v, u)
            members.setdefault(parent, set()).add(child)

        out = []
        for parent, variants in members.items():
            if len(variants) < 2:
                continue
            reach = {p for p in self.graph.neighbors(parent) if p in papers}
            for v in variants:
                reach |= {p for p in self.graph.neighbors(v) if p in papers}
            out.append({
                "name": parent,
                "type": self.graph.nodes[parent].get("type", ""),
                "variants": len(variants),
                "papers": len(reach),
            })
        out.sort(key=lambda x: (-x["papers"], -x["variants"]))
        return out[:top_n]

    def _ui_injection(self, type_counts: Counter, node_meta: dict) -> str:
        edge_counts = Counter(d.get("relation", "") for _, _, d in self.graph.edges(data=True))
        edge_counts.pop("", None)
        gr_data = {
            "TYPE_COLORS": self.NODE_COLORS,
            "TYPE_COUNTS": dict(type_counts),
            "NODE_META": node_meta,
            "TYPE_DESCRIPTIONS": self.TYPE_DESCRIPTIONS,
            "EDGE_COUNTS": dict(edge_counts),
            "EDGE_DESCRIPTIONS": self.EDGE_DESCRIPTIONS,
            "INSIGHTS": self._insights(),
            "FAMILIES": self._entity_families(),
            "TOTAL_NODES": self.graph.number_of_nodes(),
            "TOTAL_EDGES": self.graph.number_of_edges(),
        }
        # Node names/descriptions are LLM output derived from untrusted PDFs: escape <, >, & (also
        # covers "</script>" and "<!--<script>") so they can't break out of the <script> block.
        data_json = _json_for_script(gr_data)
        return (
            "<style>" + _PANEL_CSS + "</style>\n"
            + _PANEL_HTML + "\n"
            + '<script type="text/javascript">\n'
            + "const GR_DATA = " + data_json + ";\n"
            + _PANEL_JS
            + "</script>\n"
        )

    def build_html(self) -> str:
        """Build the full self-contained HTML string (DESIGN_REFERENCE Phase B + C)."""
        net = Network(
            height="100vh", width="100%",
            bgcolor="#080c1a", font_color="#e2e8f0",
            cdn_resources="in_line",
        )

        type_counts = Counter(d.get("type", "") for _, d in self.graph.nodes(data=True))
        max_type_count = max(type_counts.values()) if type_counts else 1
        node_meta = {}

        # --- Nodes ---
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("type", "Unknown")
            bg = self.NODE_COLORS.get(node_type, "#9E9E9E")  # grey fallback
            border = _darken_hex(bg, 0.7)
            degree = self.graph.degree(node)
            size = self._node_size(node_type, type_counts, max_type_count, degree)
            node_meta[node] = {"t": node_type, "bg": bg, "br": border, "d": data.get("description") or ""}

            title = f"{node}\nType: {node_type}\nConnections: {degree}"
            source_paper = data.get("source_paper")  # from Phase A
            if source_paper:
                title += f"\nPaper: {_truncate(source_paper, 70)}"

            net.add_node(
                node,
                label=_truncate(node, 28),
                title=title,
                color={"background": bg, "border": border},
                size=size,
                shape="dot",
            )

        # --- Edges ---
        for source, target, data in self.graph.edges(data=True):
            relation = data.get("relation", "")
            color = self.EDGE_COLORS.get(relation, self.DEFAULT_EDGE_COLOR)
            etitle = relation.replace("_", " ").title() if relation else ""
            desc = data.get("description")
            if desc:
                etitle += f"\n{desc[:120]}"
            net.add_edge(
                source, target,
                color={"color": color, "highlight": "#ffffff", "hover": "#ffffff"},
                width=2.5,
                title=etitle,
                rel=relation,  # custom prop so the Edge Filter JS can match by relation type
            )

        # --- Global options (DESIGN_REFERENCE §6). A dict is directly supported by pyvis. ---
        net.options = {
            "nodes": {
                "shape": "dot",
                "borderWidth": 2,
                "shadow": {"enabled": True, "color": "rgba(0,0,0,0.8)", "size": 12, "x": 0, "y": 4},
                "font": {"size": 28, "color": "#e2e8f0", "strokeWidth": 5, "strokeColor": "#080c1a"},
            },
            "edges": {
                "smooth": {"enabled": True, "type": "continuous"},
                "shadow": False,
                "font": {"size": 0},
            },
            "physics": {
                "enabled": True,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -1500,
                    "centralGravity": 0.005,
                    "springLength": 500,
                    "springConstant": 0.05,
                    "damping": 0.8,
                    "avoidOverlap": 1.0,
                },
                "stabilization": {"enabled": True, "iterations": 400, "updateInterval": 25, "fit": True},
            },
            "interaction": {
                "hover": True,
                "tooltipDelay": 80,
                "hideEdgesOnDrag": False,
                "hideNodesOnDrag": False,
                "navigationButtons": True,
                "keyboard": {"enabled": True},
            },
        }

        # pyvis's write_html() writes with the OS default encoding (cp1252 on Windows) and
        # crashes on non-latin-1 characters, so we render the string and write UTF-8 ourselves.
        html = net.generate_html(notebook=False)

        # Drop the unused Bootstrap CDN tags → the page has no external runtime dependency.
        html = _BOOTSTRAP_TAG_RE.sub("", html)
        if _EXTERNAL_REF_RE.search(html):
            print("[Viz] Warning: the page still references an external resource; the strict CSP "
                  "served by server.py will block it. Check the pyvis template.")

        # Inject the custom right-panel UI just before </body>.
        injection = self._ui_injection(type_counts, node_meta)
        if "</body>" in html:
            html = html.replace("</body>", injection + "</body>", 1)
        else:
            html += injection

        return html

    def generate_html(self, output_path: str = "data/processed/graph_visualization.html"):
        """Build the HTML and write it to a file (used by the CLI build auto-open)."""
        html = self.build_html()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Visualization generated at {output_path}")
        return output_path
