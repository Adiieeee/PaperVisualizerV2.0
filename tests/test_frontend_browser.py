"""
Tests in a REAL browser (headless Chrome/Edge) for the innerHTML sinks that receive data from the graph file
(relation names, node types). Skipped automatically if there is no Chrome/Edge. No network, no LLM.

There is a positive control: the OLD code (without esc) is put back and the test MUST detect the markup that gets
through — so it is clear this test is sensitive, rather than passing because it checks nothing.
"""
import json
import os
import re
import shutil
import subprocess

import networkx as nx
import pytest

from src.graph.visualizer import GraphVisualizer

_CANDIDATES = [
    os.environ.get("CHROME_PATH", ""),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]
CHROME = next((p for p in _CANDIDATES if p and os.path.exists(p)), None) or shutil.which("chrome") or shutil.which("chromium")

pytestmark = pytest.mark.skipif(CHROME is None, reason="Chrome/Edge tidak ditemukan (set CHROME_PATH)")

PAYLOAD = "<img src=x onerror=window.__pwn=1>"

HARNESS = """<script>
window.addEventListener('load', function () { setTimeout(function () {
  var out = {};
  try {
    var sel = document.getElementById('gr-edge-select');
    var evil = Array.from(sel.options).find(function (o) { return o.value.indexOf('img') >= 0; });
    out.optionAttrs = Array.from(sel.options).map(function (o) { return o.getAttributeNames().join(','); });
    if (evil) { sel.value = evil.value; sel.dispatchEvent(new Event('change')); }
    out.descImgs = document.getElementById('gr-edge-desc').querySelectorAll('img').length;
    var si = document.getElementById('gr-search-input'); si.value = 'Weird'; si.dispatchEvent(new Event('input'));
    out.ddImgs = document.getElementById('gr-search-dropdown').querySelectorAll('img').length;
  } catch (e) { out.err = String(e); }
  document.title = 'RESULT:' + JSON.stringify(out);
}, 2500); });
</script>"""


def _hostile_html() -> str:
    G = nx.Graph()
    G.add_node("Paper A", type="Paper", description="")
    G.add_node("Weird", type=PAYLOAD, description="")                                       # hostile node type
    G.add_edge("Paper A", "Weird", relation='x"><img src=x onerror=window.__pwn=1>', description="d")  # hostile relation
    return GraphVisualizer(G).build_html()


def _run(html: str, tmp_path, name: str) -> dict:
    page = tmp_path / f"{name}.html"
    page.write_text(html.replace("</body>", HARNESS + "</body>", 1), encoding="utf-8")
    proc = subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox", f"--user-data-dir={tmp_path / ('prof-' + name)}",
         "--virtual-time-budget=20000", "--dump-dom", page.as_uri()],
        capture_output=True, text=True, timeout=120,
    )
    m = re.search(r"<title>RESULT:(.*?)</title>", proc.stdout, re.S)
    assert m, f"hasil harness tidak ditemukan; stderr: {proc.stderr[-300:]}"
    return json.loads(m.group(1).replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))


def test_hostile_relation_and_type_do_not_inject_markup(tmp_path):
    out = _run(_hostile_html(), tmp_path, "fixed")
    assert "err" not in out, out
    assert out["descImgs"] == 0          # updateEdgeDesc sink (fallback rel) is escaped
    assert out["ddImgs"] == 0            # search dropdown sink (node type) is escaped
    assert all(not any(x.startswith("on") for x in attrs.split(",")) for attrs in out["optionAttrs"])  # no handler


def test_control_old_unescaped_code_is_detected(tmp_path):
    html = _hostile_html()
    old = html.replace("|| esc(rel);", "|| rel;").replace('esc(metaOf(m.id).t || "")', '(metaOf(m.id).t || "")')
    assert old != html, "pola kode lama tidak ditemukan: perbarui kontrol ini bila visualizer.py berubah"
    out = _run(old, tmp_path, "old")
    assert out["descImgs"] >= 1 and out["ddImgs"] >= 1, out  # hostile markup gets through in the old code
