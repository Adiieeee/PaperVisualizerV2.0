"""
Frontend: nama entitas / relasi / deskripsi adalah keluaran LLM atas PDF tak tepercaya. Tes ini membangun
halaman dari graph BERMUSUHAN lalu memeriksa dengan parser HTML sungguhan (yang memperlakukan <script>
seperti browser) bahwa tidak ada markup yang lolos, tidak ada handler inline, dan tidak ada sumber eksternal.
"""
import base64
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from html.parser import HTMLParser

import networkx as nx
import pytest

from src.graph.visualizer import GraphVisualizer, _json_for_script
from src.security.guards import build_index_csp, inline_script_hashes

BREAKOUT = "</script><img src=x onerror=alert(1)>"
COMMENT_TRICK = "<!--<script>"
ATTR_BREAK = '"><svg onload=alert(1)>'


def _graph(hostile: bool) -> nx.Graph:
    G = nx.Graph()
    if hostile:
        names = [BREAKOUT, COMMENT_TRICK, ATTR_BREAK]
        for i, n in enumerate(names):
            G.add_node(n, type="Method", description=f"desc {n}", source_paper=n)
        G.add_node("Paper A", type="Paper", description="")
        G.add_node("Weird", type=ATTR_BREAK, description="")  # tipe node bermusuhan
        for n in names:
            G.add_edge("Paper A", n, relation="uses_method", description=f"rel {n}")
        G.add_edge("Paper A", "Weird", relation=ATTR_BREAK, description=BREAKOUT)  # relasi bermusuhan
    else:
        G.add_node("Paper A", type="Paper", description="")
        G.add_node("ResNet", type="Method", description="a net")
        G.add_edge("Paper A", "ResNet", relation="uses_method", description="uses")
    return G


class _Collect(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags, self.attrs = [], []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attrs += [k for k, _ in attrs]


def _parse(html):
    p = _Collect()
    p.feed(html)
    return p


@pytest.fixture(scope="module")
def pages():
    return GraphVisualizer(_graph(False)).build_html(), GraphVisualizer(_graph(True)).build_html()


def test_hostile_names_do_not_change_the_document_structure(pages):
    benign, hostile = pages
    b, h = _parse(benign), _parse(hostile)
    # jika payload lolos dari <script>, parser akan melihat tag <img>/<svg> baru
    assert Counter(h.tags) == Counter(b.tags)
    assert "img" not in h.tags and "svg" not in h.tags


def test_no_inline_event_handler_attributes(pages):
    for html in pages:
        assert [a for a in _parse(html).attrs if a.startswith("on")] == []


def test_raw_payloads_never_appear_verbatim(pages):
    _, hostile = pages
    for raw in (BREAKOUT, COMMENT_TRICK, ATTR_BREAK):
        assert raw not in hostile


def test_no_external_resources_and_no_bootstrap_cdn(pages):
    for html in pages:
        assert "cdn.jsdelivr" not in html
        assert not re.search(r"""<(?:script|link|img|iframe)\b[^>]*\b(?:src|href)\s*=\s*["']https?://""", html, re.I)


def test_csp_hashes_match_every_inline_script(pages):
    for html in pages:
        hashes = inline_script_hashes(html)
        assert len(hashes) >= 3  # bindings/utils, vis-network, init pyvis, panel kita
        csp = build_index_csp(html)
        script_src = re.search(r"script-src ([^;]+);", csp).group(1)
        assert "'unsafe-inline'" not in script_src and "'unsafe-eval'" not in script_src
        for h in hashes:
            assert f"'sha256-{h}'" in script_src
        # hitung ulang secara independen dari isi <script> apa pun yang tidak punya src
        blocks = [m for m in re.findall(r"<script\b([^>]*)>(.*?)</script>", html, re.S | re.I) if "src=" not in m[0]]
        assert len(blocks) == len(hashes)
        d = hashlib.sha256(blocks[0][1].replace("\r\n", "\n").encode("utf-8")).digest()
        assert base64.b64encode(d).decode() == hashes[0]


def test_json_for_script_escapes_script_breakers():
    out = _json_for_script({"a": BREAKOUT, "b": COMMENT_TRICK, "c": "x & y", "d": chr(0x2028)})
    for ch in "<>&" + chr(0x2028):
        assert ch not in out
    assert json.loads(out) == {"a": BREAKOUT, "b": COMMENT_TRICK, "c": "x & y", "d": chr(0x2028)}  # round-trip utuh


# --- eksekusi renderMarkdown / esc di Node -------------------------------------------------------------

def _extract(html, start_marker, end_marker):
    a = html.index(start_marker)
    return html[a:html.index(end_marker, a)]


XSS_INPUTS = [
    "<img src=x onerror=alert(1)>",
    "<script>alert(1)</script>",
    '"><svg onload=alert(1)>',
    "**<b onclick=alert(1)>tebal</b>**",
    "[klik](javascript:alert(1))",
    "`<iframe src=//evil>`",
    "# <a href='javascript:alert(1)'>judul</a>",
    "- item <img src=x onerror=alert(1)>",
    "&lt;img src=x onerror=alert(1)&gt;",
    "' onmouseover='alert(1)",
]
ALLOWED_TAGS = {"h1", "h2", "h3", "strong", "em", "code", "li", "ul", "p"}


@pytest.mark.skipif(shutil.which("node") is None, reason="node tidak tersedia")
def test_render_markdown_and_esc_are_xss_safe_when_executed(pages, tmp_path):
    html = pages[0]
    render_src = _extract(html, "function renderMarkdown(raw) {", "  function chatAdd(")
    esc_src = _extract(html, "function esc(s) {", "  // Canonical")
    script = tmp_path / "t.js"
    script.write_text(
        render_src + "\n" + esc_src + "\n"
        + "const inputs = " + json.dumps(XSS_INPUTS) + ";\n"
        + "console.log(JSON.stringify({md: inputs.map(renderMarkdown), esc: inputs.map(esc)}));\n",
        encoding="utf-8",
    )
    result = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)

    for md in out["md"]:
        p = _parse(md)
        assert set(p.tags) <= ALLOWED_TAGS, md   # hanya tag whitelist
        assert p.attrs == [], md                 # tanpa atribut sama sekali
    for e in out["esc"]:
        assert "<" not in e and ">" not in e and '"' not in e  # aman untuk teks dan atribut ber-kutip ganda
    # teks asli tetap terbaca (di-escape, bukan dibuang)
    assert "&lt;img" in out["md"][0]
