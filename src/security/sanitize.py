"""
Netralisasi teks tak tepercaya sebelum masuk prompt LLM.

Sumber teks tak tepercaya di project ini: pertanyaan pengguna, isi PDF (chunk), dan keluaran LLM atas
isi PDF (nama entitas, deskripsi relasi, ringkasan komunitas — "injeksi orde kedua"). Semuanya
diperlakukan sebagai DATA. Fungsi di sini dipakai HANYA saat merakit prompt: teks yang tersimpan di
ChromaDB/graph TIDAK diubah, karena mengubah teks chunk berarti mengubah embedding-nya (aturan lapisan
vektor, lihat CLAUDE.md).

CATATAN: karakter tak terlihat (zero-width dsb.) dibangun dengan chr(), BUKAN ditulis sebagai literal
atau escape di sumber — literal hilang tanpa jejak saat disalin, dan sebagian alat penyunting men-decode
escape-nya menjadi literal. tests/test_sanitize.py menjaga agar sumber ini tetap bebas karakter tersebut.
"""
import re
from typing import List, Optional

# Zero-width space yang disisipkan ke dalam tag pembatas untuk merusaknya.
_ZWSP = chr(0x200B)

# Tag pembatas yang dipakai prompt di src/llm/prompts.py dan prompt build-time di llm_service.py.
# Tambahkan di sini setiap kali sebuah prompt memperkenalkan tag pembatas baru.
PROTECTED_TAGS = [
    "konteks_graf",
    "konteks_teks",
    "ringkasan_komunitas",
    "pertanyaan_user",
    "teks_paper",
    "daftar_entitas",
]

# `<` / `>` termasuk bentuk ter-encode (&lt; &#60; &#x3c;). `\s*` di sekitar `/` dan nama tag menutup
# varian "< tag>", "</ tag>", dan newline. Atribut ("<tag x=1>") ikut ditangkap karena sebagian LLM
# memperlakukannya sebagai tag pembuka.
_LT = r"(?:<|&lt;|&#0*60;|&#x0*3c;)"
_GT = r"(?:>|&gt;|&#0*62;|&#x0*3e;)"
_TAG_PATTERN = re.compile(
    _LT + r"\s*/?\s*(?:" + "|".join(PROTECTED_TAGS) + r")(?![\w-])(?:\s[^<>]*)?\s*" + _GT,
    re.IGNORECASE,
)


def neutralize_tags(text) -> str:
    """
    Rusak tag pembatas milik prompt yang muncul di dalam `text` (sisipkan zero-width space).
    `<`/`>` biasa ("< 2/3 dari jumlah") dibiarkan supaya kutipan dokumen tidak rusak.
    Idempoten: hasilnya tidak cocok lagi dengan pola, jadi aman dipanggil berkali-kali.
    """
    if text is None:
        return ""

    def _break(m: re.Match) -> str:
        s = m.group(0)
        return s[0] + _ZWSP + s[1:-1] + _ZWSP + s[-1]

    return _TAG_PATTERN.sub(_break, str(text))


# --- Kelas karakter (dibangun dengan chr(), lihat catatan di atas) ---------------------------------

# Zero-width, bidi override/isolate, word-joiner, BOM: tak terlihat, tidak punya alasan ada di nama entitas.
_INVISIBLE_CHARS = "".join(
    chr(c) for c in (*range(0x200B, 0x2010), *range(0x202A, 0x202F), 0x2060, 0xFEFF)
)
# Pemisah baris Unicode (LS/PS): diperlakukan seperti newline.
_LINE_SEPARATORS = chr(0x2028) + chr(0x2029)

_INVISIBLE_RE = re.compile("[" + re.escape(_INVISIBLE_CHARS) + "]")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f" + re.escape(_LINE_SEPARATORS) + "]")
# Untuk audit teks chunk: kontrol + tak terlihat, KECUALI tab/newline/CR yang wajar di teks PDF.
_HIDDEN_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f" + re.escape(_LINE_SEPARATORS + _INVISIBLE_CHARS) + "]"
)
_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"(?:https?://|ftp://|www\.)", re.IGNORECASE)

# --- Nama entitas (dipakai cleaner.py saat build; efek di rebuild berikutnya) -------------------

# Diukur pada graph nyata: nama Finding/Limitation terpanjang 175 char. Batas ini sengaja longgar —
# tujuannya membuang keluaran LLM yang jelas abnormal, bukan memangkas pernyataan yang sah.
MAX_ENTITY_NAME_LENGTH = 250


def sanitize_entity_name(name: str) -> Optional[str]:
    """
    Bersihkan nama entitas hasil ekstraksi LLM. Mengembalikan nama bersih, atau None bila entitas
    harus DITOLAK: kosong, terlalu panjang, memuat `<`/`>` (kemungkinan markup/tag), atau URL.
    Karakter kontrol/newline diganti spasi (lalu spasi ganda dipadatkan); karakter tak terlihat
    (zero-width, bidi) dihapus supaya nama seperti "Res<ZWSP>Net" tetap "ResNet".
    """
    cleaned = _WS_RE.sub(" ", _CONTROL_RE.sub(" ", _INVISIBLE_RE.sub("", name or ""))).strip()
    if not cleaned or len(cleaned) > MAX_ENTITY_NAME_LENGTH:
        return None
    if "<" in cleaned or ">" in cleaned or _URL_RE.search(cleaned):
        return None
    return cleaned


# --- Audit (read-only): dipakai `python corpus.py audit` -----------------------------------------

# Hanya frasa IMPERATIF pengganti-instruksi. Kata "system prompt" saja sengaja tidak dihitung: paper
# tentang LLM membahasnya di teks biasa (terukur: 5 dari 189 chunk korpus sekarang), itu bukan injeksi.
_INJECTION_HINT_RE = re.compile(
    r"(?:ignore|disregard|forget)\s+(?:(?:all|any|the|your)\s+)*(?:(?:previous|prior|above|earlier|preceding)\s+)?"
    r"(?:instructions?|rules|prompts?|guidelines)"
    r"|abaikan\s+(?:(?:semua|seluruh)\s+)?(?:instruksi|aturan|perintah)"
    r"|(?:reveal|show|print|repeat|tampilkan)\s+(?:(?:your|the)\s+)?system\s+prompt",
    re.IGNORECASE,
)


def audit_text(text: str, max_len: Optional[int] = None, *, names: bool = True) -> List[str]:
    """
    Kembalikan daftar alasan sebuah teks tampak mencurigakan (kosong = tampak wajar).
      max_len : tandai teks yang lebih panjang (mis. nama entitas).
      names   : True untuk nama/deskripsi entitas (URL dan karakter kontrol ikut dinilai). False untuk teks
                chunk/ringkasan: URL wajar di paper, dan karakter kontrol (mis. U+0010) hanya artefak font PDF.
    """
    if not text:
        return []
    reasons = []
    if _TAG_PATTERN.search(text):
        reasons.append("mirip tag pembatas prompt")
    if _INJECTION_HINT_RE.search(text):
        reasons.append("frasa mirip instruksi/injeksi")
    if names and _URL_RE.search(text):
        reasons.append("berisi URL")
    if names and _HIDDEN_RE.search(text):
        reasons.append("karakter kontrol/tak terlihat")
    if max_len is not None and len(text) > max_len:
        reasons.append(f"lebih panjang dari {max_len} char ({len(text)})")
    return reasons
