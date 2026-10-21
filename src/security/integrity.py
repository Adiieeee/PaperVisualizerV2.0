"""
Integritas artefak build (graph.graphml + ChromaDB + model embedding) dan tulis-atomik.

Masalah yang ditutup:
  * Mengganti EMBEDDING_MODEL ke model lain dengan dimensi SAMA tidak crash — cosine tetap terhitung,
    tetapi antar dua ruang embedding yang tidak sebanding: jawaban salah yang percaya diri. Manifest
    mencatat model + revisi yang membangun store, dan engine menolak start bila tidak cocok.
  * graph.graphml, chroma_db, dan model embedding harus konsisten satu sama lain. `data/processed/*`
    tidak di-commit, jadi manifest (SHA-256 graph + jumlah node/edge/chunk) adalah satu-satunya cara
    mendeteksi artefak yang berubah tanpa disadari (korupsi, restore setengah jalan, build terputus).
  * Checkpoint ekstraksi = keluaran LLM yang mahal (kuota). Tulis tidak-atomik yang terputus di tengah
    merusak JSON sehingga build tidak bisa dilanjutkan. Tulis via file sementara + os.replace.

Batasan jujur: tiap file atomik, tetapi beberapa os.replace TIDAK atomik sebagai satu kesatuan. Crash di
antaranya menghasilkan pasangan tak sinkron — yang justru terdeteksi oleh manifest saat start.
Manifest melindungi dari perubahan TIDAK DISENGAJA, bukan dari penyerang yang menulis ulang manifest juga.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Callable, List, Optional, get_args

from src import config

MANIFEST_PATH = "data/processed/manifest.json"
MANIFEST_VERSION = 1


class ArtifactIntegrityError(RuntimeError):
    """Artefak build tidak konsisten dengan manifest / konfigurasi. Tidak boleh dilanjutkan diam-diam."""


# ── Tulis atomik ────────────────────────────────────────────────────────────────────────────

def atomic_write(path: str, write_fn: Callable[[str], None]) -> None:
    """Panggil write_fn(tmp_path) lalu os.replace(tmp, path). tmp dibersihkan bila write_fn gagal."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    def _write(tmp: str) -> None:
        with open(tmp, "w", encoding=encoding, newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())

    atomic_write(path, _write)


def atomic_write_json(path: str, obj, **json_kwargs) -> None:
    json_kwargs.setdefault("ensure_ascii", False)
    atomic_write_text(path, json.dumps(obj, **json_kwargs))


# ── Manifest ────────────────────────────────────────────────────────────────────────────────

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def schema_fingerprint() -> str:
    """Sidik jari skema tipe (entitas/relasi/intent). Berubah bila daftar tipe di config.py berubah."""
    parts = [
        "E:" + ",".join(get_args(config.ENTITY_TYPES)),
        "R:" + ",".join(get_args(config.RELATION_TYPES)),
        "I:" + ",".join(get_args(config.INTENT_TYPES)),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def chroma_stats(chroma) -> dict:
    """Jumlah chunk/ringkasan komunitas + dimensi vektor yang BENAR-BENAR tersimpan di Chroma."""
    chunks = chroma.chunks_collection
    dim = None
    if chunks.count():
        emb = chunks.peek(1).get("embeddings")
        if emb is not None and len(emb):
            dim = len(emb[0])
    return {
        "chunks": chunks.count(),
        "community_summaries": chroma.community_collection.count(),
        "dimension": dim,
    }


def build_manifest(graph_path: str, graph, chroma) -> dict:
    stats = chroma_stats(chroma)
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding": {
            "model": config.EMBEDDING_MODEL_NAME,
            "revision": config.EMBEDDING_MODEL_REVISION,
            "dimension": stats["dimension"] or config.EMBEDDING_DIMENSION,
            "max_seq_length": config.EMBEDDING_MAX_SEQ_LENGTH,  # memotong teks panjang -> ikut ruang embedding
        },
        "schema_fingerprint": schema_fingerprint(),
        "graph": {
            "file": os.path.basename(graph_path),
            "sha256": sha256_file(graph_path),
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        },
        "vectorstore": {
            "chunks": stats["chunks"],
            "community_summaries": stats["community_summaries"],
        },
    }


def write_manifest(manifest: dict, path: str = MANIFEST_PATH) -> None:
    atomic_write_json(path, manifest, indent=2)


def load_manifest(path: str = MANIFEST_PATH) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise ArtifactIntegrityError(f"{path} tidak bisa dibaca ({e}). Perbaiki atau hapus, lalu buat ulang "
                                     f"dengan `python corpus.py manifest`.") from e


def verify_manifest(manifest: dict, *, graph_path: str, graph, chroma,
                    query_dimension: Optional[int] = None) -> List[str]:
    """Bandingkan manifest dengan keadaan sekarang. Mengembalikan daftar masalah (kosong = konsisten)."""
    problems: List[str] = []

    emb = manifest.get("embedding", {})
    if emb.get("model") != config.EMBEDDING_MODEL_NAME:
        problems.append(
            f"Model embedding berubah: store dibangun dengan '{emb.get('model')}', config sekarang "
            f"'{config.EMBEDDING_MODEL_NAME}'. Skor kemiripan antar dua ruang embedding tidak sebanding "
            f"(jawaban salah yang percaya diri). Kembalikan model, atau build ulang."
        )
    elif emb.get("revision") != config.EMBEDDING_MODEL_REVISION:
        problems.append(
            f"Revisi model embedding berubah: store dibangun dengan revisi '{emb.get('revision')}', config "
            f"sekarang '{config.EMBEDDING_MODEL_REVISION}'. Kembalikan revisinya, atau build ulang."
        )
    if emb.get("max_seq_length") not in (None, config.EMBEDDING_MAX_SEQ_LENGTH):  # None = manifest lama
        problems.append(
            f"EMBEDDING_MAX_SEQ_LENGTH berubah: store dibangun dengan {emb.get('max_seq_length')}, config "
            f"sekarang {config.EMBEDDING_MAX_SEQ_LENGTH} (teks panjang dipotong berbeda). Kembalikan, atau build ulang."
        )
    if query_dimension is not None and emb.get("dimension") not in (None, query_dimension):
        problems.append(
            f"Dimensi embedding tidak cocok: store {emb.get('dimension')}, model saat query {query_dimension}."
        )

    if manifest.get("schema_fingerprint") != schema_fingerprint():
        problems.append(
            "Daftar tipe entitas/relasi/intent di config.py berbeda dari saat graph dibangun. "
            "Graph lama mungkin memuat tipe yang tidak dikenal kode sekarang."
        )

    g = manifest.get("graph", {})
    if os.path.exists(graph_path) and g.get("sha256") != sha256_file(graph_path):
        problems.append(
            f"{graph_path} berubah sejak manifest ditulis (SHA-256 berbeda). Kemungkinan build terputus, "
            f"restore setengah jalan, atau file diubah. Selesaikan build, atau jika Anda yakin isinya benar: "
            f"`python corpus.py manifest`."
        )
    elif (g.get("nodes"), g.get("edges")) != (graph.number_of_nodes(), graph.number_of_edges()):
        problems.append("Jumlah node/edge graph tidak cocok dengan manifest.")

    stats = chroma_stats(chroma)
    v = manifest.get("vectorstore", {})
    if v.get("chunks") != stats["chunks"] or v.get("community_summaries") != stats["community_summaries"]:
        problems.append(
            f"Isi ChromaDB tidak cocok dengan manifest (chunk {stats['chunks']} vs {v.get('chunks')}, ringkasan "
            f"{stats['community_summaries']} vs {v.get('community_summaries')}). Sisa chunk dari korpus lama? "
            f"Gunakan `python corpus.py clear` sebelum mengganti korpus."
        )
    return problems


def check_artifacts(graph_path: str, graph, chroma, embed=None, manifest_path: str = MANIFEST_PATH) -> bool:
    """
    Dipanggil saat engine start. True = manifest ada dan konsisten. False = manifest belum ada (hanya
    peringatan: korpus lama sebelum fitur ini; buat dengan `python corpus.py manifest`).
    Raise ArtifactIntegrityError bila ada ketidakcocokan — gagal keras dan berisik.
    """
    manifest = load_manifest(manifest_path)
    if manifest is None:
        print("[Integrity] WARNING: manifest.json belum ada, integritas artefak tidak diverifikasi. "
              "Buat dengan: python corpus.py manifest")
        return False
    query_dim = len(embed.embed_text("dimension probe")) if embed is not None else None
    problems = verify_manifest(manifest, graph_path=graph_path, graph=graph, chroma=chroma, query_dimension=query_dim)
    if problems:
        raise ArtifactIntegrityError(
            "Artefak tidak konsisten dengan manifest:\n  - " + "\n  - ".join(problems)
        )
    print("[Integrity] Manifest cocok (model, graph, ChromaDB).")
    return True
