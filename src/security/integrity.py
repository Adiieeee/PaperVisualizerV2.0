"""
Integrity of the build artifacts (graph.graphml + ChromaDB + embedding model) and atomic writes.

Problems this closes:
  * Swapping EMBEDDING_MODEL for another model with the SAME dimension does not crash — cosine is still
    computed, but between two incomparable embedding spaces: confidently wrong answers. The manifest
    records the model + revision that built the store, and the engine refuses to start on a mismatch.
  * graph.graphml, chroma_db and the embedding model must be consistent with each other. `data/processed/*`
    is not committed, so the manifest (graph SHA-256 + node/edge/chunk counts) is the only way
    to detect artifacts that changed unnoticed (corruption, half-finished restore, interrupted build).
  * An extraction checkpoint = expensive LLM output (quota). A non-atomic write interrupted midway
    breaks the JSON so the build cannot resume. Write via a temporary file + os.replace.

Honest limitation: each file is atomic, but several os.replace calls are NOT atomic as a whole. A crash
between them leaves an out-of-sync pair — which the manifest does detect at startup.
The manifest protects against UNINTENTIONAL changes, not against an attacker who also rewrites the manifest.
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
    """Build artifacts are inconsistent with the manifest / configuration. Must not be continued silently."""


# ── Atomic writes ─────────────────────────────────────────────────────────────────────────────────

def atomic_write(path: str, write_fn: Callable[[str], None]) -> None:
    """Call write_fn(tmp_path) then os.replace(tmp, path). tmp is cleaned up if write_fn fails."""
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
    """Fingerprint of the type schema (entities/relations/intents). Changes when the type lists in config.py change."""
    parts = [
        "E:" + ",".join(get_args(config.ENTITY_TYPES)),
        "R:" + ",".join(get_args(config.RELATION_TYPES)),
        "I:" + ",".join(get_args(config.INTENT_TYPES)),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def chroma_stats(chroma) -> dict:
    """Number of chunks/community summaries + the vector dimension ACTUALLY stored in Chroma."""
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
            "max_seq_length": config.EMBEDDING_MAX_SEQ_LENGTH,  # truncates long text -> part of the embedding space
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
    """Compare the manifest with the current state. Returns a list of problems (empty = consistent)."""
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
    if emb.get("max_seq_length") not in (None, config.EMBEDDING_MAX_SEQ_LENGTH):  # None = old manifest
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
    Called at engine start. True = the manifest exists and is consistent. False = no manifest yet (only a
    warning: an old corpus from before this feature; create one with `python corpus.py manifest`).
    Raises ArtifactIntegrityError on any mismatch — fail hard and loudly.
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
