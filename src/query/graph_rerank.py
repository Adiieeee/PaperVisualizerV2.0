"""
Rerank fakta graph terhadap pertanyaan: pangkas node yang melebihi budget menurut relevansi (bukan
urutan BFS/GraphML) dan urutkan konteks supaya fakta yang paling relevan muncul lebih dulu.

Skor = cosine bge-m3 (model embedding yang sudah dimuat engine, tanpa model baru) antara pertanyaan asli
user dan teks fakta "source relation target — deskripsi". Meng-embed fakta mahal (bge-m3 568M parameter:
~25 ms/fakta di GTX 1650, ~80 ms di CPU; node GPT-4o punya 301 fakta), jadi embedding fakta di-cache per
teks di memori DAN di disk (FACT_EMBEDDINGS_PATH): tiap fakta dihitung sekali, restart tetap cepat.
Cache dialamatkan oleh teks dan hanya dipakai bila model/revisi/panjang-seq sama, jadi aman lintas korpus.

Aturan (diuji di tests/test_graph_rerank.py):
  - jatah hop-1 tetap <= 60% budget (sama dengan traverse()), sisanya untuk hop-2;
  - hop-1 dipilih per GRUP relasi (skor grup = skor anggota tertinggi), grup diambil utuh selama muat,
    supaya daftar seperti `uses_dataset` tidak terpotong sebagian tanpa alasan;
  - fakta hop-2 hanya disimpan bila tepi hop-1 induknya tersimpan (rantai A → B → C tidak putus);
  - urutan keluaran: grup hop-1 menurut skor grup, hop-2 menurut skor. format_traversal_context
    mempertahankan urutan list, jadi format dan prompt jawaban tidak berubah. Node di bawah budget
    tidak kehilangan fakta apa pun; hanya urutannya yang berubah.
"""
import os
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import EMBEDDING_MAX_SEQ_LENGTH, EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_REVISION
from src.security.integrity import atomic_write

HOP1_SHARE = 0.6  # sama dengan hop1_cap di GraphTraversal.traverse
_CACHE_MODEL_ID = f"{EMBEDDING_MODEL_NAME}@{EMBEDDING_MODEL_REVISION}/seq{EMBEDDING_MAX_SEQ_LENGTH}"


def fact_text(rel: Dict[str, Any]) -> str:
    text = f"{rel['source']} {rel.get('relation') or 'related_to'} {rel['target']}"
    return f"{text} — {rel['description']}" if rel.get("description") else text


def all_fact_texts(graph) -> List[str]:
    """Semua teks fakta yang bisa muncul di konteks: tiap edge dari KEDUA arah, karena arah source/target
    ditentukan oleh traversal (graph tak berarah). Atribut dibaca persis seperti traverse()."""
    texts = []
    for u, v, ed in graph.edges(data=True):
        for source, target in ((u, v), (v, u)):
            texts.append(fact_text({"source": source, "target": target,
                                    "relation": ed.get("relation"), "description": ed.get("description")}))
    return list(dict.fromkeys(texts))


def load_fact_cache(path: str) -> Dict[str, np.ndarray]:
    """{teks fakta: vektor} dari disk; {} bila tidak ada, rusak, atau dibuat oleh model lain."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["model"]) != _CACHE_MODEL_ID:
                return {}
            return {str(t): v for t, v in zip(data["texts"], data["vectors"].astype(np.float32))}
    except Exception as e:  # file rusak: hitung ulang saja, jangan gagalkan query
        print(f"[Rerank] Cache embedding fakta diabaikan ({e})")
        return {}


def save_fact_cache(path: str, cache: Dict[str, np.ndarray]) -> None:
    texts = list(cache)

    def _write(tmp: str) -> None:
        with open(tmp, "wb") as f:  # objek file: np.savez menambah ".npz" pada nama file tanpa ekstensi itu
            vectors = np.stack([cache[t] for t in texts]) if texts else np.zeros((0, 0), dtype=np.float32)
            np.savez(f, model=np.array(_CACHE_MODEL_ID), texts=np.array(texts, dtype=str), vectors=vectors)

    atomic_write(path, _write)


class GraphFactReranker:
    def __init__(self, embed_client, cache_path: Optional[str] = None):
        self.embed = embed_client
        self.cache_path = cache_path
        self._cache: Optional[Dict[str, np.ndarray]] = None  # dimuat dari disk saat pertama dipakai

    def _loaded(self) -> Dict[str, np.ndarray]:
        if self._cache is None:
            self._cache = load_fact_cache(self.cache_path)
        return self._cache

    def _save(self) -> None:
        if self.cache_path:
            try:
                save_fact_cache(self.cache_path, self._cache)
            except OSError as e:  # disk penuh / read-only: tetap jalan dengan cache di memori
                print(f"[Rerank] Cache embedding fakta tidak tersimpan ({e})")

    def _embed_missing(self, texts: List[str]) -> int:
        cache = self._loaded()
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        if missing:
            for t, vec in zip(missing, self.embed.embed_batch(missing)):
                cache[t] = np.asarray(vec, dtype=np.float32)
            self._save()
        return len(missing)

    def warm(self, graph) -> Dict[str, int]:
        """Isi cache untuk SEMUA fakta graph (dipanggil di akhir build) dan buang teks yang tak ada lagi di
        graph. Hanya fakta yang belum ada di cache yang di-embed, jadi build ulang korpus yang mirip cepat."""
        texts = all_fact_texts(graph)
        cache = self._loaded()
        wanted = set(texts)
        stale = [t for t in cache if t not in wanted]
        for t in stale:
            del cache[t]
        computed = self._embed_missing(texts)
        if stale and not computed:
            self._save()
        return {"facts": len(texts), "computed": computed, "pruned": len(stale)}

    def score(self, query: str, rels: List[Dict[str, Any]]) -> List[float]:
        texts = [fact_text(r) for r in rels]
        self._embed_missing(texts)
        q = np.asarray(self.embed.embed_text(query), dtype=np.float32)
        return [float(np.dot(q, self._cache[t])) for t in texts]

    def select(self, trav: Dict[str, Any], query: str, max_rels: int) -> Dict[str, Any]:
        """Hasil traverse() lengkap -> salinan dengan <= max_rels relasi, terurut menurut relevansi."""
        rels = trav.get("relationships") or []
        if "error" in trav or not rels:
            return trav
        scores = self.score(query, rels)
        entity = trav["entity"]

        groups: Dict[str, List[int]] = {}
        hop2: List[int] = []
        for i, r in enumerate(rels):
            if r["hop"] == 1 and r["source"] == entity:
                groups.setdefault(r.get("relation") or "", []).append(i)
            else:
                hop2.append(i)

        by_score = lambda idx: sorted(idx, key=lambda i: -scores[i])  # sorted() stabil untuk skor sama
        hop1_cap = max(1, int(max_rels * HOP1_SHARE))
        kept: List[int] = []
        for members in sorted(groups.values(), key=lambda m: -max(scores[i] for i in m)):
            room = hop1_cap - len(kept)
            if room <= 0:
                break
            kept.extend(by_score(members)[:room])

        reachable = {rels[i]["target"] for i in kept}
        room = max_rels - len(kept)
        kept.extend(by_score([i for i in hop2 if rels[i]["source"] in reachable])[:max(room, 0)])

        return {**trav, "relationships": [rels[i] for i in kept]}
