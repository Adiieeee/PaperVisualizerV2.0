"""
Rerank fakta graph terhadap pertanyaan: pangkas node yang melebihi budget menurut relevansi (bukan
urutan BFS/GraphML) dan urutkan konteks supaya fakta yang paling relevan muncul lebih dulu.

Skor = cosine bge-m3 (model embedding yang sudah dimuat engine, tanpa model baru) antara pertanyaan asli
user dan teks fakta "source relation target — deskripsi". Embedding fakta di-cache per teks, jadi query
berikutnya pada node yang sama hampir gratis.

Aturan (diuji di tests/test_graph_rerank.py):
  - jatah hop-1 tetap <= 60% budget (sama dengan traverse()), sisanya untuk hop-2;
  - hop-1 dipilih per GRUP relasi (skor grup = skor anggota tertinggi), grup diambil utuh selama muat,
    supaya daftar seperti `uses_dataset` tidak terpotong sebagian tanpa alasan;
  - fakta hop-2 hanya disimpan bila tepi hop-1 induknya tersimpan (rantai A → B → C tidak putus);
  - urutan keluaran: grup hop-1 menurut skor grup, hop-2 menurut skor. format_traversal_context
    mempertahankan urutan list, jadi format dan prompt jawaban tidak berubah. Node di bawah budget
    tidak kehilangan fakta apa pun; hanya urutannya yang berubah.
"""
from typing import Any, Dict, List

import numpy as np

HOP1_SHARE = 0.6  # sama dengan hop1_cap di GraphTraversal.traverse


def fact_text(rel: Dict[str, Any]) -> str:
    text = f"{rel['source']} {rel.get('relation') or 'related_to'} {rel['target']}"
    return f"{text} — {rel['description']}" if rel.get("description") else text


class GraphFactReranker:
    def __init__(self, embed_client):
        self.embed = embed_client
        self._cache: Dict[str, np.ndarray] = {}

    def score(self, query: str, rels: List[Dict[str, Any]]) -> List[float]:
        texts = [fact_text(r) for r in rels]
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for t, vec in zip(missing, self.embed.embed_batch(missing)):
                self._cache[t] = np.asarray(vec, dtype=np.float32)
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
