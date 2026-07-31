import os
import json
import csv
import re
from typing import List, Dict, Optional
from src.llm.gemini_client import GeminiClient
from src.llm.embedding import EmbeddingClient
from src.config import DEDUP_SIMILARITY_THRESHOLD, Entity

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")

# SEED ACRONYM DICTIONARY (small, general ML/CV safety-net).
# The PRIMARY defense against acronyms is the extraction prompt (Rule 2 → full canonical names).
# At build time `extract_acronyms_from_texts()` AUTO-EXTENDS this from each paper's own
# "Full Name (ACR)" definitions, so it grows with any new corpus instead of being hand-maintained.
# Do NOT add proper-noun model names here — "VGG"/"ResNet" are themselves canonical.
KNOWN_ACRONYMS = {
    "sgd": "Stochastic Gradient Descent",
    "bn": "Batch Normalization",
    "lstm": "Long Short-Term Memory",
    "relu": "Rectified Linear Unit",
    "fcn": "Fully Convolutional Network",
    "cnn": "Convolutional Neural Network",
    "dnn": "Deep Neural Network",
    "mlp": "Multilayer Perceptron",
    "rnn": "Recurrent Neural Network",
    "gan": "Generative Adversarial Network",
    "svm": "Support Vector Machine",
    "lrn": "Local Response Normalization",
    "nms": "Non-Maximum Suppression",
    "gap": "Global Average Pooling",
    "map": "Mean Average Precision",
}

# Matches "<Some Capitalised Words> (ACR)" — up to 6 words before the parenthesised acronym.
_ACRONYM_DEF_RE = re.compile(r"((?:[A-Za-z][A-Za-z0-9\-]*\s+){1,6})\(([A-Za-z][A-Za-z0-9\-]{1,9})\)")

def extract_acronyms_from_texts(texts) -> dict:
    """
    Auto-build an acronym→full-name map from the corpus by capturing each paper's own
    "Full Name (ACR)" definitions (e.g. "Batch Normalization (BN)"). Validated by matching the
    acronym's capital letters to the initials of the trailing words — so it uses the paper's own,
    contextually-correct definition rather than a global guess. Domain-agnostic.
    """
    found = {}
    for text in texts:
        for m in _ACRONYM_DEF_RE.finditer(text):
            words = m.group(1).split()
            acr = m.group(2).strip()
            caps = "".join(c for c in acr if c.isupper())
            if len(caps) < 2 or len(words) < len(caps):
                continue
            trailing = words[-len(caps):]
            if "".join(w[0].upper() for w in trailing) == caps:
                found.setdefault(acr.lower(), " ".join(trailing))
    return found

def register_acronyms(auto: dict):
    """Merge auto-extracted acronyms into KNOWN_ACRONYMS (seed wins on conflict)."""
    added = 0
    for acr, full in auto.items():
        if acr not in KNOWN_ACRONYMS:
            KNOWN_ACRONYMS[acr] = full
            added += 1
    if added:
        print(f"[Acronyms] Auto-extracted {added} acronym definition(s) from the corpus text.")

def _normalize_base(name: str) -> str:
    """Lowercase, punctuation-stripped, whitespace-collapsed key."""
    norm = _PUNCT_RE.sub("", name).lower()
    return _WS_RE.sub(" ", norm).strip()

def _normalize(name: str) -> str:
    """
    Normalization key with acronym expansion. The expansion is itself normalized so an
    acronym ("BN") and its written-out form ("Batch Normalization") collapse to the SAME
    key ("batch normalization"). Previously they did not — the raw dict value kept its
    capitalization and never matched the lowercased full-name key, so the fast-path silently
    failed and acronyms fell through to the embedding noise zone.
    """
    norm = _normalize_base(name)
    if norm in KNOWN_ACRONYMS:
        return _normalize_base(KNOWN_ACRONYMS[norm])
    return norm

DEDUP_CHECKPOINT_PATH = "data/processed/checkpoint_dedup.json"
SIMILARITY_DEBUG_PATH = "data/processed/similarity_debug.csv"

# ──────────────────────────────────────────
# Similarity Debug Logger
# ──────────────────────────────────────────
class SimilarityDebugLogger:
    """
    Logs ALL similarity scores (not just those above threshold) to a CSV file.
    Purpose: diagnostic only, no logic changes.
    """
    def __init__(self, path: str):
        self.path = path
        self._init_file()

    def _init_file(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "entity_baru",
                "entity_kanonik",
                "similarity_score",
                "tipe_entity_baru",
                "tipe_entity_kanonik",   # Should always == tipe_entity_baru if namespace is correct
                "above_threshold",
                "llm_decision"           # filled in later if LLM was called
            ])

    def log(self, entity_baru: str, entity_kanonik: str, similarity: float,
            tipe_baru: str, tipe_kanonik: str, above_threshold: bool, llm_decision: str = ""):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                entity_baru,
                entity_kanonik,
                f"{similarity:.6f}",
                tipe_baru,
                tipe_kanonik,
                above_threshold,
                llm_decision
            ])


# ──────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────
def _save_dedup_checkpoint(master_list_serializable: dict, name_mapping_global: dict):
    """Save dedup progress to disk."""
    os.makedirs("data/processed", exist_ok=True)
    with open(DEDUP_CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "master_list": master_list_serializable,
            "name_mapping": name_mapping_global
        }, f, ensure_ascii=False)


def _load_dedup_checkpoint():
    """Load dedup checkpoint if it exists."""
    if not os.path.exists(DEDUP_CHECKPOINT_PATH):
        return None, None
    with open(DEDUP_CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("master_list", {}), data.get("name_mapping", {})


# ──────────────────────────────────────────
# Deduplicator
# ──────────────────────────────────────────
class HybridDeduplicator:
    def __init__(self, llm_client: GeminiClient, embed_client: EmbeddingClient):
        self.llm = llm_client
        self.embed = embed_client
        self.master_list: Dict[str, Dict[str, Dict]] = {}
        # normalized_index: { type: { normalized_key: canonical_name } }
        # Used for case-insensitive exact match BEFORE embedding computation
        self.normalized_index: Dict[str, Dict[str, str]] = {}
        self.name_mapping_global: Dict[str, str] = {}
        self.debug_logger = SimilarityDebugLogger(SIMILARITY_DEBUG_PATH)
        print(f"[Debug] Similarity scores will be logged to: {SIMILARITY_DEBUG_PATH}")

    def resolve_entity(self, candidate: Entity) -> str:
        """
        Takes a candidate entity and returns its canonical name after hybrid deduplication.
        Step 0 (NEW): case-insensitive normalization match — skip embedding + LLM entirely.
        Steps 1-3 are UNCHANGED from original.
        """
        ent_type = candidate.type
        cand_name = candidate.name

        if ent_type not in self.master_list:
            self.master_list[ent_type] = {}
        if ent_type not in self.normalized_index:
            self.normalized_index[ent_type] = {}

        # 0. Fast lookup: already seen this exact name
        if cand_name in self.name_mapping_global:
            return self.name_mapping_global[cand_name]

        # 1. Exact match (case-sensitive)
        if cand_name in self.master_list[ent_type]:
            self.name_mapping_global[cand_name] = cand_name
            return cand_name

        # 2. Case-insensitive + punctuation-stripped normalization match (NEW — before embedding)
        cand_norm = _normalize(cand_name)
        if cand_norm in self.normalized_index[ent_type]:
            canonical = self.normalized_index[ent_type][cand_norm]
            print(f"  [Norm] '{cand_name}' -> '{canonical}' (case/punct match, no embedding/LLM)")
            self.name_mapping_global[cand_name] = canonical
            return canonical

        # 3. Compute embedding similarity against same-type entities
        cand_emb = self.embed.embed_text(cand_name)

        candidates_to_check = []
        for canonical_name, data in self.master_list[ent_type].items():
            sim = self.embed.cosine_similarity(cand_emb, data["embedding"])
            above = sim >= DEDUP_SIMILARITY_THRESHOLD

            # LOG EVERY comparison (diagnostic — no filtering here)
            # tipe_kanonik is always ent_type because we iterate within same type namespace
            self.debug_logger.log(
                entity_baru=cand_name,
                entity_kanonik=canonical_name,
                similarity=sim,
                tipe_baru=ent_type,
                tipe_kanonik=ent_type,  # namespace is per type — if this were different, it's a bug
                above_threshold=above,
                llm_decision=""  # will be filled by LLM call below
            )

            if above:
                candidates_to_check.append((sim, canonical_name))

        # Sort candidates by similarity descending (for transparency)
        candidates_to_check.sort(reverse=True)
        candidates_names = [name for _, name in candidates_to_check]

        # 3. LLM Confirmation (UNCHANGED LOGIC)
        llm_result = ""
        if candidates_names:
            print(f"Dedup candidate: '{cand_name}' vs {candidates_names}")
            decision = self.llm.resolve_dedup(cand_name, ent_type, candidates_names)
            llm_result = f"is_duplicate={decision.is_duplicate}, merged_into={decision.merged_name}"

            # Back-fill LLM result into the last N rows of the debug log
            # (approximate — we re-open and patch the last len(candidates_names) rows)
            self._backfill_llm_decision(cand_name, candidates_names, llm_result)

            if decision.is_duplicate and decision.merged_name in self.master_list[ent_type]:
                print(f" -> LLM merged '{cand_name}' into '{decision.merged_name}'")
                self.name_mapping_global[cand_name] = decision.merged_name
                return decision.merged_name
            else:
                print(f" -> LLM decided '{cand_name}' is NEW.")

        # Not a duplicate — add to master list AND register normalized key
        self.master_list[ent_type][cand_name] = {"embedding": cand_emb}
        self.normalized_index[ent_type][cand_norm] = cand_name
        self.name_mapping_global[cand_name] = cand_name
        return cand_name

    def _backfill_llm_decision(self, entity_baru: str, candidates: list, llm_result: str):
        """Rewrite the llm_decision column for the rows matching this entity comparison."""
        try:
            rows = []
            with open(SIMILARITY_DEBUG_PATH, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)

            candidate_set = set(candidates)
            for row in reversed(rows[1:]):  # skip header, scan from bottom
                if row[0] == entity_baru and row[1] in candidate_set and row[6] == "":
                    row[6] = llm_result
                    candidate_set.discard(row[1])
                if not candidate_set:
                    break

            with open(SIMILARITY_DEBUG_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
        except Exception:
            pass  # Best-effort; don't let logging break main flow

    def _master_list_to_serializable(self) -> dict:
        serializable = {}
        for ent_type, entities in self.master_list.items():
            serializable[ent_type] = {}
            for name, data in entities.items():
                emb = data["embedding"]
                serializable[ent_type][name] = {
                    "embedding": emb.tolist() if hasattr(emb, "tolist") else list(emb)
                }
        return serializable

    def process_extracted_knowledge(self, knowledge_list: List, resume_from_idx: int = 0) -> List:
        """
        Iterates over extracted knowledge and deduplicates entities in-place.
        LOGIC UNCHANGED — only checkpoint saves and debug logging added.
        """
        for idx, k in enumerate(knowledge_list):
            if idx < resume_from_idx:
                continue

            chunk_name_mapping = {}

            for ent in k.entities:
                canonical = self.resolve_entity(ent)
                chunk_name_mapping[ent.name] = canonical
                ent.name = canonical

            for rel in k.relationships:
                if rel.source_entity in self.name_mapping_global:
                    rel.source_entity = self.name_mapping_global[rel.source_entity]
                if rel.target_entity in self.name_mapping_global:
                    rel.target_entity = self.name_mapping_global[rel.target_entity]

            # Save checkpoint after each knowledge item
            _save_dedup_checkpoint(
                self._master_list_to_serializable(),
                self.name_mapping_global
            )
            print(f"  [Dedup] {idx + 1}/{len(knowledge_list)} knowledge items processed.")

        return knowledge_list
