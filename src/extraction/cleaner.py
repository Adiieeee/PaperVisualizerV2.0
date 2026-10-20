import re
from typing import List

from src.config import (
    Entity,
    Relationship,
    ExtractedKnowledge,
    GENERIC_ENTITY_BLOCKLIST,
    MIN_ENTITY_NAME_LENGTH,
)
from src.security.sanitize import sanitize_entity_name

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(name: str) -> str:
    """Lowercase + punctuation-stripped key, whitespace collapsed."""
    norm = _PUNCT_RE.sub("", name).lower().strip()
    return re.sub(r"\s+", " ", norm)


def clean_knowledge_list(knowledge_list: List[ExtractedKnowledge]) -> List[ExtractedKnowledge]:
    """
    Post-extraction validation pass. Runs BEFORE dedup and mutates in place.

    This is a graph-quality step only — it operates on extracted entities, NOT on
    the raw text chunks, so the vector store (chunks + community summaries) is
    completely untouched.

    Removes:
      - entities whose normalized name is in GENERIC_ENTITY_BLOCKLIST
      - entities whose normalized name is shorter than MIN_ENTITY_NAME_LENGTH
      - entities rejected by sanitize_entity_name (markup `<`/`>`, URLs, abnormal length) — LLM
        output derived from untrusted PDFs; control characters/newlines are cleaned instead
      - relationships that reference a removed entity, or that are self-loops

    Note: entity/relationship *type* validity is already guaranteed upstream by the
    Pydantic Literal schema (structured output), so it is not re-checked here.
    """
    ent_before = ent_after = 0
    rel_before = rel_after = 0
    unsafe = 0

    for k in knowledge_list:
        kept_entities = []
        removed_names = set()
        rename = {}  # original name -> sanitized name (relationships still use the original)
        for ent in k.entities:
            ent_before += 1
            original = ent.name
            safe = sanitize_entity_name(original or "")
            if safe is None:
                removed_names.add(original)
                unsafe += 1
                continue
            if safe != original:
                rename[original] = safe
                ent.name = safe
            norm = _normalize(ent.name)
            if len(norm) < MIN_ENTITY_NAME_LENGTH or norm in GENERIC_ENTITY_BLOCKLIST:
                removed_names.add(original)
                continue
            kept_entities.append(ent)
        k.entities = kept_entities
        ent_after += len(kept_entities)

        kept_rels = []
        for rel in k.relationships:
            rel_before += 1
            if rel.source_entity in removed_names or rel.target_entity in removed_names:
                continue
            rel.source_entity = rename.get(rel.source_entity, rel.source_entity)
            rel.target_entity = rename.get(rel.target_entity, rel.target_entity)
            if rel.source_entity == rel.target_entity:  # self-loop
                continue
            kept_rels.append(rel)
        k.relationships = kept_rels
        rel_after += len(kept_rels)

    print(
        f"[Cleaner] Entities: {ent_before} -> {ent_after} (-{ent_before - ent_after}). "
        f"Relationships: {rel_before} -> {rel_after} (-{rel_before - rel_after})."
        + (f" Rejected {unsafe} unsafe entity name(s)." if unsafe else "")
    )
    return knowledge_list


# --- Author preprocessing -------------------------------------------------------------

_ETAL_RE = re.compile(r"\s*,?\s*et\.?\s*al\.?", re.IGNORECASE)
_YEAR_RE = re.compile(r"[,;]?\s*\(?\b(?:19|20)\d{2}[a-z]?\)?")
_AUTHOR_SPLIT_RE = re.compile(r"\s+&\s+|\s+and\s+", re.IGNORECASE)


def _clean_author_name(name: str) -> str:
    n = _ETAL_RE.sub("", name or "")
    n = _YEAR_RE.sub("", n)
    return n.strip().strip(".,;:&")


def _split_author(name: str) -> List[str]:
    parts = [p.strip().strip(".,;:&") for p in _AUTHOR_SPLIT_RE.split(name)]
    return [p for p in parts if len(p) >= MIN_ENTITY_NAME_LENGTH]


def preprocess_authors(knowledge_list: List[ExtractedKnowledge]) -> List[ExtractedKnowledge]:
    """
    Mechanical safety net for Author entities (graph-only; vector store untouched).
    Complements the extraction prompt rules by:
      - stripping " et al." and citation years,
      - splitting combined authors ("He & Zhang" -> "He", "Zhang"),
      - duplicating any relationship referencing a combined author to each split name.
    Runs BEFORE dedup so split names get merged across papers.
    """
    split_count = 0
    for k in knowledge_list:
        author_map = {}  # original author name -> [clean split names]
        new_entities = []
        for ent in k.entities:
            if ent.type == "Author":
                parts = _split_author(_clean_author_name(ent.name))
                if not parts:
                    continue  # nothing usable left after cleaning
                if len(parts) > 1:
                    split_count += 1
                author_map[ent.name] = parts
                for p in parts:
                    new_entities.append(Entity(name=p, type="Author", description=ent.description))
            else:
                new_entities.append(ent)
        k.entities = new_entities

        new_rels = []
        for rel in k.relationships:
            srcs = author_map.get(rel.source_entity, [rel.source_entity])
            tgts = author_map.get(rel.target_entity, [rel.target_entity])
            for s in srcs:
                for t in tgts:
                    new_rels.append(Relationship(
                        source_entity=s, target_entity=t,
                        relation_type=rel.relation_type, description=rel.description,
                    ))
        k.relationships = new_rels

    if split_count:
        print(f"[Authors] Split {split_count} combined author entities into individuals.")
    return knowledge_list
