"""Build the document corpus from NER pipeline data files."""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from .models import Document

logger = logging.getLogger(__name__)

CATEGORY_META = {
    "conditions":  {"label": "Conditions",  "code_system": "ICD-10-CM"},
    "symptoms":    {"label": "Symptoms",     "code_system": "ICD-10-CM"},
    "medications": {"label": "Medications",  "code_system": "RxNorm"},
    "procedures":  {"label": "Procedures",   "code_system": "HCPCS"},
}


# ─── Loaders ─────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_data(data_dir: Path) -> tuple[dict, list, dict]:
    """Return (coded_entities, ner_records, aggregated)."""
    coded     = _load_json(data_dir / "coded_entities.json")
    ner       = _load_json(data_dir / "ner_results.json")
    aggregated = _load_json(data_dir / "aggregated.json")
    return coded, ner, aggregated


# ─── Document builders ───────────────────────────────────────────────────────

def _build_overview_document(coded: dict, ner: list) -> Document:
    total_records = len(ner)
    error_count   = sum(1 for r in ner if r.get("error"))
    good_records  = total_records - error_count

    lines = [
        "MedNER Dataset Overview",
        f"Total clinical records analyzed: {total_records:,}",
        f"Successfully processed: {good_records:,} | Errors: {error_count}",
        "Extraction model: GPT-4o-mini",
        "Code systems: ICD-10-CM (conditions & symptoms), RxNorm (medications), HCPCS (procedures)",
        "",
        "Entity counts per category:",
    ]
    for cat, meta in CATEGORY_META.items():
        entities  = coded.get(cat, [])
        coded_cnt = sum(1 for e in entities if e.get("code"))
        lines.append(
            f"  {meta['label']}: {len(entities)} top entities shown, "
            f"{coded_cnt}/{len(entities)} have {meta['code_system']} codes"
        )

    return Document(id="overview", content="\n".join(lines), category="overview")


def _build_category_document(category: str, entities: list[dict]) -> Document:
    meta  = CATEGORY_META[category]
    total = sum(e["count"] for e in entities)
    lines = [
        f"Top {meta['label']} in the MedNER dataset",
        f"Total mentions across all records: {total:,}",
        f"Code system: {meta['code_system']}",
        "",
    ]
    for i, e in enumerate(entities, 1):
        code_str = (
            f"{e['code_system']}: {e['code']} — {e['code_description']}"
            if e.get("code") else "no code assigned"
        )
        lines.append(f"{i}. {e['name'].title()} — {e['count']:,} records — {code_str}")

    return Document(
        id=f"category_{category}",
        content="\n".join(lines),
        category=category,  # type: ignore[arg-type]
        metadata={"total_mentions": total, "entity_count": len(entities)},
    )


def _build_entity_documents(category: str, entities: list[dict]) -> list[Document]:
    docs = []
    for e in entities:
        code_line = (
            f"Standardized code: {e['code']} ({e['code_system']}) — {e['code_description']}"
            if e.get("code")
            else f"No {CATEGORY_META[category]['code_system']} code found for this entity."
        )
        content = (
            f"Entity: {e['name'].title()}\n"
            f"Category: {CATEGORY_META[category]['label']}\n"
            f"Frequency: appears in {e['count']:,} out of 1,000 clinical records\n"
            f"{code_line}"
        )
        docs.append(Document(
            id=f"entity_{category}_{e['name'].replace(' ', '_')}",
            content=content,
            category=category,  # type: ignore[arg-type]
            metadata={"name": e["name"], "count": e["count"], "code": e.get("code")},
        ))
    return docs


def _build_cooccurrence_documents(
    ner_records: list[dict],
    top_entities: dict[str, list[str]],
) -> list[Document]:
    """
    For each top condition, compute which symptoms, medications, and procedures
    most frequently appear in the same record.
    """
    docs: list[Document] = []

    for condition in top_entities.get("conditions", []):
        symptom_counter:   Counter = Counter()
        med_counter:       Counter = Counter()
        procedure_counter: Counter = Counter()
        record_count = 0

        for rec in ner_records:
            if rec.get("error") or condition not in rec.get("conditions", []):
                continue
            record_count += 1
            symptom_counter.update(rec.get("symptoms", []))
            med_counter.update(rec.get("medications", []))
            procedure_counter.update(rec.get("procedures", []))

        if record_count == 0:
            continue

        def _top5(counter: Counter) -> str:
            if not counter:
                return "none recorded"
            return ", ".join(f"{name.title()} ({cnt})" for name, cnt in counter.most_common(5))

        content = (
            f"Co-occurrence analysis for: {condition.title()}\n"
            f"Appears in {record_count:,} records.\n"
            f"Top co-occurring symptoms: {_top5(symptom_counter)}\n"
            f"Top co-occurring medications: {_top5(med_counter)}\n"
            f"Top co-occurring procedures: {_top5(procedure_counter)}"
        )
        docs.append(Document(
            id=f"cooc_condition_{condition.replace(' ', '_')}",
            content=content,
            category="cooccurrence",
            metadata={"anchor": condition, "record_count": record_count},
        ))

    return docs


def _build_coding_systems_document() -> Document:
    content = (
        "Medical coding systems used in MedNER Dashboard\n\n"
        "ICD-10-CM (International Classification of Diseases, 10th Revision, Clinical Modification)\n"
        "  Used for: Conditions and Symptoms\n"
        "  Source: NLM Clinical Tables API\n"
        "  Example: I15.0 = Renovascular hypertension, U07.1 = COVID-19\n\n"
        "RxNorm\n"
        "  Used for: Medications\n"
        "  Source: NLM RxNav API\n"
        "  Provides normalized names and concept IDs for drugs\n\n"
        "HCPCS (Healthcare Common Procedure Coding System)\n"
        "  Used for: Procedures\n"
        "  Source: NLM Clinical Tables API\n"
        "  Example: G6019 = Colonoscopy lesion removal"
    )
    return Document(id="coding_systems", content=content, category="overview")


# ─── Public entry point ───────────────────────────────────────────────────────

def build_all_documents(data_dir: Path) -> list[Document]:
    """Build the full document corpus from data files."""
    logger.info("Loading data files from %s", data_dir)
    coded, ner_records, aggregated = _load_data(data_dir)

    top_entities: dict[str, list[str]] = {
        cat: [e["name"] for e in aggregated.get(cat, [])]
        for cat in CATEGORY_META
    }

    docs: list[Document] = []

    # 1. Dataset overview
    docs.append(_build_overview_document(coded, ner_records))

    # 2. Category summaries
    for cat, entities in coded.items():
        if cat in CATEGORY_META:
            docs.append(_build_category_document(cat, entities))

    # 3. Per-entity detail
    for cat, entities in coded.items():
        if cat in CATEGORY_META:
            docs.extend(_build_entity_documents(cat, entities))

    # 4. Co-occurrence patterns
    docs.extend(_build_cooccurrence_documents(ner_records, top_entities))

    # 5. Coding systems reference
    docs.append(_build_coding_systems_document())

    logger.info("Built %d documents for indexing", len(docs))
    return docs
