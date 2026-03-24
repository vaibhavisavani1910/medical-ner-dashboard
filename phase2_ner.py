"""Phase 2: NER via OpenAI gpt-4o-mini with batch processing and checkpoint/resume support."""

import json
import sys
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from config import settings
from models import ExtractedEntities, RawRecord

SYSTEM_PROMPT = """You are a clinical NLP expert. Extract medical named entities from each clinical note below.

Rules:
- EXCLUDE negated findings: "denies X", "no X", "rules out X", "without X"
- EXCLUDE family history items
- EXCLUDE uncertain findings: "possible X", "suspected X"
- NORMALIZE synonyms: SOB→dyspnea, HTN→hypertension, DM→diabetes mellitus, MI→myocardial infarction
- Use lowercase for all entities

Return ONLY a valid JSON array with no markdown, no preamble, no commentary — one object per record:
[
  {
    "record_id": "...",
    "conditions": ["..."],
    "symptoms": ["..."],
    "medications": ["..."],
    "procedures": ["..."]
  }
]"""


def build_user_prompt(batch: list[RawRecord]) -> str:
    lines = []
    for rec in batch:
        # Truncate very long notes to stay within token limits.
        text = rec.text[:2000] if len(rec.text) > 2000 else rec.text
        lines.append(f'[record_id: {rec.id}]\n{text}')
    return "\n\n---\n\n".join(lines)


def parse_llm_response(raw: str, batch_ids: list[str]) -> list[ExtractedEntities]:
    """Parse the LLM JSON response; return one ExtractedEntities per record in the batch."""
    try:
        data = json.loads(raw.strip())
        if not isinstance(data, list):
            raise ValueError("Response is not a JSON array")
    except (json.JSONDecodeError, ValueError) as exc:
        # Return error placeholders for every record in this batch.
        return [
            ExtractedEntities(record_id=rid, error=f"JSON parse error: {exc}")
            for rid in batch_ids
        ]

    results: list[ExtractedEntities] = []
    returned_ids = {item.get("record_id") for item in data if isinstance(item, dict)}

    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            results.append(ExtractedEntities(**item))
        except Exception as exc:
            rid = item.get("record_id", "unknown")
            results.append(ExtractedEntities(record_id=rid, error=str(exc)))

    # Fill in any batch records that the model silently dropped.
    for rid in batch_ids:
        if rid not in returned_ids:
            results.append(ExtractedEntities(record_id=rid, error="record not returned by model"))

    return results


def load_checkpoint(path: Path) -> dict[str, ExtractedEntities]:
    """Load existing checkpoint; returns a dict keyed by record_id."""
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        checkpoint: dict[str, ExtractedEntities] = {}
        for item in raw:
            entity = ExtractedEntities(**item)
            checkpoint[entity.record_id] = entity
        print(f"Resumed checkpoint: {len(checkpoint)} records already processed.")
        return checkpoint
    except Exception as exc:
        print(f"WARNING: Could not load checkpoint ({exc}). Starting fresh.")
        return {}


def save_checkpoint(path: Path, checkpoint: dict[str, ExtractedEntities]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            [e.model_dump() for e in checkpoint.values()],
            fh,
            indent=2,
            ensure_ascii=False,
        )


def run_ner() -> None:
    raw_path = settings.data_dir / "raw_records.json"
    checkpoint_path = settings.data_dir / "ner_results.json"

    if not raw_path.exists():
        print("ERROR: data/raw_records.json not found. Run phase1_load.py first.", file=sys.stderr)
        sys.exit(1)

    with open(raw_path, encoding="utf-8") as fh:
        raw_data = json.load(fh)

    all_records = [RawRecord(**r) for r in raw_data]
    checkpoint = load_checkpoint(checkpoint_path)

    # Identify records that still need processing.
    pending = [r for r in all_records if r.id not in checkpoint]
    print(f"Records to process: {len(pending)} (total: {len(all_records)})")

    if not pending:
        print("All records already processed. Nothing to do.")
        return

    client = OpenAI(api_key=settings.openai_api_key)
    batch_size = settings.batch_size

    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

    for batch in tqdm(batches, desc="NER batches", unit="batch"):
        batch_ids = [r.id for r in batch]
        user_prompt = build_user_prompt(batch)

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=settings.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_completion_tokens=4096,
                )
                raw_content = response.choices[0].message.content or ""
                break
            except Exception as exc:
                if attempt == 2:
                    # Mark all records in batch as errored.
                    for rid in batch_ids:
                        checkpoint[rid] = ExtractedEntities(
                            record_id=rid, error=f"API error after 3 attempts: {exc}"
                        )
                    save_checkpoint(checkpoint_path, checkpoint)
                    tqdm.write(f"Batch error (giving up): {exc}")
                    raw_content = None
                    break
                wait = 2 ** attempt
                tqdm.write(f"API error (attempt {attempt + 1}): {exc}. Retrying in {wait}s...")
                time.sleep(wait)
        else:
            raw_content = None

        if raw_content is not None:
            parsed = parse_llm_response(raw_content, batch_ids)
            for entity in parsed:
                checkpoint[entity.record_id] = entity

        # Save checkpoint after every batch.
        save_checkpoint(checkpoint_path, checkpoint)

    total_errors = sum(1 for e in checkpoint.values() if e.error)
    print(f"\nNER complete. Total records: {len(checkpoint)}, Errors: {total_errors}")
    print(f"Results saved to {checkpoint_path}")


if __name__ == "__main__":
    run_ner()
