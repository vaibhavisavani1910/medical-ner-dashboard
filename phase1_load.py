"""Phase 1: Load the first 1,000 records from ncbi/Open-Patients and save to data/raw_records.json."""

import json
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from config import settings
from models import RawRecord


def load_and_save() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.data_dir / "raw_records.json"

    print(f"Loading ncbi/Open-Patients dataset (first {settings.max_records} records)...")
    ds = load_dataset("ncbi/Open-Patients", split="train", streaming=True)

    records: list[dict] = []
    for i, row in enumerate(ds):
        if len(records) >= settings.max_records:
            break

        # Build a stable record ID — use the dataset's own id field if present,
        # otherwise fall back to the zero-padded row index.
        raw_id = str(row.get("_id", "") or row.get("id", "") or row.get("case_id", "") or "").strip()
        record_id = raw_id if raw_id else f"rec_{i:04d}"

        # The text field may be named differently across dataset versions.
        text = (
            str(row.get("description", "") or row.get("text", "") or row.get("note", "") or row.get("clinical_note", "") or "")
            .strip()
        )

        if not text:
            continue  # skip empty rows without counting toward the limit

        record = RawRecord(id=record_id, text=text)
        records.append(record.model_dump())

        if len(records) % 100 == 0:
            print(f"  Loaded {len(records)}/{settings.max_records} records...")

    if not records:
        print("ERROR: No records were loaded. Check the dataset field names.", file=sys.stderr)
        sys.exit(1)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(records)} records to {output_path}")


if __name__ == "__main__":
    load_and_save()
