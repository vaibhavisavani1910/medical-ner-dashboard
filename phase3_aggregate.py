"""Phase 3: Aggregate NER results and extract top 10 entities per category."""

import json
import sys
from collections import Counter

from config import settings
from models import EntityCount, ExtractedEntities


CATEGORIES = ("conditions", "symptoms", "medications", "procedures")
TOP_N = 10


def aggregate() -> None:
    ner_path = settings.data_dir / "ner_results.json"
    output_path = settings.data_dir / "aggregated.json"

    if not ner_path.exists():
        print("ERROR: data/ner_results.json not found. Run phase2_ner.py first.", file=sys.stderr)
        sys.exit(1)

    with open(ner_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    records = [ExtractedEntities(**item) for item in raw]
    successful = [r for r in records if not r.error]
    print(f"Loaded {len(records)} NER records ({len(successful)} successful, {len(records) - len(successful)} errors)")

    counters: dict[str, Counter] = {cat: Counter() for cat in CATEGORIES}

    for record in successful:
        for cat in CATEGORIES:
            entities: list[str] = getattr(record, cat)
            for entity in entities:
                normalized = entity.lower().strip()
                if normalized:
                    counters[cat][normalized] += 1

    result: dict[str, list[dict]] = {}
    for cat in CATEGORIES:
        top = counters[cat].most_common(TOP_N)
        result[cat] = [EntityCount(name=name, count=count).model_dump() for name, count in top]
        print(f"  {cat}: {len(top)} top entities (total unique: {len(counters[cat])})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print(f"\nAggregated top-{TOP_N} entities saved to {output_path}")


if __name__ == "__main__":
    aggregate()
