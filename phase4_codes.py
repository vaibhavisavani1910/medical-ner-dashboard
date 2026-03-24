"""Phase 4: Look up ICD-10, RxNorm, and SNOMED codes for top entities via NLM APIs."""

import asyncio
import json
import sys
import urllib.parse

import httpx

from config import settings
from models import CodedEntity, DashboardData, EntityCount


NLM_ICD10_URL = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
NLM_HCPCS_URL = "https://clinicaltables.nlm.nih.gov/api/hcpcs/v3/search"
RXNORM_SEARCH_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
RXNORM_PROP_URL = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/property.json"

TIMEOUT = 15.0
MAX_CONCURRENT = 10  # Polite concurrency limit for public NLM APIs


async def lookup_icd10(client: httpx.AsyncClient, entity: str) -> tuple[str | None, str | None]:
    """Return (code, description) from ICD-10-CM search or (None, None)."""
    try:
        params = {"sf": "code,name", "terms": entity, "maxList": 1}
        resp = await client.get(NLM_ICD10_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # Response format: [total, [codes], null, [display_strings]]
        # display_strings is a list of [code, name] pairs
        if (
            isinstance(data, list)
            and len(data) >= 4
            and data[3]
            and len(data[3]) > 0
        ):
            entry = data[3][0]  # First result: [code, name]
            code = entry[0] if len(entry) > 0 else None
            desc = entry[1] if len(entry) > 1 else None
            return code, desc
    except Exception:
        pass
    return None, None


async def lookup_hcpcs(client: httpx.AsyncClient, entity: str) -> tuple[str | None, str | None]:
    """Return (code, description) from HCPCS search or (None, None)."""
    try:
        params = {"terms": entity, "maxList": 1}
        resp = await client.get(NLM_HCPCS_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if (
            isinstance(data, list)
            and len(data) >= 4
            and data[3]
            and len(data[3]) > 0
        ):
            entry = data[3][0]
            code = entry[0] if len(entry) > 0 else None
            desc = entry[1] if len(entry) > 1 else None
            return code, desc
    except Exception:
        pass
    return None, None


async def lookup_rxnorm(client: httpx.AsyncClient, entity: str) -> tuple[str | None, str | None]:
    """Return (rxcui, name) from RxNorm search or (None, None)."""
    try:
        params = {"name": entity}
        resp = await client.get(RXNORM_SEARCH_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        rxcui = (
            data.get("idGroup", {}).get("rxnormId", [None])[0]
            if data.get("idGroup", {}).get("rxnormId")
            else None
        )
        if not rxcui:
            return None, None

        # Fetch the preferred name for this RXCUI.
        prop_url = RXNORM_PROP_URL.format(rxcui=rxcui)
        prop_resp = await client.get(
            prop_url,
            params={"propName": "RxNorm Name"},
            timeout=TIMEOUT,
        )
        prop_resp.raise_for_status()
        prop_data = prop_resp.json()
        prop_concept = prop_data.get("propConceptGroup", {})
        props = prop_concept.get("propConcept", []) if prop_concept else []
        name = props[0].get("propValue") if props else None
        return rxcui, name
    except Exception:
        pass
    return None, None


async def code_entity(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    category: str,
    entity: EntityCount,
) -> CodedEntity:
    async with semaphore:
        if category in ("conditions", "symptoms"):
            code, desc = await lookup_icd10(client, entity.name)
            code_system = "ICD-10-CM"
        elif category == "medications":
            code, desc = await lookup_rxnorm(client, entity.name)
            code_system = "RxNorm"
        else:  # procedures
            code, desc = await lookup_hcpcs(client, entity.name)
            code_system = "HCPCS"

        return CodedEntity(
            name=entity.name,
            count=entity.count,
            code=code,
            code_system=code_system,
            code_description=desc,
        )


async def lookup_all_codes(aggregated: dict[str, list[dict]]) -> DashboardData:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async with httpx.AsyncClient() as client:
        tasks: dict[str, list[asyncio.Task]] = {}
        for category, entities in aggregated.items():
            entity_counts = [EntityCount(**e) for e in entities]
            tasks[category] = [
                asyncio.create_task(code_entity(semaphore, client, category, ec))
                for ec in entity_counts
            ]

        results: dict[str, list[CodedEntity]] = {}
        for category, category_tasks in tasks.items():
            coded = await asyncio.gather(*category_tasks)
            results[category] = list(coded)
            found = sum(1 for c in coded if c.code)
            print(f"  {category}: {found}/{len(coded)} codes found")

    return DashboardData(**results)


def run_code_lookup() -> None:
    agg_path = settings.data_dir / "aggregated.json"
    output_path = settings.data_dir / "coded_entities.json"

    if not agg_path.exists():
        print("ERROR: data/aggregated.json not found. Run phase3_aggregate.py first.", file=sys.stderr)
        sys.exit(1)

    with open(agg_path, encoding="utf-8") as fh:
        aggregated = json.load(fh)

    print("Looking up standardized codes via NLM APIs...")
    dashboard_data = asyncio.run(lookup_all_codes(aggregated))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(dashboard_data.model_dump(), fh, indent=2, ensure_ascii=False)

    print(f"\nCoded entities saved to {output_path}")


if __name__ == "__main__":
    run_code_lookup()
