# Medical NER Dashboard

An end-to-end pipeline that extracts medical named entities from 1,000 clinical records in the **ncbi/Open-Patients** dataset and visualises them in an interactive Dash dashboard with standardised codes (ICD-10-CM, RxNorm, SNOMED CT).

## Architecture

```
phase1_load.py      → data/raw_records.json        (1,000 raw records)
phase2_ner.py       → data/ner_results.json         (NER via GPT-4o-mini, resumable)
phase3_aggregate.py → data/aggregated.json          (top 10 per category)
phase4_codes.py     → data/coded_entities.json      (entities + NLM codes)
dashboard.py        → http://localhost:8050          (interactive Dash app)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

`.env` format:
```
OPENAI_API_KEY=sk-...
MODEL=gpt-4o-mini
BATCH_SIZE=10
MAX_RECORDS=1000
```

### 3. Run the pipeline

```bash
python phase1_load.py       # Download & save 1,000 records  (~30 seconds)
python phase2_ner.py        # NER extraction via OpenAI       (~15-30 minutes, resumable)
python phase3_aggregate.py  # Count & rank entities           (~5 seconds)
python phase4_codes.py      # Fetch NLM codes (async)         (~1-2 minutes)
python dashboard.py         # Launch dashboard
```

Open **http://localhost:8050** in your browser.

## Dashboard Features

- **2×2 grid** of panels: Conditions, Symptoms, Medications, Procedures
- **Horizontal bar charts** (Plotly) showing top-10 entity frequencies
- **Data tables** with: Rank | Entity | Count | Code | Code System | Description
- **Summary badges** showing total mentions and coding coverage per category
- **Dark theme** (Dash Bootstrap DARKLY)

## Resumable NER Pipeline

`phase2_ner.py` checkpoints after every batch of 10 records. If the script is interrupted, re-running it resumes from where it left off — no duplicate API calls.

## Code Systems

| Category   | Code System | API                              |
|------------|-------------|----------------------------------|
| Conditions | ICD-10-CM   | NLM ClinicalTables               |
| Symptoms   | ICD-10-CM   | NLM ClinicalTables               |
| Medications| RxNorm      | NLM RxNav                        |
| Procedures | SNOMED CT   | NLM ClinicalTables               |

## Project Structure

```
medical-ner-dashboard/
├── .env                    ← your secrets (not committed)
├── .env.example            ← template
├── requirements.txt
├── config.py               ← pydantic-settings configuration
├── models.py               ← Pydantic v2 data models
├── phase1_load.py          ← HuggingFace dataset loader
├── phase2_ner.py           ← OpenAI NER + checkpointing
├── phase3_aggregate.py     ← frequency aggregation
├── phase4_codes.py         ← async NLM code lookup
├── dashboard.py            ← Dash app
└── data/
    ├── raw_records.json
    ├── ner_results.json
    ├── aggregated.json
    └── coded_entities.json
```

## Notes

- Phase 2 costs approximately **$1–3 USD** in OpenAI API usage for 1,000 records with `gpt-4o-mini`.
- Phase 4 uses only **free public NLM APIs** — no API key required.
- All NLM API calls are rate-limited to 10 concurrent requests to be a polite API consumer.
