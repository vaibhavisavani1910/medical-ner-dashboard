"""FastAPI backend for the Medical NER Dashboard (with RAG chatbot)."""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from phase2_ner import SYSTEM_PROMPT, parse_llm_response
from openai import OpenAI
from pydantic import BaseModel

from rag import ChatRequest, ChatResponse, IndexStatus, RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR   = Path("data")
DB_DIR     = DATA_DIR / "chroma_db"
STATIC_DIR = Path("static")

# ─── App lifespan ────────────────────────────────────────────────────────────

_rag: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag
    logger.info("Initialising RAG pipeline…")
    _rag = RAGPipeline(
        data_dir=DATA_DIR,
        db_dir=DB_DIR,
        openai_api_key=settings.openai_api_key,
        model=settings.model,
    )
    if not _rag.is_indexed():
        logger.info("Index empty — building now (embeddings via OpenAI)…")
        count = _rag.build_index()
        logger.info("Indexed %d documents.", count)
    else:
        logger.info("Index already populated (%d docs). Skipping rebuild.", _rag._col.count())
    yield


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="Medical NER API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_rag() -> RAGPipeline:
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready yet.")
    return _rag


# ─── Dashboard data ───────────────────────────────────────────────────────────

@app.get("/api/data", summary="Return coded entities for all categories.")
def get_dashboard_data() -> dict:
    path = DATA_DIR / "coded_entities.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run the pipeline phases first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── Live NER ─────────────────────────────────────────────────────────────────

class NERRequest(BaseModel):
    text: str


@app.post("/api/ner", summary="Extract medical entities from free text.")
def analyze_text(req: NERRequest) -> dict:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    client = OpenAI(api_key=settings.openai_api_key)
    prompt = f"[record_id: live_input]\n{text[:3000]}"

    try:
        response = client.chat.completions.create(
            model=settings.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_completion_tokens=1024,
        )
        raw    = response.choices[0].message.content or ""
        parsed = parse_llm_response(raw, ["live_input"])
        result = parsed[0]

        if result.error:
            raise HTTPException(status_code=500, detail=f"NER error: {result.error}")

        return {
            "conditions":  result.conditions,
            "symptoms":    result.symptoms,
            "medications": result.medications,
            "procedures":  result.procedures,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("NER failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─── Chat (RAG) ───────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse, summary="RAG-powered chatbot.")
def chat(req: ChatRequest) -> ChatResponse:
    rag = _get_rag()
    try:
        return rag.chat(req)
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/chat/status", response_model=IndexStatus, summary="Vector index status.")
def chat_status() -> IndexStatus:
    return _get_rag().index_status()


@app.post("/api/chat/rebuild", response_model=IndexStatus, summary="Rebuild the vector index.")
def rebuild_index() -> IndexStatus:
    rag = _get_rag()
    rag.build_index()
    return rag.index_status()


# ─── Static frontend (last — must not shadow API routes) ─────────────────────

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
