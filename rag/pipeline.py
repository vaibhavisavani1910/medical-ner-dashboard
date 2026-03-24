"""RAG pipeline: vector indexing, retrieval, and generation."""
from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

from .documents import build_all_documents
from .models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Document,
    IndexStatus,
    RetrievedDocument,
    Source,
    TokenUsage,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "medner_v1"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 6  # documents retrieved per query

SYSTEM_PROMPT = textwrap.dedent("""
    You are MedBot, an expert assistant for the MedNER Dashboard — a clinical NLP analytics platform.

    The dashboard has analyzed 1,000 real patient records from the Open Patients Dataset.
    Medical named entities (conditions, symptoms, medications, procedures) were extracted
    using GPT-4o-mini and mapped to standardized codes:
      • Conditions & Symptoms → ICD-10-CM
      • Medications → RxNorm
      • Procedures → HCPCS

    You will be given CONTEXT passages retrieved from the dataset's knowledge base.
    Use them to answer the user's question accurately and concisely.

    Guidelines:
    - Cite specific numbers, percentages, and medical codes when available.
    - Structure answers with bullet points or short paragraphs as appropriate.
    - If the context doesn't contain enough information, say so clearly — do not fabricate.
    - Keep answers focused; avoid unnecessary repetition.
""").strip()


class RAGPipeline:
    """
    End-to-end RAG pipeline:
      1. Index documents into ChromaDB using OpenAI embeddings.
      2. Retrieve semantically relevant chunks for a query.
      3. Generate a grounded answer with GPT-4o-mini.
    """

    def __init__(
        self,
        data_dir: Path,
        db_dir: Path,
        openai_api_key: str,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.data_dir = data_dir
        self.model    = model
        self.llm      = OpenAI(api_key=openai_api_key)

        logger.info("Connecting to ChromaDB at %s", db_dir)
        db_dir.mkdir(parents=True, exist_ok=True)

        ef = OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=EMBEDDING_MODEL,
        )
        self._chroma = chromadb.PersistentClient(path=str(db_dir))
        self._col    = self._chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ─── Indexing ────────────────────────────────────────────────────────────

    def is_indexed(self) -> bool:
        return self._col.count() > 0

    def build_index(self) -> int:
        """
        Build (or rebuild) the vector index from all data files.
        Returns the number of documents indexed.
        """
        logger.info("Building vector index…")
        documents = build_all_documents(self.data_dir)

        def _clean_meta(doc: Document) -> dict:
            """ChromaDB only accepts str/int/float/bool metadata values."""
            raw = {"category": doc.category, **doc.metadata}
            return {k: v for k, v in raw.items() if isinstance(v, (str, int, float, bool))}

        # Upsert in one batch (ChromaDB handles duplicates gracefully)
        self._col.upsert(
            ids=[doc.id for doc in documents],
            documents=[doc.content for doc in documents],
            metadatas=[_clean_meta(doc) for doc in documents],
        )

        count = self._col.count()
        logger.info("Index ready: %d documents in collection '%s'", count, COLLECTION_NAME)
        return count

    def index_status(self) -> IndexStatus:
        count = self._col.count()
        indexed = count > 0
        return IndexStatus(
            indexed=indexed,
            document_count=count,
            collection_name=COLLECTION_NAME,
            message="Index is ready." if indexed else "Index is empty — call /api/chat/rebuild.",
        )

    # ─── Retrieval ───────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[RetrievedDocument]:
        """Embed *query* and return the top-k most relevant documents."""
        results = self._col.query(
            query_texts=[query],
            n_results=min(top_k, self._col.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedDocument] = []
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        for content, meta, dist in zip(docs, metas, distances):
            score = max(0.0, 1.0 - dist)  # cosine distance → similarity
            retrieved.append(RetrievedDocument(
                document=Document(
                    id=str(meta.get("id", "unknown")),
                    content=content,
                    category=meta.get("category", "overview"),  # type: ignore[arg-type]
                    metadata=meta,
                ),
                score=round(score, 4),
            ))

        return retrieved

    # ─── Generation ──────────────────────────────────────────────────────────

    def _build_context_block(self, docs: list[RetrievedDocument]) -> str:
        sections = []
        for i, rd in enumerate(docs, 1):
            sections.append(f"[Source {i} | {rd.document.category}]\n{rd.document.content}")
        return "\n\n---\n\n".join(sections)

    def _build_messages(
        self,
        question: str,
        context: str,
        history: list[ChatMessage],
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Include last 6 turns of history (3 full exchanges)
        for turn in history[-6:]:
            messages.append({"role": turn.role, "content": turn.content})

        # Inject retrieved context alongside the current question
        user_content = (
            f"CONTEXT (retrieved from the MedNER knowledge base):\n"
            f"{context}\n\n"
            f"QUESTION: {question}"
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    # ─── Public interface ────────────────────────────────────────────────────

    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Full RAG pipeline:
          1. Retrieve relevant documents.
          2. Build a grounded prompt.
          3. Generate an answer.
          4. Return answer + citations + token usage.
        """
        retrieved = self.retrieve(request.message)

        if not retrieved:
            # Fallback if index is empty
            return ChatResponse(
                answer=(
                    "The knowledge index appears to be empty. "
                    "Please contact an administrator to rebuild the index."
                ),
                sources=[],
                usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )

        context  = self._build_context_block(retrieved)
        messages = self._build_messages(request.message, context, request.history)

        logger.debug("Calling %s with %d context docs", self.model, len(retrieved))
        completion = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=1024,
        )

        answer = completion.choices[0].message.content or ""
        usage  = completion.usage

        sources = [
            Source(
                id=rd.document.id,
                category=rd.document.category,
                snippet=rd.snippet,
            )
            for rd in retrieved
        ]

        return ChatResponse(
            answer=answer,
            sources=sources,
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ),
        )
