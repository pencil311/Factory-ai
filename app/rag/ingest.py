"""Ingestion: file -> parse -> chunk -> embed -> store.

Two properties this module exists to guarantee:

**Failures land on the document, not in a log.** A document that could not be
parsed ends in ``FAILED`` with the reason recorded on the record itself. It
stays visible in ``GET /knowledge/documents`` so a human can see that the
manual they uploaded never became searchable — the alternative is a document
that looks fine and silently returns nothing forever.

**Re-ingesting is idempotent.** A document is keyed by the SHA-256 of its
bytes plus its tenant. Re-uploading the same file deletes the existing chunks
and rewrites them rather than doubling every passage, which would otherwise
quietly skew retrieval toward whatever was uploaded twice.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from app.db import get_tenant_scope, normalize_tenant_id
from app.models.document import Chunk, DocType, Document, DocumentStatus
from app.rag.chunking import (
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    UnsupportedFileType,
    parse_and_chunk,
)
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from app.rag.vector_store import VectorStore, get_vector_store
from app.schemas.machine import COLLECTIONS

logger = logging.getLogger(__name__)


def content_hash(data: bytes) -> str:
    """SHA-256 of the raw bytes — the idempotency key."""
    return hashlib.sha256(data).hexdigest()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class IngestionResult:
    """Outcome of one ingestion run."""

    document_id: str
    status: DocumentStatus
    chunk_count: int
    page_count: int
    replaced_existing: bool
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.status == DocumentStatus.indexed


async def ingest_document(
    tenant_id: str,
    data: bytes,
    filename: str,
    title: Optional[str] = None,
    doc_type: DocType | str = DocType.manual,
    machine_ids: Optional[Sequence[str]] = None,
    machine_models: Optional[Sequence[str]] = None,
    *,
    provider: Optional[EmbeddingProvider] = None,
    store: Optional[VectorStore] = None,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> IngestionResult:
    """Run one file through the whole pipeline, recording status as it goes."""
    tenant_id = normalize_tenant_id(tenant_id)
    provider = provider or get_embedding_provider()
    store = store or await get_vector_store()
    scope = get_tenant_scope(tenant_id)

    digest = content_hash(data)
    doc_type = DocType(doc_type)
    machine_ids = list(machine_ids or [])
    machine_models = list(machine_models or [])

    # Idempotency: same tenant + same bytes = same document, rewritten.
    existing = await scope[COLLECTIONS.documents].find_one({"content_hash": digest})
    replaced = existing is not None
    document_id = (
        str(existing["document_id"]) if existing else f"DOC-{uuid.uuid4().hex[:12]}"
    )

    document = Document(
        tenant_id=tenant_id,
        document_id=document_id,
        title=title or filename,
        doc_type=doc_type,
        machine_ids=machine_ids,
        machine_models=machine_models,
        source_filename=filename,
        content_hash=digest,
        status=DocumentStatus.processing,
        uploaded_at=existing.get("uploaded_at") if existing else utcnow(),
    )
    await scope[COLLECTIONS.documents].upsert_many(
        ["document_id"], [document.model_dump()]
    )

    try:
        parsed, drafts = parse_and_chunk(data, filename, target_tokens, overlap_tokens)
        if not drafts:
            raise ValueError(
                f"'{filename}' parsed to {parsed.page_count} page(s) but produced no "
                "text. It may be a scanned image PDF, which needs OCR."
            )

        vectors = provider.encode([d.text for d in drafts])

        chunks: list[dict] = []
        for index, (draft, vector) in enumerate(zip(drafts, vectors)):
            chunk = Chunk(
                tenant_id=tenant_id,
                chunk_id=f"{document_id}-C{index:04d}",
                document_id=document_id,
                text=draft.text,
                page_number=draft.page_number,
                section_title=draft.section_title,
                chunk_index=index,
                embedding=[float(v) for v in vector],
                machine_ids=machine_ids,
                machine_models=machine_models,
                doc_type=doc_type,
                token_count=draft.token_count,
                document_title=document.title,
                is_table=draft.is_table,
            )
            chunks.append(chunk.model_dump())

        # Replace, never append: delete first so a re-ingest cannot double up.
        await store.delete_document_chunks(tenant_id, document_id)
        await scope[COLLECTIONS.chunks].delete_many({"document_id": document_id})
        await store.upsert_chunks(tenant_id, chunks)

        document.status = DocumentStatus.indexed
        document.page_count = parsed.page_count
        document.chunk_count = len(chunks)
        document.ingested_at = utcnow()
        document.error = None
        await scope[COLLECTIONS.documents].upsert_many(
            ["document_id"], [document.model_dump()]
        )

        logger.info(
            "Ingested '%s' for tenant '%s': %d chunks over %d page(s)%s",
            filename,
            tenant_id,
            len(chunks),
            parsed.page_count,
            " (replaced an earlier copy)" if replaced else "",
        )
        return IngestionResult(
            document_id=document_id,
            status=DocumentStatus.indexed,
            chunk_count=len(chunks),
            page_count=parsed.page_count,
            replaced_existing=replaced,
        )

    except Exception as exc:
        # The failure belongs on the document. Swallowing it would leave a
        # document that looks ingested and retrieves nothing.
        reason = f"{type(exc).__name__}: {exc}"
        document.status = DocumentStatus.failed
        document.error = reason
        document.chunk_count = 0
        await scope[COLLECTIONS.documents].upsert_many(
            ["document_id"], [document.model_dump()]
        )
        logger.exception("Ingestion failed for '%s' (tenant '%s')", filename, tenant_id)

        if isinstance(exc, UnsupportedFileType):
            raise
        return IngestionResult(
            document_id=document_id,
            status=DocumentStatus.failed,
            chunk_count=0,
            page_count=0,
            replaced_existing=replaced,
            error=reason,
        )


async def delete_document(
    tenant_id: str, document_id: str, store: Optional[VectorStore] = None
) -> tuple[bool, int]:
    """Delete a document and every chunk belonging to it.

    Returns ``(document_existed, chunks_removed)``. Chunks go first: a document
    record without chunks is a visible inconsistency, while orphaned chunks
    would keep being retrieved with no document to cite.
    """
    tenant_id = normalize_tenant_id(tenant_id)
    store = store or await get_vector_store()
    scope = get_tenant_scope(tenant_id)

    removed = await store.delete_document_chunks(tenant_id, document_id)
    db_result = await scope[COLLECTIONS.chunks].delete_many({"document_id": document_id})
    removed = max(removed, int(getattr(db_result, "deleted_count", 0)))

    result = await scope[COLLECTIONS.documents].delete_one({"document_id": document_id})
    existed = int(getattr(result, "deleted_count", 0)) > 0
    return existed, removed


async def ensure_vector_index(
    provider: Optional[EmbeddingProvider] = None,
    store: Optional[VectorStore] = None,
) -> int:
    """Create the vector index sized from the ACTIVE provider's dimension.

    The dimension is read from the provider rather than hardcoded, so changing
    ``EMBEDDING_MODEL`` cannot leave an index of the wrong width — a mismatch
    Atlas reports as "no results" rather than as an error.
    """
    provider = provider or get_embedding_provider()
    store = store or await get_vector_store()
    dimension = provider.dimension
    await store.ensure_index(dimension)
    return dimension
