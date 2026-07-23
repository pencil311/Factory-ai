"""Knowledge-base endpoints.

Every route is tenant-scoped through ``get_current_tenant``. ``/knowledge/search``
returns passages and provenance only — it never composes an answer, because
this service retrieves and downstream agents interpret.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from app.config import get_settings
from app.core.tenant_context import get_current_tenant
from app.db import get_tenant_scope
from app.models.document import DocType, DocumentStatus
from app.rag.chunking import UnsupportedFileType
from app.rag.ingest import delete_document, ingest_document
from app.rag.retriever import knowledge_stats, retrieve
from app.schemas.knowledge import (
    DeleteResponse,
    DocumentOut,
    KnowledgeStatusOut,
    RetrievedChunkOut,
    SearchRequest,
    SearchResponse,
    UploadResponse,
)
from app.schemas.machine import COLLECTIONS, strip_mongo_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _split_csv(value: Optional[str]) -> list[str]:
    """Parse a comma-separated form field into a clean list."""
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF, TXT, MD or DOCX"),
    doc_type: DocType = Form(default=DocType.manual),
    title: Optional[str] = Form(default=None),
    machine_ids: Optional[str] = Form(
        default=None, description="Comma-separated machine ids this applies to"
    ),
    machine_models: Optional[str] = Form(
        default=None, description="Comma-separated machine models this applies to"
    ),
    tenant_id: str = Depends(get_current_tenant),
) -> UploadResponse:
    """Ingest a file: parse, chunk, embed, index.

    Idempotent by content hash — re-uploading the same bytes replaces the
    existing chunks rather than duplicating them.
    """
    settings = get_settings()
    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{file.filename}' is empty.",
        )
    if len(data) > settings.rag_max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"'{file.filename}' is {len(data) / 1e6:.1f} MB, over the "
                f"{settings.rag_max_upload_bytes / 1e6:.0f} MB limit."
            ),
        )

    try:
        result = await ingest_document(
            tenant_id=tenant_id,
            data=data,
            filename=file.filename or "upload",
            title=title,
            doc_type=doc_type,
            machine_ids=_split_csv(machine_ids),
            machine_models=_split_csv(machine_models),
        )
    except UnsupportedFileType as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc

    return UploadResponse(
        document_id=result.document_id,
        status=result.status,
        chunk_count=result.chunk_count,
        page_count=result.page_count,
        replaced_existing=result.replaced_existing,
        error=result.error,
    )


@router.get(
    "/documents", response_model=list[DocumentOut], summary="List documents"
)
async def list_documents(
    doc_type: Optional[DocType] = Query(default=None),
    machine_id: Optional[str] = Query(default=None),
    document_status: Optional[DocumentStatus] = Query(default=None, alias="status"),
    tenant_id: str = Depends(get_current_tenant),
) -> list[dict]:
    """List the tenant's documents, including any that failed to ingest."""
    scope = get_tenant_scope(tenant_id)
    query: dict = {}
    if doc_type:
        query["doc_type"] = doc_type.value
    if machine_id:
        query["machine_ids"] = machine_id
    if document_status:
        query["status"] = document_status.value

    cursor = scope[COLLECTIONS.documents].find(query).sort([("uploaded_at", -1)])
    return [strip_mongo_id(doc) async for doc in cursor]


@router.get(
    "/status", response_model=KnowledgeStatusOut, summary="Active backend and index size"
)
async def knowledge_status(
    tenant_id: str = Depends(get_current_tenant),
) -> KnowledgeStatusOut:
    """Report which vector backend is live, why, and what is indexed.

    The backend is never chosen silently; this is where that choice is visible.
    """
    return KnowledgeStatusOut(**await knowledge_stats(tenant_id))


@router.post("/search", response_model=SearchResponse, summary="Retrieve passages")
async def search(
    payload: SearchRequest, tenant_id: str = Depends(get_current_tenant)
) -> SearchResponse:
    """Return the passages best matching a query.

    Passages only — no answer, no summary, no recommendation. When nothing
    clears the score threshold the result is empty and ``reason`` says why.
    """
    result = await retrieve(
        tenant_id=tenant_id,
        query=payload.query,
        machine_id=payload.machine_id,
        doc_types=[d.value for d in payload.doc_types] if payload.doc_types else None,
        top_k=payload.top_k,
    )
    return SearchResponse(
        query=result.query,
        chunks=[RetrievedChunkOut(**vars(c)) for c in result.chunks],
        backend_used=result.backend_used,
        total_candidates=result.total_candidates,
        machine_filter_applied=result.machine_filter_applied,
        machine_id=result.machine_id,
        machine_model=result.machine_model,
        doc_types=result.doc_types,
        embedding_model=result.embedding_model,
        reason=result.reason,
    )


# Declared after the fixed paths so /documents/upload is never captured here.
@router.get(
    "/documents/{document_id}", response_model=DocumentOut, summary="Get one document"
)
async def get_document(
    document_id: str, tenant_id: str = Depends(get_current_tenant)
) -> dict:
    """Return one document's record, including its failure reason if it failed."""
    scope = get_tenant_scope(tenant_id)
    doc = await scope[COLLECTIONS.documents].find_one({"document_id": document_id})
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found",
        )
    return strip_mongo_id(doc)


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    summary="Delete a document and its chunks",
)
async def remove_document(
    document_id: str, tenant_id: str = Depends(get_current_tenant)
) -> DeleteResponse:
    """Delete a document and every passage indexed from it."""
    existed, removed = await delete_document(tenant_id, document_id)
    if not existed and removed == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found",
        )
    return DeleteResponse(
        document_id=document_id, deleted=existed, chunks_removed=removed
    )
