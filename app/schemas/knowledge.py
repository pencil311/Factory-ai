"""Request/response schemas for the knowledge endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import DocType, DocumentStatus


class DocumentOut(BaseModel):
    """A knowledge-base document as the API returns it."""

    model_config = ConfigDict(use_enum_values=True)

    document_id: str
    title: str
    doc_type: DocType
    machine_ids: list[str] = []
    machine_models: list[str] = []
    source_filename: Optional[str] = None
    page_count: int = 0
    chunk_count: int = 0
    status: DocumentStatus
    uploaded_at: Optional[datetime] = None
    ingested_at: Optional[datetime] = None
    #: Populated when status is FAILED — surfaced, never hidden.
    error: Optional[str] = None


class UploadResponse(BaseModel):
    """Result of an ingestion run."""

    model_config = ConfigDict(use_enum_values=True)

    document_id: str
    status: DocumentStatus
    chunk_count: int
    page_count: int
    #: True when this upload replaced an identical earlier one.
    replaced_existing: bool
    error: Optional[str] = None


class SearchRequest(BaseModel):
    """Body of ``POST /knowledge/search``."""

    query: str = Field(..., min_length=1)
    machine_id: Optional[str] = Field(
        default=None, description="Restrict to this machine, its model, and general docs"
    )
    doc_types: Optional[list[DocType]] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=50)


class RetrievedChunkOut(BaseModel):
    """One passage with its provenance."""

    model_config = ConfigDict(use_enum_values=True)

    chunk_id: str
    document_id: str
    document_title: Optional[str] = None
    text: str
    score: float
    vector_score: float
    keyword_score: float
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    doc_type: Optional[str] = None
    machine_ids: list[str] = []
    machine_models: list[str] = []
    is_table: bool = False
    #: Exact identifier tokens shared by query and passage (codes, part numbers).
    matched_terms: list[str] = []


class SearchResponse(BaseModel):
    """Retrieval output: passages plus how they were found.

    Deliberately carries no answer, summary or recommendation — this service
    retrieves, and downstream agents interpret.
    """

    model_config = ConfigDict(use_enum_values=True)

    query: str
    chunks: list[RetrievedChunkOut] = []
    backend_used: str
    total_candidates: int
    machine_filter_applied: bool
    machine_id: Optional[str] = None
    machine_model: Optional[str] = None
    doc_types: Optional[list[str]] = None
    embedding_model: Optional[str] = None
    #: Why the result is empty, when it is.
    reason: Optional[str] = None


class KnowledgeStatusOut(BaseModel):
    """Which backend is live, and what is indexed."""

    backend: str
    backend_reason: str
    embedding_model: str
    embedding_dimension: int
    chunk_count: int
    document_count: int
    tenant_id: str


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    chunks_removed: int
