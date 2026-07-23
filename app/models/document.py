"""Knowledge-base document and chunk models.

Both are tenant-owned. A chunk carries a denormalised copy of its parent's
routing fields (``machine_ids``, ``machine_models``, ``doc_type``) because the
vector query filters on them: Atlas Vector Search cannot join to the parent
document mid-query, so the filter fields have to live on the chunk itself.

Nothing here interprets content. A chunk is a passage plus enough provenance to
cite it — document, page, section — and nothing more.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.machine import TenantOwned


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DocType(str, Enum):
    """What kind of source a document is.

    Retrieval filters on this: an operator mid-repair wants the SOP and the
    troubleshooting tree, not the spec sheet.
    """

    manual = "MANUAL"
    sop = "SOP"
    maintenance_guide = "MAINTENANCE_GUIDE"
    repair_history = "REPAIR_HISTORY"
    incident_report = "INCIDENT_REPORT"
    spec_sheet = "SPEC_SHEET"
    troubleshooting = "TROUBLESHOOTING"


class DocumentStatus(str, Enum):
    """Where a document is in the ingestion pipeline."""

    pending = "PENDING"
    processing = "PROCESSING"
    indexed = "INDEXED"
    failed = "FAILED"


class Document(TenantOwned):
    """A source document in the knowledge base."""

    document_id: str = Field(..., min_length=1)
    title: str
    doc_type: DocType

    #: Machines this document applies to. Empty on both lists means the
    #: document is tenant-wide general knowledge (site safety rules, for
    #: example) and is eligible for every machine's retrieval.
    machine_ids: list[str] = Field(default_factory=list)
    machine_models: list[str] = Field(default_factory=list)

    source_filename: Optional[str] = None
    page_count: int = 0
    uploaded_at: datetime = Field(default_factory=utcnow)
    ingested_at: Optional[datetime] = None

    status: DocumentStatus = DocumentStatus.pending
    #: SHA-256 of the source bytes. Re-ingesting identical content replaces the
    #: existing chunks instead of duplicating them.
    content_hash: Optional[str] = None
    #: Populated when status is FAILED. Never swallowed, never silently retried.
    error: Optional[str] = None
    chunk_count: int = 0

    @property
    def is_general(self) -> bool:
        """True when the document is not bound to any machine or model."""
        return not self.machine_ids and not self.machine_models


class Chunk(TenantOwned):
    """One retrievable passage of a document."""

    chunk_id: str = Field(..., min_length=1)
    document_id: str
    text: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    chunk_index: int = 0

    embedding: list[float] = Field(default_factory=list)

    # Denormalised from the parent so the vector query can filter on them.
    machine_ids: list[str] = Field(default_factory=list)
    machine_models: list[str] = Field(default_factory=list)
    doc_type: Optional[DocType] = None
    #: Derived from machine_ids/machine_models — never set directly. Atlas
    #: Vector Search's $eq operator rejects array-typed fields outright, even
    #: to test for emptiness, so "applies to every machine" cannot be
    #: expressed as machine_ids == []. This stored boolean is the field the
    #: query filters on instead (see app.rag.vector_store._machine_clause).
    is_general: bool = False

    token_count: int = 0
    #: Carried for citation display without a second lookup.
    document_title: Optional[str] = None
    #: True when the chunk is a preserved table (never split).
    is_table: bool = False

    model_config = ConfigDict(use_enum_values=True, populate_by_name=True)

    @model_validator(mode="after")
    def _derive_is_general(self) -> "Chunk":
        self.is_general = not self.machine_ids and not self.machine_models
        return self
