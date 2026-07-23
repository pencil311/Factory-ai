"""Vector storage and search, with two interchangeable backends.

``AtlasVectorStore``
    MongoDB Atlas Vector Search via the ``$vectorSearch`` aggregation stage.
    Verified available on this cluster (MongoDB 8.0 enterprise, Atlas
    replica set) — but availability is re-checked at runtime rather than
    assumed, because tier and deployment vary per environment.
``NumpyVectorStore``
    An in-process cosine index persisted to disk. Same interface, no Atlas
    dependency. This is what runs on a local mongod, in CI, and in tests.

Which one is active is **never** decided silently: :func:`build_vector_store`
logs its choice with the reason, and ``/knowledge/status`` reports it.

Tenant isolation
----------------
Every search filters by ``tenant_id`` *inside* the query — in the
``$vectorSearch`` ``filter`` clause for Atlas, and before scoring for numpy.
Post-filtering would be both a correctness bug (top-k fills with another
tenant's chunks, starving this tenant's results) and a security one.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from pymongo.errors import CollectionInvalid, OperationFailure

from app.config import get_settings
from app.db import get_tenant_scope, normalize_tenant_id
from app.schemas.machine import COLLECTIONS

logger = logging.getLogger(__name__)

ATLAS = "atlas"
NUMPY = "numpy"

#: Atlas needs a wider candidate pool than the requested k to rank well.
CANDIDATE_MULTIPLIER = 10
MIN_NUM_CANDIDATES = 100


def normalize_cosine_score(raw: float, already_raw_cosine: bool = False) -> float:
    """Put both backends on one scale: raw cosine, clamped to [0, 1].

    Atlas reports cosine as ``(1 + cos) / 2``, which squashes everything into
    [0.5, 1.0] — unrelated text scores ~0.53 and a score floor set anywhere
    sensible either rejects everything or nothing. Converting back to raw
    cosine makes the number mean what a reader assumes it means, and makes
    ``RAG_MIN_SCORE`` a real threshold rather than a coin flip.

    Negative cosines (genuinely opposed vectors) clamp to 0: for retrieval,
    "unrelated" and "anti-related" are the same answer.
    """
    cosine = raw if already_raw_cosine else (2.0 * raw - 1.0)
    return max(0.0, min(1.0, cosine))


@dataclass
class VectorHit:
    """One scored chunk returned by a backend."""

    chunk: Mapping[str, Any]
    score: float


class VectorStore(ABC):
    """Interface both backends implement."""

    @property
    @abstractmethod
    def backend(self) -> str:
        """``atlas`` or ``numpy``, reported on /knowledge/status."""

    @abstractmethod
    async def ensure_index(self, dimension: int) -> None:
        """Create the vector index for ``dimension``, if it does not exist."""

    @abstractmethod
    async def upsert_chunks(self, tenant_id: str, chunks: Sequence[Mapping[str, Any]]) -> int:
        """Insert chunk documents for one tenant."""

    @abstractmethod
    async def delete_document_chunks(self, tenant_id: str, document_id: str) -> int:
        """Remove every chunk belonging to one document."""

    @abstractmethod
    async def search(
        self,
        tenant_id: str,
        query_vector: Sequence[float],
        top_k: int,
        machine_ids: Optional[Sequence[str]] = None,
        machine_models: Optional[Sequence[str]] = None,
        doc_types: Optional[Sequence[str]] = None,
    ) -> list[VectorHit]:
        """Nearest chunks for one tenant, filtered inside the query."""

    @abstractmethod
    async def count_chunks(self, tenant_id: str) -> int:
        """How many chunks this tenant has indexed."""


def _machine_clause(
    machine_ids: Optional[Sequence[str]], machine_models: Optional[Sequence[str]]
) -> Optional[dict]:
    """Build the machine-scope clause shared by both backends.

    Chunks with empty ``machine_ids`` *and* empty ``machine_models`` are
    tenant-wide general knowledge (site safety rules, general SOPs) and stay
    eligible: excluding them would hide exactly the documents that apply to
    every machine.

    That "applies to every machine" test cannot be ``{"machine_ids": {"$eq":
    []}}``: Atlas Vector Search rejects ``$eq`` against an array-typed filter
    field outright — "must be a boolean, objectId, number, string, date,
    uuid, or null" — regardless of what the array actually holds, so an empty
    literal fails exactly like a non-empty one. ``is_general`` is a plain
    boolean stored on the chunk for exactly this test; ``$in`` (the operator
    Atlas does support on arrays) is used for the two non-empty branches.
    """
    if not machine_ids and not machine_models:
        return None
    alternatives: list[dict] = [{"is_general": {"$eq": True}}]
    if machine_ids:
        alternatives.append({"machine_ids": {"$in": list(machine_ids)}})
    if machine_models:
        alternatives.append({"machine_models": {"$in": list(machine_models)}})
    return {"$or": alternatives}


def _index_fields_match(current: Sequence[Mapping[str, Any]], desired: Sequence[Mapping[str, Any]]) -> bool:
    """True when two Atlas index field lists declare the same fields.

    Compared on the properties that matter functionally — type, path,
    dimension, similarity — rather than by strict equality, so a field-order
    difference or an Atlas-added bookkeeping key never triggers a needless
    rebuild.
    """

    def key(field: Mapping[str, Any]) -> tuple:
        return (
            field.get("type"),
            field.get("path"),
            field.get("numDimensions"),
            field.get("similarity"),
        )

    return {key(f) for f in current} == {key(f) for f in desired}


# ---------------------------------------------------------------------------
# Atlas
# ---------------------------------------------------------------------------
class AtlasVectorStore(VectorStore):
    """MongoDB Atlas Vector Search."""

    def __init__(self, index_name: str = "chunk_vector_index") -> None:
        self._index_name = index_name

    @property
    def backend(self) -> str:
        return ATLAS

    def _collection(self, tenant_id: str):
        return get_tenant_scope(tenant_id)[COLLECTIONS.chunks]

    async def ensure_index(self, dimension: int) -> None:
        """Create the vector index sized from the provider's dimension.

        ``tenant_id`` is declared as a filter field so it can be applied
        *inside* ``$vectorSearch`` rather than after it.

        Unlike an ordinary Mongo index, Atlas Search's ``createSearchIndex``
        does not implicitly create its collection — it fails with
        ``NamespaceNotFound`` against one that has never been written to. On a
        fresh database this runs before the first document is ever ingested,
        so the collection is created here first, tolerating the case where a
        concurrent caller (or an earlier run) already created it.
        """
        from app.db import get_database

        database = get_database()
        collection = database[COLLECTIONS.chunks]
        try:
            await database.create_collection(COLLECTIONS.chunks)
            logger.info("Created collection '%s'", COLLECTIONS.chunks)
        except CollectionInvalid:
            pass  # already exists — the common case after the first run
        except OperationFailure as exc:
            if exc.code != 48:  # 48 == NamespaceExists: a concurrent creator won
                raise

        fields = [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": dimension,
                "similarity": "cosine",
            },
            {"type": "filter", "path": "tenant_id"},
            {"type": "filter", "path": "machine_ids"},
            {"type": "filter", "path": "machine_models"},
            {"type": "filter", "path": "doc_type"},
            {"type": "filter", "path": "is_general"},
        ]
        index_definition = {"fields": fields}

        try:
            existing = await collection.list_search_indexes().to_list(length=50)
            current = next(
                (i for i in existing if i.get("name") == self._index_name), None
            )
            if current is not None:
                current_fields = (current.get("latestDefinition") or {}).get(
                    "fields", []
                )
                if _index_fields_match(current_fields, fields):
                    logger.info(
                        "Atlas vector index '%s' already present and up to date",
                        self._index_name,
                    )
                    return
                logger.info(
                    "Atlas vector index '%s' definition has drifted from the "
                    "declared filter fields; updating it in place",
                    self._index_name,
                )
                await collection.update_search_index(self._index_name, index_definition)
                logger.info("Updated Atlas vector index '%s'", self._index_name)
                return

            await collection.create_search_index(
                {
                    "name": self._index_name,
                    "type": "vectorSearch",
                    "definition": index_definition,
                }
            )
            logger.info(
                "Created Atlas vector index '%s' with %d dimensions",
                self._index_name,
                dimension,
            )
        except Exception:
            logger.exception(
                "Could not create or update Atlas vector index '%s'. Search will "
                "fail until it exists with the right definition; create it in the "
                "Atlas UI with numDimensions=%d.",
                self._index_name,
                dimension,
            )
            raise

    async def upsert_chunks(self, tenant_id: str, chunks: Sequence[Mapping[str, Any]]) -> int:
        if not chunks:
            return 0
        await self._collection(tenant_id).insert_many(list(chunks))
        return len(chunks)

    async def delete_document_chunks(self, tenant_id: str, document_id: str) -> int:
        result = await self._collection(tenant_id).delete_many({"document_id": document_id})
        return int(getattr(result, "deleted_count", 0))

    async def search(
        self,
        tenant_id: str,
        query_vector: Sequence[float],
        top_k: int,
        machine_ids: Optional[Sequence[str]] = None,
        machine_models: Optional[Sequence[str]] = None,
        doc_types: Optional[Sequence[str]] = None,
    ) -> list[VectorHit]:
        tenant_id = normalize_tenant_id(tenant_id)

        # The tenant clause goes INSIDE $vectorSearch. Filtering after the
        # stage would let another tenant's chunks consume the candidate pool.
        conditions: list[dict] = [{"tenant_id": {"$eq": tenant_id}}]
        machine_clause = _machine_clause(machine_ids, machine_models)
        if machine_clause:
            conditions.append(machine_clause)
        if doc_types:
            conditions.append({"doc_type": {"$in": list(doc_types)}})
        vector_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self._index_name,
                    "path": "embedding",
                    "queryVector": list(query_vector),
                    "numCandidates": max(MIN_NUM_CANDIDATES, top_k * CANDIDATE_MULTIPLIER),
                    "limit": top_k,
                    "filter": vector_filter,
                }
            },
            {"$set": {"_score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0}},
        ]

        from app.db import get_database

        cursor = get_database()[COLLECTIONS.chunks].aggregate(pipeline)
        hits = []
        async for doc in cursor:
            # Defence in depth: the filter above is authoritative, but a
            # mismatched tenant here would mean the index is misconfigured,
            # and that must never reach a caller.
            if doc.get("tenant_id") != tenant_id:
                logger.error(
                    "Atlas returned a chunk for tenant '%s' under tenant '%s' — "
                    "check the vector index filter fields.",
                    doc.get("tenant_id"),
                    tenant_id,
                )
                continue
            hits.append(
                VectorHit(
                    chunk=doc,
                    # Atlas cosine scores arrive as (1 + cos)/2; convert back.
                    score=normalize_cosine_score(float(doc.pop("_score", 0.0))),
                )
            )
        return hits

    async def count_chunks(self, tenant_id: str) -> int:
        return await self._collection(tenant_id).count_documents({})


# ---------------------------------------------------------------------------
# Numpy fallback
# ---------------------------------------------------------------------------
class NumpyVectorStore(VectorStore):
    """In-process cosine index, persisted to disk as JSONL.

    Chunks still live in Mongo when one is available; this class owns only the
    vectors and the search. With ``persist_path=None`` it is purely in-memory,
    which is what the tests use.
    """

    def __init__(self, persist_path: Optional[Path | str] = None) -> None:
        self._path = Path(persist_path) if persist_path else None
        self._chunks: list[dict] = []
        self._matrix: Optional[np.ndarray] = None
        self._loaded = False

    @property
    def backend(self) -> str:
        return NUMPY

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self._path and self._path.exists():
            with self._path.open("r", encoding="utf-8") as handle:
                self._chunks = [json.loads(line) for line in handle if line.strip()]
            self._rebuild_matrix()
            logger.info(
                "Loaded %d chunks into the numpy vector index from %s",
                len(self._chunks),
                self._path,
            )

    def _rebuild_matrix(self) -> None:
        if not self._chunks:
            self._matrix = None
            return
        self._matrix = np.asarray(
            [c.get("embedding") or [] for c in self._chunks], dtype=np.float32
        )

    def _persist(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(json.dumps(chunk, default=str) + "\n")

    async def ensure_index(self, dimension: int) -> None:
        """No index to build, but the dimension is recorded and validated."""
        self._ensure_loaded()
        self._dimension = dimension
        if self._matrix is not None and self._matrix.shape[1] != dimension:
            raise ValueError(
                f"Persisted vectors are {self._matrix.shape[1]}-dimensional but the "
                f"active embedding provider produces {dimension}. Re-ingest the "
                f"corpus, or switch back to the previous model."
            )
        logger.info("Numpy vector index ready (dimension=%d)", dimension)

    async def upsert_chunks(self, tenant_id: str, chunks: Sequence[Mapping[str, Any]]) -> int:
        self._ensure_loaded()
        tenant_id = normalize_tenant_id(tenant_id)
        for chunk in chunks:
            stored = dict(chunk)
            stored["tenant_id"] = tenant_id
            self._chunks.append(stored)
        self._rebuild_matrix()
        self._persist()
        return len(chunks)

    async def delete_document_chunks(self, tenant_id: str, document_id: str) -> int:
        self._ensure_loaded()
        tenant_id = normalize_tenant_id(tenant_id)
        before = len(self._chunks)
        self._chunks = [
            c
            for c in self._chunks
            if not (c.get("tenant_id") == tenant_id and c.get("document_id") == document_id)
        ]
        removed = before - len(self._chunks)
        self._rebuild_matrix()
        self._persist()
        return removed

    async def search(
        self,
        tenant_id: str,
        query_vector: Sequence[float],
        top_k: int,
        machine_ids: Optional[Sequence[str]] = None,
        machine_models: Optional[Sequence[str]] = None,
        doc_types: Optional[Sequence[str]] = None,
    ) -> list[VectorHit]:
        self._ensure_loaded()
        tenant_id = normalize_tenant_id(tenant_id)
        if not self._chunks or self._matrix is None:
            return []

        # Filter FIRST, then score. Same ordering guarantee as the Atlas
        # filter clause: another tenant's chunks never enter the ranking.
        wanted_ids = set(machine_ids or [])
        wanted_models = set(machine_models or [])

        indices: list[int] = []
        for position, chunk in enumerate(self._chunks):
            if chunk.get("tenant_id") != tenant_id:
                continue
            if doc_types and chunk.get("doc_type") not in set(doc_types):
                continue
            if wanted_ids or wanted_models:
                chunk_ids = set(chunk.get("machine_ids") or [])
                chunk_models = set(chunk.get("machine_models") or [])
                is_general = not chunk_ids and not chunk_models
                if not (
                    is_general
                    or (chunk_ids & wanted_ids)
                    or (chunk_models & wanted_models)
                ):
                    continue
            indices.append(position)

        if not indices:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm:
            query = query / norm
        candidates = self._matrix[indices]
        scores = candidates @ query

        order = np.argsort(-scores)[:top_k]
        hits: list[VectorHit] = []
        for rank in order:
            chunk = dict(self._chunks[indices[int(rank)]])
            chunk.pop("embedding", None)
            # Already a raw cosine (both sides are unit vectors), so it only
            # needs clamping to agree with the Atlas backend's scale.
            hits.append(
                VectorHit(
                    chunk=chunk,
                    score=normalize_cosine_score(
                        float(scores[int(rank)]), already_raw_cosine=True
                    ),
                )
            )
        return hits

    async def count_chunks(self, tenant_id: str) -> int:
        self._ensure_loaded()
        tenant_id = normalize_tenant_id(tenant_id)
        return sum(1 for c in self._chunks if c.get("tenant_id") == tenant_id)


# ---------------------------------------------------------------------------
# Backend selection — explicit, logged, never silent
# ---------------------------------------------------------------------------
_store: Optional[VectorStore] = None
_selection_reason: str = "not initialised"


async def atlas_vector_search_available() -> tuple[bool, str]:
    """Probe the live cluster for ``$listSearchIndexes`` support.

    Returns ``(available, reason)``. Read-only: it asks the server what it
    supports and writes nothing.
    """
    try:
        from app.db import get_database

        collection = get_database()[COLLECTIONS.chunks]
        await collection.list_search_indexes().to_list(length=1)
        return True, "Atlas Vector Search available (listSearchIndexes succeeded)"
    except Exception as exc:
        return False, f"Atlas Vector Search unavailable ({type(exc).__name__}: {exc})"


async def build_vector_store(kind: Optional[str] = None) -> VectorStore:
    """Choose a backend, logging which and why.

    ``VECTOR_BACKEND`` may force ``atlas`` or ``numpy``; ``auto`` (the default)
    probes the cluster and falls back. The choice is never implicit — every
    path through this function logs its reasoning.
    """
    global _selection_reason
    settings = get_settings()
    requested = (kind or settings.vector_backend or "auto").strip().lower()

    if requested == NUMPY:
        _selection_reason = "VECTOR_BACKEND=numpy (forced by configuration)"
        logger.info("Vector backend: numpy — %s", _selection_reason)
        return NumpyVectorStore(settings.vector_index_path)

    if requested == ATLAS:
        _selection_reason = "VECTOR_BACKEND=atlas (forced by configuration)"
        logger.info("Vector backend: atlas — %s", _selection_reason)
        return AtlasVectorStore(settings.vector_index_name)

    if requested != "auto":
        raise ValueError(
            f"Unknown VECTOR_BACKEND '{requested}'. Expected: auto, atlas, numpy."
        )

    available, reason = await atlas_vector_search_available()
    _selection_reason = reason
    if available:
        logger.info("Vector backend: atlas — %s", reason)
        return AtlasVectorStore(settings.vector_index_name)
    logger.warning(
        "Vector backend: numpy — %s. Falling back to the in-process index at %s.",
        reason,
        settings.vector_index_path,
    )
    return NumpyVectorStore(settings.vector_index_path)


async def get_vector_store() -> VectorStore:
    """The process-wide store, selected on first use."""
    global _store
    if _store is None:
        _store = await build_vector_store()
    return _store


def set_vector_store(store: Optional[VectorStore], reason: str = "set explicitly") -> None:
    """Replace the active store. For tests and explicit wiring."""
    global _store, _selection_reason
    _store = store
    _selection_reason = reason


def selection_reason() -> str:
    """Why the active backend was chosen — surfaced on /knowledge/status."""
    return _selection_reason
