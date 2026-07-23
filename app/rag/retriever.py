"""Retrieval. Returns passages with provenance — nothing else.

This module does not diagnose, recommend, rank by severity, or summarise. It
answers one question: *which stored passages best match this query, for this
tenant, optionally for this machine?* Downstream agents interpret. If nothing
clears the score floor, it says so and returns nothing rather than offering a
weak passage that reads as an answer.

Why hybrid scoring
------------------
Dense embeddings are good at meaning and bad at exact strings. "E104" and
"E106" are near-identical vectors, and a part number like ``SKF-6310-2RS1``
carries almost no semantic signal at all — but when an operator types one,
they mean *that* string and nothing else. So the vector score is blended with
a BM25-style lexical score, and exact matches on code-shaped tokens get an
additional boost. This is the difference between retrieving the right error's
row and retrieving a plausible neighbour's.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from app.config import get_settings
from app.db import get_tenant_scope, normalize_tenant_id
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from app.rag.vector_store import VectorStore, get_vector_store, selection_reason
from app.schemas.machine import COLLECTIONS

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_/.]*")

#: Tokens shaped like identifiers: error codes (E104, A-201), part numbers
#: (SKF-6310-2RS1), machine ids (CV-201). An exact hit on one of these is
#: strong evidence the passage is the one being asked for.
_CODE_RE = re.compile(r"^(?=.*\d)[A-Za-z]{0,4}[-_]?\d{2,5}(?:[-_][A-Za-z0-9]+)*$")

#: BM25 term-frequency saturation and length-normalisation constants. The
#: standard defaults; nothing here is tuned to the corpus.
BM25_K1 = 1.5
BM25_B = 0.75

#: Extra weight for an exact code match, added before the final clamp.
CODE_MATCH_BOOST = 0.35

_STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or that the
    to was were what when where which who why with do does did i we you my our""".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-cased tokens, stopwords dropped, identifier shapes preserved."""
    return [
        t
        for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text or ""))
        if t not in _STOPWORDS
    ]


def code_tokens(text: str) -> set[str]:
    """Identifier-shaped tokens in ``text`` (error codes, part numbers, ids)."""
    return {t for t in tokenize(text) if _CODE_RE.match(t)}


@dataclass
class RetrievedChunk:
    """One passage, with everything needed to cite it."""

    chunk_id: str
    document_id: str
    document_title: Optional[str]
    text: str
    score: float
    vector_score: float
    keyword_score: float
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    doc_type: Optional[str] = None
    machine_ids: list[str] = field(default_factory=list)
    machine_models: list[str] = field(default_factory=list)
    is_table: bool = False
    matched_terms: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """What retrieval found, and how."""

    query: str
    chunks: list[RetrievedChunk]
    backend_used: str
    total_candidates: int
    machine_filter_applied: bool
    machine_id: Optional[str] = None
    machine_model: Optional[str] = None
    doc_types: Optional[list[str]] = None
    #: Set when ``chunks`` is empty, explaining why. Never a guess.
    reason: Optional[str] = None
    embedding_model: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not self.chunks


# ---------------------------------------------------------------------------
# Lexical scoring
# ---------------------------------------------------------------------------
def bm25_scores(query: str, documents: Sequence[str]) -> list[float]:
    """BM25 over the candidate set, normalised to 0..1.

    Scored against the retrieved candidates rather than the whole corpus: this
    is a re-ranking signal, so what matters is which candidate matches best,
    not the absolute value.
    """
    query_terms = tokenize(query)
    if not query_terms or not documents:
        return [0.0] * len(documents)

    tokenized = [tokenize(d) for d in documents]
    lengths = [len(t) or 1 for t in tokenized]
    average_length = sum(lengths) / len(lengths)
    counters = [Counter(t) for t in tokenized]
    total_docs = len(documents)

    document_frequency = Counter()
    for counter in counters:
        for term in set(query_terms):
            if counter[term]:
                document_frequency[term] += 1

    raw: list[float] = []
    for counter, length in zip(counters, lengths):
        score = 0.0
        for term in query_terms:
            frequency = counter[term]
            if not frequency:
                continue
            appearances = document_frequency[term]
            # +0.5/+0.5 smoothing keeps IDF positive when a term is in every doc.
            idf = math.log(1.0 + (total_docs - appearances + 0.5) / (appearances + 0.5))
            numerator = frequency * (BM25_K1 + 1.0)
            denominator = frequency + BM25_K1 * (
                1.0 - BM25_B + BM25_B * length / average_length
            )
            score += idf * numerator / denominator
        raw.append(score)

    ceiling = max(raw)
    if ceiling <= 0:
        return [0.0] * len(documents)
    return [r / ceiling for r in raw]


# ---------------------------------------------------------------------------
# Machine scope
# ---------------------------------------------------------------------------
async def resolve_machine_model(tenant_id: str, machine_id: str) -> Optional[str]:
    """Look up a machine's model so model-level documents also match.

    A manual is usually written for the *model* ("SpanTech SB-3000"), not the
    individual asset, so retrieving for CV-201 has to reach both.
    """
    scope = get_tenant_scope(tenant_id)
    machine = await scope[COLLECTIONS.machines].find_one({"machine_id": machine_id})
    return str(machine["model"]) if machine and machine.get("model") else None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
async def retrieve(
    tenant_id: str,
    query: str,
    machine_id: Optional[str] = None,
    doc_types: Optional[Sequence[str]] = None,
    top_k: Optional[int] = None,
    *,
    provider: Optional[EmbeddingProvider] = None,
    store: Optional[VectorStore] = None,
    min_score: Optional[float] = None,
    vector_weight: Optional[float] = None,
    machine_model: Optional[str] = None,
) -> RetrievalResult:
    """Retrieve passages for a query within one tenant.

    Returns passages and provenance. Never fabricates, never summarises, and
    returns an empty result with a stated ``reason`` rather than a weak match.
    """
    settings = get_settings()
    tenant_id = normalize_tenant_id(tenant_id)
    provider = provider or get_embedding_provider()
    store = store or await get_vector_store()
    top_k = top_k or settings.rag_top_k
    min_score = settings.rag_min_score if min_score is None else min_score
    vector_weight = (
        settings.rag_vector_weight if vector_weight is None else vector_weight
    )
    keyword_weight = 1.0 - vector_weight
    doc_types = list(doc_types) if doc_types else None

    base = RetrievalResult(
        query=query,
        chunks=[],
        backend_used=store.backend,
        total_candidates=0,
        machine_filter_applied=False,
        machine_id=machine_id,
        doc_types=doc_types,
        embedding_model=provider.name,
    )

    if not (query or "").strip():
        base.reason = "Empty query — nothing to retrieve."
        return base

    # Machine scope: the asset itself, plus its model, plus tenant-wide docs.
    machine_ids: Optional[list[str]] = None
    machine_models: Optional[list[str]] = None
    if machine_id:
        if machine_model is None:
            machine_model = await resolve_machine_model(tenant_id, machine_id)
        machine_ids = [machine_id]
        machine_models = [machine_model] if machine_model else []
        base.machine_filter_applied = True
        base.machine_model = machine_model

    # Over-fetch so the lexical re-rank has room to move things: the best
    # keyword match is frequently outside the top-k by vector alone, which is
    # the entire reason hybrid scoring exists.
    fetch_k = max(top_k * 4, 20)
    query_vector = provider.encode_one(query)
    hits = await store.search(
        tenant_id=tenant_id,
        query_vector=query_vector,
        top_k=fetch_k,
        machine_ids=machine_ids,
        machine_models=machine_models,
        doc_types=doc_types,
    )
    base.total_candidates = len(hits)

    if not hits:
        base.reason = (
            f"No indexed passages matched"
            + (f" for machine '{machine_id}'" if machine_id else "")
            + (f" in doc types {doc_types}" if doc_types else "")
            + ". The knowledge base may be empty for this tenant, or the "
            "filters may exclude everything."
        )
        return base

    texts = [str(h.chunk.get("text", "")) for h in hits]
    lexical = bm25_scores(query, texts)
    query_codes = code_tokens(query)

    scored: list[RetrievedChunk] = []
    for hit, keyword_score in zip(hits, lexical):
        chunk = hit.chunk
        text = str(chunk.get("text", ""))

        # Exact identifier hits: the case dense retrieval is worst at.
        chunk_codes = code_tokens(text)
        matched_codes = sorted(query_codes & chunk_codes)
        boost = CODE_MATCH_BOOST if matched_codes else 0.0

        blended = min(
            1.0,
            vector_weight * hit.score + keyword_weight * keyword_score + boost,
        )

        scored.append(
            RetrievedChunk(
                chunk_id=str(chunk.get("chunk_id", "")),
                document_id=str(chunk.get("document_id", "")),
                document_title=chunk.get("document_title"),
                text=text,
                score=round(blended, 4),
                vector_score=round(float(hit.score), 4),
                keyword_score=round(float(keyword_score), 4),
                page_number=chunk.get("page_number"),
                section_title=chunk.get("section_title"),
                doc_type=chunk.get("doc_type"),
                machine_ids=list(chunk.get("machine_ids") or []),
                machine_models=list(chunk.get("machine_models") or []),
                is_table=bool(chunk.get("is_table", False)),
                matched_terms=matched_codes,
            )
        )

    scored.sort(key=lambda c: (-c.score, c.chunk_id))
    kept = [c for c in scored if c.score >= min_score][:top_k]

    if not kept:
        best = scored[0].score if scored else 0.0
        base.reason = (
            f"{len(scored)} passage(s) were considered but none reached the "
            f"minimum score of {min_score:.2f} (best was {best:.2f}). "
            "Returning nothing rather than a weak match."
        )
        return base

    base.chunks = kept
    return base


async def knowledge_stats(
    tenant_id: str,
    provider: Optional[EmbeddingProvider] = None,
    store: Optional[VectorStore] = None,
) -> dict[str, Any]:
    """Backend, chunk count and embedding model — for /knowledge/status."""
    tenant_id = normalize_tenant_id(tenant_id)
    provider = provider or get_embedding_provider()
    store = store or await get_vector_store()
    scope = get_tenant_scope(tenant_id)

    try:
        document_count = await scope[COLLECTIONS.documents].count_documents({})
    except Exception:  # pragma: no cover - stats must not break the endpoint
        document_count = -1

    try:
        chunk_count = await store.count_chunks(tenant_id)
    except Exception:  # pragma: no cover
        chunk_count = -1

    return {
        "backend": store.backend,
        "backend_reason": selection_reason(),
        "embedding_model": provider.name,
        "embedding_dimension": provider.dimension,
        "chunk_count": chunk_count,
        "document_count": document_count,
        "tenant_id": tenant_id,
    }
