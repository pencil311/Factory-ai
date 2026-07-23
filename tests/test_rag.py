"""Tests for the RAG knowledge engine.

No live Mongo and no network: the vector store is the in-memory numpy backend,
the embedding provider is the deterministic hashing one (no model download),
and Mongo is a fake that records what it was asked.

The tests target the properties that make retrieval trustworthy — tables kept
whole, headings propagated, tenant isolation, and exact-code precision — rather
than the mechanics of any one backend.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from app.models.document import DocType, DocumentStatus
from app.rag import chunking
from app.rag.chunking import (
    ChunkDraft,
    ParsedDocument,
    ParsedPage,
    chunk_document,
    detect_heading,
    is_table_row,
    parse_bytes,
)
from app.rag.embeddings import (
    APIEmbeddings,
    EmbeddingError,
    HashingEmbeddings,
    build_provider,
)
from app.rag.ingest import content_hash, delete_document, ingest_document
from app.rag.retriever import bm25_scores, code_tokens, retrieve
from app.rag.vector_store import NumpyVectorStore
from app.schemas.machine import COLLECTIONS

TENANT_A = "demo"
TENANT_B = "acme"


# ---------------------------------------------------------------------------
# Fake Mongo — enough for the documents/chunks collections
# ---------------------------------------------------------------------------
class FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, *_a, **_kw):
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def __aiter__(self):
        async def gen():
            for doc in self._docs:
                yield doc

        return gen()


def _matches(doc, query):
    for key, expected in query.items():
        value = doc
        for part in key.split("."):
            value = (value or {}).get(part) if isinstance(value, dict) else None
        if isinstance(expected, dict):
            if "$in" in expected and value not in expected["$in"]:
                return False
            if "$eq" in expected and value != expected["$eq"]:
                return False
        elif isinstance(value, list):
            if expected not in value:
                return False
        elif value != expected:
            return False
    return True


class FakeCollection:
    def __init__(self, name, docs=None):
        self.name = name
        self.docs = list(docs or [])

    def find(self, query=None, *_a, **_kw):
        return FakeCursor([d for d in self.docs if _matches(d, query or {})])

    async def find_one(self, query=None, *_a, **_kw):
        return next((d for d in self.docs if _matches(d, query or {})), None)

    async def count_documents(self, query=None, **_kw):
        return sum(1 for d in self.docs if _matches(d, query or {}))

    async def insert_many(self, docs, **_kw):
        self.docs.extend(docs)

    async def delete_many(self, query=None, **_kw):
        before = len(self.docs)
        self.docs = [d for d in self.docs if not _matches(d, query or {})]

        class R:
            deleted_count = before - len(self.docs)

        return R()

    async def delete_one(self, query=None, **_kw):
        for index, doc in enumerate(self.docs):
            if _matches(doc, query or {}):
                self.docs.pop(index)

                class R:
                    deleted_count = 1

                return R()

        class R:
            deleted_count = 0

        return R()

    async def bulk_write(self, ops, **_kw):
        """Apply ReplaceOne upserts, as TenantCollection.upsert_many issues them.

        A no-op double here would make re-ingest look like a first ingest and
        quietly pass the idempotency test, so it does the real replace.
        """
        upserted = modified = 0
        for op in ops:
            query, replacement = dict(op._filter), dict(op._doc)
            for index, doc in enumerate(self.docs):
                if _matches(doc, query):
                    self.docs[index] = replacement
                    modified += 1
                    break
            else:
                self.docs.append(replacement)
                upserted += 1

        class R:
            upserted_count = upserted
            modified_count = modified
            upserted_id = object() if upserted else None

        return R()


class FakeDatabase:
    def __init__(self):
        self._collections = {}

    def __getitem__(self, name):
        return self._collections.setdefault(name, FakeCollection(name))


@pytest.fixture
def fake_db(monkeypatch):
    """Route every tenant scope at an in-memory database."""
    database = FakeDatabase()
    from app import db as db_module

    monkeypatch.setattr(db_module, "get_database", lambda: database)
    return database


@pytest.fixture
def provider():
    """Deterministic embeddings: no model download, no network."""
    return HashingEmbeddings(dimension=256)


@pytest.fixture
def store():
    return NumpyVectorStore(persist_path=None)


# ---------------------------------------------------------------------------
# Chunking: tables
# ---------------------------------------------------------------------------
ERROR_TABLE_DOC = """# CV-201 Manual

## 3. Fault Codes

| Code | Description | Action |
| --- | --- | --- |
| E101 | Belt slip detected | Check belt tension and drive coupling |
| E102 | Drive motor overtemperature | Check ventilation and load |
| E103 | Belt tracking fault | Adjust tail roller alignment |
| E104 | Emergency stop circuit open | Reset all E-stops then the relay |

## 4. Routine Checks

Inspect for abnormal noise at the start of each shift.
"""


def test_table_is_not_split_across_chunks():
    """An error-code table must survive as one passage.

    Split, 'E104' lands in one chunk and its meaning in another, and neither
    answers the question.
    """
    parsed = ParsedDocument(pages=[ParsedPage(1, ERROR_TABLE_DOC)])
    chunks = chunk_document(parsed, target_tokens=40, overlap_tokens=10)

    table_chunks = [c for c in chunks if "E101" in c.text]
    assert len(table_chunks) == 1, "the table was split across chunks"

    table = table_chunks[0]
    # Every row, in one passage, despite a target of 40 tokens.
    for code in ("E101", "E102", "E103", "E104"):
        assert code in table.text
    assert "Reset all E-stops" in table.text
    assert table.is_table is True


def test_table_survives_a_target_smaller_than_the_table():
    """Table integrity beats the token target, deliberately."""
    parsed = ParsedDocument(pages=[ParsedPage(1, ERROR_TABLE_DOC)])
    chunks = chunk_document(parsed, target_tokens=5, overlap_tokens=0)

    tables = [c for c in chunks if c.is_table]
    assert len(tables) == 1
    assert tables[0].token_count > 5


@pytest.mark.parametrize(
    "line,expected",
    [
        ("| E101 | Belt slip | Check tension |", True),
        ("| --- | --- | --- |", True),
        ("Code        Description        Action", True),
        ("This is an ordinary sentence about the conveyor.", False),
        ("", False),
    ],
)
def test_table_row_detection(line, expected):
    assert is_table_row(line) is expected


# ---------------------------------------------------------------------------
# Chunking: headings
# ---------------------------------------------------------------------------
def test_section_titles_propagate_to_chunks():
    """A passage without its heading is unusable; the heading rides along."""
    parsed = ParsedDocument(pages=[ParsedPage(1, ERROR_TABLE_DOC)])
    chunks = chunk_document(parsed)

    by_section = {c.section_title for c in chunks}
    assert "3. Fault Codes" in by_section
    assert "4. Routine Checks" in by_section

    table = next(c for c in chunks if c.is_table)
    assert table.section_title == "3. Fault Codes"


def test_heading_carries_across_a_page_boundary():
    """A section continuing onto the next page keeps its title."""
    parsed = ParsedDocument(
        pages=[
            ParsedPage(1, "## 6.2 Bearing Replacement\n\nTorque the housing bolts."),
            ParsedPage(2, "Restore belt tension to the recorded position."),
        ]
    )
    chunks = chunk_document(parsed)

    page_two = [c for c in chunks if c.page_number == 2]
    assert page_two
    assert all(c.section_title == "6.2 Bearing Replacement" for c in page_two)


@pytest.mark.parametrize(
    "line,expected",
    [
        ("## 3. Fault Codes", "3. Fault Codes"),
        ("6.2 Drive Roller Bearing Replacement", "6.2 Drive Roller Bearing Replacement"),
        ("TECHNICAL SPECIFICATIONS", "TECHNICAL SPECIFICATIONS"),
        ("The bearing should be replaced when worn.", None),
        ("| E101 | Belt slip | Check |", None),
        # Numbered LIST items look like numbered headings but are prose: they
        # run long and break mid-clause. Treating one as a heading mis-cites
        # every chunk beneath it.
        ("4. Stored energy — hydraulic pressure, compressed air, suspended loads,", None),
        ("1. Every person working on isolated equipment applies their own lock.", None),
    ],
)
def test_heading_detection(line, expected):
    assert detect_heading(line) == expected


def test_numbered_list_items_do_not_become_section_titles():
    """A policy document's numbered rules must not each become a section."""
    policy = """# Lockout Policy

## Rules

1. Every person working on isolated equipment applies their own lock and tag.
2. A lock is removed only by the person who applied it, without exception.
3. Isolation must be verified by attempting a start before work begins.
"""
    chunks = chunk_document(ParsedDocument(pages=[ParsedPage(1, policy)]))

    assert {c.section_title for c in chunks} == {"Rules"}


def test_page_numbers_are_recorded_for_citation():
    parsed = ParsedDocument(
        pages=[ParsedPage(1, "First page content here."), ParsedPage(7, "Later page.")]
    )
    chunks = chunk_document(parsed)

    assert {c.page_number for c in chunks} == {1, 7}


def test_sentences_are_never_split_mid_sentence():
    text = " ".join(f"Sentence number {i} about the drive roller bearing." for i in range(40))
    parsed = ParsedDocument(pages=[ParsedPage(1, text)])

    chunks = chunk_document(parsed, target_tokens=30, overlap_tokens=5)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.endswith("."), f"chunk ends mid-sentence: {chunk.text[-40:]!r}"


def test_markdown_and_text_parsing():
    parsed = parse_bytes(b"# Title\n\nSome body text.", "manual.md")
    assert parsed.page_count == 1
    assert "Some body text." in parsed.full_text


def test_unsupported_file_type_is_rejected_loudly():
    with pytest.raises(chunking.UnsupportedFileType, match="Supported types"):
        parse_bytes(b"\x00\x01", "firmware.bin")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def test_hashing_embeddings_are_deterministic_and_normalised(provider):
    first = provider.encode(["drive roller bearing E104"])
    second = provider.encode(["drive roller bearing E104"])

    assert (first == second).all()
    assert first.shape == (1, 256)
    assert abs(float((first[0] ** 2).sum()) - 1.0) < 1e-5


def test_embedding_dimension_drives_index_creation():
    """The index must be sized from the provider, never from a literal."""
    for dimension in (128, 384, 768):
        p = HashingEmbeddings(dimension=dimension)
        assert p.dimension == dimension
        assert p.encode(["anything"]).shape == (1, dimension)


@pytest.mark.asyncio
async def test_numpy_store_rejects_a_dimension_change(store, provider):
    """A model swap must not silently feed the wrong width into the index."""
    await store.ensure_index(provider.dimension)
    await store.upsert_chunks(
        TENANT_A,
        [{"chunk_id": "c1", "document_id": "d1", "text": "x", "embedding": [0.0] * 256}],
    )

    with pytest.raises(ValueError, match="re-ingest|Re-ingest"):
        await store.ensure_index(512)


def test_api_provider_reports_its_configured_dimension():
    api = APIEmbeddings(model_name="text-embedding-3-small", dimension=1536)

    assert api.dimension == 1536
    assert api.name == "api:text-embedding-3-small"


def test_api_provider_fails_loudly_without_a_key():
    api = APIEmbeddings(model_name="m", dimension=8, api_key=None)

    with pytest.raises(EmbeddingError, match="EMBEDDING_API_KEY"):
        api.encode(["text"])


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
        build_provider("telepathy")


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
MANUAL_BYTES = ERROR_TABLE_DOC.encode("utf-8")


async def _ingest(tenant, data, filename, store, provider, **kwargs):
    return await ingest_document(
        tenant_id=tenant,
        data=data,
        filename=filename,
        store=store,
        provider=provider,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_ingestion_indexes_a_document(fake_db, store, provider):
    result = await _ingest(TENANT_A, MANUAL_BYTES, "cv-201-manual.md", store, provider,
                           doc_type=DocType.manual, machine_ids=["CV-201"])

    assert result.status == DocumentStatus.indexed
    assert result.chunk_count > 0
    assert result.replaced_existing is False
    assert await store.count_chunks(TENANT_A) == result.chunk_count


@pytest.mark.asyncio
async def test_reingest_of_identical_content_replaces_rather_than_duplicates(
    fake_db, store, provider
):
    """Content hash is the idempotency key; a re-upload must not double up."""
    first = await _ingest(TENANT_A, MANUAL_BYTES, "cv-201-manual.md", store, provider)
    after_first = await store.count_chunks(TENANT_A)

    second = await _ingest(TENANT_A, MANUAL_BYTES, "cv-201-manual.md", store, provider)
    after_second = await store.count_chunks(TENANT_A)

    assert second.replaced_existing is True
    assert second.document_id == first.document_id
    assert after_second == after_first, "re-ingest duplicated chunks"

    documents = fake_db[COLLECTIONS.documents].docs
    assert len([d for d in documents if d["tenant_id"] == TENANT_A]) == 1


@pytest.mark.asyncio
async def test_edited_content_is_a_new_document(fake_db, store, provider):
    """Different bytes are a different document, not a replacement."""
    first = await _ingest(TENANT_A, MANUAL_BYTES, "manual.md", store, provider)
    edited = (ERROR_TABLE_DOC + "\n\nAdditional revision note.").encode("utf-8")
    second = await _ingest(TENANT_A, edited, "manual.md", store, provider)

    assert second.document_id != first.document_id
    assert second.replaced_existing is False


def test_content_hash_is_stable_and_content_sensitive():
    assert content_hash(b"abc") == content_hash(b"abc")
    assert content_hash(b"abc") != content_hash(b"abd")


@pytest.mark.asyncio
async def test_failed_ingestion_records_the_reason_on_the_document(
    fake_db, store, provider
):
    """A failure that only reaches the log leaves a document that looks fine."""
    result = await _ingest(TENANT_A, b"   \n  \n ", "blank.md", store, provider)

    assert result.status == DocumentStatus.failed
    assert result.error and "no text" in result.error

    stored = await fake_db[COLLECTIONS.documents].find_one(
        {"document_id": result.document_id}
    )
    assert stored["status"] == DocumentStatus.failed.value
    assert stored["error"]


@pytest.mark.asyncio
async def test_delete_removes_document_and_its_chunks(fake_db, store, provider):
    result = await _ingest(TENANT_A, MANUAL_BYTES, "manual.md", store, provider)
    assert await store.count_chunks(TENANT_A) > 0

    existed, removed = await delete_document(TENANT_A, result.document_id, store=store)

    assert existed is True
    assert removed == result.chunk_count
    assert await store.count_chunks(TENANT_A) == 0


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
CV_MANUAL = ERROR_TABLE_DOC
MC_MANUAL = """# MC-110 Manual

## 3. Fault Codes

| Code | Description | Action |
| --- | --- | --- |
| A201 | Spindle overload | Reduce feed rate |
| A202 | Spindle bearing overtemperature | Stop and allow to cool |
"""


@pytest_asyncio.fixture
async def populated(fake_db, store, provider):
    """Two tenants, two machines, general docs — the isolation fixture."""
    await _ingest(TENANT_A, CV_MANUAL.encode(), "cv-201-manual.md", store, provider,
                  doc_type=DocType.manual, machine_ids=["CV-201"],
                  machine_models=["SpanTech SB-3000"])
    await _ingest(TENANT_A, MC_MANUAL.encode(), "mc-110-manual.md", store, provider,
                  doc_type=DocType.manual, machine_ids=["MC-110"],
                  machine_models=["Haas VF-4"])
    await _ingest(TENANT_A, b"# Safety\n\nLockout tagout applies to all equipment.",
                  "lockout.md", store, provider, doc_type=DocType.sop)
    await _ingest(TENANT_B, b"# Acme Manual\n\nAcme conveyor belt slip guidance E101.",
                  "acme-manual.md", store, provider, doc_type=DocType.manual,
                  machine_ids=["CV-201"])
    return store


@pytest.mark.asyncio
async def test_retrieval_never_returns_another_tenants_chunks(populated, provider):
    """The headline guarantee. Both tenants have a CV-201 manual mentioning E101."""
    result = await retrieve(
        TENANT_A, "belt slip E101", store=populated, provider=provider, min_score=0.0
    )

    assert result.chunks
    assert all("Acme" not in c.text for c in result.chunks)

    acme = await retrieve(
        TENANT_B, "belt slip E101", store=populated, provider=provider, min_score=0.0
    )
    assert acme.chunks
    assert all("Acme" in c.text or "acme" in c.document_id.lower() for c in acme.chunks)


@pytest.mark.asyncio
async def test_tenant_with_no_corpus_gets_an_empty_result_with_a_reason(
    populated, provider
):
    result = await retrieve(
        "empty-tenant", "anything at all", store=populated, provider=provider
    )

    assert result.is_empty
    assert result.reason and "No indexed passages" in result.reason


@pytest.mark.asyncio
async def test_machine_filter_restricts_results(populated, provider):
    """Asking about MC-110 must not return CV-201's fault table."""
    result = await retrieve(
        TENANT_A, "fault codes", machine_id="MC-110", machine_model="Haas VF-4",
        store=populated, provider=provider, min_score=0.0,
    )

    assert result.machine_filter_applied is True
    assert result.chunks

    for chunk in result.chunks:
        # Either MC-110's own content, or a tenant-wide general doc.
        assert chunk.machine_ids in ([], ["MC-110"]), chunk.machine_ids
    assert not any("E101" in c.text for c in result.chunks)


@pytest.mark.asyncio
async def test_general_documents_survive_the_machine_filter(populated, provider):
    """Site-wide rules apply to every machine and must not be filtered out."""
    result = await retrieve(
        TENANT_A, "lockout tagout", machine_id="MC-110",
        store=populated, provider=provider, min_score=0.0,
    )

    assert any("Lockout" in c.text or "lockout" in c.text.lower() for c in result.chunks)


@pytest.mark.asyncio
async def test_no_machine_filter_leaves_the_flag_false(populated, provider):
    result = await retrieve(TENANT_A, "fault codes", store=populated, provider=provider,
                            min_score=0.0)

    assert result.machine_filter_applied is False
    assert result.machine_id is None


@pytest.mark.asyncio
async def test_doc_type_filter_restricts_results(populated, provider):
    result = await retrieve(
        TENANT_A, "lockout", doc_types=[DocType.sop.value],
        store=populated, provider=provider, min_score=0.0,
    )

    assert result.chunks
    assert all(c.doc_type == DocType.sop.value for c in result.chunks)


# ---------------------------------------------------------------------------
# Hybrid scoring — the reason it exists
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_exact_error_code_outranks_semantically_similar_chunks(
    fake_db, store, provider
):
    """The case dense retrieval gets wrong.

    Three passages about overheating; only one is about E102 specifically.
    Embeddings rate them near-identically — the exact code match is what puts
    the right row on top.
    """
    corpus = """# Fault Reference

## Overheating Guidance

The motor may overheat under sustained heavy load. Check ventilation, ambient
temperature and cooling airflow around the drive motor housing regularly.

## Thermal Protection Notes

Overtemperature protection trips the drive when winding temperature is
excessive. Allow the motor to cool before restarting after a thermal trip.

## Codes

| Code | Description | Action |
| --- | --- | --- |
| E102 | Drive motor overtemperature trip | Check ventilation and load |
"""
    await _ingest(TENANT_A, corpus.encode(), "faults.md", store, provider)

    result = await retrieve(
        TENANT_A, "E102", store=store, provider=provider, min_score=0.0, top_k=3
    )

    assert result.chunks
    top = result.chunks[0]
    assert "E102" in top.text, (
        f"exact code lost to a semantic neighbour; top chunk was: {top.text[:120]!r}"
    )
    # matched_terms are normalised tokens, so the comparison is lower-case.
    assert "e102" in top.matched_terms
    assert top.keyword_score > 0


def test_code_tokens_recognises_identifier_shapes():
    assert code_tokens("fault E104 on CV-201") >= {"e104", "cv-201"}
    assert code_tokens("replace bearing SKF-6310-2RS1") >= {"skf-6310-2rs1"}
    assert code_tokens("the conveyor is making noise") == set()


def test_bm25_ranks_the_passage_containing_the_rare_term_first():
    documents = [
        "The conveyor belt moves product along the line at a steady speed.",
        "Fault E104 indicates the emergency stop circuit is open.",
        "Routine inspection of the conveyor should happen every shift.",
    ]

    scores = bm25_scores("E104 emergency stop", documents)

    assert scores[1] == max(scores)
    assert scores[1] > 0


def test_bm25_returns_zeros_when_nothing_matches():
    assert bm25_scores("xyzzy plugh", ["conveyor belt", "spindle motor"]) == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Threshold behaviour
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_empty_result_with_reason_when_nothing_clears_the_threshold(
    populated, provider
):
    """A weak passage reads as an answer. Better to return nothing and say so."""
    result = await retrieve(
        TENANT_A,
        "quantum entanglement in submarine navigation",
        store=populated,
        provider=provider,
        min_score=0.99,
    )

    assert result.is_empty
    assert result.reason is not None
    assert "minimum score" in result.reason
    assert result.total_candidates > 0, "candidates existed; they were filtered by score"


@pytest.mark.asyncio
async def test_empty_query_is_rejected_with_a_reason(populated, provider):
    result = await retrieve(TENANT_A, "   ", store=populated, provider=provider)

    assert result.is_empty
    assert result.reason and "Empty query" in result.reason


@pytest.mark.asyncio
async def test_result_carries_provenance_and_backend(populated, provider):
    """Every passage must be citable: document, page, section."""
    result = await retrieve(
        TENANT_A, "fault codes E104", store=populated, provider=provider, min_score=0.0
    )

    assert result.backend_used == "numpy"
    assert result.embedding_model.startswith("hashing:")
    assert result.total_candidates > 0

    chunk = result.chunks[0]
    assert chunk.document_id and chunk.chunk_id
    assert chunk.document_title
    assert chunk.page_number is not None
    assert 0.0 <= chunk.score <= 1.0


@pytest.mark.asyncio
async def test_retrieval_returns_passages_not_prose(populated, provider):
    """The module retrieves; it must not synthesise.

    Returned text is a verbatim slice of an ingested chunk — never a summary.
    """
    result = await retrieve(
        TENANT_A, "belt slip", store=populated, provider=provider, min_score=0.0
    )

    for chunk in result.chunks:
        assert chunk.text in CV_MANUAL or chunk.text in MC_MANUAL or chunk.text in (
            "Lockout tagout applies to all equipment."
        ), "returned text is not verbatim from the corpus"
