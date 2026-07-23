"""Tests for `app.rag.vector_store`: Atlas index bootstrap, drift, and filtering.

The fakes here are shaped like real Motor/Atlas, not like convenient
stand-ins — specifically:

* ``create_collection`` is a genuine coroutine (single await, unlike
  ``list_collections``, which needs a second await to read its cursor — see
  ``tests/test_ingestion.py`` for that distinction). Calling it on a name that
  already exists raises ``pymongo.errors.CollectionInvalid``, exactly as real
  Atlas does.
* ``create_search_index`` fails with an ``OperationFailure`` carrying code 26
  (``NamespaceNotFound``) when the collection has never been created — this is
  the exact failure `seed_documents.py` hit on a fresh database, because
  Atlas's ``createSearchIndex`` command, unlike an ordinary Mongo index, does
  not implicitly create its collection.
* ``aggregate()`` on a machine-scoped ``$vectorSearch`` filter rejects ``$eq``
  against an array-valued field with Atlas's real error text — the exact
  failure a filter built as ``{"machine_ids": {"$eq": []}}`` produces, even
  when the array being compared is empty. A fake that skipped this and just
  matched anything would pass code that is broken against real Atlas.

A fake that papered over either distinction would pass code that is broken
against real Atlas, because it would never reproduce the failure the
production code exists to avoid.
"""

from __future__ import annotations

import pytest
from pymongo.errors import CollectionInvalid, OperationFailure

from app.rag.vector_store import AtlasVectorStore
from app.schemas.machine import COLLECTIONS


class FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    async def to_list(self, length=None):
        return list(self._docs[: length] if length is not None else self._docs)


class FakeAtlasCollection:
    """One collection's worth of Atlas Search behaviour.

    ``exists`` models whether the collection has ever been created — the one
    fact ``create_search_index`` cares about and ``list_search_indexes``
    does not. Each entry in ``search_indexes`` mirrors the shape Atlas reports:
    ``{"name": ..., "latestDefinition": {"fields": [...]}}``.
    """

    def __init__(self, name: str, exists: bool) -> None:
        self.name = name
        self.exists = exists
        self.search_indexes: list[dict] = []
        #: Documents this collection would return from `aggregate()`. Only
        #: populated by tests exercising `search()`, not index bootstrap.
        self.docs: list[dict] = []

    def list_search_indexes(self, **_kw):
        # Real Motor: list_search_indexes() returns a cursor synchronously —
        # same family as find()/aggregate(), unlike list_collections(). It
        # succeeds even when the collection does not exist (verified against
        # a live Atlas cluster), so this never raises here either.
        return FakeCursor(list(self.search_indexes))

    async def create_search_index(self, model: dict) -> None:
        if not self.exists:
            # The exact failure mode from the bug report: code 26,
            # NamespaceNotFound, against a collection nothing has written to.
            raise OperationFailure(
                f"Collection 'factorypilot.{self.name}' does not exist.", code=26
            )
        self.search_indexes.append(
            {"name": model["name"], "latestDefinition": model["definition"]}
        )

    async def update_search_index(self, name: str, definition: dict) -> None:
        for entry in self.search_indexes:
            if entry["name"] == name:
                entry["latestDefinition"] = definition
                return
        raise OperationFailure(f"Index '{name}' does not exist.", code=27)

    def aggregate(self, pipeline: list[dict]):
        """Just enough $vectorSearch to prove the machine filter is legal.

        Evaluated eagerly (not lazily on iteration) — close enough to how
        Motor's aggregate() actually performs the round trip, and it means a
        regression to an illegal filter fails the test with the real
        exception rather than silently returning nothing.
        """
        stage = pipeline[0]["$vectorSearch"]
        filter_expr = stage.get("filter") or {}
        matches = [doc for doc in self.docs if _eval_atlas_filter(filter_expr, doc)]
        results = []
        for doc in matches:
            out = dict(doc)
            out["_score"] = 0.9
            out.pop("embedding", None)
            results.append(out)
        return _AsyncIter(results)


class FakeAtlasDatabase:
    """Motor-shaped enough to drive `AtlasVectorStore.ensure_index()`/`search()`.

    ``existing`` seeds which collection names already exist, mirroring a
    database that already has data in it from a previous run.
    """

    def __init__(self, existing: tuple[str, ...] = ()) -> None:
        self._existing = set(existing)
        self._collections: dict[str, FakeAtlasCollection] = {}

    def __getitem__(self, name: str) -> FakeAtlasCollection:
        return self._collections.setdefault(
            name, FakeAtlasCollection(name, exists=name in self._existing)
        )

    async def create_collection(self, name: str, **_kwargs) -> FakeAtlasCollection:
        if name in self._existing:
            # Real pymongo's exact exception for this case.
            raise CollectionInvalid(f"collection {name} already exists")
        self._existing.add(name)
        collection = self[name]
        collection.exists = True
        return collection


class _AsyncIter:
    """A plain async-iterable, matching what Motor's aggregate() returns."""

    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for item in self._items:
            yield item


def _eval_atlas_filter(expr: dict, doc: dict) -> bool:
    """A small MQL-filter interpreter, faithful to the one operator that matters here.

    Raises the real ``OperationFailure`` Atlas raises when ``$eq`` targets an
    array-valued field or an array literal — regardless of whether the array
    is empty — which is exactly the bug this whole test file guards against.
    """
    if "$and" in expr:
        return all(_eval_atlas_filter(clause, doc) for clause in expr["$and"])
    if "$or" in expr:
        return any(_eval_atlas_filter(clause, doc) for clause in expr["$or"])
    for field, cond in expr.items():
        value = doc.get(field)
        for op, operand in cond.items():
            if op == "$eq":
                if isinstance(value, list) or isinstance(operand, list):
                    raise OperationFailure(
                        f'"filter.{field}.$eq" must be a boolean, objectId, '
                        "number, string, date, uuid, or null",
                        code=9,
                    )
                if value != operand:
                    return False
            elif op == "$in":
                if isinstance(value, list):
                    if not (set(value) & set(operand)):
                        return False
                elif value not in operand:
                    return False
            else:
                raise NotImplementedError(f"unsupported operator {op!r} in test fake")
    return True


@pytest.fixture
def store() -> AtlasVectorStore:
    return AtlasVectorStore(index_name="chunk_vector_index")


def _new_index_fields() -> list[dict]:
    """The full field list `ensure_index` declares today, is_general included."""
    return [
        {"type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine"},
        {"type": "filter", "path": "tenant_id"},
        {"type": "filter", "path": "machine_ids"},
        {"type": "filter", "path": "machine_models"},
        {"type": "filter", "path": "doc_type"},
        {"type": "filter", "path": "is_general"},
    ]


# ---------------------------------------------------------------------------
# The reported bug: a fresh database, chunks never written to
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ensure_index_succeeds_when_the_chunks_collection_does_not_exist(
    monkeypatch, store
):
    """Regression test for the seed_documents NamespaceNotFound failure.

    On a brand-new database nothing has ever been inserted into ``chunks``,
    so Atlas's ``createSearchIndex`` would fail with ``NamespaceNotFound``
    unless the collection is created first.
    """
    database = FakeAtlasDatabase(existing=())
    import app.db as db_module

    monkeypatch.setattr(db_module, "get_database", lambda: database)

    await store.ensure_index(384)  # must not raise

    collection = database[COLLECTIONS.chunks]
    assert collection.exists is True
    assert [i["name"] for i in collection.search_indexes] == ["chunk_vector_index"]
    fields = collection.search_indexes[0]["latestDefinition"]["fields"]
    assert {f["path"] for f in fields} == {
        "embedding",
        "tenant_id",
        "machine_ids",
        "machine_models",
        "doc_type",
        "is_general",
    }


@pytest.mark.asyncio
async def test_ensure_index_is_idempotent_once_the_collection_and_index_exist(
    monkeypatch, store
):
    """A second run (the common case) must not fail or duplicate the index."""
    database = FakeAtlasDatabase(existing=(COLLECTIONS.chunks,))
    database[COLLECTIONS.chunks].search_indexes.append(
        {"name": "chunk_vector_index", "latestDefinition": {"fields": _new_index_fields()}}
    )
    import app.db as db_module

    monkeypatch.setattr(db_module, "get_database", lambda: database)

    await store.ensure_index(384)  # must not raise, must not duplicate or update

    assert len(database[COLLECTIONS.chunks].search_indexes) == 1


@pytest.mark.asyncio
async def test_ensure_index_updates_a_drifted_definition_instead_of_leaving_it_stale(
    monkeypatch, store
):
    """An index predating ``is_general`` is updated in place, not left behind.

    The index already exists on a live cluster from before this fix; adding a
    new filter field to the declared definition must update it rather than
    silently keep serving the old one.
    """
    database = FakeAtlasDatabase(existing=(COLLECTIONS.chunks,))
    stale_fields = [f for f in _new_index_fields() if f["path"] != "is_general"]
    database[COLLECTIONS.chunks].search_indexes.append(
        {"name": "chunk_vector_index", "latestDefinition": {"fields": stale_fields}}
    )
    import app.db as db_module

    monkeypatch.setattr(db_module, "get_database", lambda: database)

    await store.ensure_index(384)

    indexes = database[COLLECTIONS.chunks].search_indexes
    assert len(indexes) == 1, "drift must update the existing index, not add a second one"
    paths = {f["path"] for f in indexes[0]["latestDefinition"]["fields"]}
    assert "is_general" in paths


@pytest.mark.asyncio
async def test_ensure_index_tolerates_a_concurrent_collection_creation_race(
    monkeypatch, store
):
    """Two callers racing to create the collection must not fail either one.

    Mirrors the real server's NamespaceExists (code 48) response when a
    concurrent ``createCollection`` wins the race.
    """
    database = FakeAtlasDatabase(existing=())

    async def racing_create_collection(name, **_kwargs):
        # Someone else created it a moment ago; the server reports the race
        # as an OperationFailure rather than CollectionInvalid.
        database._existing.add(name)
        database[name].exists = True
        raise OperationFailure(f"Collection {name} already exists.", code=48)

    monkeypatch.setattr(database, "create_collection", racing_create_collection)
    import app.db as db_module

    monkeypatch.setattr(db_module, "get_database", lambda: database)

    await store.ensure_index(384)  # must not raise

    assert database[COLLECTIONS.chunks].search_indexes


@pytest.mark.asyncio
async def test_ensure_index_does_not_swallow_an_unrelated_operation_failure(
    monkeypatch, store
):
    """Only the known race (code 48) is tolerated — anything else must surface."""
    database = FakeAtlasDatabase(existing=())

    async def failing_create_collection(name, **_kwargs):
        raise OperationFailure("Atlas is having a bad day", code=91)

    monkeypatch.setattr(database, "create_collection", failing_create_collection)
    import app.db as db_module

    monkeypatch.setattr(db_module, "get_database", lambda: database)

    with pytest.raises(OperationFailure, match="bad day"):
        await store.ensure_index(384)


# ---------------------------------------------------------------------------
# documents is unaffected: nothing calls a command requiring pre-existence
# ---------------------------------------------------------------------------
def test_documents_collection_needs_no_such_fix():
    """Documented reasoning, not a runtime check.

    Every operation `app/rag/ingest.py` and `app/routers/knowledge.py` run
    against ``documents`` — find, find_one, upsert_many (insert-or-replace),
    count_documents, delete_one — creates the collection lazily on first
    write, or returns an empty/zero result against a missing one. Only Atlas
    Search's ``createSearchIndex`` (used exclusively for ``chunks``) requires
    the namespace to pre-exist, which is why this fix is scoped to
    ``AtlasVectorStore.ensure_index`` alone.
    """


# ---------------------------------------------------------------------------
# The machine filter: $eq-on-array is illegal, is_general + $in is the fix
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_search_with_a_machine_filter_returns_matching_and_general_chunks(
    monkeypatch,
):
    """Regression test for the "$eq must be a boolean, ... or null" failure.

    machine_ids/machine_models are arrays; Atlas rejects $eq against them
    outright, even to test for emptiness. If ``_machine_clause`` ever
    regresses to ``{"machine_ids": {"$eq": []}}`` for the "applies to every
    machine" branch, the fake collection above raises the exact Atlas error
    and this test fails with it.
    """
    tenant = "acme"
    docs = [
        {
            "tenant_id": tenant,
            "chunk_id": "c1-machine-specific",
            "machine_ids": ["CV-201"],
            "machine_models": [],
            "is_general": False,
            "embedding": [1.0, 0.0],
        },
        {
            "tenant_id": tenant,
            "chunk_id": "c2-general",
            "machine_ids": [],
            "machine_models": [],
            "is_general": True,
            "embedding": [1.0, 0.0],
        },
        {
            "tenant_id": tenant,
            "chunk_id": "c3-other-machine",
            "machine_ids": ["XV-100"],
            "machine_models": [],
            "is_general": False,
            "embedding": [1.0, 0.0],
        },
        {
            "tenant_id": "other-tenant",
            "chunk_id": "c4-other-tenant-general",
            "machine_ids": [],
            "machine_models": [],
            "is_general": True,
            "embedding": [1.0, 0.0],
        },
    ]
    collection = FakeAtlasCollection("chunks", exists=True)
    collection.docs = docs

    class FakeDatabase:
        def __getitem__(self, name):
            return collection

    import app.db as db_module

    monkeypatch.setattr(db_module, "get_database", lambda: FakeDatabase())

    store = AtlasVectorStore(index_name="chunk_vector_index")
    hits = await store.search(
        tenant, query_vector=[1.0, 0.0], top_k=5, machine_ids=["CV-201"]
    )

    ids = {hit.chunk["chunk_id"] for hit in hits}
    # CV-201-specific and tenant-wide general chunks come back...
    assert ids == {"c1-machine-specific", "c2-general"}
    # ...another machine's chunks and another tenant's general chunks do not.
    assert "c3-other-machine" not in ids
    assert "c4-other-tenant-general" not in ids
