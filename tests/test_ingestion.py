"""Tests for the ingestion bootstrap: `app.services.ingestion.ensure_collections`.

The fake database here is deliberately shaped like real Motor, not like a
convenient stand-in: Motor's async methods split into two families, and
mixing them up is exactly the bug this file exists to catch.

* ``list_collection_names`` and ``command`` are directly awaitable — the
  object returned by calling them can be awaited straight away.
* ``list_collections`` (and ``create_collection``) are genuine coroutine
  methods: calling them returns a coroutine that must itself be awaited
  before you get anything usable — here, a cursor with its own ``to_list``.

A fake that makes ``list_collections`` a plain synchronous method returning a
cursor-like object (the way ``find()`` really does behave) passes code that
is broken against real Motor, because the fake never reproduces the
"coroutine with no ``to_list`` attribute" failure. So ``FakeDatabase`` below
returns a coroutine from ``list_collections``, exactly as Motor 3.4.0 does,
and the regression test drives the code path — a pre-existing
``sensor_readings`` collection — that only that shape can catch.
"""

from __future__ import annotations

import pytest

from app.schemas.machine import COLLECTIONS
from app.services.ingestion import _assert_timeseries_layout, ensure_collections


class FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    async def to_list(self, length=None):
        return list(self._docs[: length if length is not None else None])


class FakeIndexes:
    """Records index creation without asserting on it — not this file's concern."""

    async def create_indexes(self, *_args, **_kwargs):
        return None


class FakeDatabase:
    """A fake shaped like Motor's actual split between coroutine methods.

    ``collections`` maps a name to the ``options`` dict Mongo would report for
    it (e.g. ``{"timeseries": {"metaField": "meta"}}``) — enough to drive
    :func:`_assert_timeseries_layout` without a real server.
    """

    def __init__(self, collections: dict[str, dict] | None = None) -> None:
        self._collections = dict(collections or {})
        self.created: list[tuple[str, dict]] = []

    def __getitem__(self, name: str) -> FakeIndexes:
        return FakeIndexes()

    async def list_collection_names(self) -> list[str]:
        # Directly awaitable, no second step — matches real Motor.
        return list(self._collections)

    async def list_collections(self, filter: dict | None = None):
        # A genuine two-step coroutine method: awaiting this call returns a
        # cursor, which itself must be awaited (via to_list) to get documents.
        # Collapsing these two steps into one is exactly the bug under test.
        name = (filter or {}).get("name")
        docs = [
            {"name": n, "options": opts}
            for n, opts in self._collections.items()
            if name is None or n == name
        ]
        return FakeCursor(docs)

    async def create_collection(self, name: str, **kwargs) -> None:
        self.created.append((name, kwargs))
        self._collections[name] = kwargs


# ---------------------------------------------------------------------------
# The reported bug: a pre-existing collection reaches _assert_timeseries_layout
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ensure_collections_does_not_crash_when_the_collection_already_exists():
    """Regression test: this is the exact path that raised AttributeError.

    ``list_collections(...)`` was chained directly with ``.to_list()`` instead
    of being awaited first, so the coroutine object (not the cursor) received
    the ``.to_list()`` call and blew up with
    ``AttributeError: 'coroutine' object has no attribute 'to_list'``. This
    only fires when the collection already exists — a fresh database takes
    the `create_collection` branch and never reaches the broken code.
    """
    database = FakeDatabase(
        {
            COLLECTIONS.sensor_readings: {
                "timeseries": {"timeField": "timestamp", "metaField": "meta"}
            }
        }
    )

    await ensure_collections(database)  # must not raise

    assert database.created == [], "an existing collection must not be recreated"


@pytest.mark.asyncio
async def test_assert_timeseries_layout_passes_a_matching_metafield():
    database = FakeDatabase(
        {COLLECTIONS.sensor_readings: {"timeseries": {"metaField": "meta"}}}
    )
    await _assert_timeseries_layout(database)  # must not raise


@pytest.mark.asyncio
async def test_assert_timeseries_layout_is_a_noop_when_the_collection_is_absent():
    database = FakeDatabase()
    await _assert_timeseries_layout(database)  # nothing to assert; must not raise


@pytest.mark.asyncio
async def test_assert_timeseries_layout_rejects_a_pre_tenancy_metafield():
    """A collection created before multi-tenancy used a different metaField.

    Writing into it would produce history no tenant-scoped query could ever
    find, so this must fail loudly with the remedy rather than silently
    accept the mismatch.
    """
    database = FakeDatabase(
        {COLLECTIONS.sensor_readings: {"timeseries": {"metaField": "sensor_id"}}}
    )

    with pytest.raises(RuntimeError, match="predates multi-tenancy"):
        await _assert_timeseries_layout(database)


@pytest.mark.asyncio
async def test_ensure_collections_creates_a_fresh_timeseries_collection():
    database = FakeDatabase()

    await ensure_collections(database)

    assert len(database.created) == 1
    name, options = database.created[0]
    assert name == COLLECTIONS.sensor_readings
    assert options["timeseries"]["timeField"] == "timestamp"
    assert options["timeseries"]["metaField"] == "meta"
