"""Tenant isolation tests.

The claim under test is not "queries happen to be filtered" but "an unfiltered
query is not expressible through the supported path". So these tests drive the
real guard — :class:`TenantScope` — against a fake Mongo that records exactly
what filter it was handed, rather than trusting that callers remembered.

No live Mongo, no training run.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core.tenant_context import TENANT_HEADER, resolve_tenant_id
from app.db import (
    CrossTenantError,
    MissingTenantError,
    TenantScope,
    normalize_tenant_id,
    scoped_filter,
    tenant_field_for,
)
from app.models.machine import Machine
from app.models.reading import SensorReading
from app.models.tenant import Tenant
from app.schemas.machine import COLLECTIONS
from app.schemas.resolution import ResolutionStatus
from app.sensors.simulator import MachineSimulator
from app.services.ingestion import IngestionService
from app.services.resolver import InMemoryMachineRepository, resolve_machine

TENANT_A = "demo"
TENANT_B = "acme"
NOW = datetime(2026, 7, 22, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# A fake Mongo that remembers what it was asked
# ---------------------------------------------------------------------------
class FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, *_args, **_kwargs):
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def __aiter__(self):
        async def gen():
            for doc in self._docs:
                yield doc

        return gen()


def _matches(doc: dict, query: dict) -> bool:
    """Enough of Mongo's matcher for these tests: equality and dotted paths."""
    for key, expected in query.items():
        value = doc
        for part in key.split("."):
            value = (value or {}).get(part) if isinstance(value, dict) else None
        if isinstance(expected, dict) and "$in" in expected:
            if value not in expected["$in"]:
                return False
        elif value != expected:
            return False
    return True


class FakeCollection:
    def __init__(self, name, docs=None):
        self.name = name
        self.docs = list(docs or [])
        self.seen_filters: list[dict] = []
        self.inserted: list[dict] = []

    def find(self, query=None, *_a, **_kw):
        self.seen_filters.append(dict(query or {}))
        return FakeCursor([d for d in self.docs if _matches(d, query or {})])

    async def find_one(self, query=None, *_a, **_kw):
        self.seen_filters.append(dict(query or {}))
        return next((d for d in self.docs if _matches(d, query or {})), None)

    async def count_documents(self, query=None, **_kw):
        self.seen_filters.append(dict(query or {}))
        return sum(1 for d in self.docs if _matches(d, query or {}))

    async def distinct(self, key, query=None, **_kw):
        self.seen_filters.append(dict(query or {}))
        return sorted(
            {d.get(key) for d in self.docs if _matches(d, query or {})}
        )

    async def insert_many(self, docs, **_kw):
        self.inserted.extend(docs)
        self.docs.extend(docs)
        return None

    async def bulk_write(self, ops, **_kw):
        for op in ops:
            self.seen_filters.append(dict(op._filter))
            self.inserted.append(dict(op._doc))
        return None

    async def create_indexes(self, *_a, **_kw):
        return None


class FakeDatabase:
    def __init__(self, collections=None):
        self._collections = dict(collections or {})

    def __getitem__(self, name):
        return self._collections.setdefault(name, FakeCollection(name))

    async def list_collection_names(self):
        return list(self._collections)

    async def create_collection(self, name, **kwargs):
        self.created = (name, kwargs)
        self._collections[name] = FakeCollection(name)

    async def list_collections(self, filter=None):
        # Real Motor: list_collections() is itself a coroutine that resolves to
        # a cursor — a second await (or .to_list()) is required to read it.
        # This must stay a coroutine method, not a plain sync one returning a
        # cursor-like object, or it stops catching the bug it exists to catch.
        class _Cursor:
            async def to_list(self_inner, length=None):
                return []

        return _Cursor()


# ---------------------------------------------------------------------------
# The guard: no tenant, no query
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("missing", [None, "", "   ", "\t"])
def test_query_helper_raises_when_tenant_id_is_absent(missing):
    """A blank tenant must never degrade into 'match everything'."""
    with pytest.raises(MissingTenantError):
        scoped_filter(missing, {"machine_id": "CV-201"})


def test_query_helper_raises_on_a_non_string_tenant():
    with pytest.raises(MissingTenantError, match="must be a string"):
        scoped_filter(123, {})


def test_scope_cannot_be_constructed_without_a_tenant():
    """The failure happens at construction, before any query is possible."""
    with pytest.raises(MissingTenantError):
        TenantScope(FakeDatabase(), None)


def test_scoped_filter_injects_the_tenant_at_the_top_level():
    built = scoped_filter(TENANT_A, {"machine_id": "CV-201"})

    assert built == {"tenant_id": TENANT_A, "machine_id": "CV-201"}


def test_scoped_filter_cannot_be_widened_by_a_nested_or():
    """``$or`` stays ANDed under the tenant clause, so it cannot escape."""
    built = scoped_filter(
        TENANT_A, {"$or": [{"machine_id": "CV-201"}, {"machine_id": "CV-204"}]}
    )

    assert built["tenant_id"] == TENANT_A
    assert "$or" in built


def test_scoped_filter_refuses_to_query_another_tenant():
    """Naming tenant B through tenant A's scope is a bug, not a request."""
    with pytest.raises(CrossTenantError, match="Refusing to run"):
        scoped_filter(TENANT_A, {"tenant_id": TENANT_B})


def test_readings_are_scoped_on_their_metafield_not_a_flat_field():
    """Time-series ownership lives in ``meta`` — the guard must know that."""
    assert tenant_field_for(COLLECTIONS.sensor_readings) == "meta.tenant_id"
    assert tenant_field_for(COLLECTIONS.machines) == "tenant_id"

    built = scoped_filter(TENANT_A, {}, collection=COLLECTIONS.sensor_readings)
    assert built == {"meta.tenant_id": TENANT_A}


@pytest.mark.parametrize("value", ["", "  ", None])
def test_normalize_tenant_id_rejects_empty_values(value):
    with pytest.raises(MissingTenantError):
        normalize_tenant_id(value)


def test_normalize_tenant_id_strips_padding():
    assert normalize_tenant_id("  demo ") == "demo"


# ---------------------------------------------------------------------------
# Every read through the scope carries the tenant
# ---------------------------------------------------------------------------
MACHINE_A = {
    "tenant_id": TENANT_A,
    "machine_id": "CV-201",
    "name": "Demo Infeed Conveyor",
    "model": "SpanTech SB-3000",
    "line_id": "LINE-A",
    "status": "running",
    "aliases": ["Infeed Conveyor"],
}
MACHINE_B = {
    "tenant_id": TENANT_B,
    "machine_id": "CV-201",
    "name": "Acme Infeed Conveyor",
    "model": "Acme Conveyor 9000",
    "line_id": "LINE-1",
    "status": "running",
    "aliases": ["Acme Belt"],
}


def _scope(tenant_id, docs=None):
    database = FakeDatabase(
        {COLLECTIONS.machines: FakeCollection(COLLECTIONS.machines, docs or [])}
    )
    return TenantScope(database, tenant_id), database


@pytest.mark.asyncio
async def test_machine_lookup_cannot_see_another_tenants_machine():
    """The headline guarantee: tenant A asking for CV-201 gets A's CV-201."""
    scope, _ = _scope(TENANT_A, [MACHINE_A, MACHINE_B])

    found = await scope[COLLECTIONS.machines].find_one({"machine_id": "CV-201"})

    assert found["name"] == "Demo Infeed Conveyor"
    assert found["tenant_id"] == TENANT_A


@pytest.mark.asyncio
async def test_a_tenant_with_no_machines_sees_an_empty_fleet():
    """Acme is seeded empty; isolation must be observable, not just asserted."""
    scope, _ = _scope(TENANT_B, [MACHINE_A])

    machines = [m async for m in scope[COLLECTIONS.machines].find({})]

    assert machines == []


@pytest.mark.asyncio
async def test_the_tenant_clause_reaches_the_database_on_every_read():
    """Proof the filter is applied at the driver, not merely post-filtered."""
    scope, database = _scope(TENANT_A, [MACHINE_A, MACHINE_B])
    collection = database[COLLECTIONS.machines]

    await scope[COLLECTIONS.machines].find_one({"machine_id": "CV-201"})
    [m async for m in scope[COLLECTIONS.machines].find({})]
    await scope[COLLECTIONS.machines].count_documents({})
    await scope[COLLECTIONS.machines].distinct("model")

    assert len(collection.seen_filters) == 4
    assert all(f.get("tenant_id") == TENANT_A for f in collection.seen_filters)


@pytest.mark.asyncio
async def test_aggregate_forces_the_tenant_match_into_first_position():
    scope, database = _scope(TENANT_A, [MACHINE_A, MACHINE_B])
    captured = {}

    def fake_aggregate(pipeline, **_kw):
        captured["pipeline"] = pipeline
        return FakeCursor([])

    database[COLLECTIONS.machines].aggregate = fake_aggregate
    scope[COLLECTIONS.machines].aggregate([{"$group": {"_id": "$model"}}])

    assert captured["pipeline"][0] == {"$match": {"tenant_id": TENANT_A}}


# ---------------------------------------------------------------------------
# Two tenants, one machine_id
# ---------------------------------------------------------------------------
def test_two_tenants_may_hold_the_same_machine_id():
    """CV-201 belongs to whoever is asking — uniqueness is per tenant."""
    demo = Machine(
        tenant_id=TENANT_A, machine_id="CV-201", name="Demo Conveyor",
        model="SpanTech SB-3000", manufacturer="SpanTech", site_id="SITE-DETROIT",
        line_id="LINE-A", position_in_line=1, criticality=3,
    )
    acme = Machine(
        tenant_id=TENANT_B, machine_id="CV-201", name="Acme Conveyor",
        model="Acme 9000", manufacturer="Acme", site_id="SITE-RENO",
        line_id="LINE-1", position_in_line=1, criticality=4,
    )

    assert demo.machine_id == acme.machine_id
    assert demo.tenant_id != acme.tenant_id
    # The unique index is (tenant_id, machine_id), so these two rows coexist.
    assert (demo.tenant_id, demo.machine_id) != (acme.tenant_id, acme.machine_id)


@pytest.mark.asyncio
async def test_each_tenant_sees_only_its_own_copy_of_a_shared_id():
    for tenant, expected in ((TENANT_A, "Demo Infeed Conveyor"), (TENANT_B, "Acme Infeed Conveyor")):
        scope, _ = _scope(tenant, [MACHINE_A, MACHINE_B])
        found = await scope[COLLECTIONS.machines].find_one({"machine_id": "CV-201"})
        assert found["name"] == expected


def test_tenant_model_rejects_a_blank_id():
    with pytest.raises(ValueError):
        Tenant(tenant_id="", name="Nameless", slug="nameless")


def test_tenant_model_rejects_padded_ids():
    """' demo' and 'demo' must not be able to become two tenants."""
    with pytest.raises(ValueError, match="whitespace"):
        Tenant(tenant_id=" demo", name="Demo", slug="demo")


def test_documents_cannot_be_built_without_an_owner():
    with pytest.raises(ValueError, match="tenant_id"):
        Machine(
            machine_id="CV-201", name="Orphan", model="m", manufacturer="x",
            site_id="s", line_id="l", position_in_line=1, criticality=3,
        )


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------
ERROR_CODE_A = {
    "tenant_id": TENANT_A, "code": "E104", "machine_model": "SpanTech SB-3000",
    "description": "Belt slip", "fault_class": "mechanical",
}
ERROR_CODE_B = {
    "tenant_id": TENANT_B, "code": "E104", "machine_model": "Acme Conveyor 9000",
    "description": "Acme belt slip", "fault_class": "mechanical",
}


@pytest.fixture
def two_tenant_repo() -> InMemoryMachineRepository:
    return InMemoryMachineRepository(
        machines=[MACHINE_A, MACHINE_B], error_codes=[ERROR_CODE_A, ERROR_CODE_B]
    )


@pytest.mark.asyncio
async def test_resolver_returns_only_the_requesting_tenants_machine(two_tenant_repo):
    """Both tenants own a CV-201; each must resolve to their own.

    Getting this wrong hands an operator another plant's lockout procedure.
    """
    demo = await resolve_machine("CV-201", TENANT_A, repository=two_tenant_repo)
    acme = await resolve_machine("CV-201", TENANT_B, repository=two_tenant_repo)

    assert demo.status == ResolutionStatus.resolved
    assert acme.status == ResolutionStatus.resolved
    assert demo.machine.name == "Demo Infeed Conveyor"
    assert acme.machine.name == "Acme Infeed Conveyor"
    # Same id, same confidence, different machine entirely.
    assert demo.machine.machine_id == acme.machine.machine_id == "CV-201"


@pytest.mark.asyncio
async def test_resolver_cannot_fuzzy_match_into_another_tenant(two_tenant_repo):
    """Acme's alias must not surface for demo, even as a weak candidate."""
    result = await resolve_machine("Acme Belt", TENANT_A, repository=two_tenant_repo)

    names = {c.name for c in result.candidates}
    assert "Acme Infeed Conveyor" not in names
    assert all(c.machine_id != "CV-201" or c.name == "Demo Infeed Conveyor"
               for c in result.candidates)


@pytest.mark.asyncio
async def test_resolver_error_codes_are_tenant_scoped(two_tenant_repo):
    """E104 exists for both tenants against different models."""
    demo = await resolve_machine("fault E104", TENANT_A, repository=two_tenant_repo)
    acme = await resolve_machine("fault E104", TENANT_B, repository=two_tenant_repo)

    assert demo.machine is not None and demo.machine.name == "Demo Infeed Conveyor"
    assert acme.machine is not None and acme.machine.name == "Acme Infeed Conveyor"


@pytest.mark.asyncio
async def test_resolver_finds_nothing_for_a_tenant_with_no_fleet(two_tenant_repo):
    result = await resolve_machine("CV-201", "empty-tenant", repository=two_tenant_repo)

    assert result.status == ResolutionStatus.not_found


@pytest.mark.asyncio
async def test_resolver_refuses_to_run_without_a_tenant(two_tenant_repo):
    with pytest.raises(MissingTenantError):
        await resolve_machine("CV-201", "", repository=two_tenant_repo)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
SENSOR_DOCS = [
    {
        "tenant_id": TENANT_A, "sensor_id": "CV-201-VIB-01", "machine_id": "CV-201",
        "type": "vibration", "unit": "mm/s", "normal_min": 0.0, "normal_max": 2.8,
        "warning_threshold": 4.5, "critical_threshold": 7.1,
    },
    {
        "tenant_id": TENANT_A, "sensor_id": "CV-201-TMP-01", "machine_id": "CV-201",
        "type": "temperature", "unit": "C", "normal_min": 15.0, "normal_max": 70.0,
        "warning_threshold": 80.0, "critical_threshold": 95.0,
    },
]


@pytest.mark.asyncio
async def test_ingestion_writes_readings_carrying_the_correct_tenant():
    """Both destinations must record ownership, in their own layouts."""
    database = FakeDatabase()
    simulator = MachineSimulator(tenant_id=TENANT_A, sensors=SENSOR_DOCS)
    simulator._connected = True
    service = IngestionService(
        source=simulator, tenant_id=TENANT_A, database=database,
        batch_size=1000, flush_seconds=1000,
    )

    service._buffer.extend(simulator.tick(2.0))
    await service._flush()

    history = database[COLLECTIONS.sensor_readings].inserted
    latest = database[COLLECTIONS.latest_readings].inserted

    assert len(history) == len(SENSOR_DOCS)
    # Time-series: ownership lives in meta, which is what Mongo buckets on.
    assert all(d["meta"]["tenant_id"] == TENANT_A for d in history)
    assert all("tenant_id" not in d or d["tenant_id"] == TENANT_A for d in history)
    # latest_readings is a flat collection.
    assert latest and all(d["tenant_id"] == TENANT_A for d in latest)

    # And the upsert key was scoped, so it cannot clobber another tenant's row.
    upsert_filters = database[COLLECTIONS.latest_readings].seen_filters
    assert all(f["tenant_id"] == TENANT_A for f in upsert_filters)


@pytest.mark.asyncio
async def test_ingestion_refuses_a_source_belonging_to_another_tenant():
    """Two answers to 'whose data is this' is a wiring bug, not a merge."""
    simulator = MachineSimulator(tenant_id=TENANT_B, sensors=SENSOR_DOCS)

    with pytest.raises(ValueError, match="emits readings for"):
        IngestionService(source=simulator, tenant_id=TENANT_A, database=FakeDatabase())


def test_ingestion_requires_a_tenant():
    simulator = MachineSimulator(tenant_id=TENANT_A, sensors=SENSOR_DOCS)

    with pytest.raises(MissingTenantError):
        IngestionService(source=simulator, tenant_id="", database=FakeDatabase())


def test_a_source_cannot_be_built_without_a_tenant():
    with pytest.raises(MissingTenantError):
        MachineSimulator(tenant_id="", sensors=SENSOR_DOCS)


def test_readings_require_an_owner():
    with pytest.raises(ValueError, match="tenant_id"):
        SensorReading(
            sensor_id="CV-201-VIB-01", machine_id="CV-201",
            sensor_type="vibration", value=1.0, unit="mm/s",
        )


def test_reading_document_puts_the_tenant_in_the_timeseries_metafield():
    reading = SensorReading(
        tenant_id=TENANT_A, sensor_id="CV-201-VIB-01", machine_id="CV-201",
        sensor_type="vibration", value=1.0, unit="mm/s", timestamp=NOW,
    )

    doc = reading.to_document()

    assert doc["meta"] == {
        "tenant_id": TENANT_A,
        "sensor_id": "CV-201-VIB-01",
        "machine_id": "CV-201",
    }
    assert doc["timestamp"] == NOW
    assert "tenant_id" not in doc  # it lives in meta, not at the top level


# ---------------------------------------------------------------------------
# Request-level resolution
# ---------------------------------------------------------------------------
def test_header_wins_over_the_configured_default():
    assert resolve_tenant_id(TENANT_B) == TENANT_B


def test_default_tenant_is_used_when_the_header_is_absent(monkeypatch):
    from app import config

    monkeypatch.setattr(
        config, "get_settings", lambda: type("S", (), {"default_tenant_id": "fallback"})()
    )
    from app.core import tenant_context

    monkeypatch.setattr(tenant_context, "get_settings", config.get_settings)

    assert tenant_context.resolve_tenant_id(None) == "fallback"
    assert tenant_context.resolve_tenant_id("   ") == "fallback"


def test_unresolvable_tenant_is_a_400(monkeypatch):
    from fastapi import HTTPException

    from app import config
    from app.core import tenant_context

    monkeypatch.setattr(
        config, "get_settings", lambda: type("S", (), {"default_tenant_id": ""})()
    )
    monkeypatch.setattr(tenant_context, "get_settings", config.get_settings)

    with pytest.raises(HTTPException) as excinfo:
        tenant_context.resolve_tenant_id(None)

    assert excinfo.value.status_code == 400
    assert TENANT_HEADER in excinfo.value.detail
