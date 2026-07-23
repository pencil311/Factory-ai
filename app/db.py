"""Async MongoDB (Motor) client, tenant-scoped access, and index management.

Tenant isolation is enforced *structurally*, not by discipline. Application
code does not get a raw collection handle; it gets a :class:`TenantCollection`
from :func:`get_tenant_scope`, and that wrapper injects the tenant filter into
every read and stamps the tenant onto every write. Forgetting to filter is not
possible through this path — leaking another tenant's data requires reaching
past it to :func:`get_database` on purpose, which is why that function is
named and documented as the escape hatch it is.

Two failure modes are made loud rather than silent:

* :class:`MissingTenantError` — a query was built with no tenant at all.
* :class:`CrossTenantError` — a filter or document named a *different* tenant
  than the scope it was issued through, which means a bug upstream.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)
from pymongo import ASCENDING, DESCENDING, IndexModel, ReplaceOne

from app.config import get_settings
from app.schemas.machine import COLLECTIONS

#: Field carrying ownership on ordinary documents.
TENANT_FIELD = "tenant_id"

#: ``sensor_readings`` is a time-series collection whose ``metaField`` is the
#: ``meta`` subdocument, so its tenant lives at ``meta.tenant_id``. Series are
#: bucketed by metaField — putting the tenant in there keeps one tenant's
#: readings from sharing a storage bucket with another's, which a flat
#: ``tenant_id`` measurement field would not do.
TENANT_FIELD_BY_COLLECTION: dict[str, str] = {
    COLLECTIONS.sensor_readings: "meta.tenant_id",
}


class MissingTenantError(RuntimeError):
    """Raised when a query or write is attempted without a tenant."""


class CrossTenantError(RuntimeError):
    """Raised when a filter or document names a tenant other than the scope's."""


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------
class _Database:
    """Holds the process-wide Motor client and database handle."""

    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None


_state = _Database()


def get_client() -> AsyncIOMotorClient:
    """Return the active Motor client (creating it lazily if needed)."""
    if _state.client is None:
        connect()
    assert _state.client is not None
    return _state.client


def get_database() -> AsyncIOMotorDatabase:
    """Return the RAW, UNSCOPED database handle.

    This bypasses tenant isolation. It exists for administrative work — index
    creation, collection creation, health pings, seeding — and for nothing
    else. Application queries must go through :func:`get_tenant_scope`; if you
    are reaching for this function to serve a request, that is the bug.
    """
    if _state.db is None:
        connect()
    assert _state.db is not None
    return _state.db


def connect() -> AsyncIOMotorDatabase:
    """Create the Motor client from settings and cache the database handle.

    Uses a DIRECT (non-SRV) connection string; the SRV form is rejected in
    :mod:`app.config`.
    """
    settings = get_settings()
    _state.client = AsyncIOMotorClient(
        settings.mongodb_uri,
        uuidRepresentation="standard",
        serverSelectionTimeoutMS=10_000,
    )
    _state.db = _state.client[settings.mongodb_db]
    return _state.db


async def close() -> None:
    """Close the Motor client and clear cached handles."""
    if _state.client is not None:
        _state.client.close()
    _state.client = None
    _state.db = None


async def ping() -> bool:
    """Return True if the server responds to a ping."""
    db = get_database()
    result = await db.command("ping")
    return bool(result.get("ok"))


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------
def normalize_tenant_id(tenant_id: Any) -> str:
    """Validate and normalise a tenant id, or raise :class:`MissingTenantError`.

    Deliberately strict: ``None``, ``""``, whitespace and non-strings are all
    rejected. A falsy tenant silently becoming "match everything" is exactly
    the class of bug this module exists to prevent.
    """
    if tenant_id is None:
        raise MissingTenantError(
            "No tenant_id supplied. Every query must be scoped to a tenant — "
            "use get_tenant_scope(tenant_id) rather than the raw database."
        )
    if not isinstance(tenant_id, str):
        raise MissingTenantError(
            f"tenant_id must be a string, got {type(tenant_id).__name__}."
        )
    stripped = tenant_id.strip()
    if not stripped:
        raise MissingTenantError("tenant_id must not be blank or whitespace.")
    return stripped


def tenant_field_for(collection: Optional[str]) -> str:
    """The field carrying ownership in ``collection``."""
    return TENANT_FIELD_BY_COLLECTION.get(collection or "", TENANT_FIELD)


def scoped_filter(
    tenant_id: Any,
    query: Optional[Mapping[str, Any]] = None,
    collection: Optional[str] = None,
) -> dict[str, Any]:
    """Build a query filter with the tenant injected at the top level.

    The tenant clause is ANDed at the top level, so nothing nested — ``$or``,
    ``$expr``, ``$nor`` — can widen the result set past the tenant boundary.
    """
    tenant = normalize_tenant_id(tenant_id)
    field = tenant_field_for(collection)
    built = dict(query or {})

    existing = built.get(field)
    if existing is not None and existing != tenant:
        raise CrossTenantError(
            f"Query filter names tenant '{existing}' but was issued through the "
            f"scope for tenant '{tenant}'. Refusing to run a cross-tenant query."
        )
    built[field] = tenant
    return built


def _stamp_document(
    doc: Mapping[str, Any], tenant_id: str, collection: Optional[str]
) -> dict[str, Any]:
    """Return a copy of ``doc`` owned by ``tenant_id``, or raise on conflict."""
    field = tenant_field_for(collection)
    stamped = dict(doc)

    if "." not in field:
        existing = stamped.get(field)
        if existing is not None and existing != tenant_id:
            raise CrossTenantError(
                f"Document declares tenant '{existing}' but is being written "
                f"through the scope for tenant '{tenant_id}'."
            )
        stamped[field] = tenant_id
        return stamped

    parent, child = field.split(".", 1)
    sub = dict(stamped.get(parent) or {})
    existing = sub.get(child)
    if existing is not None and existing != tenant_id:
        raise CrossTenantError(
            f"Document declares tenant '{existing}' in '{field}' but is being "
            f"written through the scope for tenant '{tenant_id}'."
        )
    sub[child] = tenant_id
    stamped[parent] = sub
    return stamped


class TenantCollection:
    """A collection handle bound to one tenant.

    Mirrors the slice of the Motor collection API the application uses. Reads
    are filtered and writes are stamped; there is no method that reaches the
    whole collection.
    """

    def __init__(
        self, collection: AsyncIOMotorCollection, tenant_id: str, name: str
    ) -> None:
        self._collection = collection
        self._tenant_id = tenant_id
        self._name = name

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def name(self) -> str:
        return self._name

    def _filter(self, query: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        return scoped_filter(self._tenant_id, query, collection=self._name)

    def _stamp(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        return _stamp_document(doc, self._tenant_id, self._name)

    # -- reads -------------------------------------------------------------
    def find(self, query: Optional[Mapping[str, Any]] = None, *args: Any, **kwargs: Any):
        return self._collection.find(self._filter(query), *args, **kwargs)

    async def find_one(
        self, query: Optional[Mapping[str, Any]] = None, *args: Any, **kwargs: Any
    ) -> Optional[Mapping[str, Any]]:
        return await self._collection.find_one(self._filter(query), *args, **kwargs)

    async def count_documents(
        self, query: Optional[Mapping[str, Any]] = None, **kwargs: Any
    ) -> int:
        return await self._collection.count_documents(self._filter(query), **kwargs)

    async def distinct(
        self, key: str, query: Optional[Mapping[str, Any]] = None, **kwargs: Any
    ) -> list[Any]:
        return await self._collection.distinct(key, self._filter(query), **kwargs)

    def aggregate(self, pipeline: Sequence[Mapping[str, Any]], **kwargs: Any):
        """Run a pipeline with a tenant ``$match`` forced into first position."""
        scoped = [{"$match": self._filter(None)}, *pipeline]
        return self._collection.aggregate(scoped, **kwargs)

    # -- writes ------------------------------------------------------------
    async def insert_one(self, document: Mapping[str, Any], **kwargs: Any):
        return await self._collection.insert_one(self._stamp(document), **kwargs)

    async def insert_many(self, documents: Iterable[Mapping[str, Any]], **kwargs: Any):
        stamped = [self._stamp(d) for d in documents]
        if not stamped:
            return None
        return await self._collection.insert_many(stamped, **kwargs)

    async def update_one(
        self, query: Mapping[str, Any], update: Mapping[str, Any], **kwargs: Any
    ):
        return await self._collection.update_one(self._filter(query), update, **kwargs)

    async def update_many(
        self, query: Mapping[str, Any], update: Mapping[str, Any], **kwargs: Any
    ):
        return await self._collection.update_many(self._filter(query), update, **kwargs)

    async def replace_one(
        self, query: Mapping[str, Any], replacement: Mapping[str, Any], **kwargs: Any
    ):
        return await self._collection.replace_one(
            self._filter(query), self._stamp(replacement), **kwargs
        )

    async def delete_one(self, query: Mapping[str, Any], **kwargs: Any):
        return await self._collection.delete_one(self._filter(query), **kwargs)

    async def delete_many(self, query: Mapping[str, Any], **kwargs: Any):
        return await self._collection.delete_many(self._filter(query), **kwargs)

    async def upsert_many(
        self, key_fields: Sequence[str], documents: Sequence[Mapping[str, Any]], **kwargs: Any
    ):
        """Replace-upsert documents keyed on ``key_fields`` *within this tenant*.

        Takes documents rather than pymongo operation objects on purpose: a
        caller cannot hand us a pre-built ``ReplaceOne`` whose filter we would
        have to reach inside and rewrite. The upsert key is always
        ``(tenant, *key_fields)``, which matches the compound unique indexes.
        """
        if not documents:
            return None
        operations = [
            ReplaceOne(
                self._filter({k: doc[k] for k in key_fields}),
                self._stamp(doc),
                upsert=True,
            )
            for doc in documents
        ]
        return await self._collection.bulk_write(operations, **kwargs)


class TenantScope:
    """A tenant-bound view of the database. ``scope[name]`` yields a collection."""

    def __init__(self, database: AsyncIOMotorDatabase, tenant_id: Any) -> None:
        self._database = database
        self._tenant_id = normalize_tenant_id(tenant_id)

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    def __getitem__(self, name: str) -> TenantCollection:
        return TenantCollection(self._database[name], self._tenant_id, name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<TenantScope tenant_id={self._tenant_id!r}>"


def get_tenant_scope(
    tenant_id: Any, database: Optional[AsyncIOMotorDatabase] = None
) -> TenantScope:
    """The tenant-scoped database handle application code should use."""
    return TenantScope(database if database is not None else get_database(), tenant_id)


# ---------------------------------------------------------------------------
# Indexes — every one compound, tenant first
# ---------------------------------------------------------------------------
async def create_indexes() -> None:
    """Create the indexes the application relies on. Idempotent.

    Every index leads with ``tenant_id`` so that (a) uniqueness is per tenant —
    two tenants may each own a machine called CV-201 — and (b) every scoped
    query, which always carries a tenant equality clause, hits an index prefix.
    """
    db = get_database()

    # tenants: the registry itself is global, keyed by its own id
    await db[COLLECTIONS.tenants].create_indexes(
        [
            IndexModel([("tenant_id", ASCENDING)], name="uq_tenant_id", unique=True),
            IndexModel([("slug", ASCENDING)], name="uq_tenant_slug", unique=True),
        ]
    )

    # machines: unique per tenant, plus lookups by alias and line
    await db[COLLECTIONS.machines].create_indexes(
        [
            IndexModel(
                [("tenant_id", ASCENDING), ("machine_id", ASCENDING)],
                name="uq_tenant_machine_id",
                unique=True,
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("aliases", ASCENDING)],
                name="ix_machine_tenant_aliases",
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("line_id", ASCENDING)],
                name="ix_machine_tenant_line_id",
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("site_id", ASCENDING)],
                name="ix_machine_tenant_site_id",
            ),
        ]
    )

    # components: unique per tenant + fast lookup by machine
    await db[COLLECTIONS.components].create_indexes(
        [
            IndexModel(
                [("tenant_id", ASCENDING), ("component_id", ASCENDING)],
                name="uq_tenant_component_id",
                unique=True,
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("machine_id", ASCENDING)],
                name="ix_component_tenant_machine_id",
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("parent_component_id", ASCENDING)],
                name="ix_component_tenant_parent_id",
            ),
        ]
    )

    # sensors: unique per tenant + lookups by machine and component
    await db[COLLECTIONS.sensors].create_indexes(
        [
            IndexModel(
                [("tenant_id", ASCENDING), ("sensor_id", ASCENDING)],
                name="uq_tenant_sensor_id",
                unique=True,
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("machine_id", ASCENDING)],
                name="ix_sensor_tenant_machine_id",
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("component_id", ASCENDING)],
                name="ix_sensor_tenant_component_id",
            ),
        ]
    )

    # error codes: unique per (tenant, code, machine_model)
    await db[COLLECTIONS.error_codes].create_indexes(
        [
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("code", ASCENDING),
                    ("machine_model", ASCENDING),
                ],
                name="uq_tenant_errorcode_code_model",
                unique=True,
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("machine_model", ASCENDING)],
                name="ix_errorcode_tenant_model",
            ),
        ]
    )

    # sites / lines: unique per tenant
    await db[COLLECTIONS.sites].create_indexes(
        [
            IndexModel(
                [("tenant_id", ASCENDING), ("site_id", ASCENDING)],
                name="uq_tenant_site_id",
                unique=True,
            )
        ]
    )
    await db[COLLECTIONS.lines].create_indexes(
        [
            IndexModel(
                [("tenant_id", ASCENDING), ("line_id", ASCENDING)],
                name="uq_tenant_line_id",
                unique=True,
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("site_id", ASCENDING)],
                name="ix_line_tenant_site_id",
            ),
        ]
    )

    # latest_readings: one document per (tenant, sensor)
    await db[COLLECTIONS.latest_readings].create_indexes(
        [
            IndexModel(
                [("tenant_id", ASCENDING), ("sensor_id", ASCENDING)],
                name="uq_tenant_latest_sensor",
                unique=True,
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("machine_id", ASCENDING)],
                name="ix_latest_tenant_machine",
            ),
        ]
    )

    # parts: unique per (tenant, part_number)
    await db[COLLECTIONS.parts].create_indexes(
        [
            IndexModel(
                [("tenant_id", ASCENDING), ("part_number", ASCENDING)],
                name="uq_tenant_part_number",
                unique=True,
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("compatible_components", ASCENDING)],
                name="ix_part_tenant_compatible_components",
            ),
        ]
    )

    # conversations: one document per (tenant, session)
    await db[COLLECTIONS.conversations].create_indexes(
        [
            IndexModel(
                [("tenant_id", ASCENDING), ("session_id", ASCENDING)],
                name="uq_tenant_session_id",
                unique=True,
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("updated_at", DESCENDING)],
                name="ix_conversation_tenant_updated",
            ),
        ]
    )

    # sensor_readings is a time-series collection; its index is created in
    # app.services.ingestion.ensure_collections, alongside the collection
    # itself, because the metaField layout and the index must agree.
