"""Background ingestion: sensor stream -> MongoDB.

Two destinations, because they answer different questions:

``sensor_readings``
    A time-series collection — the full history, for trends and diagnostics.
``latest_readings``
    One document per sensor, replaced in place. A dashboard asking "what is
    everything doing right now" reads N small docs instead of scanning a
    time-series bucket per sensor.

Writes are batched. At a 2-second interval across a seeded fleet that is a
trickle, but the same code runs a plant with thousands of tags, where a write
per reading would be the bottleneck.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from pymongo import ASCENDING, DESCENDING, IndexModel, ReplaceOne

from app.config import get_settings
from app.models.reading import SensorReading
from app.schemas.machine import COLLECTIONS
from app.sensors.base import SensorSource

logger = logging.getLogger(__name__)


async def ensure_collections(database: Any) -> None:
    """Create the time-series collection and indexes. Idempotent.

    A time-series collection cannot be converted from a normal one after the
    fact, so it must be created explicitly before the first insert — otherwise
    Mongo silently makes an ordinary collection and the bucketing is lost.
    """
    existing = await database.list_collection_names()

    if COLLECTIONS.sensor_readings not in existing:
        options: dict[str, Any] = {
            "timeseries": {
                "timeField": "timestamp",
                # The whole series identity — tenant, sensor, machine — is the
                # metaField, so Mongo buckets one tenant's readings separately
                # from another's rather than interleaving them and relying on
                # a query filter to pull them apart again.
                "metaField": "meta",
                "granularity": "seconds",
            }
        }
        retention_days = get_settings().readings_retention_days
        if retention_days > 0:
            options["expireAfterSeconds"] = retention_days * 86400
        await database.create_collection(COLLECTIONS.sensor_readings, **options)
        logger.info("Created time-series collection '%s'", COLLECTIONS.sensor_readings)
    else:
        await _assert_timeseries_layout(database)

    # Tenant leads the index so every scoped query hits a prefix.
    await database[COLLECTIONS.sensor_readings].create_indexes(
        [
            IndexModel(
                [
                    ("meta.tenant_id", ASCENDING),
                    ("meta.sensor_id", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="ix_reading_tenant_sensor_time",
            ),
            IndexModel(
                [
                    ("meta.tenant_id", ASCENDING),
                    ("meta.machine_id", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="ix_reading_tenant_machine_time",
            ),
        ]
    )
    # latest_readings indexes are created in app.db.create_indexes alongside
    # every other tenant-compound index.


async def _assert_timeseries_layout(database: Any) -> None:
    """Refuse to write into a pre-tenancy time-series collection.

    A collection created with ``metaField: sensor_id`` cannot be altered in
    place, and writing ``meta`` documents into it would produce history that no
    tenant-scoped query can ever find. Fail with the remedy instead.
    """
    cursor = await database.list_collections(
        filter={"name": COLLECTIONS.sensor_readings}
    )
    infos = await cursor.to_list(length=1)
    if not infos:
        return
    meta_field = (
        infos[0].get("options", {}).get("timeseries", {}).get("metaField")
    )
    if meta_field not in (None, "meta"):
        raise RuntimeError(
            f"Collection '{COLLECTIONS.sensor_readings}' was created with "
            f"metaField='{meta_field}', which predates multi-tenancy. A "
            "time-series metaField cannot be changed in place. Drop the "
            "collection and let it be recreated:\n"
            f"  db.{COLLECTIONS.sensor_readings}.drop()\n"
            "Historic readings in it carry no tenant and cannot be migrated "
            "automatically — decide who owns them before discarding."
        )


class IngestionService:
    """Consumes a :class:`SensorSource` stream and persists it.

    Source-agnostic by construction: it only knows ``stream()`` and
    :class:`SensorReading`.
    """

    def __init__(
        self,
        source: SensorSource,
        tenant_id: str,
        database: Any = None,
        batch_size: Optional[int] = None,
        flush_seconds: Optional[float] = None,
    ) -> None:
        from app.db import normalize_tenant_id

        settings = get_settings()
        self._source = source
        self._tenant_id = normalize_tenant_id(tenant_id)
        if source.tenant_id != self._tenant_id:
            # Two different answers to "whose data is this" means a wiring bug;
            # refuse rather than pick one and write mislabelled history.
            raise ValueError(
                f"Ingestion is scoped to tenant '{self._tenant_id}' but its "
                f"source '{source.name}' emits readings for "
                f"'{source.tenant_id}'."
            )
        self._db = database
        self._batch_size = batch_size or settings.ingestion_batch_size
        self._flush_seconds = flush_seconds or settings.ingestion_flush_seconds

        self._task: Optional[asyncio.Task] = None
        self._buffer: list[SensorReading] = []
        self._running = False
        self._last_flush = 0.0

        self.readings_ingested = 0
        self.batches_written = 0
        self.errors = 0

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def database(self) -> Any:
        """RAW database handle — administrative use only (see ``scope``)."""
        if self._db is None:
            from app.db import get_database

            self._db = get_database()
        return self._db

    @property
    def scope(self) -> Any:
        """Tenant-bound handle used for every write this service makes."""
        from app.db import get_tenant_scope

        return get_tenant_scope(self._tenant_id, self._db)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def source(self) -> SensorSource:
        return self._source

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        """Connect the source, prepare collections, and start consuming."""
        if self._running:
            return
        await ensure_collections(self.database)
        await self._source.connect()
        self._running = True
        self._last_flush = asyncio.get_event_loop().time()
        self._task = asyncio.create_task(self._run(), name="sensor-ingestion")
        logger.info(
            "Ingestion started from source '%s' for tenant '%s'",
            self._source.name,
            self._tenant_id,
        )

    async def stop(self) -> None:
        """Stop consuming, flush what is buffered, and release the source.

        Buffered readings are flushed rather than dropped — on a clean shutdown
        there is no reason to lose the last few seconds of history.
        """
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        try:
            await self._flush()
        except Exception:  # pragma: no cover - best effort on shutdown
            logger.exception("Final ingestion flush failed")
        await self._source.disconnect()
        logger.info(
            "Ingestion stopped — %d readings in %d batches",
            self.readings_ingested,
            self.batches_written,
        )

    # -- consumption -------------------------------------------------------
    async def _run(self) -> None:
        """Consume the stream until cancelled, flushing on size or age."""
        try:
            async for reading in self._source.stream():
                self._buffer.append(reading)
                if self._should_flush():
                    await self._flush()
        except asyncio.CancelledError:
            raise
        except Exception:
            # A source failure must not kill the API process; surface and stop.
            self.errors += 1
            self._running = False
            logger.exception("Ingestion loop failed for source '%s'", self._source.name)

    def _should_flush(self) -> bool:
        if len(self._buffer) >= self._batch_size:
            return True
        age = asyncio.get_event_loop().time() - self._last_flush
        return bool(self._buffer) and age >= self._flush_seconds

    async def _flush(self) -> None:
        """Write the buffer to history and refresh the latest-value cache."""
        if not self._buffer:
            return
        batch, self._buffer = self._buffer, []
        scope = self.scope

        # Both writes go through the tenant scope, which stamps ownership; a
        # reading from another tenant's source would be rejected, not stored.
        await scope[COLLECTIONS.sensor_readings].insert_many(
            [r.to_document() for r in batch], ordered=False
        )

        # Keep only the newest reading per sensor within this batch, so a batch
        # spanning several ticks does not issue redundant upserts.
        newest: dict[str, dict] = {}
        for reading in batch:
            current = newest.get(reading.sensor_id)
            if current is None or reading.timestamp >= current["timestamp"]:
                newest[reading.sensor_id] = reading.to_latest_document()

        if newest:
            await scope[COLLECTIONS.latest_readings].upsert_many(
                ["sensor_id"], list(newest.values()), ordered=False
            )

        self.readings_ingested += len(batch)
        self.batches_written += 1
        self._last_flush = asyncio.get_event_loop().time()

    def stats(self) -> dict:
        """Counters for the health endpoint."""
        return {
            "running": self._running,
            "tenant_id": self._tenant_id,
            "source": self._source.name,
            "readings_ingested": self.readings_ingested,
            "batches_written": self.batches_written,
            "buffered": len(self._buffer),
            "errors": self.errors,
        }


_service: Optional[IngestionService] = None


def get_ingestion_service() -> Optional[IngestionService]:
    """The process-wide ingestion service, if one has been started."""
    return _service


def set_ingestion_service(service: Optional[IngestionService]) -> None:
    global _service
    _service = service


async def start_ingestion() -> Optional[IngestionService]:
    """Build and start ingestion from the configured source. Called by lifespan."""
    settings = get_settings()
    if not settings.ingestion_enabled:
        logger.info("Ingestion disabled by configuration")
        return None

    from app.sensors.registry import get_active_source

    source = get_active_source()
    service = IngestionService(source=source, tenant_id=source.tenant_id)
    await service.start()
    set_ingestion_service(service)
    return service


async def stop_ingestion() -> None:
    """Stop the process-wide ingestion service, if running. Called by lifespan."""
    global _service
    if _service is not None:
        await _service.stop()
        _service = None
