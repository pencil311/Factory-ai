"""The sensor reading contract.

This is the seam of the whole pipeline: every source — simulator, OPC UA, MQTT,
a replayed dataset — normalises into :class:`SensorReading`, and nothing
downstream (ingestion, endpoints, dashboards, diagnostics) ever asks where a
reading came from. ``source`` is carried for provenance and debugging only; no
business logic should branch on it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.machine import SensorType


class ReadingQuality(str, Enum):
    """Trustworthiness of the *signal*, not of the process value.

    A genuine over-critical temperature is ``GOOD`` quality and bad news; a
    reading outside the physically plausible envelope is ``SUSPECT`` because the
    sensor, not the machine, is the likely problem. Keeping these separate stops
    a drifting transducer from being read as a failing bearing.
    """

    good = "GOOD"
    suspect = "SUSPECT"
    bad = "BAD"


class ReadingSource(str, Enum):
    """Where a reading originated. Provenance only — never branch on this."""

    simulator = "SIMULATOR"
    opcua = "OPCUA"
    mqtt = "MQTT"
    dataset = "DATASET"


def utcnow() -> datetime:
    """Timezone-aware UTC now. Readings are always stored in UTC."""
    return datetime.now(timezone.utc)


class SensorReading(BaseModel):
    """One measurement from one sensor at one instant."""

    model_config = ConfigDict(use_enum_values=True)

    tenant_id: str = Field(..., min_length=1, description="Owning tenant id")
    sensor_id: str
    machine_id: str
    component_id: Optional[str] = None

    sensor_type: SensorType
    value: float
    unit: str

    timestamp: datetime = Field(default_factory=utcnow)
    quality: ReadingQuality = ReadingQuality.good
    source: ReadingSource = ReadingSource.simulator

    def to_document(self) -> dict:
        """Render for the ``sensor_readings`` time-series collection.

        The identifiers that define the series — tenant, sensor, machine — go
        into the ``meta`` subdocument, which is the collection's ``metaField``.
        Mongo buckets by metaField, so putting the tenant there keeps one
        tenant's readings out of another's storage buckets rather than merely
        filtering them apart at query time. ``timestamp`` stays a real
        ``datetime`` so the time index works.
        """
        doc = self.model_dump(mode="python")
        return {
            "meta": {
                "tenant_id": doc.pop("tenant_id"),
                "sensor_id": doc.pop("sensor_id"),
                "machine_id": doc.pop("machine_id"),
            },
            **doc,
        }

    def to_latest_document(self) -> dict:
        """Render for ``latest_readings``, which is an ordinary flat collection."""
        return self.model_dump(mode="python")


def flatten_reading_document(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Lift a time-series reading's ``meta`` fields back to the top level.

    Repositories flatten on the way out so that everything downstream sees one
    reading shape, whichever collection it came from.
    """
    flat = {k: v for k, v in doc.items() if k not in ("_id", "meta")}
    flat.update(doc.get("meta") or {})
    return flat
