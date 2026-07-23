"""OPC UA sensor source.

Structurally complete: address-space mapping, connection lifecycle, subscription
model, and normalisation from OPC UA DataValues into :class:`SensorReading`.
Only the four places that would touch a real PLC raise
:class:`NotImplementedError` — everything around them is real code, so wiring a
client library in is a localised change rather than a rewrite.

Driver of record would be ``asyncua``; it is deliberately not a dependency yet,
since importing it would install a client this build never opens.

Mapping is config-driven: each sensor is bound to a node id, so swapping a PLC
tag never touches application code.

    {
      "CV-201-TMP-01": {"node_id": "ns=2;s=Line_A.CV201.Motor.Temp", "scale": 1.0},
      "CV-201-VIB-01": {"node_id": "ns=2;s=Line_A.CV201.Brg.Vib_RMS"}
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, Optional

from app.models.machine import SensorType
from app.models.reading import ReadingQuality, ReadingSource, SensorReading, utcnow
from app.sensors.base import SensorSource

#: OPC UA StatusCode severity lives in the top two bits of the 32-bit code.
_STATUS_GOOD = 0x00000000
_STATUS_UNCERTAIN = 0x40000000
_STATUS_BAD = 0x80000000


@dataclass(frozen=True)
class NodeBinding:
    """Binds one sensor to one OPC UA node."""

    sensor_id: str
    node_id: str
    machine_id: str
    sensor_type: SensorType
    unit: str
    component_id: Optional[str] = None
    #: Applied as ``raw * scale + offset`` — PLCs often serve scaled integers.
    scale: float = 1.0
    offset: float = 0.0

    def to_engineering_units(self, raw: float) -> float:
        return float(raw) * self.scale + self.offset


def load_node_map(path: str | Path, sensor_catalogue: Mapping[str, Mapping]) -> dict[str, NodeBinding]:
    """Build bindings from a JSON node map plus the sensor catalogue.

    The node map carries only PLC addressing; units and types come from the
    sensor registry, so the two cannot drift apart.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    bindings: dict[str, NodeBinding] = {}
    for sensor_id, entry in raw.items():
        sensor = sensor_catalogue.get(sensor_id)
        if sensor is None:
            raise KeyError(
                f"Node map references unknown sensor '{sensor_id}' — "
                "the sensor catalogue and the PLC map are out of sync"
            )
        bindings[sensor_id] = NodeBinding(
            sensor_id=sensor_id,
            node_id=str(entry["node_id"]),
            machine_id=str(sensor["machine_id"]),
            sensor_type=SensorType(sensor["type"]),
            unit=str(sensor["unit"]),
            component_id=sensor.get("component_id"),
            scale=float(entry.get("scale", 1.0)),
            offset=float(entry.get("offset", 0.0)),
        )
    return bindings


def quality_from_status_code(status_code: int) -> ReadingQuality:
    """Map an OPC UA StatusCode onto our three-level quality.

    OPC UA's Uncertain maps to SUSPECT and Bad to BAD, which is exactly the
    distinction our contract wants: the server is telling us how much to trust
    the transducer, not how the machine is doing.
    """
    severity = status_code & 0xC0000000
    if severity == _STATUS_GOOD:
        return ReadingQuality.good
    if severity == _STATUS_UNCERTAIN:
        return ReadingQuality.suspect
    return ReadingQuality.bad


class OpcUaSensorSource(SensorSource):
    """Reads sensors from an OPC UA server."""

    source_type = ReadingSource.opcua

    def __init__(
        self,
        tenant_id: str,
        endpoint: str,
        bindings: Mapping[str, NodeBinding],
        username: Optional[str] = None,
        password: Optional[str] = None,
        publishing_interval_ms: int = 1000,
        security_policy: str = "None",
    ) -> None:
        from app.db import normalize_tenant_id

        self._tenant_id = normalize_tenant_id(tenant_id)
        self._endpoint = endpoint
        self._bindings = dict(bindings)
        self._by_node = {b.node_id: b for b in self._bindings.values()}
        self._username = username
        self._password = password
        self._publishing_interval_ms = publishing_interval_ms
        self._security_policy = security_policy
        self._client: Any = None
        self._subscription: Any = None
        self._connected = False

    # -- SensorSource interface -------------------------------------------
    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def name(self) -> str:
        return "opcua"

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        if self._connected:
            return
        raise NotImplementedError(
            "OPC UA client I/O is not wired up. To enable: add `asyncua` to "
            "requirements, open Client(url=self._endpoint) here, apply "
            "self._security_policy / credentials, await client.connect(), and "
            "set self._client and self._connected."
        )

    async def disconnect(self) -> None:
        """Tear down subscription then session. Safe when never connected."""
        if self._subscription is not None:
            await self._delete_subscription()
            self._subscription = None
        if self._client is not None:
            await self._close_client()
            self._client = None
        self._connected = False

    async def stream(self) -> AsyncIterator[SensorReading]:
        """Yield readings from an OPC UA subscription as they arrive."""
        self._require_connected()
        raise NotImplementedError(
            "OPC UA subscription I/O is not wired up. To enable: create a "
            "subscription at self._publishing_interval_ms, subscribe to every "
            "binding's node, and yield self.normalize(binding, value, status, "
            "source_timestamp) from the datachange handler queue."
        )
        yield  # pragma: no cover - marks this an async generator

    async def read_once(self, sensor_id: str) -> SensorReading:
        binding = self._bindings.get(sensor_id)
        if binding is None:
            raise KeyError(f"Sensor '{sensor_id}' is not mapped to an OPC UA node")
        self._require_connected()
        raise NotImplementedError(
            "OPC UA read I/O is not wired up. To enable: "
            f"node = self._client.get_node('{binding.node_id}'), "
            "dv = await node.read_data_value(), then return "
            "self.normalize(binding, dv.Value.Value, dv.StatusCode.value, "
            "dv.SourceTimestamp)."
        )

    # -- normalisation (fully implemented — the part that must be right) ---
    def normalize(
        self,
        binding: NodeBinding,
        raw_value: float,
        status_code: int = _STATUS_GOOD,
        source_timestamp: Optional[Any] = None,
    ) -> SensorReading:
        """Turn one OPC UA DataValue into a :class:`SensorReading`.

        Prefers the server's SourceTimestamp — the instant the PLC sampled the
        signal — over local time, which can be seconds later under load.
        """
        return SensorReading(
            tenant_id=self._tenant_id,
            sensor_id=binding.sensor_id,
            machine_id=binding.machine_id,
            component_id=binding.component_id,
            sensor_type=binding.sensor_type,
            value=binding.to_engineering_units(raw_value),
            unit=binding.unit,
            timestamp=source_timestamp or utcnow(),
            quality=quality_from_status_code(status_code),
            source=ReadingSource.opcua,
        )

    @property
    def bindings(self) -> Mapping[str, NodeBinding]:
        return dict(self._bindings)

    def binding_for_node(self, node_id: str) -> NodeBinding:
        """Reverse lookup used by subscription callbacks, which carry node ids."""
        return self._by_node[node_id]

    # -- teardown seams ----------------------------------------------------
    async def _delete_subscription(self) -> None:
        raise NotImplementedError(
            "OPC UA subscription teardown is not wired up: "
            "await self._subscription.delete()."
        )

    async def _close_client(self) -> None:
        raise NotImplementedError(
            "OPC UA session teardown is not wired up: await self._client.disconnect()."
        )
