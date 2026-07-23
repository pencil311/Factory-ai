"""MQTT sensor source.

Structurally complete: topic pattern matching (including ``+``/``#`` wildcards),
payload decoding for the common broker dialects, connection lifecycle, and
normalisation into :class:`SensorReading`. Only actual broker I/O raises
:class:`NotImplementedError`.

Driver of record would be ``aiomqtt``; not a dependency yet for the same reason
as the OPC UA client.

Topic mapping is config-driven and wildcard-aware, so a gateway that publishes
``factory/LINE-A/CV-201/vibration`` needs one pattern, not one line per sensor:

    {
      "factory/+/+/vibration": {"sensor_id_from": "topic", "value_key": "v"},
      "factory/LINE-A/CV-201/temp": {"sensor_id": "CV-201-TMP-01"}
    }
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Mapping, Optional

from app.models.machine import SensorType
from app.models.reading import ReadingQuality, ReadingSource, SensorReading, utcnow
from app.sensors.base import SensorSource


@dataclass(frozen=True)
class TopicBinding:
    """Binds an MQTT topic (possibly wildcarded) to a sensor."""

    topic: str
    sensor_id: str
    machine_id: str
    sensor_type: SensorType
    unit: str
    component_id: Optional[str] = None
    #: JSON key holding the measurement; ``None`` means the payload is a bare number.
    value_key: Optional[str] = "value"
    #: JSON key holding the sample time, if the device stamps its own.
    timestamp_key: Optional[str] = "timestamp"
    #: JSON key holding device-reported quality.
    quality_key: Optional[str] = "quality"
    scale: float = 1.0
    offset: float = 0.0

    def to_engineering_units(self, raw: float) -> float:
        return float(raw) * self.scale + self.offset


def topic_to_regex(topic: str) -> re.Pattern[str]:
    """Compile an MQTT topic filter into a regex.

    ``+`` matches exactly one level, ``#`` matches the remainder — the broker's
    own semantics, so a filter behaves here the way it does on the wire.
    """
    parts = []
    for level in topic.split("/"):
        if level == "+":
            parts.append(r"[^/]+")
        elif level == "#":
            parts.append(r".*")
            break
        else:
            parts.append(re.escape(level))
    return re.compile("^" + "/".join(parts) + "$")


def parse_payload(payload: bytes | str) -> Any:
    """Decode a payload as JSON, falling back to a bare numeric string.

    Field gateways publish both dialects; accepting either keeps one broker from
    needing two integrations.
    """
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    text = text.strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return float(text)


def coerce_timestamp(value: Any) -> datetime:
    """Accept epoch seconds, epoch millis, or ISO-8601; always return UTC.

    Devices disagree about time formats far more than they should.
    """
    if value is None:
        return utcnow()
    if isinstance(value, (int, float)):
        # Anything past ~2001 in seconds is milliseconds if it is this large.
        seconds = value / 1000.0 if value > 1e11 else float(value)
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def coerce_quality(value: Any) -> ReadingQuality:
    """Map device-reported quality onto our three levels, defaulting to GOOD."""
    if value is None:
        return ReadingQuality.good
    if isinstance(value, bool):
        return ReadingQuality.good if value else ReadingQuality.bad
    text = str(value).strip().upper()
    if text in {"GOOD", "OK", "1", "TRUE"}:
        return ReadingQuality.good
    if text in {"SUSPECT", "UNCERTAIN", "STALE", "DEGRADED"}:
        return ReadingQuality.suspect
    if text in {"BAD", "FAULT", "ERROR", "0", "FALSE"}:
        return ReadingQuality.bad
    return ReadingQuality.good


class MqttSensorSource(SensorSource):
    """Reads sensors from MQTT topics published by field gateways."""

    source_type = ReadingSource.mqtt

    def __init__(
        self,
        tenant_id: str,
        host: str,
        bindings: Mapping[str, TopicBinding],
        port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: str = "factorypilot-ingest",
        qos: int = 1,
        tls: bool = False,
    ) -> None:
        from app.db import normalize_tenant_id

        self._tenant_id = normalize_tenant_id(tenant_id)
        self._host = host
        self._port = port
        self._bindings = dict(bindings)
        self._matchers = [(topic_to_regex(t), b) for t, b in self._bindings.items()]
        self._username = username
        self._password = password
        self._client_id = client_id
        self._qos = qos
        self._tls = tls
        self._client: Any = None
        self._connected = False

    # -- SensorSource interface -------------------------------------------
    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def name(self) -> str:
        return "mqtt"

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        if self._connected:
            return
        raise NotImplementedError(
            "MQTT broker I/O is not wired up. To enable: add `aiomqtt` to "
            "requirements, construct Client(hostname=self._host, port=self._port, "
            "identifier=self._client_id, username/password, tls_context if "
            "self._tls), await its __aenter__, subscribe to self.topic_filters "
            "at self._qos, and set self._client and self._connected."
        )

    async def disconnect(self) -> None:
        """Close the broker session. Safe when never connected."""
        if self._client is not None:
            await self._close_client()
            self._client = None
        self._connected = False

    async def stream(self) -> AsyncIterator[SensorReading]:
        """Yield a reading for every matched message on the subscribed topics."""
        self._require_connected()
        raise NotImplementedError(
            "MQTT message loop is not wired up. To enable: iterate "
            "`async for message in self._client.messages:` and yield "
            "self.normalize(message.topic.value, message.payload), skipping "
            "topics with no binding."
        )
        yield  # pragma: no cover - marks this an async generator

    async def read_once(self, sensor_id: str) -> SensorReading:
        """MQTT is push-only; a point read means the last retained message."""
        if sensor_id not in {b.sensor_id for b in self._bindings.values()}:
            raise KeyError(f"Sensor '{sensor_id}' is not mapped to an MQTT topic")
        self._require_connected()
        raise NotImplementedError(
            "MQTT point reads are not wired up. MQTT has no request/response: to "
            "enable, subscribe to the sensor's topic and take the broker's "
            "retained message, or serve the newest value from `latest_readings`."
        )

    # -- normalisation (fully implemented — the part that must be right) ---
    def binding_for_topic(self, topic: str) -> Optional[TopicBinding]:
        """First binding whose filter matches, honouring MQTT wildcards."""
        for matcher, binding in self._matchers:
            if matcher.match(topic):
                return binding
        return None

    def normalize(self, topic: str, payload: bytes | str) -> SensorReading:
        """Turn one MQTT message into a :class:`SensorReading`.

        Raises :class:`KeyError` when no binding matches — an unmapped topic is
        a configuration error worth surfacing, not a message to drop silently.
        """
        binding = self.binding_for_topic(topic)
        if binding is None:
            raise KeyError(f"No sensor binding matches MQTT topic '{topic}'")

        data = parse_payload(payload)
        if isinstance(data, Mapping):
            raw_value = data[binding.value_key] if binding.value_key else data["value"]
            timestamp = coerce_timestamp(
                data.get(binding.timestamp_key) if binding.timestamp_key else None
            )
            quality = coerce_quality(
                data.get(binding.quality_key) if binding.quality_key else None
            )
        else:
            raw_value, timestamp, quality = data, utcnow(), ReadingQuality.good

        return SensorReading(
            tenant_id=self._tenant_id,
            sensor_id=binding.sensor_id,
            machine_id=binding.machine_id,
            component_id=binding.component_id,
            sensor_type=binding.sensor_type,
            value=binding.to_engineering_units(raw_value),
            unit=binding.unit,
            timestamp=timestamp,
            quality=quality,
            source=ReadingSource.mqtt,
        )

    @property
    def topic_filters(self) -> list[str]:
        """Topics to subscribe to on connect."""
        return list(self._bindings)

    async def _close_client(self) -> None:
        raise NotImplementedError(
            "MQTT session teardown is not wired up: await self._client.__aexit__()."
        )
