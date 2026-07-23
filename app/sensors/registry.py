"""Source selection.

One process-wide source instance, chosen by ``SENSOR_SOURCE``. It is a singleton
on purpose: the ingestion task and the ``/simulator/*`` endpoints must operate on
the *same* simulator, or injecting a fault would have no effect on the readings
being written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.sensors.base import SensorSource
from app.sensors.mqtt_source import MqttSensorSource, TopicBinding
from app.sensors.opcua_source import OpcUaSensorSource, load_node_map
from app.sensors.simulator import MachineSimulator

SIMULATOR = "simulator"
OPCUA = "opcua"
MQTT = "mqtt"
VALID_SOURCES = (SIMULATOR, OPCUA, MQTT)

_active_source: Optional[SensorSource] = None


def _tenant_id() -> str:
    """The tenant the process-wide source ingests for.

    One source per process today, so it belongs to the default tenant. When
    ingestion becomes per-tenant, this is the function that grows a parameter —
    everything downstream already carries the tenant explicitly.
    """
    from app.db import normalize_tenant_id

    return normalize_tenant_id(get_settings().default_tenant_id)


def _sensor_catalogue() -> dict[str, dict]:
    """Sensor definitions used to give hardware bindings their units and types.

    Read from the seed catalogue rather than Mongo so building a source never
    requires a live database; the seed is the same data that gets loaded.
    """
    from app.seed.seed_machines import SENSORS

    return {s.sensor_id: s.model_dump() for s in SENSORS}


def _build_simulator() -> MachineSimulator:
    settings = get_settings()
    # sensors=None means the simulator loads its catalogue from Mongo on
    # connect(), which is what we want in a running app.
    return MachineSimulator(
        tenant_id=_tenant_id(),
        interval_seconds=settings.sensor_interval_seconds,
        time_scale=settings.sensor_time_scale,
    )


def _build_opcua() -> OpcUaSensorSource:
    settings = get_settings()
    path = Path(settings.opcua_node_map_path)
    if not path.exists():
        raise FileNotFoundError(
            f"SENSOR_SOURCE=opcua but no node map at '{path}'. "
            "Set OPCUA_NODE_MAP_PATH to a JSON map of sensor_id -> node_id."
        )
    return OpcUaSensorSource(
        tenant_id=_tenant_id(),
        endpoint=settings.opcua_endpoint,
        bindings=load_node_map(path, _sensor_catalogue()),
        username=settings.opcua_username or None,
        password=settings.opcua_password or None,
        publishing_interval_ms=settings.opcua_publishing_interval_ms,
        security_policy=settings.opcua_security_policy,
    )


def _build_mqtt() -> MqttSensorSource:
    settings = get_settings()
    path = Path(settings.mqtt_topic_map_path)
    if not path.exists():
        raise FileNotFoundError(
            f"SENSOR_SOURCE=mqtt but no topic map at '{path}'. "
            "Set MQTT_TOPIC_MAP_PATH to a JSON map of topic -> sensor binding."
        )
    catalogue = _sensor_catalogue()
    raw = json.loads(path.read_text(encoding="utf-8"))

    bindings: dict[str, TopicBinding] = {}
    for topic, entry in raw.items():
        sensor_id = str(entry["sensor_id"])
        sensor = catalogue.get(sensor_id)
        if sensor is None:
            raise KeyError(
                f"Topic map references unknown sensor '{sensor_id}' — "
                "the sensor catalogue and the broker map are out of sync"
            )
        bindings[topic] = TopicBinding(
            topic=topic,
            sensor_id=sensor_id,
            machine_id=str(sensor["machine_id"]),
            sensor_type=sensor["type"],
            unit=str(sensor["unit"]),
            component_id=sensor.get("component_id"),
            value_key=entry.get("value_key", "value"),
            timestamp_key=entry.get("timestamp_key", "timestamp"),
            quality_key=entry.get("quality_key", "quality"),
            scale=float(entry.get("scale", 1.0)),
            offset=float(entry.get("offset", 0.0)),
        )

    return MqttSensorSource(
        tenant_id=_tenant_id(),
        host=settings.mqtt_host,
        port=settings.mqtt_port,
        bindings=bindings,
        username=settings.mqtt_username or None,
        password=settings.mqtt_password or None,
        client_id=settings.mqtt_client_id,
        qos=settings.mqtt_qos,
        tls=settings.mqtt_tls,
    )


def build_source(kind: Optional[str] = None) -> SensorSource:
    """Construct a source without touching the process-wide singleton."""
    name = (kind or get_settings().sensor_source or SIMULATOR).strip().lower()
    if name == SIMULATOR:
        return _build_simulator()
    if name == OPCUA:
        return _build_opcua()
    if name == MQTT:
        return _build_mqtt()
    raise ValueError(
        f"Unknown SENSOR_SOURCE '{name}'. Expected one of: {', '.join(VALID_SOURCES)}"
    )


def get_active_source() -> SensorSource:
    """Return the process-wide source, creating it on first use."""
    global _active_source
    if _active_source is None:
        _active_source = build_source()
    return _active_source


def set_active_source(source: Optional[SensorSource]) -> None:
    """Replace the active source. For tests and for explicit app wiring."""
    global _active_source
    _active_source = source


def reset_active_source() -> None:
    """Drop the cached source so the next call re-reads configuration."""
    set_active_source(None)


def get_simulator() -> MachineSimulator:
    """Return the active source as a simulator, or explain why it is not one.

    The ``/simulator/*`` endpoints use this so they fail with a clear message
    instead of an AttributeError when running against real hardware.
    """
    source = get_active_source()
    if not isinstance(source, MachineSimulator):
        raise LookupError(
            f"Simulator controls are unavailable: SENSOR_SOURCE is "
            f"'{source.name}', not '{SIMULATOR}'."
        )
    return source
