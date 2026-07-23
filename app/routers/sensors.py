"""Sensor read endpoints and simulator controls.

The read endpoints are deliberately source-agnostic: they serve whatever
ingestion has written, so they behave identically against the simulator, a PLC,
or a broker. Only the ``/simulator/*`` controls know a simulator exists, and
they refuse to run when it is not the active source.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.tenant_context import get_current_tenant
from app.db import get_tenant_scope
from app.models.reading import flatten_reading_document, utcnow
from app.schemas.machine import COLLECTIONS, strip_mongo_id
from app.schemas.sensor import (
    ClearFaultRequest,
    HistoryPoint,
    InjectFaultRequest,
    MachineHealthOut,
    ReadingOut,
    SensorHealthOut,
    SensorHistoryOut,
    SensorStatus,
    SimulatorStateOut,
)
from app.sensors.registry import get_simulator
from app.sensors.simulator import FAULT_MODELS, FaultType

router = APIRouter(tags=["sensors"])


# ---------------------------------------------------------------------------
# Health scoring
# ---------------------------------------------------------------------------
def classify(value: float, warning: float, critical: float, normal_max: float) -> SensorStatus:
    """Bucket a value against its sensor's thresholds."""
    if value >= critical:
        return SensorStatus.critical
    if value >= warning:
        return SensorStatus.warning
    return SensorStatus.normal


def score_sensor(value: float, normal_max: float, warning: float, critical: float) -> float:
    """Score a reading 1.0 (healthy) to 0.0 (at or past critical).

    Linear between the normal ceiling and critical: it degrades smoothly rather
    than stepping at the alarm, so a machine drifting toward trouble shows it
    before any threshold trips.
    """
    if value <= normal_max:
        return 1.0
    if value >= critical:
        return 0.0
    return round(1.0 - (value - normal_max) / max(1e-9, critical - normal_max), 4)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------
@router.get(
    "/sensors/latest",
    response_model=list[ReadingOut],
    summary="Newest reading for every sensor",
)
async def latest_readings(
    machine_id: Optional[str] = Query(default=None, description="Filter by machine"),
    tenant_id: str = Depends(get_current_tenant),
) -> list[dict]:
    """Return the newest reading per sensor, optionally for one machine."""
    database = get_tenant_scope(tenant_id)
    query = {"machine_id": machine_id} if machine_id else {}
    cursor = database[COLLECTIONS.latest_readings].find(query).sort(
        [("sensor_id", 1)]
    )
    return [strip_mongo_id(doc) async for doc in cursor]


@router.get(
    "/sensors/{sensor_id}/history",
    response_model=SensorHistoryOut,
    summary="A sensor's readings over a window",
)
async def sensor_history(
    sensor_id: str,
    minutes: int = Query(default=60, ge=1, le=10080, description="Look-back window"),
    tenant_id: str = Depends(get_current_tenant),
) -> SensorHistoryOut:
    """Return one sensor's readings for the last ``minutes``, oldest first."""
    database = get_tenant_scope(tenant_id)
    since = utcnow() - timedelta(minutes=minutes)

    # Identifiers live under `meta` in the time-series collection; flatten on
    # the way out so the response shape is independent of storage layout.
    cursor = (
        database[COLLECTIONS.sensor_readings]
        .find({"meta.sensor_id": sensor_id, "timestamp": {"$gte": since}})
        .sort([("timestamp", 1)])
    )
    docs = [flatten_reading_document(doc) async for doc in cursor]

    sensor = await database[COLLECTIONS.sensors].find_one({"sensor_id": sensor_id})
    if sensor is None and not docs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sensor '{sensor_id}' not found",
        )

    return SensorHistoryOut(
        sensor_id=sensor_id,
        machine_id=(sensor or {}).get("machine_id") or (docs[0]["machine_id"] if docs else None),
        sensor_type=(sensor or {}).get("type") or (docs[0]["sensor_type"] if docs else None),
        unit=(sensor or {}).get("unit") or (docs[0]["unit"] if docs else None),
        minutes=minutes,
        count=len(docs),
        points=[
            HistoryPoint(
                timestamp=d["timestamp"], value=d["value"], quality=d["quality"]
            )
            for d in docs
        ],
    )


@router.get(
    "/machines/{machine_id}/health",
    response_model=MachineHealthOut,
    summary="Machine health derived from its newest readings",
)
async def machine_health(
    machine_id: str, tenant_id: str = Depends(get_current_tenant)
) -> MachineHealthOut:
    """Assess a machine from its latest readings against configured thresholds.

    Computed from readings alone, never from simulator internals, so the number
    means the same thing whatever is feeding the pipeline.
    """
    database = get_tenant_scope(tenant_id)

    machine = await database[COLLECTIONS.machines].find_one({"machine_id": machine_id})
    if machine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Machine '{machine_id}' not found",
        )

    sensors = [s async for s in database[COLLECTIONS.sensors].find({"machine_id": machine_id})]
    latest = {
        doc["sensor_id"]: doc
        async for doc in database[COLLECTIONS.latest_readings].find(
            {"machine_id": machine_id}
        )
    }

    reports: list[SensorHealthOut] = []
    scores: list[float] = []
    worst = SensorStatus.normal
    last_updated = None

    for sensor in sensors:
        reading = latest.get(sensor["sensor_id"])
        if reading is None:
            reports.append(
                SensorHealthOut(
                    sensor_id=sensor["sensor_id"],
                    sensor_type=sensor["type"],
                    value=None,
                    unit=sensor["unit"],
                    timestamp=None,
                    status=SensorStatus.unknown,
                    score=1.0,
                    normal_min=sensor["normal_min"],
                    normal_max=sensor["normal_max"],
                    warning_threshold=sensor["warning_threshold"],
                    critical_threshold=sensor["critical_threshold"],
                )
            )
            continue

        value = float(reading["value"])
        sensor_status = classify(
            value,
            sensor["warning_threshold"],
            sensor["critical_threshold"],
            sensor["normal_max"],
        )
        sensor_score = score_sensor(
            value,
            sensor["normal_max"],
            sensor["warning_threshold"],
            sensor["critical_threshold"],
        )
        scores.append(sensor_score)

        if sensor_status == SensorStatus.critical or (
            sensor_status == SensorStatus.warning and worst != SensorStatus.critical
        ):
            worst = sensor_status

        timestamp = reading.get("timestamp")
        if timestamp is not None and (last_updated is None or timestamp > last_updated):
            last_updated = timestamp

        reports.append(
            SensorHealthOut(
                sensor_id=sensor["sensor_id"],
                sensor_type=sensor["type"],
                value=value,
                unit=sensor["unit"],
                timestamp=timestamp,
                status=sensor_status,
                score=sensor_score,
                normal_min=sensor["normal_min"],
                normal_max=sensor["normal_max"],
                warning_threshold=sensor["warning_threshold"],
                critical_threshold=sensor["critical_threshold"],
            )
        )

    # Weight the worst sensor as heavily as the average: one critical bearing
    # makes the machine unhealthy no matter how well everything else reads.
    if scores:
        health_score = round(
            0.5 * (sum(scores) / len(scores)) + 0.5 * min(scores), 4
        )
    else:
        health_score = 1.0

    return MachineHealthOut(
        machine_id=machine_id,
        name=machine.get("name", ""),
        status=worst if scores else SensorStatus.unknown,
        health_score=health_score,
        sensor_count=len(sensors),
        stale=not scores,
        sensors=reports,
        last_updated=last_updated,
    )


# ---------------------------------------------------------------------------
# Simulator controls — available only when SENSOR_SOURCE=simulator
# ---------------------------------------------------------------------------
def _simulator():
    """Fetch the active simulator or fail with a clear 409."""
    try:
        return get_simulator()
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


def _parse_fault_type(value: str) -> FaultType:
    try:
        return FaultType(value.strip().upper())
    except ValueError as exc:
        known = ", ".join(f.value for f in FAULT_MODELS)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown fault_type '{value}'. Known types: {known}",
        ) from exc


@router.post("/simulator/inject-fault", summary="Start a fault on a machine")
async def inject_fault(payload: InjectFaultRequest) -> dict:
    """Inject a modelled fault, which then progresses toward failure."""
    simulator = _simulator()
    fault_type = _parse_fault_type(payload.fault_type)
    try:
        fault = simulator.inject_fault(
            machine_id=payload.machine_id,
            fault_type=fault_type,
            severity=payload.severity,
            progression_rate=payload.progression_rate,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    return {
        "injected": fault.to_dict(),
        "machine": simulator.get_machine_state(payload.machine_id).to_dict(),
    }


@router.post("/simulator/clear-fault", summary="Repair a fault on a machine")
async def clear_fault(payload: ClearFaultRequest) -> dict:
    """Clear a fault; readings return toward baseline."""
    simulator = _simulator()
    fault_type = _parse_fault_type(payload.fault_type)
    try:
        cleared = simulator.clear_fault(payload.machine_id, fault_type)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    return {
        "cleared": cleared,
        "fault_type": fault_type.value,
        "machine": simulator.get_machine_state(payload.machine_id).to_dict(),
    }


@router.post("/simulator/reset", summary="Reset the whole simulated fleet")
async def reset_simulator() -> dict:
    """Return every machine to hour zero, full health, no faults."""
    simulator = _simulator()
    simulator.reset()
    return {
        "reset": True,
        "machines": [s.to_dict() for s in simulator.get_all_states().values()],
    }


@router.get(
    "/simulator/state",
    response_model=SimulatorStateOut,
    summary="Current simulator state",
)
async def simulator_state() -> SimulatorStateOut:
    """Full fleet state: health, runtime, load, and every active fault."""
    simulator = _simulator()
    return SimulatorStateOut(
        source=simulator.name,
        time_scale=simulator.time_scale,
        interval_seconds=simulator.interval_seconds,
        machines=[s.to_dict() for s in simulator.get_all_states().values()],
    )
