"""Request/response schemas for the sensor endpoints."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.machine import SensorType
from app.models.reading import ReadingQuality, ReadingSource


class ReadingOut(BaseModel):
    """One reading as returned by the API."""

    model_config = ConfigDict(use_enum_values=True)

    sensor_id: str
    machine_id: str
    component_id: Optional[str] = None
    sensor_type: SensorType
    value: float
    unit: str
    timestamp: datetime
    quality: ReadingQuality
    source: ReadingSource


class SensorStatus(str, Enum):
    """Where a reading sits relative to its sensor's configured thresholds."""

    normal = "NORMAL"
    warning = "WARNING"
    critical = "CRITICAL"
    unknown = "UNKNOWN"


class SensorHealthOut(BaseModel):
    """Per-sensor assessment inside a machine health report."""

    model_config = ConfigDict(use_enum_values=True)

    sensor_id: str
    sensor_type: SensorType
    value: Optional[float] = None
    unit: str
    timestamp: Optional[datetime] = None
    status: SensorStatus
    #: 1.0 at or below normal_max, 0.0 at critical.
    score: float = Field(..., ge=0.0, le=1.0)
    normal_min: float
    normal_max: float
    warning_threshold: float
    critical_threshold: float


class MachineHealthOut(BaseModel):
    """Health of one machine, derived from its newest readings."""

    model_config = ConfigDict(use_enum_values=True)

    machine_id: str
    name: str
    status: SensorStatus = Field(..., description="Worst status across all sensors")
    health_score: float = Field(..., ge=0.0, le=1.0)
    sensor_count: int
    stale: bool = Field(
        ..., description="True when no reading has arrived for this machine yet"
    )
    sensors: list[SensorHealthOut] = []
    last_updated: Optional[datetime] = None


class HistoryPoint(BaseModel):
    """One point on a sensor's history series."""

    timestamp: datetime
    value: float
    quality: ReadingQuality


class SensorHistoryOut(BaseModel):
    """A sensor's readings over a time window."""

    model_config = ConfigDict(use_enum_values=True)

    sensor_id: str
    machine_id: Optional[str] = None
    sensor_type: Optional[SensorType] = None
    unit: Optional[str] = None
    minutes: int
    count: int
    points: list[HistoryPoint] = []


# ---------------------------------------------------------------------------
# Simulator control
# ---------------------------------------------------------------------------
class InjectFaultRequest(BaseModel):
    """Body of ``POST /simulator/inject-fault``."""

    machine_id: str
    fault_type: str = Field(
        ..., description="BEARING_WEAR | MOTOR_OVERHEAT | LUBRICATION_LOSS | "
        "BELT_MISALIGNMENT | SEAL_LEAK | TOOL_WEAR"
    )
    severity: float = Field(default=0.1, ge=0.0, le=1.0)
    progression_rate: float = Field(
        default=1.0, gt=0.0, description="Multiplier on the fault's nominal pace"
    )


class ClearFaultRequest(BaseModel):
    """Body of ``POST /simulator/clear-fault``."""

    machine_id: str
    fault_type: str


class SimulatorStateOut(BaseModel):
    """Full simulator state, for demo control panels."""

    source: str
    time_scale: float
    interval_seconds: float
    machines: list[dict]
