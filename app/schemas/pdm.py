"""Request/response schemas for the predictive-maintenance endpoints."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TrendDirection(str, Enum):
    """Where the machine's condition is heading, from signal evidence."""

    improving = "IMPROVING"
    stable = "STABLE"
    degrading = "DEGRADING"


class ContributingFeature(BaseModel):
    """One feature's role in the prediction — what RCA consumes downstream."""

    name: str
    value: float
    importance: float = Field(..., ge=0.0)


class PdmPredictionOut(BaseModel):
    """The full prediction for one machine."""

    model_config = ConfigDict(use_enum_values=True)

    machine_id: str
    failure_probability: float = Field(..., ge=0.0, le=1.0)
    remaining_useful_life_hours: float = Field(..., ge=0.0)
    health_score: float = Field(..., ge=0.0, le=1.0)
    predicted_failure_time: Optional[datetime] = None
    predicted_failure_mode: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    contributing_features: list[ContributingFeature] = []
    trend_direction: TrendDirection
    #: Provenance: how many readings and channels fed this prediction.
    readings_used: int
    channels_present: list[str] = []
    generated_at: datetime


class TrendPointOut(BaseModel):
    """One bucket of the health trend series."""

    timestamp: datetime
    health_score: float = Field(..., ge=0.0, le=1.0)
    deviation: float = Field(
        ..., description="Mean absolute deviation from healthy baseline, z-units"
    )
    readings: int


class TrendOut(BaseModel):
    """A machine's health trajectory over a look-back window."""

    model_config = ConfigDict(use_enum_values=True)

    machine_id: str
    hours: int
    direction: TrendDirection
    points: list[TrendPointOut] = []


class FleetEntryOut(BaseModel):
    """One machine in the fleet risk ranking."""

    model_config = ConfigDict(use_enum_values=True)

    machine_id: str
    name: str
    line_id: str
    prediction: Optional[PdmPredictionOut] = None
    #: Set when a prediction could not be made (e.g. no readings yet).
    error: Optional[str] = None


class ModelInfoOut(BaseModel):
    """Loaded model versions and their held-out metrics."""

    artifacts_dir: str
    loaded_at: datetime
    schema_version: str
    feature_count: int
    window: int
    rul_model: dict
    failure_classifier: dict
