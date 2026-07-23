"""Mongo collection names and API response schemas.

The response schemas mirror the persisted document models but are the shapes the
API contractually returns. Keeping them separate lets the storage models evolve
(extra internal fields, audit metadata, ...) without changing the public API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict

from app.models.machine import ComponentType, MachineStatus, SensorType


# ---------------------------------------------------------------------------
# Collection names (single source of truth, shared by db/router/seed)
# ---------------------------------------------------------------------------
class COLLECTIONS:
    """Canonical MongoDB collection names."""

    #: The tenant registry itself — the one collection that is not tenant-scoped.
    tenants = "tenants"
    sites = "sites"
    lines = "production_lines"
    machines = "machines"
    components = "components"
    sensors = "sensors"
    error_codes = "error_codes"
    #: Time-series collection of every reading ever ingested.
    sensor_readings = "sensor_readings"
    #: Newest reading per sensor — one small doc each, for fast dashboard reads.
    latest_readings = "latest_readings"
    #: Knowledge-base source documents.
    documents = "documents"
    #: Retrievable passages, carrying the vectors the index is built over.
    chunks = "chunks"
    #: Spare parts inventory.
    parts = "parts"
    #: Orchestrator conversation state: turn history and cached module output.
    conversations = "conversations"


def strip_mongo_id(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of a Mongo document without its ``_id`` field."""
    return {k: v for k, v in doc.items() if k != "_id"}


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class SiteOut(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    site_id: str
    name: str
    location: Optional[str] = None
    timezone: str = "UTC"


class ProductionLineOut(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    line_id: str
    site_id: str
    name: str
    description: Optional[str] = None


class MachineOut(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    machine_id: str
    name: str
    model: str
    manufacturer: str
    site_id: str
    line_id: str
    position_in_line: int
    criticality: int
    status: MachineStatus
    aliases: list[str] = []
    installed_at: Optional[datetime] = None
    last_maintenance_at: Optional[datetime] = None


class ComponentOut(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    component_id: str
    machine_id: str
    name: str
    type: ComponentType
    part_number: Optional[str] = None
    parent_component_id: Optional[str] = None


class SensorOut(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    sensor_id: str
    machine_id: str
    component_id: Optional[str] = None
    type: SensorType
    unit: str
    normal_min: float
    normal_max: float
    warning_threshold: float
    critical_threshold: float


class ErrorCodeOut(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    code: str
    machine_model: str
    description: str
    fault_class: str
