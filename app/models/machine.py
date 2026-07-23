"""Persisted document models for the Site -> Line -> Machine -> Component hierarchy.

These Pydantic v2 models describe how documents are stored in MongoDB. Each model
carries its own natural/business key (``machine_id``, ``component_id``, ...) which
is what the rest of the application references, independent of Mongo's ``_id``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class MachineStatus(str, Enum):
    """Operational status of a machine."""

    running = "running"
    stopped = "stopped"
    maintenance = "maintenance"
    fault = "fault"


class SensorType(str, Enum):
    """Physical quantity measured by a sensor."""

    temperature = "temperature"
    vibration = "vibration"
    pressure = "pressure"
    rpm = "rpm"
    power = "power"


class ComponentType(str, Enum):
    """Coarse classification of a machine component."""

    motor = "motor"
    bearing = "bearing"
    pump = "pump"
    spindle = "spindle"
    gearbox = "gearbox"
    belt = "belt"
    valve = "valve"
    cylinder = "cylinder"
    roller = "roller"
    controller = "controller"
    sensor = "sensor"
    frame = "frame"
    other = "other"


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------
class TenantOwned(BaseModel):
    """Base for every document owned by a tenant.

    ``tenant_id`` is required, so a document with no owner cannot be
    constructed. Subclassing rather than repeating the field means a new model
    inherits the ownership rule instead of having to remember it.
    """

    model_config = ConfigDict(use_enum_values=True, populate_by_name=True)

    tenant_id: str = Field(..., min_length=1, description="Owning tenant id")


# ---------------------------------------------------------------------------
# Hierarchy: Site -> ProductionLine
# ---------------------------------------------------------------------------
class Site(TenantOwned):
    """A physical plant / facility."""

    site_id: str = Field(..., description="Canonical site id, e.g. 'SITE-DETROIT'")
    name: str
    location: Optional[str] = None
    timezone: str = "UTC"


class ProductionLine(TenantOwned):
    """A production line within a site."""

    line_id: str = Field(..., description="Canonical line id, e.g. 'LINE-A'")
    site_id: str
    name: str
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# Machine
# ---------------------------------------------------------------------------
class Machine(TenantOwned):
    """An industrial machine positioned within a production line."""

    machine_id: str = Field(..., description="Canonical machine id, e.g. 'CV-201'")
    name: str
    model: str
    manufacturer: str

    site_id: str
    line_id: str
    position_in_line: int = Field(..., ge=0, description="Ordinal position along the line")

    criticality: int = Field(..., ge=1, le=5, description="1 (low) .. 5 (mission critical)")
    status: MachineStatus = MachineStatus.running

    aliases: list[str] = Field(
        default_factory=list,
        description="Alternate names: floor names, ERP names, drawing/tag names",
    )

    installed_at: Optional[datetime] = None
    last_maintenance_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Component (self-nesting via parent_component_id)
# ---------------------------------------------------------------------------
class Component(TenantOwned):
    """A component of a machine. Components may nest via ``parent_component_id``."""

    component_id: str = Field(..., description="Canonical component id, e.g. 'CV-201-MTR'")
    machine_id: str
    name: str
    type: ComponentType = ComponentType.other
    part_number: Optional[str] = None
    parent_component_id: Optional[str] = Field(
        default=None, description="Parent component id for nested sub-assemblies"
    )


# ---------------------------------------------------------------------------
# Sensor
# ---------------------------------------------------------------------------
class Sensor(TenantOwned):
    """A sensor attached to a machine (and optionally a specific component)."""

    sensor_id: str = Field(..., description="Canonical sensor id, e.g. 'CV-201-VIB-01'")
    machine_id: str
    component_id: Optional[str] = None
    type: SensorType
    unit: str = Field(..., description="Engineering unit, e.g. '°C', 'mm/s', 'bar', 'rpm', 'kW'")

    normal_min: float
    normal_max: float
    warning_threshold: float
    critical_threshold: float


# ---------------------------------------------------------------------------
# Error / fault codes
# ---------------------------------------------------------------------------
class ErrorCode(TenantOwned):
    """A fault/error code emitted by a class of machine model."""

    code: str = Field(..., description="Fault code, e.g. 'E104'")
    machine_model: str = Field(..., description="Machine model the code applies to")
    description: str
    fault_class: str = Field(..., description="e.g. 'mechanical', 'electrical', 'hydraulic'")
