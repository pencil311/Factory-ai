"""Typed output schemas for the four domain agents.

Every agent returns STRUCTURED DATA, not prose. Natural language is composed
once, at aggregation time, later. Each schema is the contract downstream
consumers depend on.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Agent base
# ---------------------------------------------------------------------------
class AgentStatus(str, Enum):
    ok = "OK"
    partial = "PARTIAL"
    unavailable = "UNAVAILABLE"


class AgentResult(BaseModel):
    """Wrapper every agent returns."""

    model_config = ConfigDict(use_enum_values=True)

    agent_name: str
    status: AgentStatus
    data: Optional[Any] = None
    reason: Optional[str] = None
    #: Full exception text when ``reason`` was derived from one. Logged and
    #: kept on the result for debugging; never composed into the narrative.
    error_detail: Optional[str] = None
    #: Names of this agent's SOFT upstream inputs that were unavailable, so it
    #: ran on less than the full picture (e.g. no identified root cause).
    degraded_inputs: list[str] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    elapsed_ms: int = 0


class AgentContext(BaseModel):
    """Input context every agent receives."""

    model_config = ConfigDict(use_enum_values=True)

    tenant_id: str
    machine_id: str
    rca_result: Optional[dict[str, Any]] = None
    pdm_result: Optional[dict[str, Any]] = None
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    user_role: Optional[str] = None
    raw_query: Optional[str] = None
    #: Upstream agent output the orchestrator forwards. Agents still never call
    #: each other — the orchestrator passes what a downstream agent needs, and
    #: an agent given nothing here falls back to its own lookups.
    maintenance_plan: Optional[dict[str, Any]] = None
    inventory_status: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Maintenance Agent
# ---------------------------------------------------------------------------
class ProcedureSource(str, Enum):
    documented = "DOCUMENTED"
    derived = "DERIVED"


class SkillLevel(str, Enum):
    basic = "BASIC"
    intermediate = "INTERMEDIATE"
    specialist = "SPECIALIST"


class ProcedureCitation(BaseModel):
    document_id: Optional[str] = None
    page_number: Optional[int] = None


class ProcedureStep(BaseModel):
    order: int
    instruction: str
    component_id: Optional[str] = None
    tools_required: list[str] = Field(default_factory=list)
    estimated_minutes: int = 0
    caution: Optional[str] = None
    citation: Optional[ProcedureCitation] = None


class RequiredPart(BaseModel):
    part_number: str
    description: str
    quantity: int = 1
    component_id: Optional[str] = None


class MaintenancePlan(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    procedure_steps: list[ProcedureStep] = Field(default_factory=list)
    required_parts: list[RequiredPart] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    total_estimated_minutes: int = 0
    skill_level: SkillLevel = SkillLevel.intermediate
    procedure_source: ProcedureSource = ProcedureSource.derived


# ---------------------------------------------------------------------------
# Inventory Agent
# ---------------------------------------------------------------------------
class StockStatus(str, Enum):
    in_stock = "IN_STOCK"
    low_stock = "LOW_STOCK"
    out_of_stock = "OUT_OF_STOCK"


class InventoryItem(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    part_number: str
    description: str
    required_qty: int = 1
    available_qty: int = 0
    status: StockStatus = StockStatus.out_of_stock
    location: Optional[str] = None
    alternatives: list[str] = Field(default_factory=list)
    lead_time_days: int = 0


class InventoryStatus(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    items: list[InventoryItem] = Field(default_factory=list)
    all_parts_available: bool = False
    blocking_parts: list[str] = Field(default_factory=list)
    earliest_full_availability_days: int = 0


# ---------------------------------------------------------------------------
# Safety Agent
# ---------------------------------------------------------------------------
class HazardSeverity(str, Enum):
    low = "LOW"
    medium = "MEDIUM"
    high = "HIGH"
    critical = "CRITICAL"


class EnergySourceType(str, Enum):
    electrical = "ELECTRICAL"
    hydraulic = "HYDRAULIC"
    pneumatic = "PNEUMATIC"
    thermal = "THERMAL"
    mechanical = "MECHANICAL"
    chemical = "CHEMICAL"


class SafetySource(str, Enum):
    documented = "DOCUMENTED"
    generic = "GENERIC"


class Hazard(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    hazard_type: str
    description: str
    severity: HazardSeverity = HazardSeverity.medium
    source_component_id: Optional[str] = None


class LotoStep(BaseModel):
    order: int
    instruction: str
    verification: str


class EnergySource(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    type: EnergySourceType
    location: str
    isolation_method: str


class SafetyBriefing(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    hazards: list[Hazard] = Field(default_factory=list)
    required_ppe: list[str] = Field(default_factory=list)
    lockout_tagout_steps: list[LotoStep] = Field(default_factory=list)
    energy_sources_to_isolate: list[EnergySource] = Field(default_factory=list)
    permits_required: list[str] = Field(default_factory=list)
    #: Genuine, machine-state-specific conditions that must be resolved before
    #: work may begin at all — an unisolated energy source, a pressurised
    #: vessel, a fault mode that makes the machine unsafe to approach. Never
    #: routine steps; those belong in ``standard_preconditions``. Driving a
    #: "lead with a warning banner" UI off this field only works if it stays
    #: reserved for conditions that are actually exceptional.
    blocking_conditions: list[str] = Field(default_factory=list)
    #: Routine, always-applicable steps and requirements — standard LOTO
    #: sequence, standard PPE. True for every job on every machine, so they
    #: render as ordinary briefing content rather than leading with a warning.
    standard_preconditions: list[str] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    source: SafetySource = SafetySource.generic


# ---------------------------------------------------------------------------
# Production Agent
# ---------------------------------------------------------------------------
class DowntimeRecommendation(str, Enum):
    repair_now = "REPAIR_NOW"
    schedule_next_window = "SCHEDULE_NEXT_WINDOW"
    monitor = "MONITOR"


class DowntimeEstimate(BaseModel):
    repair_time: int = 0  # minutes
    total_including_parts_wait: int = 0  # minutes


class CostEstimate(BaseModel):
    downtime_cost: float = 0.0
    parts_cost: float = 0.0
    total: float = 0.0
    currency: str = "USD"


class ProductionImpact(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    downtime_estimate_minutes: DowntimeEstimate = Field(default_factory=DowntimeEstimate)
    units_lost_estimate: int = 0
    is_bottleneck: bool = False
    downstream_machines_affected: list[str] = Field(default_factory=list)
    cost_estimate: CostEstimate = Field(default_factory=CostEstimate)
    recommendation: DowntimeRecommendation = DowntimeRecommendation.monitor
    recommendation_rationale: str = ""
    assumptions: list[str] = Field(default_factory=list)
