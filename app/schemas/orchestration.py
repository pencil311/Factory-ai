"""The orchestration output contract.

One shape comes back from ``POST /orchestrate`` no matter how many modules ran,
how many failed, or which role asked. Everything downstream — the UI, the
streaming layer, the audit log — reads this and nothing else.

Two rules are encoded here rather than left to convention:

* Every module that was *selected* appears in ``modules_run`` with its outcome,
  including the ones that failed or were skipped. A missing module is never
  silently dropped; the request degrades to ``PARTIAL`` and says why.
* ``narrative`` is prose composed once, at the end, from the structured fields
  in this same object. ``narrative_source`` records whether an LLM wrote it or
  the deterministic template did, so a reader always knows which they are
  looking at.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.agents import (
    InventoryStatus,
    MaintenancePlan,
    ProductionImpact,
    SafetyBriefing,
)
from app.schemas.pdm import PdmPredictionOut
from app.schemas.rca import RCAResult


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class ModuleName(str, Enum):
    """Every module the orchestrator knows how to run.

    The router LLM may only return names from this set; anything else is
    dropped rather than trusted.
    """

    resolver = "RESOLVER"
    rag = "RAG"
    pdm = "PDM"
    rca = "RCA"
    maintenance = "MAINTENANCE"
    inventory = "INVENTORY"
    safety = "SAFETY"
    production = "PRODUCTION"


class Intent(str, Enum):
    report_fault = "REPORT_FAULT"
    ask_question = "ASK_QUESTION"
    check_status = "CHECK_STATUS"
    request_procedure = "REQUEST_PROCEDURE"
    assess_impact = "ASSESS_IMPACT"


class Urgency(str, Enum):
    low = "LOW"
    normal = "NORMAL"
    high = "HIGH"
    critical = "CRITICAL"


class UserRole(str, Enum):
    technician = "TECHNICIAN"
    engineer = "ENGINEER"
    manager = "MANAGER"
    safety_officer = "SAFETY_OFFICER"


class ModuleStatus(str, Enum):
    """Outcome of one module within one request.

    ``UNAVAILABLE`` means the module ran and could not answer (or timed out).
    ``SKIPPED`` means it never ran because something it depends on was
    unavailable — the reason always names that dependency.
    """

    ok = "OK"
    partial = "PARTIAL"
    unavailable = "UNAVAILABLE"
    skipped = "SKIPPED"
    reused = "REUSED"


class OrchestrationStatus(str, Enum):
    complete = "COMPLETE"
    partial = "PARTIAL"
    clarification_needed = "CLARIFICATION_NEEDED"
    not_found = "NOT_FOUND"


class NarrativeSource(str, Enum):
    llm = "LLM"
    template = "TEMPLATE"


class RoutingSource(str, Enum):
    """Where the module selection came from.

    Anything beginning with ``FALLBACK`` means the LLM did not produce a usable
    answer and the safe default set was used instead.
    """

    llm = "LLM"
    fallback_no_client = "FALLBACK_NO_CLIENT"
    fallback_timeout = "FALLBACK_TIMEOUT"
    fallback_invalid = "FALLBACK_INVALID"
    fallback_error = "FALLBACK_ERROR"
    explicit = "EXPLICIT"


class ProgressEventType(str, Enum):
    module_started = "MODULE_STARTED"
    module_finished = "MODULE_FINISHED"
    module_skipped = "MODULE_SKIPPED"
    module_reused = "MODULE_REUSED"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class ProgressEvent(BaseModel):
    """Emitted as modules start and finish. The streaming layer replays these."""

    model_config = ConfigDict(use_enum_values=True)

    type: ProgressEventType
    module: ModuleName
    status: Optional[ModuleStatus] = None
    elapsed_ms: int = 0
    reason: Optional[str] = None
    #: The module's raw output, carried on finish/reuse events only. The
    #: executor stays ignorant of what modules mean — this exists so the
    #: streaming layer can summarise a result *as it lands* rather than
    #: waiting for the aggregated response. Never part of the HTTP contract.
    data: Optional[Any] = None
    at: datetime = Field(default_factory=utcnow)


class ModuleRun(BaseModel):
    """One row of the per-request module ledger."""

    model_config = ConfigDict(use_enum_values=True)

    name: ModuleName
    status: ModuleStatus
    elapsed_ms: int = 0
    reason: Optional[str] = None
    #: Full exception text when ``reason`` was derived from one. Logged and
    #: kept in the API result for debugging; never composed into the narrative.
    error_detail: Optional[str] = None
    #: Names of this module's SOFT dependencies that produced nothing usable —
    #: it ran anyway, on less than the full picture.
    degraded_inputs: list[str] = Field(default_factory=list)
    #: True when the result came from the conversation cache rather than a run.
    reused: bool = False


class RoutingDecision(BaseModel):
    """What the router chose, and why — surfaced in the UI."""

    model_config = ConfigDict(use_enum_values=True)

    selected_modules: list[ModuleName] = Field(default_factory=list)
    reasoning: str = ""
    source: RoutingSource = RoutingSource.llm
    intent: Intent = Intent.ask_question
    urgency: Urgency = Urgency.normal
    machine_reference: Optional[str] = None
    #: Names the LLM returned that are not known modules. Kept for debugging;
    #: they never influence execution.
    dropped_modules: list[str] = Field(default_factory=list)


class MachineSummary(BaseModel):
    """The resolved machine, flattened to what the answer needs."""

    machine_id: str
    name: str
    model: str
    line_id: str
    status: str


class ClarificationCandidate(BaseModel):
    """One machine the operator might have meant."""

    machine_id: str
    name: str
    model: str
    line_id: str
    status: str
    confidence: float = 0.0
    matched_by: Optional[str] = None


class Clarification(BaseModel):
    """Returned instead of an answer when the machine is ambiguous."""

    question: str
    candidates: list[ClarificationCandidate] = Field(default_factory=list)


class OrchestrationCitation(BaseModel):
    document_id: str
    title: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None


class BlockedStep(BaseModel):
    """A maintenance step that cannot proceed because a part is unavailable."""

    order: int
    instruction: str
    blocked_by_parts: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------
class OrchestrationResult(BaseModel):
    """Everything one orchestrated request produced."""

    model_config = ConfigDict(use_enum_values=True)

    request_id: str
    tenant_id: str
    user_role: UserRole
    intent: Intent = Intent.ask_question
    urgency: Urgency = Urgency.normal

    status: OrchestrationStatus

    machine: Optional[MachineSummary] = None
    clarification: Optional[Clarification] = None

    narrative: str = ""
    narrative_source: NarrativeSource = NarrativeSource.template
    #: ISO 639-1 code of the language the operator wrote in, and the language
    #: ``narrative`` is written in whenever an LLM composed it.
    detected_language: str = "en"
    #: True when the answer came back in English because the deterministic
    #: template was used and the operator did not write in English. Templates
    #: are never machine-translated — a mistranslated safety instruction is a
    #: hazard — so the reader is told instead.
    language_fallback: bool = False

    modules_run: list[ModuleRun] = Field(default_factory=list)
    routing_decision: RoutingDecision = Field(default_factory=RoutingDecision)

    rca: Optional[RCAResult] = None
    pdm: Optional[PdmPredictionOut] = None
    maintenance: Optional[MaintenancePlan] = None
    inventory: Optional[InventoryStatus] = None
    safety: Optional[SafetyBriefing] = None
    production: Optional[ProductionImpact] = None

    citations: list[OrchestrationCitation] = Field(default_factory=list)
    #: Human-readable statements of every disagreement between modules. Never
    #: resolved silently — if two modules disagree, the reader is told.
    conflicts_surfaced: list[str] = Field(default_factory=list)

    total_elapsed_ms: int = 0

    # --- flags the composer and the UI both key off ------------------------
    session_id: Optional[str] = None
    #: Set when Safety reported a CRITICAL hazard or a blocking condition.
    safety_critical: bool = False
    #: Set when a repair recommendation is contingent on safety sign-off.
    safety_clearance_required: bool = False
    #: Set when RCA reported insufficient data — all recommendations downstream
    #: of it are provisional.
    provisional: bool = False
    blocked_steps: list[BlockedStep] = Field(default_factory=list)
    #: Sections withheld for this role (never includes safety).
    omitted_for_role: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Endpoint payloads
# ---------------------------------------------------------------------------
class OrchestrateRequest(BaseModel):
    """Body of ``POST /orchestrate``."""

    model_config = ConfigDict(use_enum_values=True)

    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    user_role: UserRole = UserRole.technician
    #: Set when the caller already knows the machine (e.g. the operator tapped
    #: it on a dashboard). Skips the guessing half of resolution.
    machine_id: Optional[str] = None


class SessionTurn(BaseModel):
    """One exchange in a conversation."""

    model_config = ConfigDict(use_enum_values=True)

    request_id: str
    message: str
    user_role: UserRole
    machine_id: Optional[str] = None
    status: OrchestrationStatus
    narrative: str = ""
    modules_run: list[str] = Field(default_factory=list)
    at: datetime = Field(default_factory=utcnow)


class SessionOut(BaseModel):
    """Body of ``GET /orchestrate/sessions/{session_id}``."""

    session_id: str
    tenant_id: str
    last_machine_id: Optional[str] = None
    turns: list[SessionTurn] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    #: Modules whose results are still fresh enough to reuse, and for how long.
    cached_modules: list[str] = Field(default_factory=list)
    cache_age_seconds: Optional[float] = None
