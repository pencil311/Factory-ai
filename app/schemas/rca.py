"""Root-cause analysis output contract.

Pinned exactly: Inventory, Safety, Maintenance and AR all consume these shapes,
and ``component_id`` on a hypothesis is what lets them act without guessing
which part of the machine is at fault.

RCA explains WHY. It does not prescribe repairs — that is the Maintenance
Agent's job — and nothing in these schemas carries a remedy.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EvidenceSource(str, Enum):
    """Where a piece of evidence came from.

    Independence is counted across these categories: two sensors agreeing is
    one source, a sensor plus the PdM model agreeing is two.
    """

    sensor = "SENSOR"
    pdm_model = "PDM_MODEL"
    history = "HISTORY"
    document = "DOCUMENT"
    threshold = "THRESHOLD"


class EvidenceStrength(str, Enum):
    weak = "WEAK"
    moderate = "MODERATE"
    strong = "STRONG"


class Citation(BaseModel):
    """Pointer back to a retrieved passage."""

    document_id: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None


class Evidence(BaseModel):
    """One observation supporting or contradicting a hypothesis."""

    model_config = ConfigDict(use_enum_values=True)

    evidence_id: str
    source: EvidenceSource
    description: str
    strength: EvidenceStrength
    value: Optional[Union[str, float]] = None
    citation: Optional[Citation] = None


class CausalHypothesis(BaseModel):
    """A candidate explanation for the observed condition."""

    cause_id: str
    description: str
    #: The component believed to be at fault. Downstream agents key off this,
    #: so it is None only when the signals genuinely do not localise.
    component_id: Optional[str] = None
    fault_mode: str
    probability: float = Field(..., ge=0.0, le=1.0)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)


class CausalStep(BaseModel):
    """One link in the chain from initiating cause to observed symptom."""

    order: int
    description: str
    #: The physical why — what is actually happening in the metal.
    mechanism: str
    evidence_ids: list[str] = Field(default_factory=list)
    sensor_signals: list[str] = Field(default_factory=list)


class RCAResult(BaseModel):
    """The full analysis."""

    model_config = ConfigDict(use_enum_values=True)

    machine_id: str
    primary_cause: Optional[CausalHypothesis] = None
    alternative_causes: list[CausalHypothesis] = Field(default_factory=list)
    causal_chain: list[CausalStep] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    confidence: float = Field(..., ge=0.0, le=1.0)
    #: One line explaining how the confidence number was arrived at. Derived
    #: from evidence counts and agreement, never self-reported by a model.
    confidence_basis: str
    analysis_timestamp: datetime = Field(default_factory=utcnow)

    #: True when the evidence is too thin to conclude. Confidence is capped
    #: at 0.5 whenever this is set.
    insufficient_data: bool = False
    missing_data: list[str] = Field(default_factory=list)

    #: False when the chain was composed mechanically because no LLM was
    #: available. The analysis itself is unaffected — only the prose.
    narrative_generated: bool = False

    tenant_id: Optional[str] = None


class RCARequest(BaseModel):
    """Body of ``POST /rca/analyze``."""

    machine_id: str = Field(..., min_length=1)
    include_narrative: bool = True
