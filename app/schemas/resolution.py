"""Schemas for entity resolution: messy text in, exactly one machine out.

Resolution is safety-critical. A wrong machine means the wrong manual, the wrong
spare part, and the wrong lockout/tagout procedure — so the contract here is
deliberately explicit: a result is either ``RESOLVED`` (and carries exactly one
machine), or it BLOCKS (``AMBIGUOUS``/``NOT_FOUND``) and the caller must stop and
ask the human. There is no "best guess" path.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.machine import MachineStatus


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class MatchMethod(str, Enum):
    """Which link in the resolution chain produced a candidate."""

    exact_id = "EXACT_ID"
    alias = "ALIAS"
    error_code = "ERROR_CODE"
    context = "CONTEXT"
    fuzzy = "FUZZY"


class ResolutionStatus(str, Enum):
    """Outcome of a resolution attempt.

    ``AMBIGUOUS`` and ``NOT_FOUND`` are both blocking states — callers must not
    proceed to act on a machine.
    """

    resolved = "RESOLVED"
    ambiguous = "AMBIGUOUS"
    not_found = "NOT_FOUND"


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
class ResolutionContext(BaseModel):
    """Conversational and operational context used to disambiguate candidates.

    Every field is optional; an empty context simply means the CONTEXT stage
    contributes no boosts.
    """

    model_config = ConfigDict(use_enum_values=True)

    user_id: Optional[str] = Field(default=None, description="Requesting user")
    assigned_line_id: Optional[str] = Field(
        default=None, description="Production line the user is assigned to"
    )
    last_machine_id: Optional[str] = Field(
        default=None, description="Machine resolved on the previous conversational turn"
    )
    active_alarm_machine_ids: list[str] = Field(
        default_factory=list, description="Machines currently in alarm"
    )


class ResolutionRequest(BaseModel):
    """Body of ``POST /resolve``."""

    model_config = ConfigDict(use_enum_values=True)

    text: str = Field(..., description="Raw natural-language input from the operator")
    context: ResolutionContext = Field(default_factory=ResolutionContext)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
class ResolutionCandidate(BaseModel):
    """One machine the input might refer to, with an audit trail of why."""

    model_config = ConfigDict(use_enum_values=True)

    machine_id: str
    name: str
    model: str
    line_id: str
    status: MachineStatus

    confidence: float = Field(..., ge=0.0, le=1.0)
    matched_by: MatchMethod
    matched_value: str = Field(..., description="The value that actually matched")
    error_code: Optional[str] = Field(
        default=None, description="Error code that produced this candidate, if any"
    )


class ResolutionResult(BaseModel):
    """The full outcome, including why it is or is not actionable."""

    model_config = ConfigDict(use_enum_values=True)

    status: ResolutionStatus
    machine: Optional[ResolutionCandidate] = Field(
        default=None, description="Set only when status is RESOLVED"
    )
    candidates: list[ResolutionCandidate] = Field(
        default_factory=list, description="Ranked descending; populated when AMBIGUOUS"
    )
    clarification_question: Optional[str] = Field(
        default=None, description="Question to put to the operator when AMBIGUOUS"
    )
    extracted_error_codes: list[str] = Field(default_factory=list)
    raw_input: str

    @property
    def is_blocking(self) -> bool:
        """True when the caller must stop and ask rather than act.

        Compared with ``!=`` rather than ``is not``: ``use_enum_values`` stores
        the plain string, which compares equal to the member but is not it.
        """
        return self.status != ResolutionStatus.resolved
