"""The Server-Sent Events contract for ``POST /chat/stream``.

One orchestrated request produces a stream of typed events. The frontend does
not exist yet, so this contract is deliberately **complete** rather than
minimal: a client may ignore an event it does not need, but it cannot render
one that was never sent. Every intermediate state the orchestrator passes
through — the route it chose, the machine it confirmed, each module starting
and finishing, each token of prose, each citation, each conflict — is on the
wire, in the order it happened.

Two things are worth knowing about the wire format.

* **Each event is emitted twice-addressed.** The SSE frame carries a named
  ``event:`` line *and* repeats the type inside the JSON body::

      event: module_finish
      data: {"type": "module_finish", "data": {...}}

  A client using ``addEventListener("module_finish", ...)`` and a client using
  a single ``onmessage`` handler that switches on ``payload.type`` both work.
  Neither has to know about the other.

* **``module_finish.summary`` is load-bearing.** It is a one-line,
  human-readable statement of what the module actually concluded
  ("Bearing wear, 0.82 confidence"; "3 parts required, 1 out of stock") so a
  client can render a live timeline without parsing six different typed
  payloads. It is derived from the module's own output, never from an LLM.

Comments (``: heartbeat``) are interleaved on a timer so that proxies with an
idle-connection timeout do not drop a long-running orchestration.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.orchestration import (
    Intent,
    ModuleName,
    ModuleStatus,
    OrchestrationCitation,
    OrchestrationResult,
    RoutingDecision,
    Urgency,
    UserRole,
)
from app.schemas.resolution import ResolutionResult, ResolutionStatus


# ---------------------------------------------------------------------------
# Event names
# ---------------------------------------------------------------------------
class StreamEventType(str, Enum):
    """Every event type the stream can carry.

    Lowercase on purpose: these are SSE event names read by a browser, not
    internal enums, and they appear verbatim in client code.
    """

    session = "session"
    routing = "routing"
    resolution = "resolution"
    module_start = "module_start"
    module_finish = "module_finish"
    narrative_delta = "narrative_delta"
    citation = "citation"
    conflict = "conflict"
    result = "result"
    error = "error"
    done = "done"


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------
class SessionData(BaseModel):
    """First event on every stream. Identifies the conversation and the turn.

    ``session_id`` is generated here when the caller did not supply one, so a
    client always has an id to send back on the next turn.
    """

    session_id: str
    request_id: str


class RoutingData(BaseModel):
    """What the router decided, before anything ran."""

    model_config = ConfigDict(use_enum_values=True)

    selected_modules: list[ModuleName] = Field(default_factory=list)
    intent: Intent = Intent.ask_question
    urgency: Urgency = Urgency.normal
    reasoning: str = ""


class ResolutionMachine(BaseModel):
    """The confirmed machine, flattened to what a header needs."""

    machine_id: str
    name: str
    model: str


class ResolutionCandidateData(BaseModel):
    """One machine the operator might have meant, when the reference is ambiguous."""

    machine_id: str
    name: str
    model: str
    line_id: Optional[str] = None
    status: Optional[str] = None
    confidence: float = 0.0
    matched_by: Optional[str] = None


class ResolutionData(BaseModel):
    """The gate's verdict, emitted the moment it is reached.

    ``status`` is ``RESOLVED``, ``AMBIGUOUS`` or ``NOT_FOUND``. Only the first
    carries a ``machine``; the second carries ``candidates`` and a
    ``clarification_question`` to put to the operator, and in both blocking
    cases no module runs.
    """

    model_config = ConfigDict(use_enum_values=True)

    status: ResolutionStatus
    machine: Optional[ResolutionMachine] = None
    candidates: list[ResolutionCandidateData] = Field(default_factory=list)
    clarification_question: Optional[str] = None


class ModuleStartData(BaseModel):
    """A module began. ``level`` is its depth in the dependency graph.

    Modules sharing a level run concurrently, so a client can lay them out
    side by side rather than as a single-file list.
    """

    model_config = ConfigDict(use_enum_values=True)

    module: ModuleName
    level: int = 0


class ModuleFinishData(BaseModel):
    """A module ended — successfully, degraded, unavailable, skipped or reused."""

    model_config = ConfigDict(use_enum_values=True)

    module: ModuleName
    status: ModuleStatus
    elapsed_ms: int = 0
    reason: Optional[str] = None
    #: One line a human can read without opening the typed payload.
    summary: str = ""


class NarrativeDeltaData(BaseModel):
    """A chunk of the composed answer, as it is written."""

    text: str


class CitationData(BaseModel):
    """A source the answer drew on."""

    document_id: str
    title: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None


class ConflictData(BaseModel):
    """A disagreement between modules, stated rather than resolved."""

    description: str


class ErrorData(BaseModel):
    """Something went wrong.

    ``recoverable=True`` means the stream continues and the client will still
    receive a usable answer — the canonical case is a composed narrative that
    failed number validation, after which the deterministic template is
    re-streamed. ``recoverable=False`` means this turn produced nothing.
    """

    message: str
    recoverable: bool = False


class DoneData(BaseModel):
    """Last event on every stream."""

    total_elapsed_ms: int = 0


StreamPayload = Union[
    SessionData,
    RoutingData,
    ResolutionData,
    ModuleStartData,
    ModuleFinishData,
    NarrativeDeltaData,
    CitationData,
    ConflictData,
    OrchestrationResult,
    ErrorData,
    DoneData,
]


# ---------------------------------------------------------------------------
# The frame
# ---------------------------------------------------------------------------
class StreamEvent(BaseModel):
    """One SSE frame: a type and its payload."""

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)

    type: StreamEventType
    data: Any = None

    # -- serialisation -----------------------------------------------------
    def payload(self) -> dict[str, Any]:
        """The JSON body of this frame, type included."""
        data = self.data
        if isinstance(data, BaseModel):
            data = data.model_dump(mode="json")
        return {"type": str(StreamEventType(self.type).value), "data": data}

    def to_sse(self) -> str:
        """Render as an SSE frame, named event line included."""
        body = json.dumps(self.payload(), default=str)
        return f"event: {StreamEventType(self.type).value}\ndata: {body}\n\n"

    # -- constructors ------------------------------------------------------
    # Named after the events themselves so call sites read like the contract.
    @classmethod
    def for_session(cls, session_id: str, request_id: str) -> "StreamEvent":
        return cls(
            type=StreamEventType.session,
            data=SessionData(session_id=session_id, request_id=request_id),
        )

    @classmethod
    def for_routing(cls, decision: RoutingDecision) -> "StreamEvent":
        return cls(
            type=StreamEventType.routing,
            data=RoutingData(
                selected_modules=[ModuleName(m) for m in decision.selected_modules],
                intent=Intent(decision.intent),
                urgency=Urgency(decision.urgency),
                reasoning=decision.reasoning,
            ),
        )

    @classmethod
    def for_resolution(cls, resolution: ResolutionResult) -> "StreamEvent":
        machine = None
        if resolution.machine is not None:
            machine = ResolutionMachine(
                machine_id=resolution.machine.machine_id,
                name=resolution.machine.name,
                model=resolution.machine.model,
            )
        return cls(
            type=StreamEventType.resolution,
            data=ResolutionData(
                status=ResolutionStatus(resolution.status),
                machine=machine,
                candidates=[
                    ResolutionCandidateData(
                        machine_id=c.machine_id,
                        name=c.name,
                        model=c.model,
                        line_id=c.line_id,
                        status=str(c.status),
                        confidence=c.confidence,
                        matched_by=str(c.matched_by),
                    )
                    for c in resolution.candidates
                ],
                clarification_question=resolution.clarification_question,
            ),
        )

    @classmethod
    def for_module_start(cls, module: ModuleName, level: int) -> "StreamEvent":
        return cls(
            type=StreamEventType.module_start,
            data=ModuleStartData(module=ModuleName(module), level=level),
        )

    @classmethod
    def for_module_finish(
        cls,
        module: ModuleName,
        status: ModuleStatus,
        *,
        elapsed_ms: int = 0,
        reason: Optional[str] = None,
        summary: str = "",
    ) -> "StreamEvent":
        return cls(
            type=StreamEventType.module_finish,
            data=ModuleFinishData(
                module=ModuleName(module),
                status=ModuleStatus(status),
                elapsed_ms=elapsed_ms,
                reason=reason,
                summary=summary,
            ),
        )

    @classmethod
    def for_narrative_delta(cls, text: str) -> "StreamEvent":
        return cls(
            type=StreamEventType.narrative_delta, data=NarrativeDeltaData(text=text)
        )

    @classmethod
    def for_citation(cls, citation: OrchestrationCitation) -> "StreamEvent":
        return cls(
            type=StreamEventType.citation,
            data=CitationData(
                document_id=citation.document_id,
                title=citation.title,
                page_number=citation.page_number,
                section_title=citation.section_title,
            ),
        )

    @classmethod
    def for_conflict(cls, description: str) -> "StreamEvent":
        return cls(
            type=StreamEventType.conflict, data=ConflictData(description=description)
        )

    @classmethod
    def for_result(cls, result: OrchestrationResult) -> "StreamEvent":
        return cls(type=StreamEventType.result, data=result)

    @classmethod
    def for_error(cls, message: str, *, recoverable: bool) -> "StreamEvent":
        return cls(
            type=StreamEventType.error,
            data=ErrorData(message=message, recoverable=recoverable),
        )

    @classmethod
    def for_done(cls, total_elapsed_ms: int) -> "StreamEvent":
        return cls(
            type=StreamEventType.done, data=DoneData(total_elapsed_ms=total_elapsed_ms)
        )


class SSEComment:
    """A comment frame. Carries no data; exists to keep the connection warm.

    Proxies and load balancers close idle connections, and an orchestration
    with a slow module can be silent for a while. A comment is legal SSE that
    every client implementation ignores, which makes it the right keep-alive.
    """

    __slots__ = ("text",)

    def __init__(self, text: str = "heartbeat") -> None:
        self.text = text

    def to_sse(self) -> str:
        return f": {self.text}\n\n"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<SSEComment {self.text!r}>"


#: The keep-alive emitted on the heartbeat timer.
HEARTBEAT = SSEComment("heartbeat")

#: Anything the stream may yield.
SSEMessage = Union[StreamEvent, SSEComment]


# ---------------------------------------------------------------------------
# Endpoint payload
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Body of ``POST /chat`` and ``POST /chat/stream``."""

    model_config = ConfigDict(use_enum_values=True)

    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    user_role: UserRole = UserRole.technician
    #: Set when the caller already knows the machine (the operator tapped it on
    #: a dashboard). Skips the guessing half of resolution.
    machine_id: Optional[str] = None
    #: ISO 639-1 code. Supplied by a client that already knows the operator's
    #: language; when absent it is detected from the message itself.
    language: Optional[str] = None
