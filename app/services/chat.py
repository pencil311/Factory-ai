"""The streaming chat service: one orchestration, rendered as it happens.

``Orchestrator.handle`` is a coroutine that returns one finished answer. This
module turns that into a live event stream without duplicating any of its
logic: it subscribes to the orchestrator's lifecycle hooks and the executor's
progress events, pushes each onto a queue, and drains that queue while the
orchestration runs as a background task. Nothing here waits for completion
before emitting.

Four behaviours are worth stating outright, because they are the ones a client
depends on and the ones that are easy to get subtly wrong.

**The gate is visible.** ``resolution`` is emitted the instant the gate
decides, before the branch. An ambiguous machine reference produces
``resolution`` carrying the candidates and the question, then the result and
``done`` — and no module event at all, because no module ran.

**Narration is identical with or without an LLM.** When a key is present the
composer streams and each token becomes a ``narrative_delta``. When it is
absent the deterministic template is emitted as a small number of
``narrative_delta`` chunks. A client renders one code path either way; it
never has to ask which composer produced the text.

**A fabricated number is a recoverable failure, not a silent one.** The
composer's output is validated against the structured data *after* streaming,
because that is the only time the full text exists. If it fails, the client
has already rendered prose that is about to be replaced — so it is told:
``error`` with ``recoverable=true``, then the template re-streamed as
``narrative_delta`` chunks. Whatever was on screen is superseded by what
follows.

**A disconnected client stops the work.** The orchestration runs as a task; if
the generator is closed — the client hung up, the proxy dropped, the request
was cancelled — the task is cancelled, which propagates into the executor's
``asyncio.gather`` and cancels every in-flight module. A vanished operator
should not keep an RCA query running against the sensor store.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from app.config import get_settings
from app.orchestrator.aggregator import composer_system
from app.orchestrator.graph import execution_levels, expand_selection
from app.orchestrator.orchestrator import (
    OrchestrationHooks,
    Orchestrator,
    get_orchestrator,
)
from app.schemas.orchestration import (
    ModuleName,
    ModuleStatus,
    NarrativeSource,
    OrchestrationResult,
    OrchestrationStatus,
    ProgressEvent,
    ProgressEventType,
    RoutingDecision,
    UserRole,
)
from app.schemas.resolution import ResolutionResult
from app.schemas.stream import HEARTBEAT, SSEMessage, StreamEvent
from app.services.language import detect_language

logger = logging.getLogger(__name__)

#: How often a comment frame goes out while nothing else is happening. Proxies
#: commonly drop connections idle for 30–60s; 15s leaves generous margin.
HEARTBEAT_SECONDS = 15.0

#: Roughly how many chunks a template narrative is split into. Small on
#: purpose — the point is that the client's rendering path is the same as for
#: a streamed LLM narrative, not to fake token-by-token generation.
TEMPLATE_CHUNK_COUNT = 6

DEFAULT_COMPOSER_MODEL = get_settings().anthropic_model

#: An async generator of text chunks, given a prompt and a system prompt.
#: Injectable so tests drive the streaming path without a network or a key.
NarrativeStream = Callable[..., AsyncIterator[str]]


# ---------------------------------------------------------------------------
# Queue sentinels
# ---------------------------------------------------------------------------
@dataclass
class _Completed:
    result: OrchestrationResult


@dataclass
class _Failed:
    error: BaseException


@dataclass
class _Cancelled:
    pass


# ---------------------------------------------------------------------------
# Module summaries
# ---------------------------------------------------------------------------
def _pct(value: Any) -> str:
    try:
        return f"{round(float(value) * 100)}%"
    except (TypeError, ValueError):
        return "unknown"


def _rag_summary(data: Any) -> str:
    chunks = list((data or {}).get("chunks") or [])
    if not chunks:
        return "No passages matched."
    documents = {c.get("document_id") for c in chunks if isinstance(c, dict)}
    return (
        f"{len(chunks)} passage(s) retrieved from "
        f"{len(documents)} document(s)."
    )


def _pdm_summary(data: Any) -> str:
    if not data:
        return "No prediction available."
    return (
        f"Failure probability {_pct(data.get('failure_probability'))}, "
        f"{round(float(data.get('remaining_useful_life_hours') or 0), 1)}h remaining "
        f"useful life, trend {data.get('trend_direction', 'UNKNOWN')}."
    )


def _rca_summary(data: Any) -> str:
    if not data:
        return "No root cause analysis."
    cause = (data or {}).get("primary_cause") or {}
    confidence = data.get("confidence")
    try:
        confidence_text = f"{float(confidence):.2f} confidence"
    except (TypeError, ValueError):
        confidence_text = "confidence unknown"
    if not cause:
        return f"No single root cause identified, {confidence_text}."
    headline = cause.get("description") or cause.get("fault_mode") or "Cause identified"
    return f"{headline}, {confidence_text}."


def _maintenance_summary(data: Any) -> str:
    if not data:
        return "No repair procedure."
    steps = list(data.get("procedure_steps") or [])
    parts = list(data.get("required_parts") or [])
    return (
        f"{len(steps)} step(s), about {data.get('total_estimated_minutes', 0)} minutes, "
        f"{len(parts)} part(s) required."
    )


def _inventory_summary(data: Any) -> str:
    if not data:
        return "No parts checked."
    items = list(data.get("items") or [])
    blocking = list(data.get("blocking_parts") or [])
    if not items:
        return "No parts required."
    if not blocking:
        return f"{len(items)} part(s) required, all in stock."
    return (
        f"{len(items)} part(s) required, {len(blocking)} out of stock "
        f"({', '.join(blocking)})."
    )


def _safety_summary(data: Any) -> str:
    if not data:
        return "No safety briefing."
    hazards = list(data.get("hazards") or [])
    blocking = list(data.get("blocking_conditions") or [])
    severe = sum(
        1
        for hazard in hazards
        if str(hazard.get("severity", "")).upper() in ("CRITICAL", "HIGH")
    )
    summary = f"{len(hazards)} hazard(s), {severe} at CRITICAL or HIGH"
    if blocking:
        summary += f", {len(blocking)} blocking condition(s)"
    return summary + "."


def _production_summary(data: Any) -> str:
    if not data:
        return "No production impact."
    downtime = data.get("downtime_estimate_minutes") or {}
    cost = data.get("cost_estimate") or {}
    return (
        f"{data.get('recommendation', 'UNKNOWN')} — "
        f"{downtime.get('total_including_parts_wait', 0)} minutes downtime, "
        f"{data.get('units_lost_estimate', 0)} units lost, "
        f"{cost.get('currency', 'USD')} {cost.get('total', 0)} total."
    )


_SUMMARISERS: dict[ModuleName, Callable[[Any], str]] = {
    ModuleName.rag: _rag_summary,
    ModuleName.pdm: _pdm_summary,
    ModuleName.rca: _rca_summary,
    ModuleName.maintenance: _maintenance_summary,
    ModuleName.inventory: _inventory_summary,
    ModuleName.safety: _safety_summary,
    ModuleName.production: _production_summary,
}


def summarize_module(
    module: ModuleName,
    status: ModuleStatus,
    data: Any,
    reason: Optional[str] = None,
) -> str:
    """One human-readable line stating what a module concluded.

    This is what lets a client render a live timeline — "Bearing wear, 0.82
    confidence", "3 parts required, 1 out of stock" — without unpacking six
    different typed payloads. It is computed from the module's own structured
    output; no model is asked to describe anything.

    A module that did not produce a usable result summarises as its reason,
    because "why there is nothing" is the only useful thing left to say.
    """
    module = ModuleName(module)
    status = ModuleStatus(status)

    if status in (ModuleStatus.unavailable, ModuleStatus.skipped):
        return reason or f"{module.value} produced no result."

    summariser = _SUMMARISERS.get(module)
    try:
        summary = summariser(data) if summariser else ""
    except Exception:  # a malformed payload must not break the stream
        logger.warning("Could not summarise %s output", module.value, exc_info=True)
        summary = ""

    if not summary:
        summary = f"{module.value} completed."
    if status == ModuleStatus.partial and reason:
        summary = f"{summary} ({reason})"
    return summary


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, max_chunks: int = TEMPLATE_CHUNK_COUNT) -> list[str]:
    """Split ``text`` into at most ``max_chunks`` pieces along paragraph breaks.

    Concatenating the result reproduces ``text`` byte for byte — the client's
    accumulated narrative must equal the one in the ``result`` event, and an
    off-by-one newline would break that quietly.
    """
    if not text:
        return []

    paragraphs = text.split("\n\n")
    pieces = [p + "\n\n" for p in paragraphs[:-1]] + [paragraphs[-1]]
    pieces = [p for p in pieces if p]
    if not pieces:
        return [text]
    if len(pieces) <= max_chunks:
        return pieces

    size = -(-len(pieces) // max_chunks)  # ceiling division
    return ["".join(pieces[i : i + size]) for i in range(0, len(pieces), size)]


# ---------------------------------------------------------------------------
# Streaming composer
# ---------------------------------------------------------------------------
async def anthropic_stream_compose(
    prompt: str,
    *,
    system: str,
    model: str = DEFAULT_COMPOSER_MODEL,
    max_tokens: int = 2048,
) -> AsyncIterator[str]:
    """Stream the composed answer from Claude, token chunk by token chunk.

    Uses the async client so the event loop keeps draining the queue while the
    model writes; the synchronous client would need a worker thread and would
    deliver the whole answer at once, which is the thing this file exists to
    avoid.
    """
    api_key = get_settings().anthropic_api_key
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    import anthropic  # optional dependency

    client = anthropic.AsyncAnthropic(api_key=api_key)
    async with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------
class ChatService:
    """Wraps the orchestrator and renders one request as a live event stream."""

    def __init__(
        self,
        orchestrator: Optional[Orchestrator] = None,
        *,
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
        stream_llm: Optional[NarrativeStream] = None,
        model: str = DEFAULT_COMPOSER_MODEL,
    ) -> None:
        self._orchestrator = orchestrator
        self._heartbeat_seconds = heartbeat_seconds
        self._stream_llm = stream_llm
        self._model = model

    @property
    def orchestrator(self) -> Orchestrator:
        """The orchestrator to use — injected, or the process-wide one."""
        return self._orchestrator or get_orchestrator()

    # -- non-streaming -----------------------------------------------------
    async def handle(
        self,
        *,
        tenant_id: str,
        message: str,
        user_role: UserRole = UserRole.technician,
        session_id: Optional[str] = None,
        machine_id: Optional[str] = None,
        language: Optional[str] = None,
    ) -> OrchestrationResult:
        """The same request, answered in one piece. For clients without SSE."""
        return await self.orchestrator.handle(
            tenant_id=tenant_id,
            message=message,
            user_role=UserRole(user_role),
            session_id=session_id,
            machine_id=machine_id,
            language=language,
        )

    # -- streaming ---------------------------------------------------------
    async def stream(
        self,
        *,
        tenant_id: str,
        message: str,
        user_role: UserRole = UserRole.technician,
        session_id: Optional[str] = None,
        machine_id: Optional[str] = None,
        language: Optional[str] = None,
        is_disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
    ) -> AsyncIterator[SSEMessage]:
        """Yield SSE frames for one orchestrated request, as it happens."""
        started = time.monotonic()
        session_id = session_id or uuid.uuid4().hex
        request_id = uuid.uuid4().hex

        # Detected here rather than left to the orchestrator: the streaming
        # composer needs the target language when it builds its system prompt,
        # which happens before ``handle`` would have worked it out. Passing the
        # answer down as ``language`` keeps one detection per request.
        detection = detect_language(message, declared=language)

        queue: asyncio.Queue[Any] = asyncio.Queue()
        levels: dict[ModuleName, int] = {}
        streamed: list[str] = []

        # -- hooks: fired from inside the orchestration task ---------------
        def on_routing(decision: RoutingDecision) -> None:
            # Levels come from the selection, so a client learns which modules
            # are concurrent before any of them starts.
            levels.update(_level_index(decision.selected_modules))
            queue.put_nowait(StreamEvent.for_routing(decision))

        def on_resolution(resolution: ResolutionResult) -> None:
            queue.put_nowait(StreamEvent.for_resolution(resolution))

        def on_progress(event: ProgressEvent) -> None:
            for frame in _progress_frames(event, levels):
                queue.put_nowait(frame)

        def on_aggregated(result: OrchestrationResult) -> None:
            for citation in result.citations:
                queue.put_nowait(StreamEvent.for_citation(citation))
            for conflict in result.conflicts_surfaced:
                queue.put_nowait(StreamEvent.for_conflict(conflict))

        hooks = OrchestrationHooks(
            on_routing=on_routing,
            on_resolution=on_resolution,
            on_progress=on_progress,
            on_aggregated=on_aggregated,
        )

        # -- the composer, wired to publish deltas as it writes ------------
        stream_llm = self._stream_llm
        if stream_llm is None and get_settings().anthropic_api_key:
            stream_llm = anthropic_stream_compose

        composer = None
        if stream_llm is not None:

            async def composer(prompt: str) -> str:  # noqa: F811 - per request
                system = composer_system(detection.language)
                async for chunk in stream_llm(prompt, system=system, model=self._model):
                    if not chunk:
                        continue
                    streamed.append(chunk)
                    queue.put_nowait(StreamEvent.for_narrative_delta(chunk))
                return "".join(streamed)

        async def run() -> None:
            try:
                result = await self.orchestrator.handle(
                    tenant_id=tenant_id,
                    message=message,
                    user_role=UserRole(user_role),
                    session_id=session_id,
                    machine_id=machine_id,
                    language=detection.language,
                    request_id=request_id,
                    hooks=hooks,
                    compose_llm=composer,
                )
                queue.put_nowait(_Completed(result))
            except asyncio.CancelledError:
                queue.put_nowait(_Cancelled())
                raise
            except Exception as exc:  # pragma: no cover - orchestrator degrades
                logger.exception("Orchestration failed for request %s", request_id)
                queue.put_nowait(_Failed(exc))

        yield StreamEvent.for_session(session_id, request_id)

        task = asyncio.create_task(run())
        final: Optional[OrchestrationResult] = None
        failure: Optional[BaseException] = None

        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(), timeout=self._heartbeat_seconds
                    )
                except asyncio.TimeoutError:
                    if is_disconnected is not None and await is_disconnected():
                        logger.info(
                            "Client disconnected during request %s; stopping",
                            request_id,
                        )
                        return
                    yield HEARTBEAT
                    continue

                if isinstance(item, _Completed):
                    final = item.result
                    break
                if isinstance(item, _Failed):
                    failure = item.error
                    break
                if isinstance(item, _Cancelled):
                    return
                yield item

            if failure is not None:
                yield StreamEvent.for_error(
                    f"{type(failure).__name__}: {failure}", recoverable=False
                )
                yield StreamEvent.for_done(_elapsed_ms(started))
                return

            if final is None:  # unreachable: every other branch returned
                return
            for frame in _narrative_frames(final, streamed):
                yield frame

            yield StreamEvent.for_result(final)
            yield StreamEvent.for_done(final.total_elapsed_ms or _elapsed_ms(started))
        finally:
            await _cancel(task, request_id)


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------
def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _level_index(selected: Any) -> dict[ModuleName, int]:
    """Map each module in the plan to its concurrency level."""
    try:
        levels = execution_levels(expand_selection(selected))
    except Exception:  # a bad selection is the router's problem, not the stream's
        logger.warning("Could not compute execution levels", exc_info=True)
        return {}
    return {module: index for index, level in enumerate(levels) for module in level}


def _progress_frames(
    event: ProgressEvent, levels: dict[ModuleName, int]
) -> list[StreamEvent]:
    """Translate one executor progress event into stream frames.

    Skipped and reused modules get a synthetic ``module_start`` immediately
    before their ``module_finish``. They never "ran", but a timeline that
    shows a row finishing without ever starting is a timeline with a hole in
    it — so every ``module_finish`` on the wire is preceded by its start.
    """
    module = ModuleName(event.module)
    level = levels.get(module, 0)
    kind = ProgressEventType(event.type)

    if kind == ProgressEventType.module_started:
        return [StreamEvent.for_module_start(module, level)]

    status = ModuleStatus(event.status) if event.status else ModuleStatus.ok
    finish = StreamEvent.for_module_finish(
        module,
        status,
        elapsed_ms=event.elapsed_ms,
        reason=event.reason,
        summary=summarize_module(module, status, event.data, event.reason),
    )
    if kind == ProgressEventType.module_finished:
        return [finish]
    return [StreamEvent.for_module_start(module, level), finish]


def _narrative_frames(
    result: OrchestrationResult, streamed: list[str]
) -> list[StreamEvent]:
    """The narrative frames still owed to the client once the answer exists.

    Three cases:

    * A blocking resolution — nothing to narrate beyond the question already
      carried by the ``resolution`` event and the result.
    * The composer streamed and its output survived validation — the client
      already has every token; nothing more is owed.
    * Anything else — the template is the answer, and it goes out as chunks.
      If tokens were already streamed, they came from a composition that
      failed number validation, so the client is told before the replacement
      arrives.
    """
    if OrchestrationStatus(result.status) in (
        OrchestrationStatus.clarification_needed,
        OrchestrationStatus.not_found,
    ):
        return []

    source = NarrativeSource(result.narrative_source)
    if source == NarrativeSource.llm:
        return []

    frames: list[StreamEvent] = []
    if streamed:
        frames.append(
            StreamEvent.for_error(
                "The composed narrative contained figures the structured data "
                "does not support and was discarded. Replacing it with the "
                "deterministic answer.",
                recoverable=True,
            )
        )
    frames.extend(
        StreamEvent.for_narrative_delta(chunk) for chunk in chunk_text(result.narrative)
    )
    return frames


async def _cancel(task: "asyncio.Task[None]", request_id: str) -> None:
    """Stop in-flight module work, and say so.

    Cancelling the orchestration task propagates into the executor's
    ``asyncio.gather``, which cancels each module coroutine. Without this a
    closed browser tab would leave a full RCA + agent fan-out running to
    completion against the database.
    """
    if task.done():
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.exception()
        return

    logger.warning(
        "Chat stream for request %s closed before completion; cancelling "
        "in-flight module tasks",
        request_id,
    )
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


# ---------------------------------------------------------------------------
# Process-wide instance
# ---------------------------------------------------------------------------
_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """The shared chat service, built on first use."""
    global _service
    if _service is None:
        _service = ChatService()
    return _service


def set_chat_service(service: Optional[ChatService]) -> None:
    """Install a different chat service (tests, or a custom wiring)."""
    global _service
    _service = service
