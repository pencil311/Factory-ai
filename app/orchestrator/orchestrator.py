"""The orchestrator itself: route, gate on resolution, execute, aggregate.

The shape of one request:

    route  ->  RESOLUTION GATE  ->  execute the graph  ->  aggregate  ->  compose

The gate is absolute. Until the machine is confirmed, nothing runs: an
ambiguous reference returns the candidates and the question, a missing one
returns not-found, and in both cases exactly zero modules execute. A wrong
machine means the wrong manual, the wrong part, and the wrong lockout
procedure, so there is no "best guess" path through this file.

Everything else degrades rather than fails. A module that errors or times out
becomes UNAVAILABLE, its dependents become SKIPPED naming it, and the request
comes back PARTIAL with prose that says what is missing.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional

from app.agents.inventory_agent import InventoryAgent
from app.agents.maintenance_agent import MaintenanceAgent
from app.agents.production_agent import ProductionAgent
from app.agents.safety_agent import SafetyAgent
from app.config import get_settings
from app.db import get_tenant_scope, normalize_tenant_id
from app.orchestrator.aggregator import aggregate, compose_narrative, render_template, build_brief
from app.orchestrator.executor import (
    ExecutionContext,
    ModuleExecutor,
    ModuleOutcome,
    ModuleOutput,
    ProgressCallback,
    soft_degraded_inputs,
)
from app.orchestrator.graph import expand_selection
from app.orchestrator.router_llm import LLMRouter
from app.schemas.agents import AgentContext, AgentResult, AgentStatus
from app.schemas.machine import COLLECTIONS
from app.schemas.orchestration import (
    Clarification,
    ClarificationCandidate,
    MachineSummary,
    ModuleName,
    ModuleRun,
    ModuleStatus,
    NarrativeSource,
    OrchestrationResult,
    OrchestrationStatus,
    RoutingDecision,
    SessionTurn,
    UserRole,
)
from app.schemas.resolution import ResolutionContext, ResolutionResult, ResolutionStatus
from app.services.language import LanguageDetection, detect_language
from app.services.resolver import resolve_machine

logger = logging.getLogger(__name__)

#: How many turns of history a session keeps. Enough for pronoun resolution and
#: an audit trail; not a transcript store.
MAX_SESSION_TURNS = 20

#: Distinguishes "caller passed no per-request composer" from "caller passed
#: None to mean *no LLM*". The two must not collapse into each other.
_UNSET: Any = object()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------
@dataclass
class OrchestrationHooks:
    """Optional callbacks fired as a request moves through its phases.

    The streaming layer subscribes to these so it can emit events *as work
    happens* rather than reconstructing a timeline from the finished result.
    Each is optional, each may be sync or async, and a listener that raises is
    logged and ignored — a broken observer must never fail a request.

    The phases, in order:

    ``on_routing``      the module selection, before anything runs
    ``on_resolution``   the gate's verdict, before anything runs
    ``on_progress``     every module start/finish (handed to the executor)
    ``on_aggregated``   the structured result, after conflict rules and role
                        scoping have settled but *before* prose is composed —
                        so citations and conflicts reach a client ahead of the
                        narrative that references them
    """

    on_routing: Optional[Callable[[RoutingDecision], Any]] = None
    on_resolution: Optional[Callable[[ResolutionResult], Any]] = None
    on_progress: Optional[ProgressCallback] = None
    on_aggregated: Optional[Callable[[OrchestrationResult], Any]] = None


async def _notify(callback: Optional[Callable[..., Any]], payload: Any) -> None:
    """Deliver one hook, tolerating sync callbacks and broken ones."""
    if callback is None:
        return
    try:
        outcome = callback(payload)
        if asyncio.iscoroutine(outcome):
            await outcome
    except Exception:  # a broken listener must not fail the request
        logger.exception("Orchestration hook raised; continuing")


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------
class ConversationStore:
    """Session history and module-output cache, in Mongo, tenant-scoped.

    Everything goes through :func:`get_tenant_scope`, so one tenant's sessions
    are structurally unreachable from another's — a session id from tenant A
    simply does not exist for tenant B.
    """

    collection = COLLECTIONS.conversations

    def __init__(self, cache_ttl_seconds: float = 120.0) -> None:
        self.cache_ttl_seconds = cache_ttl_seconds

    def _collection(self, tenant_id: str):
        return get_tenant_scope(normalize_tenant_id(tenant_id))[self.collection]

    async def load(self, tenant_id: str, session_id: str) -> Optional[dict[str, Any]]:
        if not session_id:
            return None
        try:
            doc = await self._collection(tenant_id).find_one({"session_id": session_id})
        except Exception:
            logger.exception("Could not load conversation '%s'", session_id)
            return None
        return dict(doc) if doc else None

    async def delete(self, tenant_id: str, session_id: str) -> bool:
        """Drop a conversation. True if one existed for *this* tenant.

        Scoped like every other access, so a session id belonging to another
        tenant is not deletable and not distinguishable from a missing one.
        """
        if not session_id:
            return False
        try:
            outcome = await self._collection(tenant_id).delete_one(
                {"session_id": session_id}
            )
        except Exception:
            logger.exception("Could not delete conversation '%s'", session_id)
            return False
        return bool(getattr(outcome, "deleted_count", 0))

    def fresh_cache(
        self, session: Optional[Mapping[str, Any]], machine_id: str
    ) -> tuple[dict[ModuleName, ModuleOutcome], Optional[float]]:
        """Reusable module outcomes for ``machine_id``, and the cache's age.

        A cache entry is reusable only when it is for the same machine and
        younger than the TTL, and only for modules that actually produced
        something. Anything else re-runs.
        """
        if not session:
            return {}, None
        cache = session.get("cache") or {}
        if not cache or cache.get("machine_id") != machine_id:
            return {}, None

        cached_at = cache.get("cached_at")
        if isinstance(cached_at, str):
            try:
                cached_at = datetime.fromisoformat(cached_at)
            except ValueError:
                return {}, None
        if not isinstance(cached_at, datetime):
            return {}, None
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)

        age = (_utcnow() - cached_at).total_seconds()
        if age > self.cache_ttl_seconds:
            return {}, age

        outcomes: dict[ModuleName, ModuleOutcome] = {}
        for name, payload in (cache.get("modules") or {}).items():
            try:
                module = ModuleName(name)
                status = ModuleStatus(payload.get("status", "OK"))
            except ValueError:
                continue
            if status not in (ModuleStatus.ok, ModuleStatus.partial):
                continue
            outcomes[module] = ModuleOutcome(
                name=module,
                status=status,
                data=payload.get("data"),
                reason=payload.get("reason"),
                error_detail=payload.get("error_detail"),
                elapsed_ms=int(payload.get("elapsed_ms", 0)),
                citations=list(payload.get("citations") or []),
                degraded_inputs=list(payload.get("degraded_inputs") or []),
                reused=True,
            )
        return outcomes, age

    async def record_turn(
        self,
        tenant_id: str,
        session_id: str,
        *,
        turn: SessionTurn,
        last_machine_id: Optional[str],
        outcomes: Optional[Mapping[ModuleName, ModuleOutcome]] = None,
        machine_id: Optional[str] = None,
    ) -> None:
        """Append a turn and refresh the module cache. Never fatal."""
        if not session_id:
            return

        existing = await self.load(tenant_id, session_id) or {}
        turns = list(existing.get("turns") or [])
        turns.append(turn.model_dump())
        turns = turns[-MAX_SESSION_TURNS:]

        document: dict[str, Any] = {
            "session_id": session_id,
            "created_at": existing.get("created_at") or _utcnow(),
            "updated_at": _utcnow(),
            "last_machine_id": last_machine_id or existing.get("last_machine_id"),
            "turns": turns,
            "cache": existing.get("cache"),
        }

        if outcomes and machine_id:
            modules: dict[str, Any] = {}
            for name, outcome in outcomes.items():
                if ModuleStatus(outcome.status) not in (
                    ModuleStatus.ok,
                    ModuleStatus.partial,
                ):
                    continue
                modules[ModuleName(name).value] = {
                    "status": ModuleStatus(outcome.status).value,
                    "data": outcome.data,
                    "reason": outcome.reason,
                    "error_detail": outcome.error_detail,
                    "elapsed_ms": outcome.elapsed_ms,
                    "citations": outcome.citations,
                    "degraded_inputs": outcome.degraded_inputs,
                }
            document["cache"] = {
                "machine_id": machine_id,
                "cached_at": _utcnow(),
                "modules": modules,
            }

        try:
            await self._collection(tenant_id).replace_one(
                {"session_id": session_id}, document, upsert=True
            )
        except Exception:
            logger.exception("Could not persist conversation '%s'", session_id)


# ---------------------------------------------------------------------------
# Module runners
# ---------------------------------------------------------------------------
async def _run_rag(context: ExecutionContext) -> ModuleOutput:
    from app.rag.retriever import retrieve

    query = context.message.strip() or context.machine_id
    result = await retrieve(
        context.tenant_id,
        query=query,
        machine_id=context.machine_id,
        machine_model=context.machine.get("model"),
    )
    chunks = [dataclasses.asdict(chunk) for chunk in result.chunks]
    if not chunks:
        # An empty knowledge base is a thin answer, not a broken module: RCA
        # and Maintenance can both work without documents.
        return ModuleOutput(
            status=ModuleStatus.partial,
            data={"chunks": []},
            reason=result.reason or "No passages matched this query.",
        )

    citations = [
        {
            "document_id": chunk.document_id,
            "title": chunk.document_title,
            "page_number": chunk.page_number,
            "section_title": chunk.section_title,
        }
        for chunk in result.chunks
    ]
    return ModuleOutput(
        status=ModuleStatus.ok, data={"chunks": chunks}, citations=citations
    )


async def _run_pdm(context: ExecutionContext) -> ModuleOutput:
    from app.services.pdm import (
        InsufficientDataError,
        PdmArtifactsMissingError,
        get_pdm_service,
    )

    try:
        service = get_pdm_service()
        prediction = await service.predict(context.tenant_id, context.machine_id)
    except (PdmArtifactsMissingError, InsufficientDataError) as exc:
        # The model has nothing to say — no trained artifacts, or no readings.
        # That is a known, explainable gap rather than a module failure, so
        # downstream analysis proceeds without a prediction.
        return ModuleOutput(status=ModuleStatus.partial, data=None, reason=str(exc))

    return ModuleOutput(status=ModuleStatus.ok, data=prediction.model_dump())


async def _run_rca(context: ExecutionContext) -> ModuleOutput:
    from app.services.rca import get_rca_service

    service = get_rca_service()
    result = await service.analyze(
        tenant_id=context.tenant_id,
        machine_id=context.machine_id,
        pdm_result=context.data(ModuleName.pdm),
        include_narrative=False,
    )
    data = result.model_dump()
    if result.insufficient_data:
        return ModuleOutput(
            status=ModuleStatus.partial,
            data=data,
            reason=(
                "Evidence was insufficient for a firm conclusion; findings are "
                "provisional."
            ),
        )
    return ModuleOutput(status=ModuleStatus.ok, data=data)


def _retrieved_chunks(context: ExecutionContext) -> list[dict[str, Any]]:
    rag = context.data(ModuleName.rag) or {}
    return list(rag.get("chunks") or [])


def _agent_context(context: ExecutionContext) -> AgentContext:
    return AgentContext(
        tenant_id=context.tenant_id,
        machine_id=context.machine_id,
        rca_result=context.data(ModuleName.rca),
        pdm_result=context.data(ModuleName.pdm),
        retrieved_chunks=_retrieved_chunks(context),
        user_role=context.user_role,
        raw_query=context.message,
        maintenance_plan=context.data(ModuleName.maintenance),
        inventory_status=context.data(ModuleName.inventory),
    )


def _from_agent(result: AgentResult) -> ModuleOutput:
    """Translate an agent's own status vocabulary into the executor's."""
    status = AgentStatus(result.status)
    mapping = {
        AgentStatus.ok: ModuleStatus.ok,
        AgentStatus.partial: ModuleStatus.partial,
        AgentStatus.unavailable: ModuleStatus.unavailable,
    }
    return ModuleOutput(
        status=mapping[status],
        data=result.data,
        reason=result.reason,
        error_detail=result.error_detail,
        citations=list(result.citations or []),
    )


async def _agent_result(agent, module: ModuleName, context: ExecutionContext) -> AgentResult:
    """Run one agent and record which of its SOFT inputs were missing.

    The agent itself has no notion of the module graph — that knowledge lives
    in app.orchestrator.graph — so it is attached here rather than asking each
    agent to work it out independently.
    """
    result = await agent.run(_agent_context(context))
    result.degraded_inputs = soft_degraded_inputs(module, context)
    return result


async def _run_maintenance(context: ExecutionContext) -> ModuleOutput:
    return _from_agent(
        await _agent_result(MaintenanceAgent(), ModuleName.maintenance, context)
    )


async def _run_inventory(context: ExecutionContext) -> ModuleOutput:
    return _from_agent(
        await _agent_result(InventoryAgent(), ModuleName.inventory, context)
    )


async def _run_safety(context: ExecutionContext) -> ModuleOutput:
    return _from_agent(await _agent_result(SafetyAgent(), ModuleName.safety, context))


async def _run_production(context: ExecutionContext) -> ModuleOutput:
    return _from_agent(
        await _agent_result(ProductionAgent(), ModuleName.production, context)
    )


def default_module_runners() -> dict[ModuleName, Any]:
    """The production wiring: every module called in-process, never over HTTP."""
    return {
        ModuleName.rag: _run_rag,
        ModuleName.pdm: _run_pdm,
        ModuleName.rca: _run_rca,
        ModuleName.maintenance: _run_maintenance,
        ModuleName.inventory: _run_inventory,
        ModuleName.safety: _run_safety,
        ModuleName.production: _run_production,
    }


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------
class Orchestrator:
    """Runs one request end to end."""

    def __init__(
        self,
        *,
        router: Optional[LLMRouter] = None,
        module_runners: Optional[Mapping[ModuleName, Any]] = None,
        session_store: Optional[ConversationStore] = None,
        module_timeout_seconds: Optional[float] = None,
        cache_ttl_seconds: Optional[float] = None,
        resolver=resolve_machine,
        compose=compose_narrative,
        compose_llm=None,
    ) -> None:
        settings = get_settings()
        self.router = router or LLMRouter(
            timeout_seconds=settings.orchestrator_router_timeout_seconds
        )
        self.runners = dict(module_runners or default_module_runners())
        self.module_timeout_seconds = (
            settings.orchestrator_module_timeout_seconds
            if module_timeout_seconds is None
            else module_timeout_seconds
        )
        ttl = (
            settings.orchestrator_cache_ttl_seconds
            if cache_ttl_seconds is None
            else cache_ttl_seconds
        )
        self.sessions = session_store or ConversationStore(cache_ttl_seconds=ttl)
        self.sessions.cache_ttl_seconds = ttl
        self._resolve = resolver
        self._compose = compose
        self._compose_llm = compose_llm

    # -- resolution gate ---------------------------------------------------
    async def _resolve_machine(
        self,
        *,
        tenant_id: str,
        message: str,
        machine_id: Optional[str],
        routing: RoutingDecision,
        last_machine_id: Optional[str],
    ) -> ResolutionResult:
        """Confirm exactly one machine, or return a blocking result.

        The operator's own words come first: the resolver is built to take raw
        input and already handles ids, aliases and error codes in one pass.
        Only if that finds nothing do we fall back to the reference the router
        *believed* it saw, and then to the machine from the previous turn — so
        "is it safe to run it?" resolves the way the operator means it. The
        order matters: a model's paraphrase must never outrank what was
        actually typed, or a hallucinated designator could redirect the answer
        to the wrong machine.
        """
        context = ResolutionContext(last_machine_id=last_machine_id)

        if machine_id:
            attempts = [machine_id]
        else:
            attempts = [message]
            if routing.machine_reference and routing.machine_reference != message:
                attempts.append(routing.machine_reference)
            if last_machine_id:
                attempts.append(last_machine_id)

        blocking: Optional[ResolutionResult] = None
        for text in attempts:
            if not (text or "").strip():
                continue
            result = await self._resolve(
                text=text, tenant_id=tenant_id, context=context
            )
            if ResolutionStatus(result.status) == ResolutionStatus.resolved:
                return result
            if (
                ResolutionStatus(result.status) == ResolutionStatus.ambiguous
                and blocking is None
            ):
                # Ambiguity is an answer in its own right: stop and ask rather
                # than trying a looser reference that might pick the wrong one.
                return result
            blocking = blocking or result

        return blocking or ResolutionResult(
            status=ResolutionStatus.not_found, raw_input=message
        )

    # -- request -----------------------------------------------------------
    async def handle(
        self,
        *,
        tenant_id: str,
        message: str,
        user_role: UserRole = UserRole.technician,
        session_id: Optional[str] = None,
        machine_id: Optional[str] = None,
        language: Optional[str] = None,
        request_id: Optional[str] = None,
        progress: Optional[ProgressCallback] = None,
        hooks: Optional[OrchestrationHooks] = None,
        compose_llm: Any = _UNSET,
    ) -> OrchestrationResult:
        """Run one request end to end.

        ``request_id`` may be supplied by a caller that has already told a
        client which turn this is — the streaming layer announces the id in
        its first event, before any work starts, so it cannot let the
        orchestrator mint a different one afterwards.

        ``compose_llm`` overrides the instance-level composer for this request
        only. The streaming layer uses it to install a token-by-token composer
        while leaving the shared orchestrator untouched.
        """
        tenant_id = normalize_tenant_id(tenant_id)
        started = time.monotonic()
        request_id = request_id or uuid.uuid4().hex
        hooks = hooks or OrchestrationHooks()
        progress = hooks.on_progress or progress
        composer = self._compose_llm if compose_llm is _UNSET else compose_llm

        detection = detect_language(message, declared=language)
        logger.debug("Language for request %s: %s", request_id, detection.reason)

        session = await self.sessions.load(tenant_id, session_id) if session_id else None
        last_machine_id = (session or {}).get("last_machine_id")

        routing = await self.router.route(
            message, user_role=str(UserRole(user_role).value), last_machine_id=last_machine_id
        )
        await _notify(hooks.on_routing, routing)

        result = OrchestrationResult(
            request_id=request_id,
            tenant_id=tenant_id,
            user_role=UserRole(user_role),
            intent=routing.intent,
            urgency=routing.urgency,
            status=OrchestrationStatus.complete,
            routing_decision=routing,
            session_id=session_id,
            detected_language=detection.language,
        )

        # --- THE GATE -----------------------------------------------------
        resolve_started = time.monotonic()
        resolution = await self._resolve_machine(
            tenant_id=tenant_id,
            message=message,
            machine_id=machine_id,
            routing=routing,
            last_machine_id=last_machine_id,
        )
        resolve_ms = int((time.monotonic() - resolve_started) * 1000)
        status = ResolutionStatus(resolution.status)

        # Emitted before the branch: a blocked request is still an answer, and
        # a client needs the candidates and the question either way.
        await _notify(hooks.on_resolution, resolution)

        if status != ResolutionStatus.resolved or resolution.machine is None:
            return await self._blocked(
                result,
                resolution=resolution,
                elapsed_ms=resolve_ms,
                started=started,
                message=message,
                session_id=session_id,
                detection=detection,
            )

        candidate = resolution.machine
        result.machine = MachineSummary(
            machine_id=candidate.machine_id,
            name=candidate.name,
            model=candidate.model,
            line_id=candidate.line_id,
            status=str(candidate.status),
        )
        result.modules_run = [
            ModuleRun(
                name=ModuleName.resolver,
                status=ModuleStatus.ok,
                elapsed_ms=resolve_ms,
                reason=f"Resolved by {candidate.matched_by} on '{candidate.matched_value}'.",
            )
        ]

        # --- execution ----------------------------------------------------
        selected = expand_selection(routing.selected_modules)
        cached, cache_age = self.sessions.fresh_cache(session, candidate.machine_id)
        reusable = {
            module: outcome
            for module, outcome in cached.items()
            if module in selected and module != ModuleName.resolver
        }
        if reusable:
            logger.info(
                "Reusing %d cached module result(s) for %s (age %.0fs): %s",
                len(reusable),
                candidate.machine_id,
                cache_age or 0.0,
                ", ".join(sorted(ModuleName(m).value for m in reusable)),
            )

        context = ExecutionContext(
            tenant_id=tenant_id,
            machine_id=candidate.machine_id,
            message=message,
            user_role=str(UserRole(user_role).value),
            session_id=session_id,
            machine={
                "machine_id": candidate.machine_id,
                "name": candidate.name,
                "model": candidate.model,
                "line_id": candidate.line_id,
            },
            outcomes={
                ModuleName.resolver: ModuleOutcome(
                    name=ModuleName.resolver,
                    status=ModuleStatus.ok,
                    data=result.machine.model_dump(),
                    elapsed_ms=resolve_ms,
                )
            },
        )

        executor = ModuleExecutor(
            self.runners, timeout_seconds=self.module_timeout_seconds
        )
        report = await executor.run(
            selected,
            context,
            prior_outcomes=reusable,
            progress=progress,
            skip=[ModuleName.resolver],
        )

        # --- aggregation + one composition --------------------------------
        aggregate(result, report.outcomes, selected)
        await _notify(hooks.on_aggregated, result)

        narrative, source = await self._compose(
            result, call_llm=composer, language=detection.language
        )
        result.narrative = narrative
        result.narrative_source = source
        self._mark_language(result, detection, source)
        result.total_elapsed_ms = int((time.monotonic() - started) * 1000)

        await self._save_turn(
            result,
            message=message,
            session_id=session_id,
            outcomes=report.outcomes,
            machine_id=candidate.machine_id,
        )
        return result

    # -- language ----------------------------------------------------------
    @staticmethod
    def _mark_language(
        result: OrchestrationResult,
        detection: LanguageDetection,
        source: NarrativeSource,
    ) -> None:
        """Record the answer's language, and whether it is the one asked for.

        The deterministic template is English and is never machine-translated:
        a mistranslated lockout step or torque figure is a safety problem, not
        a cosmetic one. So when the template answers a non-English operator we
        flag it rather than pretending the language matched.
        """
        result.detected_language = detection.language
        result.language_fallback = (
            NarrativeSource(source) == NarrativeSource.template
            and not detection.is_english
        )

    # -- blocking paths ----------------------------------------------------
    async def _blocked(
        self,
        result: OrchestrationResult,
        *,
        resolution: ResolutionResult,
        elapsed_ms: int,
        started: float,
        message: str,
        session_id: Optional[str],
        detection: Optional[LanguageDetection] = None,
    ) -> OrchestrationResult:
        """Return a clarification or not-found answer, having run nothing."""
        status = ResolutionStatus(resolution.status)

        if status == ResolutionStatus.ambiguous:
            result.status = OrchestrationStatus.clarification_needed
            result.clarification = Clarification(
                question=(
                    resolution.clarification_question
                    or "Which machine do you mean?"
                ),
                candidates=[
                    ClarificationCandidate(
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
            )
            reason = "Ambiguous machine reference — asked the operator to confirm."
        else:
            result.status = OrchestrationStatus.not_found
            reason = (
                f"No machine in this tenant matches '{message}'. "
                "Nothing was run against an unconfirmed machine."
            )

        result.modules_run = [
            ModuleRun(
                name=ModuleName.resolver,
                status=ModuleStatus.ok,
                elapsed_ms=elapsed_ms,
                reason=reason,
            )
        ]

        # Composed from the template only: there is no analysis to narrate, and
        # a model has nothing to add to "which machine did you mean?".
        brief = build_brief(result)
        if result.clarification is None:
            result.narrative = (
                f"{reason} Give the machine id (for example CV-201) or its "
                f"full name and I will take it from there."
            )
        else:
            result.narrative = render_template(brief)
        result.narrative_source = NarrativeSource.template
        if detection is not None:
            self._mark_language(result, detection, NarrativeSource.template)
        result.total_elapsed_ms = int((time.monotonic() - started) * 1000)

        await self._save_turn(
            result, message=message, session_id=session_id, outcomes=None, machine_id=None
        )
        return result

    async def _save_turn(
        self,
        result: OrchestrationResult,
        *,
        message: str,
        session_id: Optional[str],
        outcomes: Optional[Mapping[ModuleName, ModuleOutcome]],
        machine_id: Optional[str],
    ) -> None:
        if not session_id:
            return
        turn = SessionTurn(
            request_id=result.request_id,
            message=message,
            user_role=UserRole(result.user_role),
            machine_id=machine_id,
            status=OrchestrationStatus(result.status),
            narrative=result.narrative,
            modules_run=[
                f"{ModuleName(row.name).value}:{ModuleStatus(row.status).value}"
                for row in result.modules_run
            ],
        )
        await self.sessions.record_turn(
            result.tenant_id,
            session_id,
            turn=turn,
            last_machine_id=machine_id,
            outcomes=outcomes,
            machine_id=machine_id,
        )


# ---------------------------------------------------------------------------
# Process-wide instance
# ---------------------------------------------------------------------------
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """The shared orchestrator, built on first use."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def set_orchestrator(orchestrator: Optional[Orchestrator]) -> None:
    """Install a different orchestrator (tests, or a custom wiring)."""
    global _orchestrator
    _orchestrator = orchestrator
