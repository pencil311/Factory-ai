"""Level-by-level execution of the module graph.

Everything in one level runs concurrently; levels run in order. A module that
fails, raises, or times out does not fail the request — it yields
``UNAVAILABLE`` with a reason, and its dependents are marked ``SKIPPED``
naming the dependency that was missing. The caller then aggregates whatever
exists.

The executor knows nothing about what the modules do. It is handed a mapping of
module name to an async runner, so the same machinery drives the real services
and a test's stubs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence

from app.core.errors import describe_failure
from app.orchestrator.graph import (
    DEPENDENCIES,
    DependencyKind,
    dependency_kind,
    execution_levels,
    sort_modules,
)
from app.schemas.orchestration import (
    ModuleName,
    ModuleStatus,
    ProgressEvent,
    ProgressEventType,
)

logger = logging.getLogger(__name__)

DEFAULT_MODULE_TIMEOUT_SECONDS = 15.0

#: Statuses that mean a dependent cannot proceed on this module's output.
_UNUSABLE = frozenset({ModuleStatus.unavailable, ModuleStatus.skipped})

#: Operator-facing name for each module, used to build a safe failure reason
#: that never quotes the raised exception's own message (see app.core.errors).
_MODULE_LABELS: dict[ModuleName, str] = {
    ModuleName.resolver: "Machine resolution",
    ModuleName.rag: "Knowledge base query",
    ModuleName.pdm: "Predictive model",
    ModuleName.rca: "Root cause analysis",
    ModuleName.maintenance: "Maintenance procedure lookup",
    ModuleName.inventory: "Inventory check",
    ModuleName.safety: "Safety briefing",
    ModuleName.production: "Production impact estimate",
}


@dataclass
class ModuleOutput:
    """What a runner returns: a status, its data, and a reason if degraded."""

    status: ModuleStatus
    data: Any = None
    reason: Optional[str] = None
    #: Full exception text when ``reason`` was derived from one. Never shown
    #: to an operator — see app.core.errors.describe_failure.
    error_detail: Optional[str] = None
    citations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ModuleOutcome:
    """What actually happened when the executor ran (or did not run) a module."""

    name: ModuleName
    status: ModuleStatus
    data: Any = None
    reason: Optional[str] = None
    #: Full exception text when ``reason`` was derived from one. Logged and
    #: kept for debugging; never composed into the narrative.
    error_detail: Optional[str] = None
    elapsed_ms: int = 0
    citations: list[dict[str, Any]] = field(default_factory=list)
    #: Names of this module's SOFT dependencies that produced nothing usable —
    #: it ran anyway, on less than the full picture. Never populated for a
    #: SKIPPED module: that already means it did not run at all.
    degraded_inputs: list[str] = field(default_factory=list)
    #: Monotonic clock readings. Kept so callers (and tests) can prove which
    #: modules genuinely overlapped rather than inferring it from durations.
    started_at: float = 0.0
    finished_at: float = 0.0
    reused: bool = False

    @property
    def usable(self) -> bool:
        """True when a dependent may build on this outcome."""
        return ModuleStatus(self.status) not in _UNUSABLE


@dataclass
class ExecutionContext:
    """Everything a runner is given. Runners read; only the executor writes."""

    tenant_id: str
    machine_id: str
    message: str = ""
    user_role: Optional[str] = None
    session_id: Optional[str] = None
    machine: dict[str, Any] = field(default_factory=dict)
    outcomes: dict[ModuleName, ModuleOutcome] = field(default_factory=dict)

    def outcome(self, module: ModuleName) -> Optional[ModuleOutcome]:
        return self.outcomes.get(ModuleName(module))

    def data(self, module: ModuleName) -> Any:
        """Data from an upstream module, or ``None`` if it did not produce any."""
        outcome = self.outcome(module)
        if outcome is None or not outcome.usable:
            return None
        return outcome.data


@dataclass
class ExecutionReport:
    outcomes: dict[ModuleName, ModuleOutcome]
    events: list[ProgressEvent] = field(default_factory=list)
    levels: list[list[ModuleName]] = field(default_factory=list)
    total_elapsed_ms: int = 0


Runner = Callable[[ExecutionContext], Awaitable[ModuleOutput]]
#: Progress callbacks may be sync or async; both are supported.
ProgressCallback = Callable[[ProgressEvent], Any]


def soft_degraded_inputs(
    module: ModuleName,
    context: ExecutionContext,
    dependencies: Mapping[ModuleName, Sequence[ModuleName]] = DEPENDENCIES,
) -> list[str]:
    """Names of ``module``'s SOFT dependencies that produced nothing usable.

    Shared by the executor, which records this on ``ModuleOutcome``, and the
    orchestrator's per-module runners, which record the same fact on an
    agent's own result — both derive it from the same graph rather than
    tracking it independently.
    """
    degraded: list[str] = []
    for dep in dependencies.get(ModuleName(module), ()):
        if dependency_kind(module, dep) != DependencyKind.soft:
            continue
        outcome = context.outcomes.get(ModuleName(dep))
        if outcome is None or not outcome.usable:
            degraded.append(ModuleName(dep).value)
    return degraded


class ModuleExecutor:
    """Runs a selection of modules in dependency order."""

    def __init__(
        self,
        runners: Mapping[ModuleName, Runner],
        *,
        timeout_seconds: float = DEFAULT_MODULE_TIMEOUT_SECONDS,
        dependencies: Mapping[ModuleName, Sequence[ModuleName]] = DEPENDENCIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._runners = {ModuleName(k): v for k, v in runners.items()}
        self._timeout = timeout_seconds
        self._dependencies = dependencies
        self._clock = clock

    async def run(
        self,
        modules: Iterable[ModuleName],
        context: ExecutionContext,
        *,
        prior_outcomes: Optional[Mapping[ModuleName, ModuleOutcome]] = None,
        progress: Optional[ProgressCallback] = None,
        skip: Iterable[ModuleName] = (),
    ) -> ExecutionReport:
        """Execute ``modules``, reusing ``prior_outcomes`` where supplied.

        ``skip`` names modules the caller has already handled itself (the
        resolver, which runs ahead of everything as a gate). They are excluded
        from the plan but their outcomes may still be seeded into ``context``.
        """
        levels = execution_levels(modules, self._dependencies)
        excluded = {ModuleName(m) for m in skip}
        reused = {ModuleName(k): v for k, v in (prior_outcomes or {}).items()}

        events: list[ProgressEvent] = []
        started = self._clock()

        for level in levels:
            runnable = [m for m in level if m not in excluded]
            if not runnable:
                continue

            fresh: list[ModuleName] = []
            for module in runnable:
                cached = reused.get(module)
                if cached is not None:
                    outcome = ModuleOutcome(
                        name=module,
                        status=ModuleStatus(cached.status),
                        data=cached.data,
                        reason=cached.reason,
                        error_detail=cached.error_detail,
                        elapsed_ms=cached.elapsed_ms,
                        citations=list(cached.citations),
                        degraded_inputs=list(cached.degraded_inputs),
                        started_at=self._clock(),
                        finished_at=self._clock(),
                        reused=True,
                    )
                    context.outcomes[module] = outcome
                    events.append(
                        ProgressEvent(
                            type=ProgressEventType.module_reused,
                            module=module,
                            status=outcome.status,
                            elapsed_ms=outcome.elapsed_ms,
                            reason=outcome.reason,
                            data=outcome.data,
                        )
                    )
                    await _emit(progress, events[-1])
                    continue
                fresh.append(module)

            # Dependency gate: anything whose hard dependency is missing is
            # skipped before we spend a task on it.
            to_run: list[ModuleName] = []
            for module in fresh:
                missing = self._missing_dependency(module, context)
                if missing is None:
                    to_run.append(module)
                    continue
                outcome = ModuleOutcome(
                    name=module,
                    status=ModuleStatus.skipped,
                    reason=missing,
                    started_at=self._clock(),
                    finished_at=self._clock(),
                )
                context.outcomes[module] = outcome
                events.append(
                    ProgressEvent(
                        type=ProgressEventType.module_skipped,
                        module=module,
                        status=ModuleStatus.skipped,
                        reason=missing,
                    )
                )
                await _emit(progress, events[-1])

            if not to_run:
                continue

            for module in to_run:
                event = ProgressEvent(
                    type=ProgressEventType.module_started, module=module
                )
                events.append(event)
                await _emit(progress, event)

            results = await asyncio.gather(
                *(self._run_one(module, context) for module in to_run),
                return_exceptions=False,
            )

            for outcome in results:
                outcome.degraded_inputs = soft_degraded_inputs(
                    outcome.name, context, self._dependencies
                )
                context.outcomes[outcome.name] = outcome
                event = ProgressEvent(
                    type=ProgressEventType.module_finished,
                    module=outcome.name,
                    status=ModuleStatus(outcome.status),
                    elapsed_ms=outcome.elapsed_ms,
                    reason=outcome.reason,
                    data=outcome.data,
                )
                events.append(event)
                await _emit(progress, event)

        total_ms = int((self._clock() - started) * 1000)
        return ExecutionReport(
            outcomes=dict(context.outcomes),
            events=events,
            levels=[sort_modules(level) for level in levels],
            total_elapsed_ms=total_ms,
        )

    # -- internals ---------------------------------------------------------
    def _missing_dependency(
        self, module: ModuleName, context: ExecutionContext
    ) -> Optional[str]:
        """Reason string naming the first unusable HARD dependency, or None.

        SOFT dependencies never block: a module missing only a SOFT input
        still runs (see ``run()``), so they are skipped here entirely.
        """
        for dep in self._dependencies.get(module, ()):  # declared order
            if dependency_kind(module, dep) != DependencyKind.hard:
                continue
            outcome = context.outcomes.get(ModuleName(dep))
            if outcome is None:
                return (
                    f"Skipped: depends on {ModuleName(dep).value}, which did not run."
                )
            if not outcome.usable:
                status = ModuleStatus(outcome.status)
                # A SKIPPED dependency's own reason may itself name a further
                # dependency several hops back; quoting it verbatim here would
                # nest the whole chain into one sentence. Naming only the
                # immediate dependency and a short, non-recursive detail keeps
                # every skip message one hop deep, however long the chain.
                detail = (
                    "an upstream dependency was unavailable"
                    if status == ModuleStatus.skipped
                    else (outcome.reason or "no reason recorded")
                )
                return (
                    f"Skipped: depends on {ModuleName(dep).value}, which was "
                    f"{status.value} ({detail})."
                )
        return None

    async def _run_one(
        self, module: ModuleName, context: ExecutionContext
    ) -> ModuleOutcome:
        runner = self._runners.get(module)
        start = self._clock()

        if runner is None:
            return ModuleOutcome(
                name=module,
                status=ModuleStatus.unavailable,
                reason=f"No runner registered for module {module.value}.",
                started_at=start,
                finished_at=self._clock(),
            )

        try:
            output = await asyncio.wait_for(runner(context), timeout=self._timeout)
        except asyncio.TimeoutError:
            finished = self._clock()
            logger.warning("Module %s timed out after %ss", module.value, self._timeout)
            return ModuleOutcome(
                name=module,
                status=ModuleStatus.unavailable,
                reason=f"Timed out after {self._timeout:g}s.",
                elapsed_ms=int((finished - start) * 1000),
                started_at=start,
                finished_at=finished,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            finished = self._clock()
            logger.warning(
                "Module %s failed: %s: %s", module.value, type(exc).__name__, exc
            )
            # The exception's own message is exactly where server/driver
            # internals (cluster timestamps, connection strings, signature
            # bytes) leak into a narrative. reason stays generic and safe to
            # display; error_detail keeps the full text for logs and debugging.
            reason, detail = describe_failure(
                exc, label=_MODULE_LABELS.get(module, module.value.title())
            )
            return ModuleOutcome(
                name=module,
                status=ModuleStatus.unavailable,
                reason=reason,
                error_detail=detail,
                elapsed_ms=int((finished - start) * 1000),
                started_at=start,
                finished_at=finished,
            )

        finished = self._clock()
        if not isinstance(output, ModuleOutput):
            return ModuleOutcome(
                name=module,
                status=ModuleStatus.unavailable,
                reason=(
                    f"Runner for {module.value} returned "
                    f"{type(output).__name__}, expected ModuleOutput."
                ),
                elapsed_ms=int((finished - start) * 1000),
                started_at=start,
                finished_at=finished,
            )

        return ModuleOutcome(
            name=module,
            status=ModuleStatus(output.status),
            data=output.data,
            reason=output.reason,
            error_detail=output.error_detail,
            elapsed_ms=int((finished - start) * 1000),
            citations=list(output.citations),
            started_at=start,
            finished_at=finished,
        )


async def _emit(callback: Optional[ProgressCallback], event: ProgressEvent) -> None:
    """Deliver a progress event, tolerating sync callbacks and bad ones."""
    if callback is None:
        return
    try:
        result = callback(event)
        if asyncio.iscoroutine(result):
            await result
    except Exception:  # a broken listener must not fail the request
        logger.exception("Progress callback raised for %s", event.module)
