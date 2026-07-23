"""The module dependency graph.

Dependencies are declared once, here, and everything else — what runs, in what
order, what may run in parallel, what gets skipped when something upstream
fails — is derived from this table. No module knows about any other module;
the edges live in this file alone.

Two properties matter:

* **Selecting a module selects its dependencies.** Asking only for INVENTORY
  cannot produce a parts list without MAINTENANCE, RCA, RAG, PDM and RESOLVER,
  so :func:`expand_selection` pulls them in rather than letting the request
  fail halfway through.
* **A cycle is a bug, not a runtime condition.** :func:`execution_levels`
  raises rather than dropping edges or picking an arbitrary order, and the
  declared graph is validated at import time so a bad edit fails loudly on the
  first import instead of quietly on the first request.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Mapping, Sequence

from app.schemas.orchestration import ModuleName

#: Module -> the modules whose output it needs before it can run.
#:
#: RESOLVER has no dependencies and is blocking: nothing may run against an
#: unconfirmed machine. SAFETY deliberately hangs off RCA alone so it can start
#: while MAINTENANCE is still working — a safety briefing must never wait on a
#: repair plan. PRODUCTION needs INVENTORY because a downtime estimate that
#: ignores the parts wait is worse than no estimate.
#:
#: This table is topology only — which edges exist, for ordering and level
#: grouping. Whether an edge actually *blocks* its dependent is a separate
#: question, answered by :data:`DEPENDENCY_KINDS` below.
DEPENDENCIES: dict[ModuleName, tuple[ModuleName, ...]] = {
    ModuleName.resolver: (),
    ModuleName.rag: (ModuleName.resolver,),
    ModuleName.pdm: (ModuleName.resolver,),
    ModuleName.rca: (ModuleName.pdm, ModuleName.rag),
    ModuleName.maintenance: (ModuleName.rca, ModuleName.rag),
    ModuleName.inventory: (ModuleName.maintenance,),
    ModuleName.safety: (ModuleName.rca,),
    ModuleName.production: (ModuleName.rca, ModuleName.inventory),
}


class DependencyKind(str, Enum):
    """Whether a dependent can run without this dependency having succeeded.

    ``HARD`` means the dependent has nothing to say without it — it is
    SKIPPED, naming the missing dependency, exactly as before this kind
    existed. ``SOFT`` means the dependent's own analysis stands on its own; it
    still runs, receives ``None`` for that input, and records the gap in
    ``degraded_inputs`` rather than refusing to answer.
    """

    hard = "HARD"
    soft = "SOFT"


#: Kind of each edge declared in :data:`DEPENDENCIES`. An edge with no entry
#: here defaults to HARD — in particular every module's dependency on
#: RESOLVER, which is never listed because nothing may run against an
#: unconfirmed machine in the first place.
#:
#: RAG and PDM only *corroborate* RCA's own sensor-and-fault-signature
#: analysis, so their absence degrades RCA rather than blocking it. RCA itself
#: is a hard dependency of MAINTENANCE (no cause means no procedure to write)
#: but only a soft one of SAFETY (a generic briefing is valid, and required,
#: even with no identified cause) and of PRODUCTION (an impact estimate can
#: still be given without one). INVENTORY is a hard dependency of PRODUCTION's
#: neighbour relation only in the sense that its absence degrades the downtime
#: estimate to repair time alone — it does not block the estimate outright.
DEPENDENCY_KINDS: dict[ModuleName, dict[ModuleName, DependencyKind]] = {
    ModuleName.rca: {
        ModuleName.pdm: DependencyKind.soft,
        ModuleName.rag: DependencyKind.soft,
    },
    ModuleName.maintenance: {
        ModuleName.rca: DependencyKind.hard,  # no cause, no procedure
        ModuleName.rag: DependencyKind.soft,
    },
    ModuleName.inventory: {
        ModuleName.maintenance: DependencyKind.hard,  # needs the parts list
    },
    ModuleName.safety: {
        ModuleName.rca: DependencyKind.soft,  # a generic briefing is valid and required
    },
    ModuleName.production: {
        ModuleName.rca: DependencyKind.soft,
        ModuleName.inventory: DependencyKind.soft,  # falls back to repair time alone
    },
}


def dependency_kind(
    module: ModuleName,
    dependency: ModuleName,
    kinds: Mapping[ModuleName, Mapping[ModuleName, DependencyKind]] = DEPENDENCY_KINDS,
) -> DependencyKind:
    """HARD unless the edge is explicitly declared SOFT in ``kinds``."""
    return kinds.get(ModuleName(module), {}).get(ModuleName(dependency), DependencyKind.hard)

#: Stable ordering for anything user-visible, so two identical requests produce
#: identical output. Mirrors the natural flow of an investigation.
MODULE_ORDER: tuple[ModuleName, ...] = (
    ModuleName.resolver,
    ModuleName.rag,
    ModuleName.pdm,
    ModuleName.rca,
    ModuleName.maintenance,
    ModuleName.inventory,
    ModuleName.safety,
    ModuleName.production,
)

_ORDER_INDEX = {name: i for i, name in enumerate(MODULE_ORDER)}


class GraphCycleError(RuntimeError):
    """Raised when the dependency declaration contains a cycle."""


class UnknownModuleError(ValueError):
    """Raised when a module has no entry in the dependency table."""


def sort_modules(modules: Iterable[ModuleName]) -> list[ModuleName]:
    """Sort modules into the canonical display order."""
    return sorted(set(modules), key=lambda m: _ORDER_INDEX.get(ModuleName(m), 99))


def dependencies_of(
    module: ModuleName,
    dependencies: Mapping[ModuleName, Sequence[ModuleName]] = DEPENDENCIES,
) -> tuple[ModuleName, ...]:
    """Direct dependencies of one module."""
    module = ModuleName(module)
    if module not in dependencies:
        raise UnknownModuleError(f"Module '{module}' has no dependency declaration.")
    return tuple(dependencies[module])


def dependents_of(
    module: ModuleName,
    dependencies: Mapping[ModuleName, Sequence[ModuleName]] = DEPENDENCIES,
) -> list[ModuleName]:
    """Modules that depend on ``module`` directly."""
    module = ModuleName(module)
    return sort_modules(
        name for name, deps in dependencies.items() if module in deps
    )


def expand_selection(
    selected: Iterable[ModuleName],
    dependencies: Mapping[ModuleName, Sequence[ModuleName]] = DEPENDENCIES,
) -> list[ModuleName]:
    """Close a selection over its dependencies.

    RESOLVER is always included: no module may run against an unconfirmed
    machine, so it is part of every plan whether or not anyone asked for it.
    """
    pending = [ModuleName(m) for m in selected]
    pending.append(ModuleName.resolver)
    resolved: set[ModuleName] = set()

    while pending:
        module = pending.pop()
        if module in resolved:
            continue
        resolved.add(module)
        pending.extend(dependencies_of(module, dependencies))

    return sort_modules(resolved)


def execution_levels(
    modules: Iterable[ModuleName],
    dependencies: Mapping[ModuleName, Sequence[ModuleName]] = DEPENDENCIES,
) -> list[list[ModuleName]]:
    """Group ``modules`` into levels that may each run concurrently.

    Level *n* contains every module whose dependencies all live in levels
    ``< n``. Dependencies outside the selection are ignored — the caller is
    expected to have run :func:`expand_selection` first, and a partial
    selection should not silently stall.

    Raises :class:`GraphCycleError` if the edges among ``modules`` cycle.
    """
    wanted = set(expand_selection(modules, dependencies))

    remaining = {
        module: {d for d in dependencies_of(module, dependencies) if d in wanted}
        for module in wanted
    }

    levels: list[list[ModuleName]] = []
    done: set[ModuleName] = set()

    while remaining:
        ready = [m for m, deps in remaining.items() if deps <= done]
        if not ready:
            # Nothing can run and nothing has finished: the leftovers form at
            # least one cycle. Name them rather than deadlocking.
            stuck = ", ".join(sorted(str(ModuleName(m).value) for m in remaining))
            raise GraphCycleError(
                f"Dependency cycle detected among modules: {stuck}. "
                f"The graph in app/orchestrator/graph.py must be acyclic."
            )
        level = sort_modules(ready)
        levels.append(level)
        done.update(level)
        for module in level:
            del remaining[module]

    return levels


def validate_graph(
    dependencies: Mapping[ModuleName, Sequence[ModuleName]] = DEPENDENCIES,
) -> None:
    """Fail loudly on an unknown edge or a cycle. Run at import time."""
    for module, deps in dependencies.items():
        for dep in deps:
            if ModuleName(dep) not in dependencies:
                raise UnknownModuleError(
                    f"Module '{module}' depends on '{dep}', which is not declared."
                )
    execution_levels(dependencies.keys(), dependencies)


validate_graph()
