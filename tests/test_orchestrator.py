"""Tests for the AI Orchestrator.

No live Mongo, no LLM, no network. The executor takes injected runners, the
orchestrator takes an injected resolver and session store, and both composer
and router take injected LLM callables — so every path here, including the
"LLM available" ones, runs offline and deterministically.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from app.orchestrator.aggregator import (
    aggregate,
    build_brief,
    compose_narrative,
    render_template,
    validate_narrative,
)
from app.orchestrator.executor import (
    ExecutionContext,
    ModuleExecutor,
    ModuleOutcome,
    ModuleOutput,
)
from app.orchestrator.graph import (
    DEPENDENCIES,
    GraphCycleError,
    execution_levels,
    expand_selection,
)
from app.orchestrator.orchestrator import (
    ConversationStore,
    Orchestrator,
    set_orchestrator,
)
from app.orchestrator.router_llm import DEFAULT_MODULES, LLMRouter
from app.schemas.agents import AgentContext
from app.schemas.machine import COLLECTIONS
from app.schemas.orchestration import (
    ModuleName,
    ModuleStatus,
    NarrativeSource,
    OrchestrationResult,
    OrchestrationStatus,
    RoutingSource,
    UserRole,
)
from app.schemas.resolution import (
    MatchMethod,
    ResolutionCandidate,
    ResolutionResult,
    ResolutionStatus,
)

TENANT = "demo"
OTHER_TENANT = "acme"


# ---------------------------------------------------------------------------
# Fake Mongo (same shape as test_agents.py, plus the writes sessions need)
# ---------------------------------------------------------------------------
class FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, *_a, **_kw):
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def __aiter__(self):
        async def gen():
            for doc in self._docs:
                yield doc

        return gen()


def _matches(doc, query):
    for key, expected in query.items():
        value = doc
        for part in key.split("."):
            value = (value or {}).get(part) if isinstance(value, dict) else None
        if isinstance(expected, dict):
            if "$gt" in expected and not (value is not None and value > expected["$gt"]):
                return False
            if "$in" in expected and value not in expected["$in"]:
                return False
        elif isinstance(value, list):
            if expected not in value:
                return False
        elif value != expected:
            return False
    return True


class FakeCollection:
    def __init__(self, name, docs=None):
        self.name = name
        self.docs = list(docs or [])

    def find(self, query=None, *_a, **_kw):
        return FakeCursor([d for d in self.docs if _matches(d, query or {})])

    async def find_one(self, query=None, *_a, **_kw):
        return next((d for d in self.docs if _matches(d, query or {})), None)

    async def count_documents(self, query=None, **_kw):
        return sum(1 for d in self.docs if _matches(d, query or {}))

    async def replace_one(self, query, replacement, upsert=False, **_kw):
        for index, doc in enumerate(self.docs):
            if _matches(doc, query):
                self.docs[index] = dict(replacement)
                return SimpleNamespace(matched_count=1, upserted_id=None)
        if upsert:
            self.docs.append(dict(replacement))
            return SimpleNamespace(matched_count=0, upserted_id="fake")
        return SimpleNamespace(matched_count=0, upserted_id=None)


class FakeDatabase:
    def __init__(self):
        self._collections: dict[str, FakeCollection] = {}

    def __getitem__(self, name):
        return self._collections.setdefault(name, FakeCollection(name))

    def add_collection(self, name, docs):
        self._collections[name] = FakeCollection(name, docs)


@pytest.fixture
def fake_db(monkeypatch):
    database = FakeDatabase()
    from app import db as db_module

    monkeypatch.setattr(db_module, "get_database", lambda: database)
    return database


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """Every test runs as if ANTHROPIC_API_KEY were unset unless it says otherwise."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _reset_process_orchestrator():
    yield
    set_orchestrator(None)


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------
MACHINES = [
    {
        "tenant_id": TENANT,
        "machine_id": "CV-201",
        "name": "Infeed Belt Conveyor",
        "model": "SpanTech SB-3000",
        "line_id": "LINE-A",
        "site_id": "SITE-DETROIT",
        "status": "running",
        "aliases": ["infeed conveyor"],
    },
    {
        "tenant_id": TENANT,
        "machine_id": "MC-110",
        "name": "3-Axis CNC Milling Center",
        "model": "Haas VF-4",
        "line_id": "LINE-A",
        "site_id": "SITE-DETROIT",
        "status": "running",
        "aliases": ["the mill"],
    },
    {
        "tenant_id": TENANT,
        "machine_id": "CV-202",
        "name": "Outfeed Belt Conveyor",
        "model": "SpanTech SB-3000",
        "line_id": "LINE-B",
        "site_id": "SITE-DETROIT",
        "status": "running",
        "aliases": ["outfeed conveyor"],
    },
    {
        "tenant_id": OTHER_TENANT,
        "machine_id": "PR-900",
        "name": "Acme Stamping Press",
        "model": "Acme SP-900",
        "line_id": "LINE-Z",
        "site_id": "SITE-AUSTIN",
        "status": "running",
        "aliases": [],
    },
]


def _candidate(machine_id: str = "CV-201") -> ResolutionCandidate:
    machine = next(m for m in MACHINES if m["machine_id"] == machine_id)
    return ResolutionCandidate(
        machine_id=machine["machine_id"],
        name=machine["name"],
        model=machine["model"],
        line_id=machine["line_id"],
        status=machine["status"],
        confidence=0.95,
        matched_by=MatchMethod.exact_id,
        matched_value=machine["machine_id"],
    )


def resolver_returning(result: ResolutionResult):
    """A stand-in resolver that always returns ``result`` and records calls."""

    async def _resolve(text, tenant_id, context=None, **_kw):
        _resolve.calls.append((text, tenant_id))
        return result

    _resolve.calls = []
    return _resolve


def real_resolver():
    """The genuine resolver over an in-memory fleet — used for tenant scoping."""
    from app.services.resolver import InMemoryMachineRepository, resolve_machine

    repository = InMemoryMachineRepository(MACHINES, [])

    async def _resolve(text, tenant_id, context=None, **_kw):
        return await resolve_machine(
            text=text, tenant_id=tenant_id, context=context, repository=repository
        )

    return _resolve


RCA_DATA = {
    "machine_id": "CV-201",
    "tenant_id": TENANT,
    "primary_cause": {
        "cause_id": "RCA-BEARING-001",
        "description": "Drive roller bearing degradation",
        "component_id": "CV-201-BRG-D",
        "fault_mode": "BEARING_WEAR",
        "probability": 0.85,
        "supporting_evidence_ids": ["SIG-VIB"],
        "contradicting_evidence_ids": [],
    },
    "alternative_causes": [],
    "causal_chain": [
        {
            "order": 1,
            "description": "Lubricant film breakdown",
            "mechanism": "Metal-to-metal contact raises friction",
            "evidence_ids": ["SIG-VIB"],
            "sensor_signals": ["vibration"],
        }
    ],
    "evidence": [
        {
            "evidence_id": "SIG-VIB",
            "source": "SENSOR",
            "description": "Vibration rising 3x over 24h",
            "strength": "STRONG",
        }
    ],
    "confidence": 0.72,
    "confidence_basis": "Two independent strong sources agree.",
    "insufficient_data": False,
    "missing_data": [],
}

PDM_DATA = {
    "machine_id": "CV-201",
    "failure_probability": 0.81,
    "remaining_useful_life_hours": 96.0,
    "health_score": 0.42,
    "predicted_failure_mode": "BEARING_WEAR",
    "confidence": 0.7,
    "contributing_features": [],
    "trend_direction": "DEGRADING",
    "readings_used": 480,
    "channels_present": ["vibration"],
    "generated_at": "2026-07-23T10:00:00Z",
}

MAINTENANCE_DATA = {
    "procedure_steps": [
        {
            "order": 1,
            "instruction": "Isolate the machine and apply lock and tag.",
            "component_id": None,
            "tools_required": ["lockout/tagout kit"],
            "estimated_minutes": 10,
            "caution": "Verify zero energy before proceeding.",
        },
        {
            "order": 2,
            "instruction": "Remove the drive roller bearing.",
            "component_id": "CV-201-BRG-D",
            "tools_required": ["bearing puller"],
            "estimated_minutes": 45,
        },
        {
            "order": 3,
            "instruction": "Install the replacement bearing and torque to spec.",
            "component_id": "CV-201-BRG-D",
            "tools_required": ["torque wrench"],
            "estimated_minutes": 35,
        },
    ],
    "required_parts": [
        {
            "part_number": "SKF-6310-2RS1",
            "description": "Deep groove ball bearing",
            "quantity": 1,
            "component_id": "CV-201-BRG-D",
        }
    ],
    "required_tools": ["bearing puller", "torque wrench"],
    "total_estimated_minutes": 90,
    "skill_level": "INTERMEDIATE",
    "procedure_source": "DERIVED",
}

INVENTORY_AVAILABLE = {
    "items": [
        {
            "part_number": "SKF-6310-2RS1",
            "description": "Deep groove ball bearing",
            "required_qty": 1,
            "available_qty": 4,
            "status": "IN_STOCK",
            "location": "A-03-12",
            "alternatives": [],
            "lead_time_days": 3,
        }
    ],
    "all_parts_available": True,
    "blocking_parts": [],
    "earliest_full_availability_days": 0,
}

INVENTORY_BLOCKED = {
    "items": [
        {
            "part_number": "SKF-6310-2RS1",
            "description": "Deep groove ball bearing",
            "required_qty": 1,
            "available_qty": 0,
            "status": "OUT_OF_STOCK",
            "location": "A-03-12",
            "alternatives": [],
            "lead_time_days": 21,
        }
    ],
    "all_parts_available": False,
    "blocking_parts": ["SKF-6310-2RS1"],
    "earliest_full_availability_days": 21,
}

SAFETY_ROUTINE = {
    "hazards": [
        {
            "hazard_type": "Entanglement",
            "description": "Moving belt can catch loose clothing",
            "severity": "HIGH",
            "source_component_id": "CV-201-BRG-D",
        }
    ],
    "required_ppe": ["safety glasses", "gloves"],
    "lockout_tagout_steps": [
        {
            "order": 1,
            "instruction": "Open the local disconnect",
            "verification": "Confirm no motion",
        }
    ],
    "energy_sources_to_isolate": [
        {
            "type": "ELECTRICAL",
            "location": "Local disconnect panel",
            "isolation_method": "Lock in the open position",
        }
    ],
    "permits_required": [],
    "blocking_conditions": [],
    "citations": [],
    "source": "GENERIC",
}

SAFETY_CRITICAL = {
    **SAFETY_ROUTINE,
    "hazards": [
        {
            "hazard_type": "Stored energy",
            "description": "Belt tensioner retains stored energy and can release without warning",
            "severity": "CRITICAL",
            "source_component_id": "CV-201-BRG-D",
        }
    ],
    "blocking_conditions": ["Guard interlock is bypassed — do not work on this machine"],
}

PRODUCTION_REPAIR_NOW = {
    "downtime_estimate_minutes": {"repair_time": 180, "total_including_parts_wait": 180},
    "units_lost_estimate": 1800,
    "is_bottleneck": True,
    "downstream_machines_affected": ["MC-110"],
    "cost_estimate": {
        "downtime_cost": 1350.0,
        "parts_cost": 500.0,
        "total": 1850.0,
        "currency": "USD",
    },
    "recommendation": "REPAIR_NOW",
    "recommendation_rationale": "High failure probability on a bottleneck machine.",
    "assumptions": ["Repair time estimated from fault mode BEARING_WEAR"],
}


def _outcome(
    module: ModuleName,
    data: Any = None,
    status: ModuleStatus = ModuleStatus.ok,
    reason: Optional[str] = None,
) -> ModuleOutcome:
    return ModuleOutcome(name=module, status=status, data=data, reason=reason)


def full_outcomes(**overrides) -> dict[ModuleName, ModuleOutcome]:
    """A complete, healthy set of module outcomes for aggregation tests."""
    outcomes = {
        ModuleName.resolver: _outcome(ModuleName.resolver, {"machine_id": "CV-201"}),
        ModuleName.rag: _outcome(ModuleName.rag, {"chunks": []}, ModuleStatus.partial),
        ModuleName.pdm: _outcome(ModuleName.pdm, PDM_DATA),
        ModuleName.rca: _outcome(ModuleName.rca, RCA_DATA),
        ModuleName.maintenance: _outcome(ModuleName.maintenance, MAINTENANCE_DATA),
        ModuleName.inventory: _outcome(ModuleName.inventory, INVENTORY_AVAILABLE),
        ModuleName.safety: _outcome(ModuleName.safety, SAFETY_ROUTINE),
        ModuleName.production: _outcome(ModuleName.production, PRODUCTION_REPAIR_NOW),
    }
    outcomes.update(overrides)
    return outcomes


def _result(role: UserRole = UserRole.engineer) -> OrchestrationResult:
    from app.schemas.orchestration import MachineSummary

    return OrchestrationResult(
        request_id="req-test",
        tenant_id=TENANT,
        user_role=role,
        status=OrchestrationStatus.complete,
        machine=MachineSummary(
            machine_id="CV-201",
            name="Infeed Belt Conveyor",
            model="SpanTech SB-3000",
            line_id="LINE-A",
            status="running",
        ),
    )


def aggregated(
    role: UserRole = UserRole.engineer, **overrides
) -> OrchestrationResult:
    outcomes = full_outcomes(**overrides)
    return aggregate(_result(role), outcomes, list(outcomes))


# ---------------------------------------------------------------------------
# Stub runners for the executor / orchestrator
# ---------------------------------------------------------------------------
def stub_runner(
    data: Any = None,
    *,
    status: ModuleStatus = ModuleStatus.ok,
    delay: float = 0.0,
    raises: Optional[Exception] = None,
    calls: Optional[list] = None,
    name: str = "",
):
    async def _run(context: ExecutionContext) -> ModuleOutput:
        if calls is not None:
            calls.append(name)
        if delay:
            await asyncio.sleep(delay)
        if raises is not None:
            raise raises
        return ModuleOutput(status=status, data=data)

    return _run


def healthy_runners(calls: Optional[list] = None, **overrides):
    runners = {
        ModuleName.rag: stub_runner({"chunks": []}, calls=calls, name="RAG"),
        ModuleName.pdm: stub_runner(PDM_DATA, calls=calls, name="PDM"),
        ModuleName.rca: stub_runner(RCA_DATA, calls=calls, name="RCA"),
        ModuleName.maintenance: stub_runner(
            MAINTENANCE_DATA, calls=calls, name="MAINTENANCE"
        ),
        ModuleName.inventory: stub_runner(
            INVENTORY_AVAILABLE, calls=calls, name="INVENTORY"
        ),
        ModuleName.safety: stub_runner(SAFETY_ROUTINE, calls=calls, name="SAFETY"),
        ModuleName.production: stub_runner(
            PRODUCTION_REPAIR_NOW, calls=calls, name="PRODUCTION"
        ),
    }
    runners.update(overrides)
    return runners


def router_selecting(*modules: ModuleName, reasoning: str = "test route") -> LLMRouter:
    # No machine_reference: the router must not be the source of the machine
    # the request runs against — the operator's own words are.
    payload = json.dumps(
        {
            "modules": [ModuleName(m).value for m in modules],
            "intent": "REPORT_FAULT",
            "urgency": "HIGH",
            "reasoning": reasoning,
        }
    )

    async def _call(_prompt: str) -> str:
        return payload

    return LLMRouter(call_llm=_call)


def make_orchestrator(
    *,
    runners=None,
    resolver=None,
    router: Optional[LLMRouter] = None,
    cache_ttl_seconds: float = 120.0,
    module_timeout_seconds: float = 15.0,
    compose_llm=None,
    calls: Optional[list] = None,
) -> Orchestrator:
    return Orchestrator(
        router=router or router_selecting(*DEFAULT_MODULES),
        module_runners=runners if runners is not None else healthy_runners(calls),
        resolver=resolver or resolver_returning(
            ResolutionResult(
                status=ResolutionStatus.resolved,
                machine=_candidate(),
                raw_input="CV-201",
            )
        ),
        session_store=ConversationStore(cache_ttl_seconds=cache_ttl_seconds),
        cache_ttl_seconds=cache_ttl_seconds,
        module_timeout_seconds=module_timeout_seconds,
        compose_llm=compose_llm,
    )


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def test_selecting_a_module_selects_its_dependencies():
    selected = expand_selection([ModuleName.inventory])
    assert set(selected) == {
        ModuleName.resolver,
        ModuleName.rag,
        ModuleName.pdm,
        ModuleName.rca,
        ModuleName.maintenance,
        ModuleName.inventory,
    }
    # PRODUCTION was not asked for and is not a dependency of INVENTORY.
    assert ModuleName.production not in selected


def test_execution_levels_are_topological():
    levels = execution_levels(DEPENDENCIES.keys())
    position = {m: i for i, level in enumerate(levels) for m in level}

    for module, deps in DEPENDENCIES.items():
        for dep in deps:
            assert position[dep] < position[module], f"{dep} must precede {module}"

    assert levels[0] == [ModuleName.resolver]
    # RAG and PDM depend only on the resolver, so they share a level.
    assert set(levels[1]) == {ModuleName.rag, ModuleName.pdm}
    # SAFETY needs only RCA, so it starts before MAINTENANCE has finished.
    assert position[ModuleName.safety] == position[ModuleName.maintenance]
    assert position[ModuleName.inventory] > position[ModuleName.maintenance]
    assert position[ModuleName.production] > position[ModuleName.inventory]


def test_cycle_is_detected_and_named():
    cyclic = {
        ModuleName.resolver: (),
        ModuleName.rag: (ModuleName.pdm,),
        ModuleName.pdm: (ModuleName.rag,),
    }
    with pytest.raises(GraphCycleError) as excinfo:
        execution_levels([ModuleName.rag], cyclic)
    assert "RAG" in str(excinfo.value) and "PDM" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_independent_modules_run_in_parallel_dependents_wait():
    runners = healthy_runners()
    runners[ModuleName.rag] = stub_runner({"chunks": []}, delay=0.05)
    runners[ModuleName.pdm] = stub_runner(PDM_DATA, delay=0.05)

    executor = ModuleExecutor(runners, timeout_seconds=5.0)
    context = ExecutionContext(tenant_id=TENANT, machine_id="CV-201")
    context.outcomes[ModuleName.resolver] = _outcome(ModuleName.resolver, {})

    report = await executor.run(
        list(DEPENDENCIES), context, skip=[ModuleName.resolver]
    )

    rag = report.outcomes[ModuleName.rag]
    pdm = report.outcomes[ModuleName.pdm]
    rca = report.outcomes[ModuleName.rca]

    # Recorded start times prove the overlap: each began before the other ended.
    assert rag.started_at < pdm.finished_at
    assert pdm.started_at < rag.finished_at
    # And the dependent genuinely waited for both.
    assert rca.started_at >= rag.finished_at
    assert rca.started_at >= pdm.finished_at
    # Two 50ms modules in parallel, not 100ms in series.
    assert report.total_elapsed_ms < 150


@pytest.mark.asyncio
async def test_failed_module_is_unavailable_and_hard_dependent_is_skipped_naming_it():
    """RCA fails: its HARD dependents (MAINTENANCE, then INVENTORY) cascade to
    SKIPPED, but its SOFT dependents (SAFETY, PRODUCTION) still run degraded —
    a safety briefing and an impact estimate are both valid without a
    confirmed cause.
    """
    runners = healthy_runners()
    runners[ModuleName.rca] = stub_runner(raises=RuntimeError("sensor store down"))

    executor = ModuleExecutor(runners, timeout_seconds=5.0)
    context = ExecutionContext(tenant_id=TENANT, machine_id="CV-201")
    context.outcomes[ModuleName.resolver] = _outcome(ModuleName.resolver, {})

    report = await executor.run(
        list(DEPENDENCIES), context, skip=[ModuleName.resolver]
    )

    rca = report.outcomes[ModuleName.rca]
    assert ModuleStatus(rca.status) == ModuleStatus.unavailable
    # The raw exception text is never in the safe, display-facing reason...
    assert "sensor store down" not in rca.reason
    assert "Root cause analysis" in rca.reason and "RuntimeError" in rca.reason
    # ...but it is preserved, in full, for debugging.
    assert rca.error_detail == "RuntimeError: sensor store down"

    # SOFT dependents of RCA still run: they degrade, they do not refuse.
    safety = report.outcomes[ModuleName.safety]
    assert ModuleStatus(safety.status) == ModuleStatus.ok
    assert safety.degraded_inputs == ["RCA"]

    production = report.outcomes[ModuleName.production]
    assert ModuleStatus(production.status) == ModuleStatus.ok
    assert set(production.degraded_inputs) == {"RCA", "INVENTORY"}

    # MAINTENANCE has a HARD dependency on RCA: no cause, no procedure.
    maintenance = report.outcomes[ModuleName.maintenance]
    assert ModuleStatus(maintenance.status) == ModuleStatus.skipped
    assert "RCA" in maintenance.reason and "UNAVAILABLE" in maintenance.reason

    # The HARD skip still cascades: INVENTORY needs MAINTENANCE's parts list.
    inventory = report.outcomes[ModuleName.inventory]
    assert ModuleStatus(inventory.status) == ModuleStatus.skipped
    assert "MAINTENANCE" in inventory.reason
    # And it names only the immediate dependency: MAINTENANCE's own reason
    # (which names RCA) is not nested inside INVENTORY's, however long the
    # HARD chain gets.
    assert "RCA" not in inventory.reason
    assert "sensor store down" not in inventory.reason

    # Modules that did not depend on RCA at all still ran normally.
    assert ModuleStatus(report.outcomes[ModuleName.pdm].status) == ModuleStatus.ok


@pytest.mark.asyncio
async def test_rag_unavailable_still_allows_rca_safety_production_to_run():
    """RAG only corroborates RCA — its own sensor/fault-signature analysis, and
    everything downstream of it, must not be held hostage by an empty or
    broken knowledge base.
    """
    runners = healthy_runners()
    runners[ModuleName.rag] = stub_runner(raises=RuntimeError("vector index down"))

    executor = ModuleExecutor(runners, timeout_seconds=5.0)
    context = ExecutionContext(tenant_id=TENANT, machine_id="CV-201")
    context.outcomes[ModuleName.resolver] = _outcome(ModuleName.resolver, {})

    report = await executor.run(
        list(DEPENDENCIES), context, skip=[ModuleName.resolver]
    )

    assert ModuleStatus(report.outcomes[ModuleName.rag].status) == ModuleStatus.unavailable

    rca = report.outcomes[ModuleName.rca]
    assert ModuleStatus(rca.status) == ModuleStatus.ok
    assert rca.degraded_inputs == ["RAG"]

    assert ModuleStatus(report.outcomes[ModuleName.safety].status) == ModuleStatus.ok
    assert ModuleStatus(report.outcomes[ModuleName.production].status) == ModuleStatus.ok
    # MAINTENANCE's only SOFT dependency (RAG) is missing but its HARD one
    # (RCA) is fine, so it still runs — just noting the gap.
    maintenance = report.outcomes[ModuleName.maintenance]
    assert ModuleStatus(maintenance.status) == ModuleStatus.ok
    assert maintenance.degraded_inputs == ["RAG"]


@pytest.mark.asyncio
async def test_maintenance_unavailable_still_skips_inventory_hard_dependency():
    """INVENTORY's dependency on MAINTENANCE stays HARD: no procedure means no
    parts list to check.
    """
    runners = healthy_runners()
    runners[ModuleName.maintenance] = stub_runner(raises=RuntimeError("plan service down"))

    executor = ModuleExecutor(runners, timeout_seconds=5.0)
    context = ExecutionContext(tenant_id=TENANT, machine_id="CV-201")
    context.outcomes[ModuleName.resolver] = _outcome(ModuleName.resolver, {})

    report = await executor.run(
        list(DEPENDENCIES), context, skip=[ModuleName.resolver]
    )

    assert (
        ModuleStatus(report.outcomes[ModuleName.maintenance].status)
        == ModuleStatus.unavailable
    )
    inventory = report.outcomes[ModuleName.inventory]
    assert ModuleStatus(inventory.status) == ModuleStatus.skipped
    assert "MAINTENANCE" in inventory.reason and "UNAVAILABLE" in inventory.reason


@pytest.mark.asyncio
async def test_module_timeout_is_unavailable_not_an_exception():
    runners = healthy_runners()
    runners[ModuleName.pdm] = stub_runner(PDM_DATA, delay=1.0)

    executor = ModuleExecutor(runners, timeout_seconds=0.05)
    context = ExecutionContext(tenant_id=TENANT, machine_id="CV-201")
    context.outcomes[ModuleName.resolver] = _outcome(ModuleName.resolver, {})

    report = await executor.run(
        [ModuleName.pdm, ModuleName.rag], context, skip=[ModuleName.resolver]
    )

    pdm = report.outcomes[ModuleName.pdm]
    assert ModuleStatus(pdm.status) == ModuleStatus.unavailable
    assert "Timed out" in pdm.reason
    assert ModuleStatus(report.outcomes[ModuleName.rag].status) == ModuleStatus.ok


@pytest.mark.asyncio
async def test_progress_events_are_emitted_for_start_and_finish():
    events = []
    executor = ModuleExecutor(healthy_runners(), timeout_seconds=5.0)
    context = ExecutionContext(tenant_id=TENANT, machine_id="CV-201")
    context.outcomes[ModuleName.resolver] = _outcome(ModuleName.resolver, {})

    await executor.run(
        [ModuleName.rag, ModuleName.pdm],
        context,
        skip=[ModuleName.resolver],
        progress=events.append,
    )

    types = [(e.type, ModuleName(e.module).value) for e in events]
    assert ("MODULE_STARTED", "RAG") in types
    assert ("MODULE_FINISHED", "RAG") in types
    assert ("MODULE_STARTED", "PDM") in types
    assert ("MODULE_FINISHED", "PDM") in types


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_router_garbage_falls_back_to_the_default_module_set():
    async def garbage(_prompt: str) -> str:
        return "I'm sorry, I can't do that. <<<not json>>>"

    decision = await LLMRouter(call_llm=garbage).route("CV-201 is grinding")

    assert list(decision.selected_modules) == [
        ModuleName(m).value for m in DEFAULT_MODULES
    ]
    assert RoutingSource(decision.source) == RoutingSource.fallback_invalid
    assert decision.reasoning


@pytest.mark.asyncio
async def test_router_drops_unknown_module_names_but_keeps_valid_ones():
    async def partly_hallucinated(_prompt: str) -> str:
        return json.dumps(
            {
                "modules": ["RCA", "WEATHER", "SAFETY", "launch_missiles"],
                "intent": "REPORT_FAULT",
                "urgency": "HIGH",
                "reasoning": "fault reported",
            }
        )

    decision = await LLMRouter(call_llm=partly_hallucinated).route("CV-201 grinding")

    assert set(decision.selected_modules) == {"RCA", "SAFETY"}
    assert set(decision.dropped_modules) == {"WEATHER", "launch_missiles"}
    assert RoutingSource(decision.source) == RoutingSource.llm


@pytest.mark.asyncio
async def test_router_timeout_falls_back_to_defaults():
    async def slow(_prompt: str) -> str:
        await asyncio.sleep(1.0)
        return "{}"

    decision = await LLMRouter(call_llm=slow, timeout_seconds=0.05).route("help")

    assert list(decision.selected_modules) == [
        ModuleName(m).value for m in DEFAULT_MODULES
    ]
    assert RoutingSource(decision.source) == RoutingSource.fallback_timeout


@pytest.mark.asyncio
async def test_router_with_no_key_and_no_client_falls_back(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    decision = await LLMRouter().route("CV-201 is overheating badly")

    assert RoutingSource(decision.source) == RoutingSource.fallback_no_client
    assert list(decision.selected_modules) == [
        ModuleName(m).value for m in DEFAULT_MODULES
    ]
    # The deterministic classifier still produces usable intent and urgency.
    assert decision.intent == "REPORT_FAULT"
    assert decision.urgency in ("HIGH", "CRITICAL")
    assert decision.machine_reference == "CV-201"


# ---------------------------------------------------------------------------
# Resolution gate
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ambiguous_machine_returns_clarification_and_runs_zero_modules(fake_db):
    calls: list[str] = []
    ambiguous = ResolutionResult(
        status=ResolutionStatus.ambiguous,
        candidates=[_candidate("CV-201"), _candidate("CV-202")],
        clarification_question="Do you mean CV-201 or CV-202?",
        raw_input="the conveyor is making a noise",
    )
    orchestrator = make_orchestrator(
        resolver=resolver_returning(ambiguous),
        runners=healthy_runners(calls),
    )

    result = await orchestrator.handle(
        tenant_id=TENANT, message="the conveyor is making a noise"
    )

    assert OrchestrationStatus(result.status) == OrchestrationStatus.clarification_needed
    assert result.clarification is not None
    assert result.clarification.question == "Do you mean CV-201 or CV-202?"
    assert {c.machine_id for c in result.clarification.candidates} == {"CV-201", "CV-202"}
    # The hard rule: nothing ran against an unconfirmed machine.
    assert calls == []
    assert [ModuleName(r.name).value for r in result.modules_run] == ["RESOLVER"]
    assert result.rca is None and result.safety is None
    assert result.narrative and "CV-201" in result.narrative


@pytest.mark.asyncio
async def test_not_found_machine_stops_with_not_found(fake_db):
    calls: list[str] = []
    orchestrator = make_orchestrator(
        resolver=resolver_returning(
            ResolutionResult(status=ResolutionStatus.not_found, raw_input="ZZ-999")
        ),
        runners=healthy_runners(calls),
    )

    result = await orchestrator.handle(tenant_id=TENANT, message="ZZ-999 is down")

    assert OrchestrationStatus(result.status) == OrchestrationStatus.not_found
    assert calls == []
    assert result.machine is None
    assert result.narrative


# ---------------------------------------------------------------------------
# End-to-end orchestration
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_healthy_request_completes_with_every_module(fake_db):
    calls: list[str] = []
    orchestrator = make_orchestrator(calls=calls)

    result = await orchestrator.handle(
        tenant_id=TENANT,
        message="CV-201 is grinding",
        user_role=UserRole.engineer,
    )

    assert OrchestrationStatus(result.status) in (
        OrchestrationStatus.complete,
        OrchestrationStatus.partial,
    )
    assert result.machine.machine_id == "CV-201"
    assert result.rca is not None and result.rca.primary_cause.fault_mode == "BEARING_WEAR"
    assert result.safety is not None
    assert result.production is not None
    assert set(calls) == {
        "RAG",
        "PDM",
        "RCA",
        "MAINTENANCE",
        "INVENTORY",
        "SAFETY",
        "PRODUCTION",
    }
    assert result.total_elapsed_ms >= 0
    assert all(row.elapsed_ms >= 0 for row in result.modules_run)


@pytest.mark.asyncio
async def test_failing_module_yields_partial_not_an_error(fake_db):
    runners = healthy_runners()
    runners[ModuleName.production] = stub_runner(raises=RuntimeError("cost service down"))
    orchestrator = make_orchestrator(runners=runners)

    result = await orchestrator.handle(tenant_id=TENANT, message="CV-201 is grinding")

    assert OrchestrationStatus(result.status) == OrchestrationStatus.partial
    production_row = next(
        r for r in result.modules_run if ModuleName(r.name) == ModuleName.production
    )
    assert ModuleStatus(production_row.status) == ModuleStatus.unavailable
    # The raw exception text must never reach the operator-facing reason...
    assert "cost service down" not in production_row.reason
    assert "RuntimeError" in production_row.reason
    # ...only the full error_detail, kept for debugging.
    assert production_row.error_detail == "RuntimeError: cost service down"
    # Everything else still made it into the answer.
    assert result.rca is not None and result.safety is not None
    assert result.narrative


@pytest.mark.asyncio
async def test_narrative_never_contains_a_raw_driver_error(fake_db):
    """A raw MongoDB error document — cluster timestamps, signature bytes,
    codeName, the lot — must never reach the composed narrative, however a
    module's underlying implementation happens to fail.
    """
    from pymongo.errors import OperationFailure

    raw = OperationFailure(
        "Collection 'factorypilot.chunks' does not exist. Raw server reply: "
        "{'ok': 0.0, 'errmsg': 'ns not found', 'code': 26, 'codeName': "
        "'NamespaceNotFound', '$clusterTime': {'clusterTime': Timestamp(1753, 2), "
        "'signature': {'hash': b'\\x00\\x01', 'keyId': 7}}}",
        code=26,
    )
    runners = healthy_runners()
    runners[ModuleName.rag] = stub_runner(raises=raw)
    orchestrator = make_orchestrator(runners=runners)

    result = await orchestrator.handle(tenant_id=TENANT, message="CV-201 is grinding")

    # The exception's type name (e.g. "OperationFailure") is safe and may
    # appear — it is the raw driver payload that must never reach an operator.
    assert "clusterTime" not in result.narrative
    assert "signature" not in result.narrative
    assert "Timestamp(1753" not in result.narrative

    rag_row = next(r for r in result.modules_run if ModuleName(r.name) == ModuleName.rag)
    assert ModuleStatus(rag_row.status) == ModuleStatus.unavailable
    assert "clusterTime" not in rag_row.reason
    # The full detail survives, but only off to the side, for debugging.
    assert rag_row.error_detail is not None
    assert "clusterTime" in rag_row.error_detail


@pytest.mark.asyncio
async def test_works_entirely_without_anthropic_api_key(monkeypatch, fake_db):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # A default router (no injected callable) and no composer callable: both
    # must fall back rather than fail.
    orchestrator = Orchestrator(
        router=LLMRouter(),
        module_runners=healthy_runners(),
        resolver=resolver_returning(
            ResolutionResult(
                status=ResolutionStatus.resolved,
                machine=_candidate(),
                raw_input="CV-201",
            )
        ),
        session_store=ConversationStore(cache_ttl_seconds=120.0),
    )

    result = await orchestrator.handle(
        tenant_id=TENANT, message="CV-201 is grinding", user_role=UserRole.technician
    )

    assert NarrativeSource(result.narrative_source) == NarrativeSource.template
    assert RoutingSource(result.routing_decision.source) == RoutingSource.fallback_no_client
    assert result.narrative.strip()
    assert OrchestrationStatus(result.status) in (
        OrchestrationStatus.complete,
        OrchestrationStatus.partial,
    )


# ---------------------------------------------------------------------------
# Conflict rules
# ---------------------------------------------------------------------------
def test_safety_critical_leads_the_response_even_for_a_manager():
    result = aggregated(
        UserRole.manager,
        **{ModuleName.safety: _outcome(ModuleName.safety, SAFETY_CRITICAL)},
    )
    narrative = render_template(build_brief(result))

    assert result.safety_critical is True
    assert result.safety is not None, "safety is never scoped away"
    assert result.safety_clearance_required is True

    # Safety appears before the production impact a manager asked for.
    assert "SAFETY CRITICAL" in narrative
    assert narrative.index("SAFETY CRITICAL") < narrative.index("PRODUCTION IMPACT")
    assert "Guard interlock is bypassed" in narrative

    # REPAIR_NOW is annotated, not silently overruled.
    assert "safety clearance" in result.production.recommendation_rationale.lower()
    assert any(
        "REPAIR_NOW" in conflict and "Safety" in conflict
        for conflict in result.conflicts_surfaced
    )


@pytest.mark.asyncio
async def test_safety_banner_only_leads_for_genuine_blocking_conditions(fake_db):
    """The banner tracks blocking_conditions, not routine LOTO preconditions.

    Runs the real SafetyAgent for two machine states and feeds its actual
    output through the aggregator, so this exercises the agent's new
    blocking_conditions/standard_preconditions split and the aggregator's
    banner rule together.
    """
    from app.agents.safety_agent import SafetyAgent

    fake_db.add_collection(
        COLLECTIONS.machines,
        [
            {**MACHINES[0]},
            {
                "tenant_id": TENANT,
                "machine_id": "HP-150",
                "name": "150-Ton Hydraulic Press",
                "model": "Beckwood BX-150",
                "line_id": "LINE-A",
                "site_id": "SITE-DETROIT",
                "status": "fault",
                "aliases": [],
            },
        ],
    )
    fake_db.add_collection(
        COLLECTIONS.components,
        [
            {
                "tenant_id": TENANT,
                "component_id": "CV-201-BRG-D",
                "machine_id": "CV-201",
                "name": "Drive Roller Bearing",
                "type": "bearing",
            },
            {
                "tenant_id": TENANT,
                "component_id": "HP-150-CYL",
                "machine_id": "HP-150",
                "name": "Main Ram Cylinder",
                "type": "cylinder",
            },
        ],
    )

    agent = SafetyAgent()

    routine_briefing = await agent.run(
        AgentContext(
            tenant_id=TENANT,
            machine_id="CV-201",
            rca_result={
                "primary_cause": {
                    "component_id": "CV-201-BRG-D",
                    "fault_mode": "BEARING_WEAR",
                }
            },
        )
    )
    routine_result = aggregated(
        UserRole.engineer,
        **{ModuleName.safety: _outcome(ModuleName.safety, routine_briefing.data)},
    )
    routine_narrative = render_template(build_brief(routine_result))

    assert routine_result.safety_critical is False
    assert "SAFETY CRITICAL" not in routine_narrative
    # The routine preconditions still appear — just not as a banner.
    assert "Standard preconditions" in routine_narrative

    leaking_briefing = await agent.run(
        AgentContext(
            tenant_id=TENANT,
            machine_id="HP-150",
            rca_result={
                "primary_cause": {
                    "component_id": "HP-150-CYL",
                    "fault_mode": "SEAL_LEAK",
                }
            },
        )
    )
    leaking_result = aggregated(
        UserRole.engineer,
        **{ModuleName.safety: _outcome(ModuleName.safety, leaking_briefing.data)},
    )
    leaking_narrative = render_template(build_brief(leaking_result))

    assert leaking_result.safety_critical is True
    assert "SAFETY CRITICAL" in leaking_narrative


def test_blocking_parts_force_production_to_use_the_parts_wait_downtime():
    result = aggregated(
        UserRole.engineer,
        **{ModuleName.inventory: _outcome(ModuleName.inventory, INVENTORY_BLOCKED)},
    )

    downtime = result.production.downtime_estimate_minutes
    assert downtime.repair_time == 180
    # 21 days of waiting, not the bare repair time.
    assert downtime.total_including_parts_wait == 180 + 21 * 24 * 60
    assert result.production.cost_estimate.downtime_cost > 1350.0

    # The steps that need the missing part are marked blocked.
    blocked_orders = {step.order for step in result.blocked_steps}
    assert 3 in blocked_orders
    assert all(
        "SKF-6310-2RS1" in step.blocked_by_parts for step in result.blocked_steps
    )
    assert any("blocked" in c.lower() for c in result.conflicts_surfaced)
    assert any("downtime estimate excluded" in c for c in result.conflicts_surfaced)


def test_insufficient_rca_data_marks_everything_provisional():
    thin = {
        **RCA_DATA,
        "confidence": 0.35,
        "insufficient_data": True,
        "missing_data": ["no vibration sensor on the drive end"],
    }
    result = aggregated(
        UserRole.engineer, **{ModuleName.rca: _outcome(ModuleName.rca, thin)}
    )
    narrative = render_template(build_brief(result))

    assert result.provisional is True
    assert any("insufficient data" in c.lower() for c in result.conflicts_surfaced)
    assert "PROVISIONAL" in narrative
    assert "35%" in narrative


def test_module_disagreement_is_surfaced_not_reconciled():
    monitoring = {
        **PRODUCTION_REPAIR_NOW,
        "recommendation": "MONITOR",
        "recommendation_rationale": "Low risk.",
    }
    result = aggregated(
        UserRole.engineer,
        **{ModuleName.production: _outcome(ModuleName.production, monitoring)},
    )

    assert any(
        "MONITOR" in c and "failure probability" in c
        for c in result.conflicts_surfaced
    ), result.conflicts_surfaced
    # Both sides survive; the orchestrator does not pick a winner.
    assert str(result.production.recommendation) == "MONITOR"
    assert result.pdm.failure_probability == pytest.approx(0.81)


# ---------------------------------------------------------------------------
# Role scoping
# ---------------------------------------------------------------------------
def test_role_scoping_omits_cost_for_a_technician_but_never_safety():
    result = aggregated(
        UserRole.technician,
        **{ModuleName.safety: _outcome(ModuleName.safety, SAFETY_CRITICAL)},
    )
    narrative = render_template(build_brief(result))

    # Cost lives on the production impact, which a technician does not receive.
    assert result.production is None
    assert "production" in result.omitted_for_role
    assert "1850" not in narrative and "1,850" not in narrative
    assert "cost" not in narrative.lower()

    # Safety survives in full, and leads.
    assert result.safety is not None
    assert "SAFETY CRITICAL" in narrative
    assert "Guard interlock is bypassed" in narrative
    # And the things a technician does need are present.
    assert result.maintenance is not None
    assert "A-03-12" in narrative, "parts location is technician-relevant"


def test_manager_gets_impact_and_cost_but_not_the_step_list():
    result = aggregated(UserRole.manager)
    narrative = render_template(build_brief(result))

    assert result.production is not None
    assert result.maintenance is None
    assert "maintenance" in result.omitted_for_role
    assert "1850" in narrative or "1850.0" in narrative
    assert "Remove the drive roller bearing" not in narrative


def test_engineer_gets_everything_including_rca_detail():
    result = aggregated(UserRole.engineer)
    narrative = render_template(build_brief(result))

    assert result.omitted_for_role == []
    assert result.rca is not None and result.pdm is not None
    assert result.maintenance is not None and result.production is not None
    assert "Lubricant film breakdown" in narrative, "full causal chain for engineers"
    assert "Confidence: 72%" in narrative


def test_degraded_modules_appear_as_a_brief_note_not_a_stack_trace():
    """A module that ran on a missing SOFT input gets one short line, not a dump."""
    degraded_rca = ModuleOutcome(
        name=ModuleName.rca,
        status=ModuleStatus.ok,
        data=RCA_DATA,
        degraded_inputs=["RAG"],
    )
    result = aggregated(UserRole.engineer, **{ModuleName.rca: degraded_rca})
    brief = build_brief(result)
    narrative = render_template(brief)

    assert brief["degraded_modules"] == [{"name": "RCA", "missing": ["RAG"]}]
    assert "PARTIAL INPUT" in narrative
    assert "RCA proceeded without: RAG" in narrative


def test_safety_officer_leads_with_the_briefing():
    result = aggregated(
        UserRole.safety_officer,
        **{ModuleName.safety: _outcome(ModuleName.safety, SAFETY_CRITICAL)},
    )
    narrative = render_template(build_brief(result))

    assert result.safety is not None
    assert result.production is None
    assert narrative.index("SAFETY") < narrative.index("ROOT CAUSE")
    assert "Lockout/tagout" in narrative
    assert "Energy sources to isolate" in narrative


# ---------------------------------------------------------------------------
# Narrative composition and validation
# ---------------------------------------------------------------------------
def test_template_rendering_never_contains_an_unbacked_number():
    for role in UserRole:
        brief = build_brief(aggregated(role))
        assert validate_narrative(render_template(brief), brief) == []


@pytest.mark.asyncio
async def test_narrative_validation_catches_a_fabricated_number_and_falls_back():
    result = aggregated(UserRole.manager)

    async def hallucinating(_prompt: str) -> str:
        return (
            "The conveyor bearing is failing. Repairing it will cost $47,500 "
            "and take 14 days, so I recommend waiting."
        )

    narrative, source = await compose_narrative(result, call_llm=hallucinating)

    assert NarrativeSource(source) == NarrativeSource.template
    assert "47,500" not in narrative
    assert narrative == render_template(build_brief(result))


@pytest.mark.asyncio
async def test_narrative_from_llm_is_kept_when_every_number_checks_out():
    result = aggregated(UserRole.manager)

    async def faithful(_prompt: str) -> str:
        return (
            "CV-201 has a degrading drive roller bearing. Expected downtime is "
            "180 minutes, costing USD 1850.0 in total, and the recommendation "
            "is REPAIR_NOW."
        )

    narrative, source = await compose_narrative(result, call_llm=faithful)

    assert NarrativeSource(source) == NarrativeSource.llm
    assert "REPAIR_NOW" in narrative


@pytest.mark.asyncio
async def test_composer_failure_falls_back_to_the_template():
    result = aggregated(UserRole.engineer)

    async def exploding(_prompt: str) -> str:
        raise RuntimeError("model overloaded")

    narrative, source = await compose_narrative(result, call_llm=exploding)

    assert NarrativeSource(source) == NarrativeSource.template
    assert narrative == render_template(build_brief(result))


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_followup_within_ttl_reuses_cached_module_output(fake_db):
    calls: list[str] = []
    orchestrator = make_orchestrator(calls=calls, cache_ttl_seconds=120.0)

    first = await orchestrator.handle(
        tenant_id=TENANT, message="CV-201 is grinding", session_id="sess-1"
    )
    assert len(calls) == 7
    assert all(not row.reused for row in first.modules_run)

    calls.clear()
    second = await orchestrator.handle(
        tenant_id=TENANT, message="is it safe to keep running it?", session_id="sess-1"
    )

    assert calls == [], "a fresh follow-up must not re-run the graph"
    reused = {
        ModuleName(row.name).value for row in second.modules_run if row.reused
    }
    assert reused == {"RAG", "PDM", "RCA", "MAINTENANCE", "INVENTORY", "SAFETY", "PRODUCTION"}
    assert second.rca is not None, "reused output still populates the answer"


@pytest.mark.asyncio
async def test_stale_cache_reruns_the_graph(fake_db):
    calls: list[str] = []
    orchestrator = make_orchestrator(calls=calls, cache_ttl_seconds=0.0)

    await orchestrator.handle(
        tenant_id=TENANT, message="CV-201 is grinding", session_id="sess-2"
    )
    calls.clear()

    # TTL of zero: nothing is ever fresh enough to reuse.
    await asyncio.sleep(0.01)
    second = await orchestrator.handle(
        tenant_id=TENANT, message="and now?", session_id="sess-2"
    )

    assert len(calls) == 7
    assert all(not row.reused for row in second.modules_run)


@pytest.mark.asyncio
async def test_session_records_turns_and_last_machine(fake_db):
    orchestrator = make_orchestrator()
    await orchestrator.handle(
        tenant_id=TENANT, message="CV-201 is grinding", session_id="sess-3"
    )

    session = await orchestrator.sessions.load(TENANT, "sess-3")
    assert session is not None
    assert session["last_machine_id"] == "CV-201"
    assert len(session["turns"]) == 1
    assert session["turns"][0]["message"] == "CV-201 is grinding"


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resolution_is_tenant_scoped(fake_db):
    orchestrator = make_orchestrator(resolver=real_resolver())

    mine = await orchestrator.handle(tenant_id=TENANT, message="CV-201 is grinding")
    assert OrchestrationStatus(mine.status) != OrchestrationStatus.not_found
    assert mine.machine.machine_id == "CV-201"

    # The other tenant's fleet does not contain CV-201, and must not see it.
    theirs = await orchestrator.handle(
        tenant_id=OTHER_TENANT, message="CV-201 is grinding"
    )
    assert OrchestrationStatus(theirs.status) == OrchestrationStatus.not_found
    assert theirs.machine is None

    # And the reverse: our tenant cannot reach theirs.
    cross = await orchestrator.handle(tenant_id=TENANT, message="PR-900 is down")
    assert OrchestrationStatus(cross.status) == OrchestrationStatus.not_found


@pytest.mark.asyncio
async def test_sessions_are_tenant_scoped(fake_db):
    orchestrator = make_orchestrator()
    await orchestrator.handle(
        tenant_id=TENANT, message="CV-201 is grinding", session_id="shared-id"
    )

    assert await orchestrator.sessions.load(TENANT, "shared-id") is not None
    # Same session id, different tenant: invisible.
    assert await orchestrator.sessions.load(OTHER_TENANT, "shared-id") is None

    # A second tenant writing the same id gets its own document, and neither
    # tenant's cache can serve the other.
    orchestrator2 = make_orchestrator(
        resolver=resolver_returning(
            ResolutionResult(
                status=ResolutionStatus.resolved,
                machine=_candidate("MC-110"),
                raw_input="MC-110",
            )
        )
    )
    await orchestrator2.handle(
        tenant_id=OTHER_TENANT, message="MC-110 is grinding", session_id="shared-id"
    )

    mine = await orchestrator.sessions.load(TENANT, "shared-id")
    theirs = await orchestrator2.sessions.load(OTHER_TENANT, "shared-id")
    assert mine["last_machine_id"] == "CV-201"
    assert theirs["last_machine_id"] == "MC-110"


@pytest.mark.asyncio
async def test_module_execution_carries_the_tenant(fake_db):
    seen: list[str] = []

    async def tenant_recording_runner(context: ExecutionContext) -> ModuleOutput:
        seen.append(context.tenant_id)
        return ModuleOutput(status=ModuleStatus.ok, data=RCA_DATA)

    runners = healthy_runners()
    runners[ModuleName.rca] = tenant_recording_runner
    orchestrator = make_orchestrator(runners=runners)

    await orchestrator.handle(tenant_id=OTHER_TENANT, message="CV-201 is grinding")
    assert seen == [OTHER_TENANT]


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------
@pytest.fixture
def client(fake_db):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.core.tenant_context import get_current_tenant
    from app.routers.orchestrate import router as orchestrate_router

    app = FastAPI()
    app.include_router(orchestrate_router)
    app.dependency_overrides[get_current_tenant] = lambda: TENANT
    return TestClient(app)


def test_endpoint_returns_partial_not_a_500_when_a_module_fails(client):
    runners = healthy_runners()
    runners[ModuleName.rca] = stub_runner(raises=RuntimeError("rca exploded"))
    set_orchestrator(make_orchestrator(runners=runners))

    response = client.post(
        "/orchestrate",
        json={"message": "CV-201 is grinding", "user_role": "ENGINEER"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "PARTIAL"
    assert body["narrative"]
    rca_row = next(r for r in body["modules_run"] if r["name"] == "RCA")
    assert rca_row["status"] == "UNAVAILABLE"
    # The raw exception text must never leak into the API-visible reason...
    assert "rca exploded" not in rca_row["reason"]
    # ...only into error_detail, kept for debugging.
    assert rca_row["error_detail"] == "RuntimeError: rca exploded"
    assert "rca exploded" not in body["narrative"]


def test_endpoint_returns_clarification_with_candidates(client):
    set_orchestrator(
        make_orchestrator(
            resolver=resolver_returning(
                ResolutionResult(
                    status=ResolutionStatus.ambiguous,
                    candidates=[_candidate("CV-201"), _candidate("CV-202")],
                    clarification_question="Do you mean CV-201 or CV-202?",
                    raw_input="the conveyor",
                )
            )
        )
    )

    response = client.post("/orchestrate", json={"message": "the conveyor is noisy"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "CLARIFICATION_NEEDED"
    assert len(body["clarification"]["candidates"]) == 2
    assert body["modules_run"] == [
        r for r in body["modules_run"] if r["name"] == "RESOLVER"
    ]


def test_session_endpoint_returns_history_and_404s_across_tenants(client):
    set_orchestrator(make_orchestrator())

    client.post(
        "/orchestrate",
        json={"message": "CV-201 is grinding", "session_id": "http-sess"},
    )

    ok = client.get("/orchestrate/sessions/http-sess")
    assert ok.status_code == 200
    body = ok.json()
    assert body["last_machine_id"] == "CV-201"
    assert len(body["turns"]) == 1
    assert set(body["cached_modules"]) >= {"RCA", "SAFETY"}

    missing = client.get("/orchestrate/sessions/never-existed")
    assert missing.status_code == 404
