"""Tests for the streaming chat API.

No live Mongo, no LLM, no network. The orchestrator takes injected runners, a
resolver and a session store; the chat service takes an injected narrative
stream. Every path here — including the ones that exercise "an LLM is
available" — runs offline and deterministically.

The properties under test are the ones a client actually depends on: that
events arrive in the order the work happened, that a blocking resolution
produces no module events at all, that the narrative a client accumulates
equals the one in the final result, and that a client which hangs up stops the
work rather than leaving it running.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from app.orchestrator.executor import ExecutionContext, ModuleOutput
from app.orchestrator.orchestrator import (
    ConversationStore,
    Orchestrator,
    set_orchestrator,
)
from app.orchestrator.router_llm import DEFAULT_MODULES, LLMRouter
from app.schemas.machine import COLLECTIONS
from app.schemas.orchestration import (
    ModuleName,
    ModuleStatus,
    NarrativeSource,
    OrchestrationStatus,
    UserRole,
)
from app.schemas.resolution import (
    MatchMethod,
    ResolutionCandidate,
    ResolutionResult,
    ResolutionStatus,
)
from app.schemas.stream import SSEComment, StreamEvent
from app.services.chat import ChatService, chunk_text, set_chat_service, summarize_module
from app.services.language import (
    LanguageSource,
    detect_language,
    linguistic_text,
)

TENANT = "demo"
OTHER_TENANT = "acme"


# ---------------------------------------------------------------------------
# Fake Mongo (same shape as test_orchestrator.py, plus the delete sessions need)
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

    async def delete_one(self, query, **_kw):
        for index, doc in enumerate(self.docs):
            if _matches(doc, query):
                del self.docs[index]
                return SimpleNamespace(deleted_count=1)
        return SimpleNamespace(deleted_count=0)


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
def _reset_process_singletons():
    yield
    set_orchestrator(None)
    set_chat_service(None)


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
    "causal_chain": [],
    "evidence": [
        {
            "evidence_id": "SIG-VIB",
            "source": "SENSOR",
            "description": "Vibration rising over 24h",
            "strength": "STRONG",
        }
    ],
    "confidence": 0.82,
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

RAG_DATA = {
    "chunks": [
        {
            "document_id": "DOC-SOP-12",
            "document_title": "SB-3000 Maintenance Manual",
            "page_number": 41,
            "section_title": "Drive roller bearing replacement",
            "text": "Remove the drive roller bearing using a puller.",
        }
    ]
}

RAG_CITATIONS = [
    {
        "document_id": "DOC-SOP-12",
        "title": "SB-3000 Maintenance Manual",
        "page_number": 41,
        "section_title": "Drive roller bearing replacement",
    }
]

MAINTENANCE_DATA = {
    "procedure_steps": [
        {
            "order": 1,
            "instruction": "Isolate the machine and apply lock and tag.",
            "tools_required": ["lockout/tagout kit"],
            "estimated_minutes": 10,
        },
        {
            "order": 2,
            "instruction": "Install the replacement bearing and torque to spec.",
            "component_id": "CV-201-BRG-D",
            "tools_required": ["torque wrench"],
            "estimated_minutes": 80,
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
    "required_tools": ["torque wrench"],
    "total_estimated_minutes": 90,
    "skill_level": "INTERMEDIATE",
    "procedure_source": "DERIVED",
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

SAFETY_DATA = {
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

PRODUCTION_DATA = {
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
    async def _resolve(text, tenant_id, context=None, **_kw):
        return result

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


# ---------------------------------------------------------------------------
# Stub runners and wiring
# ---------------------------------------------------------------------------
def stub_runner(
    data: Any = None,
    *,
    status: ModuleStatus = ModuleStatus.ok,
    citations: Optional[list] = None,
    delay: float = 0.0,
    raises: Optional[Exception] = None,
):
    async def _run(context: ExecutionContext) -> ModuleOutput:
        if delay:
            await asyncio.sleep(delay)
        if raises is not None:
            raise raises
        return ModuleOutput(
            status=status, data=data, citations=list(citations or [])
        )

    return _run


def healthy_runners(**overrides):
    runners = {
        ModuleName.rag: stub_runner(RAG_DATA, citations=RAG_CITATIONS),
        ModuleName.pdm: stub_runner(PDM_DATA),
        ModuleName.rca: stub_runner(RCA_DATA),
        ModuleName.maintenance: stub_runner(MAINTENANCE_DATA),
        ModuleName.inventory: stub_runner(INVENTORY_BLOCKED),
        ModuleName.safety: stub_runner(SAFETY_DATA),
        ModuleName.production: stub_runner(PRODUCTION_DATA),
    }
    runners.update(overrides)
    return runners


def router_selecting(*modules: ModuleName) -> LLMRouter:
    payload = json.dumps(
        {
            "modules": [ModuleName(m).value for m in modules],
            "intent": "REPORT_FAULT",
            "urgency": "HIGH",
            "reasoning": "fault reported on a conveyor",
        }
    )

    async def _call(_prompt: str) -> str:
        return payload

    return LLMRouter(call_llm=_call)


def make_orchestrator(*, runners=None, resolver=None, router=None) -> Orchestrator:
    return Orchestrator(
        router=router or router_selecting(*DEFAULT_MODULES),
        module_runners=healthy_runners() if runners is None else runners,
        resolver=resolver
        or resolver_returning(
            ResolutionResult(
                status=ResolutionStatus.resolved,
                machine=_candidate(),
                raw_input="CV-201",
            )
        ),
        session_store=ConversationStore(cache_ttl_seconds=120.0),
        cache_ttl_seconds=120.0,
        module_timeout_seconds=5.0,
    )


def fake_narrative_stream(text: str, *, calls: Optional[list] = None, chunks: int = 5):
    """A stand-in composer that streams ``text`` in pieces and records its prompt."""
    size = max(1, -(-len(text) // chunks))
    pieces = [text[i : i + size] for i in range(0, len(text), size)]

    async def _stream(prompt: str, *, system: str, model: str = "", **_kw):
        if calls is not None:
            calls.append({"prompt": prompt, "system": system})
        for piece in pieces:
            await asyncio.sleep(0)
            yield piece

    return _stream


def make_service(orchestrator=None, **kwargs) -> ChatService:
    return ChatService(orchestrator or make_orchestrator(), **kwargs)


# ---------------------------------------------------------------------------
# Stream helpers
# ---------------------------------------------------------------------------
async def collect(service: ChatService, **kwargs) -> list:
    """Drain a stream to completion, returning the raw frames."""
    kwargs.setdefault("tenant_id", TENANT)
    kwargs.setdefault("message", "CV-201 is grinding")
    return [frame async for frame in service.stream(**kwargs)]


def event_types(frames) -> list[str]:
    return [
        str(f.type) for f in frames if isinstance(f, StreamEvent)
    ]


def events_of(frames, kind: str) -> list[StreamEvent]:
    return [f for f in frames if isinstance(f, StreamEvent) and str(f.type) == kind]


def only(frames, kind: str) -> StreamEvent:
    matches = events_of(frames, kind)
    assert len(matches) == 1, f"expected exactly one '{kind}', got {len(matches)}"
    return matches[0]


def parse_sse(body: str) -> list[dict]:
    """Parse an SSE response body back into event payloads."""
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


# ---------------------------------------------------------------------------
# Event ordering
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_events_arrive_in_the_order_the_work_happened(fake_db):
    frames = await collect(make_service())
    types = event_types(frames)

    assert types[0] == "session"
    assert types[1] == "routing"
    assert types[2] == "resolution"
    assert types[-1] == "done"
    assert types[-2] == "result"

    # Modules finish before the answer is composed, and the composed answer
    # arrives before the structured result that contains it.
    assert types.index("module_start") < types.index("module_finish")
    assert types.index("module_finish") < types.index("narrative_delta")
    assert types.index("narrative_delta") < types.index("result")

    # Citations and conflicts land after the work, before the prose that
    # references them.
    assert types.index("citation") < types.index("narrative_delta")
    assert types.index("conflict") < types.index("narrative_delta")


@pytest.mark.asyncio
async def test_session_event_leads_and_carries_ids_the_result_agrees_with(fake_db):
    frames = await collect(make_service(), session_id="sess-stream")

    session = only(frames, "session")
    assert session.data.session_id == "sess-stream"

    result = only(frames, "result").data
    assert result.request_id == session.data.request_id
    assert result.session_id == "sess-stream"


@pytest.mark.asyncio
async def test_session_id_is_minted_when_the_caller_supplies_none(fake_db):
    frames = await collect(make_service())
    session = only(frames, "session")

    assert session.data.session_id
    assert only(frames, "result").data.session_id == session.data.session_id


@pytest.mark.asyncio
async def test_routing_event_carries_the_selection_intent_and_reasoning(fake_db):
    frames = await collect(make_service())
    routing = only(frames, "routing").data

    assert set(routing.selected_modules) == {
        ModuleName(m).value for m in DEFAULT_MODULES
    }
    assert routing.intent == "REPORT_FAULT"
    assert routing.urgency == "HIGH"
    assert routing.reasoning


@pytest.mark.asyncio
async def test_resolution_event_carries_the_confirmed_machine(fake_db):
    frames = await collect(make_service())
    resolution = only(frames, "resolution").data

    assert resolution.status == "RESOLVED"
    assert resolution.machine.machine_id == "CV-201"
    assert resolution.machine.model == "SpanTech SB-3000"
    assert resolution.candidates == []


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ambiguous_resolution_emits_resolution_then_done_and_no_module_events(
    fake_db,
):
    ambiguous = ResolutionResult(
        status=ResolutionStatus.ambiguous,
        candidates=[_candidate("CV-201"), _candidate("CV-202")],
        clarification_question="Do you mean CV-201 or CV-202?",
        raw_input="the conveyor is making a noise",
    )
    service = make_service(make_orchestrator(resolver=resolver_returning(ambiguous)))

    frames = await collect(service, message="the conveyor is making a noise")
    types = event_types(frames)

    resolution = only(frames, "resolution").data
    assert resolution.status == "AMBIGUOUS"
    assert resolution.clarification_question == "Do you mean CV-201 or CV-202?"
    assert {c.machine_id for c in resolution.candidates} == {"CV-201", "CV-202"}

    # The hard rule: nothing ran against an unconfirmed machine, so there is
    # nothing to report about modules — and no prose to stream either.
    assert "module_start" not in types
    assert "module_finish" not in types
    assert "narrative_delta" not in types

    assert types == ["session", "routing", "resolution", "result", "done"]
    assert (
        OrchestrationStatus(only(frames, "result").data.status)
        == OrchestrationStatus.clarification_needed
    )


@pytest.mark.asyncio
async def test_not_found_resolution_also_runs_nothing(fake_db):
    service = make_service(
        make_orchestrator(
            resolver=resolver_returning(
                ResolutionResult(status=ResolutionStatus.not_found, raw_input="ZZ-999")
            )
        )
    )

    frames = await collect(service, message="ZZ-999 is down")
    types = event_types(frames)

    assert only(frames, "resolution").data.status == "NOT_FOUND"
    assert only(frames, "resolution").data.machine is None
    assert "module_start" not in types and "module_finish" not in types
    assert (
        OrchestrationStatus(only(frames, "result").data.status)
        == OrchestrationStatus.not_found
    )


# ---------------------------------------------------------------------------
# Module lifecycle
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_module_start_precedes_module_finish_for_every_module(fake_db):
    frames = await collect(make_service())

    started_at: dict[str, int] = {}
    for index, frame in enumerate(frames):
        if not isinstance(frame, StreamEvent):
            continue
        if str(frame.type) == "module_start":
            started_at.setdefault(frame.data.module, index)
        elif str(frame.type) == "module_finish":
            assert frame.data.module in started_at, (
                f"{frame.data.module} finished without ever starting"
            )
            assert started_at[frame.data.module] < index

    assert set(started_at) == {
        "RAG",
        "PDM",
        "RCA",
        "MAINTENANCE",
        "INVENTORY",
        "SAFETY",
        "PRODUCTION",
    }
    assert len(events_of(frames, "module_finish")) == len(started_at)


@pytest.mark.asyncio
async def test_parallel_modules_interleave(fake_db):
    """RAG and PDM share a level, so both start before either finishes."""
    runners = healthy_runners(
        **{
            ModuleName.rag: stub_runner(RAG_DATA, citations=RAG_CITATIONS, delay=0.05),
            ModuleName.pdm: stub_runner(PDM_DATA, delay=0.05),
        }
    )
    frames = await collect(make_service(make_orchestrator(runners=runners)))

    starts = [
        index
        for index, f in enumerate(frames)
        if isinstance(f, StreamEvent) and str(f.type) == "module_start"
    ]
    first_finish = next(
        index
        for index, f in enumerate(frames)
        if isinstance(f, StreamEvent) and str(f.type) == "module_finish"
    )

    assert starts[1] < first_finish, "two modules must start before either finishes"

    concurrent = {
        frames[i].data.module for i in starts if i < first_finish
    }
    assert concurrent == {"RAG", "PDM"}
    # And they were reported as sharing a concurrency level.
    assert {frames[i].data.level for i in starts if i < first_finish} == {1}


@pytest.mark.asyncio
async def test_module_finish_carries_a_readable_one_line_summary(fake_db):
    frames = await collect(make_service())
    summaries = {f.data.module: f.data.summary for f in events_of(frames, "module_finish")}

    assert "0.82 confidence" in summaries["RCA"]
    assert "Drive roller bearing degradation" in summaries["RCA"]
    assert "1 out of stock" in summaries["INVENTORY"]
    assert "SKF-6310-2RS1" in summaries["INVENTORY"]
    assert "REPAIR_NOW" in summaries["PRODUCTION"]
    assert summaries["SAFETY"].startswith("1 hazard(s), 1 at CRITICAL or HIGH")
    assert all(summary for summary in summaries.values())


@pytest.mark.asyncio
async def test_a_failed_module_finishes_unavailable_with_its_reason_as_the_summary(
    fake_db,
):
    runners = healthy_runners(
        **{ModuleName.production: stub_runner(raises=RuntimeError("cost service down"))}
    )
    frames = await collect(make_service(make_orchestrator(runners=runners)))

    production = next(
        f.data for f in events_of(frames, "module_finish") if f.data.module == "PRODUCTION"
    )
    assert production.status == "UNAVAILABLE"
    # The raw exception text must never reach the wire...
    assert "cost service down" not in production.summary
    assert "cost service down" not in (production.reason or "")
    # ...the summary still says something useful.
    assert "RuntimeError" in production.summary
    # The stream still completes with an answer.
    assert only(frames, "result").data.status == "PARTIAL"
    assert event_types(frames)[-1] == "done"


@pytest.mark.asyncio
async def test_skipped_modules_still_report_a_start_and_a_finish(fake_db):
    """A dependent that never runs is a row in the timeline, not a gap.

    MAINTENANCE fails, so INVENTORY — whose only dependency is MAINTENANCE,
    and a HARD one: no procedure means no parts list — is SKIPPED rather than
    run. (RCA failing no longer skips anything: SAFETY and PRODUCTION only
    depend on it SOFTly, and run degraded instead of being skipped.)
    """
    runners = healthy_runners(
        **{ModuleName.maintenance: stub_runner(raises=RuntimeError("plan service down"))}
    )
    frames = await collect(make_service(make_orchestrator(runners=runners)))

    finishes = {f.data.module: f.data for f in events_of(frames, "module_finish")}
    starts = {f.data.module for f in events_of(frames, "module_start")}

    assert finishes["INVENTORY"].status == "SKIPPED"
    assert "MAINTENANCE" in finishes["INVENTORY"].summary
    assert "INVENTORY" in starts, "a skipped module still gets a start so the row renders"
    assert set(finishes) == starts


# ---------------------------------------------------------------------------
# Citations and conflicts
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_citations_and_conflicts_are_emitted_as_events(fake_db):
    frames = await collect(make_service())

    citations = [f.data for f in events_of(frames, "citation")]
    assert [c.document_id for c in citations] == ["DOC-SOP-12"]
    assert citations[0].page_number == 41
    assert citations[0].section_title == "Drive roller bearing replacement"

    conflicts = [f.data.description for f in events_of(frames, "conflict")]
    assert conflicts, "the blocked part must surface as a conflict"
    assert any("SKF-6310-2RS1" in c for c in conflicts)
    # Every conflict on the wire is also in the result — one source of truth.
    assert conflicts == only(frames, "result").data.conflicts_surfaced


# ---------------------------------------------------------------------------
# Narrative streaming
# ---------------------------------------------------------------------------
FAITHFUL_NARRATIVE = (
    "CV-201 has a degrading drive roller bearing. The repair needs part "
    "SKF-6310-2RS1, which is out of stock. Recommendation: REPAIR_NOW, subject "
    "to safety clearance."
)


@pytest.mark.asyncio
async def test_narrative_deltas_concatenate_to_the_final_narrative(fake_db, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    service = make_service(stream_llm=fake_narrative_stream(FAITHFUL_NARRATIVE))

    frames = await collect(service, user_role=UserRole.engineer)

    deltas = [f.data.text for f in events_of(frames, "narrative_delta")]
    result = only(frames, "result").data

    assert len(deltas) > 1, "the answer must arrive in pieces, not one block"
    assert "".join(deltas) == result.narrative
    assert NarrativeSource(result.narrative_source) == NarrativeSource.llm
    assert not events_of(frames, "error")


@pytest.mark.asyncio
async def test_template_narrative_is_streamed_as_chunks_when_no_llm_is_available(
    fake_db,
):
    frames = await collect(make_service(), user_role=UserRole.engineer)

    deltas = [f.data.text for f in events_of(frames, "narrative_delta")]
    result = only(frames, "result").data

    assert deltas, "a client must see narrative_delta whether or not an LLM ran"
    assert "".join(deltas) == result.narrative
    assert NarrativeSource(result.narrative_source) == NarrativeSource.template


@pytest.mark.asyncio
async def test_failed_number_validation_emits_a_recoverable_error_then_the_template(
    fake_db, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    hallucinated = (
        "The conveyor bearing is failing. Repairing it will cost $47,500 and "
        "take 14 days, so I recommend waiting."
    )
    service = make_service(stream_llm=fake_narrative_stream(hallucinated))

    frames = await collect(service, user_role=UserRole.engineer)
    types = event_types(frames)

    errors = events_of(frames, "error")
    assert len(errors) == 1
    assert errors[0].data.recoverable is True

    deltas = [f.data.text for f in events_of(frames, "narrative_delta")]
    error_index = types.index("error")
    before = [
        f.data.text
        for f in events_of(frames[:error_index], "narrative_delta")
    ]
    after = [
        f.data.text
        for f in events_of(frames[error_index:], "narrative_delta")
    ]

    result = only(frames, "result").data
    assert "".join(before) == hallucinated, "the discarded prose was already streamed"
    assert "".join(after) == result.narrative, "the template replaces it"
    assert NarrativeSource(result.narrative_source) == NarrativeSource.template
    assert "47,500" not in result.narrative
    assert len(deltas) == len(before) + len(after)
    # The error is recoverable: the stream still finishes normally.
    assert types[-2:] == ["result", "done"]


@pytest.mark.asyncio
async def test_composer_failure_falls_back_to_the_template_without_a_stream(
    fake_db, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    async def exploding(_prompt, *, system, model="", **_kw):
        raise RuntimeError("model overloaded")
        yield ""  # pragma: no cover - makes this an async generator

    service = make_service(stream_llm=exploding)
    frames = await collect(service)

    result = only(frames, "result").data
    deltas = [f.data.text for f in events_of(frames, "narrative_delta")]

    assert NarrativeSource(result.narrative_source) == NarrativeSource.template
    assert "".join(deltas) == result.narrative
    # Nothing was streamed before the failure, so there is nothing to retract.
    assert not events_of(frames, "error")


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------
GERMAN_MESSAGE = (
    "Das Förderband CV-201 macht seit gestern ein sehr lautes Schleifgeräusch "
    "und wird immer wärmer."
)

GERMAN_NARRATIVE = (
    "Das Förderband CV-201 zeigt einen Lagerschaden am Antriebsrollenlager "
    "(BEARING_WEAR). Das benötigte Ersatzteil SKF-6310-2RS1 ist nicht auf "
    "Lager. Empfehlung: REPAIR_NOW."
)


@pytest.mark.asyncio
async def test_non_english_input_is_answered_in_that_language_when_a_key_is_set(
    fake_db, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls: list[dict] = []
    service = make_service(
        stream_llm=fake_narrative_stream(GERMAN_NARRATIVE, calls=calls)
    )

    frames = await collect(service, message=GERMAN_MESSAGE, user_role=UserRole.engineer)
    result = only(frames, "result").data

    assert result.detected_language == "de"
    assert result.language_fallback is False
    assert NarrativeSource(result.narrative_source) == NarrativeSource.llm
    assert result.narrative == GERMAN_NARRATIVE

    # The composer was told the language, and told not to translate identifiers.
    assert "German" in calls[0]["system"]
    assert "never translated" in calls[0]["system"]

    # Structured data is language-neutral and survives untouched.
    assert "SKF-6310-2RS1" in result.narrative
    assert "BEARING_WEAR" in result.narrative
    assert "CV-201" in result.narrative
    assert result.inventory.blocking_parts == ["SKF-6310-2RS1"]
    assert result.rca.primary_cause.fault_mode == "BEARING_WEAR"


@pytest.mark.asyncio
async def test_language_fallback_is_marked_when_the_template_answers_in_english(
    fake_db,
):
    frames = await collect(make_service(), message=GERMAN_MESSAGE)
    result = only(frames, "result").data

    assert result.detected_language == "de"
    assert result.language_fallback is True, (
        "an English template answering a German operator must say so"
    )
    assert NarrativeSource(result.narrative_source) == NarrativeSource.template
    # The template is not machine-translated: it is still English.
    assert "SAFETY BRIEFING" in result.narrative


@pytest.mark.asyncio
async def test_english_input_never_sets_the_language_fallback_flag(fake_db):
    frames = await collect(make_service())
    result = only(frames, "result").data

    assert result.detected_language == "en"
    assert result.language_fallback is False


@pytest.mark.asyncio
async def test_a_declared_language_overrides_detection(fake_db, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls: list[dict] = []
    service = make_service(
        stream_llm=fake_narrative_stream(GERMAN_NARRATIVE, calls=calls)
    )

    frames = await collect(
        service, message="CV-201 is grinding", language="de", user_role=UserRole.engineer
    )
    result = only(frames, "result").data

    assert result.detected_language == "de"
    assert "German" in calls[0]["system"]


def test_identifier_heavy_text_is_not_mistaken_for_a_language():
    message = "CV-201 E4471 SKF-6310-2RS1 BEARING_WEAR 8.2mm/s 47.5C"
    assert linguistic_text(message).strip() == ""

    detection = detect_language(message)
    assert detection.language == "en"
    assert LanguageSource(detection.source) == LanguageSource.default
    assert "identifiers were removed" in detection.reason


def test_declared_language_is_normalised_and_trusted():
    detection = detect_language("anything at all", declared="de-DE")
    assert detection.language == "de"
    assert LanguageSource(detection.source) == LanguageSource.declared
    assert detection.is_english is False


def test_german_is_detected_from_a_realistic_fault_report():
    detection = detect_language(GERMAN_MESSAGE)
    assert detection.language == "de"
    assert LanguageSource(detection.source) == LanguageSource.detected


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_client_disconnect_cancels_in_flight_module_tasks(fake_db, caplog):
    cancelled: list[str] = []

    async def slow_rca(context: ExecutionContext) -> ModuleOutput:
        try:
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            cancelled.append("RCA")
            raise
        return ModuleOutput(status=ModuleStatus.ok, data=RCA_DATA)  # pragma: no cover

    runners = healthy_runners(**{ModuleName.rca: slow_rca})
    service = make_service(make_orchestrator(runners=runners))

    stream = service.stream(tenant_id=TENANT, message="CV-201 is grinding")
    with caplog.at_level("WARNING"):
        async for frame in stream:
            if (
                isinstance(frame, StreamEvent)
                and str(frame.type) == "module_start"
                and frame.data.module == "RCA"
            ):
                break  # the operator closed the tab mid-analysis
        await stream.aclose()

    assert cancelled == ["RCA"], "the in-flight module must be cancelled, not awaited"
    assert any("cancelling" in record.message.lower() for record in caplog.records)


@pytest.mark.asyncio
async def test_a_stream_abandoned_after_the_first_event_still_cancels_cleanly(fake_db):
    runners = healthy_runners(**{ModuleName.rag: stub_runner(RAG_DATA, delay=2.0)})
    service = make_service(make_orchestrator(runners=runners))

    stream = service.stream(tenant_id=TENANT, message="CV-201 is grinding")
    first = await stream.__anext__()
    assert str(first.type) == "session"
    await stream.aclose()  # must not hang or raise


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_heartbeat_comments_are_emitted_while_work_is_slow(fake_db):
    runners = healthy_runners(
        **{ModuleName.rag: stub_runner(RAG_DATA, citations=RAG_CITATIONS, delay=0.2)}
    )
    service = make_service(
        make_orchestrator(runners=runners), heartbeat_seconds=0.02
    )

    frames = await collect(service)
    comments = [f for f in frames if isinstance(f, SSEComment)]

    assert comments, "a slow orchestration must keep the connection warm"
    assert comments[0].to_sse() == ": heartbeat\n\n"
    # Heartbeats never displace real events.
    assert event_types(frames)[-1] == "done"


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------
def test_sse_frames_carry_both_a_named_event_and_a_typed_body():
    frame = StreamEvent.for_narrative_delta("hello")
    text = frame.to_sse()

    assert text.startswith("event: narrative_delta\ndata: ")
    assert text.endswith("\n\n")

    body = json.loads(text.split("data: ", 1)[1].strip())
    assert body == {"type": "narrative_delta", "data": {"text": "hello"}}


def test_template_chunking_reproduces_the_text_exactly():
    text = "\n\n".join(f"BLOCK {i}\n  detail line" for i in range(20))
    chunks = chunk_text(text, max_chunks=4)

    assert 1 < len(chunks) <= 4
    assert "".join(chunks) == text
    assert chunk_text("") == []


def test_module_summaries_degrade_to_the_reason_when_there_is_no_result():
    summary = summarize_module(
        ModuleName.pdm,
        ModuleStatus.unavailable,
        None,
        "No trained artifacts for this machine model.",
    )
    assert summary == "No trained artifacts for this machine model."


def test_module_summaries_survive_a_malformed_payload():
    summary = summarize_module(ModuleName.rca, ModuleStatus.ok, {"unexpected": True})
    assert summary, "a broken payload must not produce an empty timeline row"


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_streams_are_tenant_scoped(fake_db):
    service = make_service(make_orchestrator(resolver=real_resolver()))

    mine = await collect(service, tenant_id=TENANT, message="CV-201 is grinding")
    assert only(mine, "resolution").data.status == "RESOLVED"
    assert only(mine, "result").data.machine.machine_id == "CV-201"

    # The other tenant's fleet does not contain CV-201 and must not see it.
    theirs = await collect(
        service, tenant_id=OTHER_TENANT, message="CV-201 is grinding"
    )
    assert only(theirs, "resolution").data.status == "NOT_FOUND"
    assert only(theirs, "result").data.machine is None
    assert "module_start" not in event_types(theirs)

    # And the reverse.
    cross = await collect(service, tenant_id=TENANT, message="PR-900 is down")
    assert only(cross, "resolution").data.status == "NOT_FOUND"


@pytest.mark.asyncio
async def test_stream_sessions_are_tenant_scoped(fake_db):
    service = make_service()
    await collect(service, tenant_id=TENANT, session_id="shared-id")

    store = service.orchestrator.sessions
    assert await store.load(TENANT, "shared-id") is not None
    assert await store.load(OTHER_TENANT, "shared-id") is None
    assert await store.delete(OTHER_TENANT, "shared-id") is False
    assert await store.load(TENANT, "shared-id") is not None


@pytest.mark.asyncio
async def test_module_execution_carries_the_streaming_request_tenant(fake_db):
    seen: list[str] = []

    async def tenant_recording_runner(context: ExecutionContext) -> ModuleOutput:
        seen.append(context.tenant_id)
        return ModuleOutput(status=ModuleStatus.ok, data=RCA_DATA)

    runners = healthy_runners(**{ModuleName.rca: tenant_recording_runner})
    await collect(
        make_service(make_orchestrator(runners=runners)), tenant_id=OTHER_TENANT
    )
    assert seen == [OTHER_TENANT]


# ---------------------------------------------------------------------------
# Works entirely offline
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_works_entirely_without_an_anthropic_api_key(monkeypatch, fake_db):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # A default router and no injected composer: both must fall back rather
    # than fail, and the stream must be indistinguishable in shape.
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

    frames = await collect(ChatService(orchestrator), user_role=UserRole.technician)
    types = event_types(frames)
    result = only(frames, "result").data

    assert types[0] == "session" and types[-1] == "done"
    for expected in (
        "routing",
        "resolution",
        "module_start",
        "module_finish",
        "narrative_delta",
        "result",
    ):
        assert expected in types, f"'{expected}' must be emitted without a key"

    assert NarrativeSource(result.narrative_source) == NarrativeSource.template
    assert result.narrative.strip()
    assert "".join(f.data.text for f in events_of(frames, "narrative_delta")) == (
        result.narrative
    )


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------
@pytest.fixture
def client(fake_db):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.core.tenant_context import get_current_tenant
    from app.routers.chat import router as chat_router

    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[get_current_tenant] = lambda: TENANT
    return TestClient(app)


def _install(service: ChatService) -> ChatService:
    """Make ``service`` (and its orchestrator) the process-wide instances."""
    set_orchestrator(service.orchestrator)
    set_chat_service(service)
    return service


def test_stream_endpoint_returns_an_event_stream(client):
    _install(make_service())

    response = client.post(
        "/chat/stream",
        json={"message": "CV-201 is grinding", "user_role": "ENGINEER"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"

    events = parse_sse(response.text)
    types = [event["type"] for event in events]
    assert types[0] == "session"
    assert types[-1] == "done"
    assert "module_finish" in types

    narrative = "".join(
        e["data"]["text"] for e in events if e["type"] == "narrative_delta"
    )
    result = next(e for e in events if e["type"] == "result")
    assert narrative == result["data"]["narrative"]
    assert "event: module_finish" in response.text


def test_stream_endpoint_reports_a_clarification_without_running_modules(client):
    _install(
        make_service(
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
    )

    response = client.post("/chat/stream", json={"message": "the conveyor is noisy"})
    events = parse_sse(response.text)
    types = [event["type"] for event in events]

    assert response.status_code == 200
    assert types == ["session", "routing", "resolution", "result", "done"]
    resolution = next(e for e in events if e["type"] == "resolution")["data"]
    assert len(resolution["candidates"]) == 2
    assert resolution["clarification_question"] == "Do you mean CV-201 or CV-202?"


def test_chat_endpoint_returns_the_complete_result(client):
    _install(make_service())

    response = client.post(
        "/chat",
        json={"message": "CV-201 is grinding", "user_role": "ENGINEER"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("COMPLETE", "PARTIAL")
    assert body["machine"]["machine_id"] == "CV-201"
    assert body["narrative"]
    assert body["detected_language"] == "en"
    assert body["language_fallback"] is False
    assert {row["name"] for row in body["modules_run"]} >= {"RCA", "SAFETY"}


def test_chat_endpoint_returns_partial_not_a_500_when_a_module_fails(client):
    runners = healthy_runners(
        **{ModuleName.rca: stub_runner(raises=RuntimeError("rca exploded"))}
    )
    _install(make_service(make_orchestrator(runners=runners)))

    response = client.post("/chat", json={"message": "CV-201 is grinding"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "PARTIAL"
    rca_row = next(r for r in body["modules_run"] if r["name"] == "RCA")
    assert rca_row["status"] == "UNAVAILABLE"
    assert "rca exploded" not in rca_row["reason"]
    assert rca_row["error_detail"] == "RuntimeError: rca exploded"


def test_session_endpoints_read_and_delete_within_the_tenant(client):
    _install(make_service())

    client.post(
        "/chat/stream",
        json={"message": "CV-201 is grinding", "session_id": "http-sess"},
    )

    ok = client.get("/chat/sessions/http-sess")
    assert ok.status_code == 200
    body = ok.json()
    assert body["last_machine_id"] == "CV-201"
    assert len(body["turns"]) == 1
    assert set(body["cached_modules"]) >= {"RCA", "SAFETY"}

    assert client.get("/chat/sessions/never-existed").status_code == 404

    assert client.delete("/chat/sessions/http-sess").status_code == 204
    assert client.get("/chat/sessions/http-sess").status_code == 404
    assert client.delete("/chat/sessions/http-sess").status_code == 404


def test_session_endpoints_are_invisible_across_tenants(client, fake_db):
    from app.core.tenant_context import get_current_tenant

    _install(make_service())
    client.post(
        "/chat/stream",
        json={"message": "CV-201 is grinding", "session_id": "cross-tenant"},
    )
    assert client.get("/chat/sessions/cross-tenant").status_code == 200

    client.app.dependency_overrides[get_current_tenant] = lambda: OTHER_TENANT
    assert client.get("/chat/sessions/cross-tenant").status_code == 404
    assert client.delete("/chat/sessions/cross-tenant").status_code == 404

    # Still there for the tenant that owns it.
    client.app.dependency_overrides[get_current_tenant] = lambda: TENANT
    assert client.get("/chat/sessions/cross-tenant").status_code == 200
