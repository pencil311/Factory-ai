"""Tests for the four domain agents.

No live Mongo, no LLM: agents are tested against in-memory data using the
FakeDatabase pattern from test_rag.py. Each agent must return its typed shape,
never fabricate on missing data, and work without ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from app.agents.base import Agent
from app.agents.inventory_agent import InventoryAgent
from app.agents.maintenance_agent import MaintenanceAgent
from app.agents.production_agent import ProductionAgent
from app.agents.safety_agent import SafetyAgent
from app.schemas.agents import (
    AgentContext,
    AgentStatus,
    InventoryStatus,
    MaintenancePlan,
    ProductionImpact,
    SafetyBriefing,
    SafetySource,
    StockStatus,
)
from app.schemas.machine import COLLECTIONS

TENANT = "demo"

# ---------------------------------------------------------------------------
# Fake Mongo (same pattern as test_rag.py)
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


class FakeDatabase:
    def __init__(self):
        self._collections: dict[str, FakeCollection] = {}

    def __getitem__(self, name):
        return self._collections.setdefault(name, FakeCollection(name))

    def add_collection(self, name, docs):
        self._collections[name] = FakeCollection(name, docs)


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------
MACHINE_CV201 = {
    "tenant_id": TENANT, "machine_id": "CV-201", "name": "Infeed Belt Conveyor",
    "model": "SpanTech SB-3000", "manufacturer": "SpanTech",
    "site_id": "SITE-DETROIT", "line_id": "LINE-A", "position_in_line": 1,
    "criticality": 3, "status": "running",
    "units_per_hour": 600, "cost_per_hour_downtime": 450.0,
}

MACHINE_MC110 = {
    "tenant_id": TENANT, "machine_id": "MC-110", "name": "3-Axis CNC Milling Center",
    "model": "Haas VF-4", "manufacturer": "Haas Automation",
    "site_id": "SITE-DETROIT", "line_id": "LINE-A", "position_in_line": 2,
    "criticality": 5, "status": "running",
    "units_per_hour": 30, "cost_per_hour_downtime": 1200.0,
}

MACHINE_HP150 = {
    "tenant_id": TENANT, "machine_id": "HP-150", "name": "150-Ton Hydraulic Press",
    "model": "Beckwood BX-150", "manufacturer": "Beckwood Press",
    "site_id": "SITE-DETROIT", "line_id": "LINE-A", "position_in_line": 4,
    "criticality": 5, "status": "fault",
    "units_per_hour": 120, "cost_per_hour_downtime": 950.0,
}

COMPONENT_BRG = {
    "tenant_id": TENANT, "component_id": "CV-201-BRG-D", "machine_id": "CV-201",
    "name": "Drive Roller Bearing", "type": "bearing", "part_number": "SKF-6310-2RS1",
}

COMPONENT_SPN_BRG = {
    "tenant_id": TENANT, "component_id": "MC-110-SPN-BRG", "machine_id": "MC-110",
    "name": "Spindle Bearing Pack", "type": "bearing", "part_number": "NSK-7014A5-P4",
}

COMPONENT_CYL = {
    "tenant_id": TENANT, "component_id": "HP-150-CYL", "machine_id": "HP-150",
    "name": "Main Ram Cylinder", "type": "cylinder", "part_number": "BW-CYL-150T",
}

PART_SKF = {
    "tenant_id": TENANT, "part_number": "SKF-6310-2RS1",
    "description": "Deep groove ball bearing", "category": "bearing",
    "compatible_components": ["CV-201-BRG-D"],
    "compatible_machine_models": ["SpanTech SB-3000"],
    "quantity_on_hand": 4, "reorder_level": 2,
    "warehouse_location": "A-03-12", "unit_cost": 42.50, "lead_time_days": 3,
    "supplier": "SKF", "alternative_part_numbers": [],
}

PART_NSK_OOS = {
    "tenant_id": TENANT, "part_number": "NSK-7014A5-P4",
    "description": "Spindle bearing pack", "category": "bearing",
    "compatible_components": ["MC-110-SPN-BRG"],
    "compatible_machine_models": ["Haas VF-4"],
    "quantity_on_hand": 0, "reorder_level": 1,  # OUT OF STOCK
    "warehouse_location": "A-05-02", "unit_cost": 1850.00, "lead_time_days": 21,
    "supplier": "NSK", "alternative_part_numbers": ["SKF-7014A5-P4"],
}

PART_SKF_ALT = {
    "tenant_id": TENANT, "part_number": "SKF-7014A5-P4",
    "description": "Angular contact bearing (alternative)", "category": "bearing",
    "compatible_components": ["MC-110-SPN-BRG"],
    "compatible_machine_models": ["Haas VF-4"],
    "quantity_on_hand": 1, "reorder_level": 1,
    "warehouse_location": "A-05-03", "unit_cost": 1920.00, "lead_time_days": 14,
    "supplier": "SKF", "alternative_part_numbers": [],
}

PART_SEAL = {
    "tenant_id": TENANT, "part_number": "BW-SEAL-KIT-150",
    "description": "Seal kit for BX-150 cylinder", "category": "seal",
    "compatible_components": ["HP-150-CYL"],
    "compatible_machine_models": ["Beckwood BX-150"],
    "quantity_on_hand": 3, "reorder_level": 2,
    "warehouse_location": "D-02-03", "unit_cost": 280.00, "lead_time_days": 7,
    "supplier": "Beckwood", "alternative_part_numbers": [],
}


@pytest.fixture
def fake_db(monkeypatch):
    """Route tenant scope to an in-memory database with seed data."""
    database = FakeDatabase()
    database.add_collection(COLLECTIONS.machines, [MACHINE_CV201, MACHINE_MC110, MACHINE_HP150])
    database.add_collection(COLLECTIONS.components, [COMPONENT_BRG, COMPONENT_SPN_BRG, COMPONENT_CYL])
    database.add_collection(COLLECTIONS.parts, [PART_SKF, PART_NSK_OOS, PART_SKF_ALT, PART_SEAL])

    from app import db as db_module
    monkeypatch.setattr(db_module, "get_database", lambda: database)
    return database


def _bearing_rca():
    return {
        "machine_id": "CV-201",
        "primary_cause": {
            "cause_id": "RCA-BEARING-001",
            "description": "Bearing wear",
            "component_id": "CV-201-BRG-D",
            "fault_mode": "BEARING_WEAR",
            "probability": 0.85,
            "supporting_evidence_ids": ["SIG-VIB"],
            "contradicting_evidence_ids": [],
        },
        "confidence": 0.7,
    }


def _spindle_rca():
    return {
        "machine_id": "MC-110",
        "primary_cause": {
            "cause_id": "RCA-BEARING-002",
            "description": "Spindle bearing wear",
            "component_id": "MC-110-SPN-BRG",
            "fault_mode": "BEARING_WEAR",
            "probability": 0.8,
            "supporting_evidence_ids": ["SIG-VIB"],
            "contradicting_evidence_ids": [],
        },
        "confidence": 0.6,
    }


def _seal_rca():
    return {
        "machine_id": "HP-150",
        "primary_cause": {
            "cause_id": "RCA-SEAL-001",
            "description": "Seal leak on hydraulic circuit",
            "component_id": "HP-150-CYL",
            "fault_mode": "SEAL_LEAK",
            "probability": 0.75,
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
        },
        "confidence": 0.65,
    }


# ---------------------------------------------------------------------------
# Maintenance Agent
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_maintenance_returns_typed_shape(fake_db):
    ctx = AgentContext(tenant_id=TENANT, machine_id="CV-201", rca_result=_bearing_rca())
    agent = MaintenanceAgent()
    result = await agent.run(ctx)

    assert result.agent_name == "maintenance"
    assert result.status == AgentStatus.ok
    data = result.data
    assert isinstance(data, dict)
    assert "procedure_steps" in data
    assert "required_tools" in data
    assert data["total_estimated_minutes"] > 0
    steps = data["procedure_steps"]
    assert len(steps) > 0
    # Steps should be ordered
    orders = [s["order"] for s in steps]
    assert orders == sorted(orders)


@pytest.mark.asyncio
async def test_maintenance_unavailable_without_rca(fake_db):
    ctx = AgentContext(tenant_id=TENANT, machine_id="CV-201")
    agent = MaintenanceAgent()
    result = await agent.run(ctx)

    assert result.status == AgentStatus.unavailable
    assert result.reason


@pytest.mark.asyncio
async def test_maintenance_works_without_anthropic_key(fake_db, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ctx = AgentContext(tenant_id=TENANT, machine_id="CV-201", rca_result=_bearing_rca())
    agent = MaintenanceAgent()
    result = await agent.run(ctx)

    assert result.status == AgentStatus.ok


# ---------------------------------------------------------------------------
# Inventory Agent
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_inventory_finds_alternative_for_oos_part(fake_db):
    """The NSK spindle bearing is OUT_OF_STOCK but SKF alternative is in stock."""
    ctx = AgentContext(tenant_id=TENANT, machine_id="MC-110", rca_result=_spindle_rca())
    agent = InventoryAgent()
    result = await agent.run(ctx)

    assert result.status == AgentStatus.ok
    data = result.data
    assert isinstance(data, dict)
    items = data["items"]

    # Find the OOS part
    oos = [i for i in items if i["part_number"] == "NSK-7014A5-P4"]
    assert len(oos) == 1
    assert oos[0]["status"] == StockStatus.out_of_stock
    assert oos[0]["available_qty"] == 0
    # Alternative should be listed
    assert "SKF-7014A5-P4" in oos[0]["alternatives"]


@pytest.mark.asyncio
async def test_inventory_reports_in_stock_correctly(fake_db):
    ctx = AgentContext(tenant_id=TENANT, machine_id="CV-201", rca_result=_bearing_rca())
    agent = InventoryAgent()
    result = await agent.run(ctx)

    assert result.status == AgentStatus.ok
    data = result.data
    items = data["items"]
    skf = [i for i in items if i["part_number"] == "SKF-6310-2RS1"]
    assert len(skf) == 1
    assert skf[0]["status"] == StockStatus.in_stock
    assert skf[0]["available_qty"] == 4
    assert data["all_parts_available"] is True


@pytest.mark.asyncio
async def test_inventory_unavailable_without_rca(fake_db):
    ctx = AgentContext(tenant_id=TENANT, machine_id="CV-201")
    agent = InventoryAgent()
    result = await agent.run(ctx)

    assert result.status == AgentStatus.unavailable


@pytest.mark.asyncio
async def test_inventory_tenant_isolation(fake_db):
    """Inventory for a different tenant should find no parts."""
    ctx = AgentContext(tenant_id="acme", machine_id="CV-201", rca_result=_bearing_rca())
    agent = InventoryAgent()
    result = await agent.run(ctx)

    # No parts for acme tenant
    if result.status == AgentStatus.ok:
        assert result.data["items"] == [] or len(result.data["items"]) == 0


@pytest.mark.asyncio
async def test_inventory_works_without_anthropic_key(fake_db, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ctx = AgentContext(tenant_id=TENANT, machine_id="CV-201", rca_result=_bearing_rca())
    agent = InventoryAgent()
    result = await agent.run(ctx)

    assert result.status == AgentStatus.ok


# ---------------------------------------------------------------------------
# Safety Agent
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_safety_returns_generic_rather_than_unavailable(fake_db):
    """Safety must NEVER return UNAVAILABLE for a known machine."""
    ctx = AgentContext(tenant_id=TENANT, machine_id="CV-201", rca_result=_bearing_rca())
    agent = SafetyAgent()
    result = await agent.run(ctx)

    assert result.status == AgentStatus.ok
    data = result.data
    assert isinstance(data, dict)
    assert len(data["hazards"]) > 0
    assert len(data["required_ppe"]) > 0
    assert len(data["lockout_tagout_steps"]) > 0
    assert len(data["energy_sources_to_isolate"]) > 0
    # Routine LOTO/PPE preconditions apply to every job and always appear...
    assert len(data["standard_preconditions"]) > 0
    # ...but a bearing wear fault carries no CRITICAL hazard, so nothing here
    # is a genuine blocking condition. See test_safety_blocking_conditions_
    # reflect_genuine_machine_state below for the contrast with a fault that
    # does produce one.
    assert data["blocking_conditions"] == []


@pytest.mark.asyncio
async def test_safety_blocking_conditions_reflect_genuine_machine_state(fake_db):
    """blocking_conditions is reserved for genuine, machine-state hazards.

    A bearing-wear fault (component type "bearing") only produces MEDIUM
    hazards, so it must not lead with a warning banner. A seal leak on a
    cylinder produces a CRITICAL pressurized-fluid hazard, which must.
    """
    routine = await SafetyAgent().run(
        AgentContext(tenant_id=TENANT, machine_id="CV-201", rca_result=_bearing_rca())
    )
    assert routine.status == AgentStatus.ok
    assert routine.data["blocking_conditions"] == []
    assert routine.data["standard_preconditions"], "routine LOTO steps still surface"
    assert all(
        h["severity"] != "CRITICAL" for h in routine.data["hazards"]
    ), "bearing hazards are MEDIUM — nothing here should be treated as blocking"

    leaking = await SafetyAgent().run(
        AgentContext(tenant_id=TENANT, machine_id="HP-150", rca_result=_seal_rca())
    )
    assert leaking.status == AgentStatus.ok
    assert leaking.data["blocking_conditions"], "a pressurised, unresolved hazard blocks work"
    assert any(
        "pressur" in condition.lower() for condition in leaking.data["blocking_conditions"]
    )
    # Standard preconditions still show up alongside the genuine blocker.
    assert leaking.data["standard_preconditions"]


@pytest.mark.asyncio
async def test_safety_returns_generic_without_rca(fake_db):
    """Even without RCA, safety should return generic precautions."""
    ctx = AgentContext(tenant_id=TENANT, machine_id="CV-201")
    agent = SafetyAgent()
    result = await agent.run(ctx)

    assert result.status == AgentStatus.ok
    data = result.data
    assert len(data["hazards"]) > 0
    assert data["source"] == SafetySource.generic


@pytest.mark.asyncio
async def test_safety_seal_leak_adds_pressure_hazard(fake_db):
    """Seal leak should add pressurized fluid hazard."""
    ctx = AgentContext(tenant_id=TENANT, machine_id="HP-150", rca_result=_seal_rca())
    agent = SafetyAgent()
    result = await agent.run(ctx)

    assert result.status == AgentStatus.ok
    data = result.data
    hazard_types = [h["hazard_type"] for h in data["hazards"]]
    assert any("Pressurized" in ht or "pressurized" in ht.lower() or "Stored" in ht for ht in hazard_types)


@pytest.mark.asyncio
async def test_safety_unknown_machine_is_unavailable(fake_db):
    ctx = AgentContext(tenant_id=TENANT, machine_id="NOPE-999")
    agent = SafetyAgent()
    result = await agent.run(ctx)

    assert result.status == AgentStatus.unavailable


@pytest.mark.asyncio
async def test_safety_tenant_isolation(fake_db):
    """Safety for a different tenant should not find the machine."""
    ctx = AgentContext(tenant_id="acme", machine_id="CV-201")
    agent = SafetyAgent()
    result = await agent.run(ctx)

    assert result.status == AgentStatus.unavailable


@pytest.mark.asyncio
async def test_safety_works_without_anthropic_key(fake_db, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ctx = AgentContext(tenant_id=TENANT, machine_id="CV-201", rca_result=_bearing_rca())
    agent = SafetyAgent()
    result = await agent.run(ctx)

    assert result.status == AgentStatus.ok


# ---------------------------------------------------------------------------
# Production Agent
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_production_returns_typed_shape(fake_db):
    ctx = AgentContext(tenant_id=TENANT, machine_id="CV-201", rca_result=_bearing_rca())
    agent = ProductionAgent()
    result = await agent.run(ctx)

    assert result.agent_name == "production"
    assert result.status == AgentStatus.ok
    data = result.data
    assert isinstance(data, dict)
    assert "downtime_estimate_minutes" in data
    assert "units_lost_estimate" in data
    assert "is_bottleneck" in data
    assert "cost_estimate" in data
    assert "recommendation" in data
    assert "assumptions" in data
    assert len(data["assumptions"]) > 0


@pytest.mark.asyncio
async def test_production_flags_bottleneck_machine(fake_db):
    """MC-110 has criticality 5 and position 2 — it should be a bottleneck."""
    ctx = AgentContext(
        tenant_id=TENANT, machine_id="MC-110",
        rca_result=_spindle_rca(),
        pdm_result={"failure_probability": 0.8},
    )
    agent = ProductionAgent()
    result = await agent.run(ctx)

    assert result.status == AgentStatus.ok
    data = result.data
    assert data["is_bottleneck"] is True
    # Downstream machines should be listed (HP-150 is at position 4, AC-301 at position 3)
    assert len(data["downstream_machines_affected"]) > 0


@pytest.mark.asyncio
async def test_production_recommendation_repair_now_for_critical(fake_db):
    """High failure probability on critical machine -> REPAIR_NOW."""
    ctx = AgentContext(
        tenant_id=TENANT, machine_id="HP-150",
        rca_result=_seal_rca(),
        pdm_result={"failure_probability": 0.85},
    )
    agent = ProductionAgent()
    result = await agent.run(ctx)

    assert result.status == AgentStatus.ok
    assert result.data["recommendation"] == "REPAIR_NOW"


@pytest.mark.asyncio
async def test_production_unknown_machine(fake_db):
    ctx = AgentContext(tenant_id=TENANT, machine_id="NOPE-999", rca_result=_bearing_rca())
    agent = ProductionAgent()
    result = await agent.run(ctx)

    assert result.status == AgentStatus.unavailable


@pytest.mark.asyncio
async def test_production_works_without_anthropic_key(fake_db, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ctx = AgentContext(tenant_id=TENANT, machine_id="CV-201", rca_result=_bearing_rca())
    agent = ProductionAgent()
    result = await agent.run(ctx)

    assert result.status == AgentStatus.ok


@pytest.mark.asyncio
async def test_production_tenant_isolation(fake_db):
    ctx = AgentContext(tenant_id="acme", machine_id="CV-201", rca_result=_bearing_rca())
    agent = ProductionAgent()
    result = await agent.run(ctx)

    assert result.status == AgentStatus.unavailable


# ---------------------------------------------------------------------------
# Cross-agent: all return their shape, never fabricate
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_all_agents_return_their_shape(fake_db):
    rca = _bearing_rca()
    ctx = AgentContext(tenant_id=TENANT, machine_id="CV-201", rca_result=rca)

    for AgentClass in (MaintenanceAgent, InventoryAgent, SafetyAgent, ProductionAgent):
        agent = AgentClass()
        result = await agent.run(ctx)
        assert result.agent_name in ("maintenance", "inventory", "safety", "production")
        assert result.status in (AgentStatus.ok, AgentStatus.partial, AgentStatus.unavailable)
        assert result.elapsed_ms >= 0
