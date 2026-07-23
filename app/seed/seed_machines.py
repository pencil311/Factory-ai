"""Idempotent seed: two tenants, with a fleet under the demo tenant.

Run with:

    python -m app.seed.seed_machines

Creates two tenants:

* ``demo``  — Demo Manufacturing, owns the whole seeded fleet.
* ``acme``  — Acme Industrial, deliberately EMPTY.

The empty tenant is the point: isolation you cannot observe is isolation you
cannot trust. With ``acme`` present, ``GET /machines`` with
``X-Tenant-Id: acme`` returning ``[]`` while ``demo`` returns four machines is
a demonstration rather than an assertion.

Every document is upserted on ``(tenant_id, canonical key)`` through the tenant
scope, so re-running converges to the same state rather than duplicating.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app import db
from app.core.tenant_context import clear_tenant_cache
from app.models.machine import (
    Component,
    ComponentType,
    ErrorCode,
    Machine,
    MachineStatus,
    ProductionLine,
    Sensor,
    SensorType,
    Site,
)
from app.models.tenant import Tenant, TenantSettings, UnitSystem
from app.schemas.machine import COLLECTIONS

#: The tenant that owns everything below.
DEMO_TENANT_ID = "demo"

TENANTS: list[Tenant] = [
    Tenant(
        tenant_id=DEMO_TENANT_ID,
        name="Demo Manufacturing",
        slug="demo",
        settings=TenantSettings(timezone="America/Detroit", units=UnitSystem.metric),
    ),
    Tenant(
        tenant_id="acme",
        name="Acme Industrial",
        slug="acme",
        settings=TenantSettings(timezone="UTC", units=UnitSystem.imperial),
    ),
]


def _dt(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Site & line
# ---------------------------------------------------------------------------
SITE = Site(
    tenant_id=DEMO_TENANT_ID,
    site_id="SITE-DETROIT",
    name="Detroit Assembly Plant",
    location="Detroit, MI, USA",
    timezone="America/Detroit",
)

LINE = ProductionLine(
    tenant_id=DEMO_TENANT_ID,
    line_id="LINE-A",
    site_id=SITE.site_id,
    name="Line A — Machining & Finishing",
    description="Primary machining and finishing line feeding final assembly.",
)


# ---------------------------------------------------------------------------
# Machines
# ---------------------------------------------------------------------------
#: Production fields stored as extra keys on the machine document. The Machine
#: Pydantic model uses ``model_config = ConfigDict(extra="ignore")``, but Mongo
#: stores whatever we give it. The production agent reads these directly from
#: the document rather than hardcoding numbers.
_PRODUCTION_FIELDS: dict[str, dict] = {
    "CV-201": {"units_per_hour": 600, "cost_per_hour_downtime": 450.0},
    "MC-110": {"units_per_hour": 30, "cost_per_hour_downtime": 1200.0},
    "AC-301": {"units_per_hour": 0, "cost_per_hour_downtime": 800.0},   # utility, not direct production
    "HP-150": {"units_per_hour": 120, "cost_per_hour_downtime": 950.0},
}

MACHINES: list[Machine] = [
    Machine(
        tenant_id=DEMO_TENANT_ID,
        machine_id="CV-201",
        name="Infeed Belt Conveyor",
        model="SpanTech SB-3000",
        manufacturer="SpanTech",
        site_id=SITE.site_id,
        line_id=LINE.line_id,
        position_in_line=1,
        criticality=3,
        status=MachineStatus.running,
        aliases=["Infeed Conveyor", "Line A Belt 1", "ERP-CNV-0201"],
        installed_at=_dt(2019, 4, 12),
        last_maintenance_at=_dt(2026, 6, 2),
    ),
    Machine(
        tenant_id=DEMO_TENANT_ID,
        machine_id="MC-110",
        name="3-Axis CNC Milling Center",
        model="Haas VF-4",
        manufacturer="Haas Automation",
        site_id=SITE.site_id,
        line_id=LINE.line_id,
        position_in_line=2,
        criticality=5,
        status=MachineStatus.running,
        aliases=["VF-4 Mill", "Cell 2 CNC", "ERP-MILL-0110"],
        installed_at=_dt(2020, 9, 27),
        last_maintenance_at=_dt(2026, 7, 6),
    ),
    Machine(
        tenant_id=DEMO_TENANT_ID,
        machine_id="AC-301",
        name="Rotary Screw Air Compressor",
        model="Atlas Copco GA-75",
        manufacturer="Atlas Copco",
        site_id=SITE.site_id,
        line_id=LINE.line_id,
        position_in_line=3,
        criticality=4,
        status=MachineStatus.maintenance,
        aliases=["Plant Air Compressor", "GA-75 Unit", "ERP-COMP-0301"],
        installed_at=_dt(2018, 1, 18),
        last_maintenance_at=_dt(2026, 7, 15),
    ),
    Machine(
        tenant_id=DEMO_TENANT_ID,
        machine_id="HP-150",
        name="150-Ton Hydraulic Press",
        model="Beckwood BX-150",
        manufacturer="Beckwood Press",
        site_id=SITE.site_id,
        line_id=LINE.line_id,
        position_in_line=4,
        criticality=5,
        status=MachineStatus.fault,
        aliases=["Big Press", "Line A Press", "ERP-PRESS-0150"],
        installed_at=_dt(2017, 11, 3),
        last_maintenance_at=_dt(2026, 5, 21),
    ),
]


# ---------------------------------------------------------------------------
# Components (flat list; nesting expressed via parent_component_id)
# ---------------------------------------------------------------------------
COMPONENTS: list[Component] = [
    # --- CV-201 Belt Conveyor ---------------------------------------------
    Component(tenant_id=DEMO_TENANT_ID, component_id="CV-201-DRV", machine_id="CV-201", name="Drive Unit",
              type=ComponentType.other, part_number="ST-DRV-3000"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="CV-201-MTR", machine_id="CV-201", name="Drive Motor",
              type=ComponentType.motor, part_number="WEG-W22-5.5KW",
              parent_component_id="CV-201-DRV"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="CV-201-GBX", machine_id="CV-201", name="Right-Angle Gearbox",
              type=ComponentType.gearbox, part_number="NORD-SK9032.1",
              parent_component_id="CV-201-DRV"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="CV-201-DRLR", machine_id="CV-201", name="Drive Roller",
              type=ComponentType.roller, part_number="ST-RLR-114",
              parent_component_id="CV-201-DRV"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="CV-201-BRG-D", machine_id="CV-201", name="Drive Roller Bearing",
              type=ComponentType.bearing, part_number="SKF-6310-2RS1",
              parent_component_id="CV-201-DRLR"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="CV-201-BRG-T", machine_id="CV-201", name="Tail Roller Bearing",
              type=ComponentType.bearing, part_number="SKF-6308-2RS1"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="CV-201-BELT", machine_id="CV-201", name="Conveyor Belt",
              type=ComponentType.belt, part_number="HAB-PVC-1200-GRN"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="CV-201-FRM", machine_id="CV-201", name="Conveyor Frame",
              type=ComponentType.frame, part_number="ST-FRM-3000"),

    # --- MC-110 CNC Mill --------------------------------------------------
    Component(tenant_id=DEMO_TENANT_ID, component_id="MC-110-SPN", machine_id="MC-110", name="Spindle Assembly",
              type=ComponentType.spindle, part_number="HAAS-40T-8100"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="MC-110-SPN-MTR", machine_id="MC-110", name="Spindle Motor",
              type=ComponentType.motor, part_number="HAAS-VMTR-30HP",
              parent_component_id="MC-110-SPN"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="MC-110-SPN-BRG", machine_id="MC-110", name="Spindle Bearing Pack",
              type=ComponentType.bearing, part_number="NSK-7014A5-P4",
              parent_component_id="MC-110-SPN"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="MC-110-XAXIS", machine_id="MC-110", name="X-Axis Ballscrew Drive",
              type=ComponentType.other, part_number="HAAS-BS-X-40"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="MC-110-YAXIS", machine_id="MC-110", name="Y-Axis Ballscrew Drive",
              type=ComponentType.other, part_number="HAAS-BS-Y-40"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="MC-110-COOL", machine_id="MC-110", name="Coolant Pump",
              type=ComponentType.pump, part_number="GRUNDFOS-CR3-11"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="MC-110-CTRL", machine_id="MC-110", name="CNC Controller",
              type=ComponentType.controller, part_number="HAAS-NGC-CTRL"),

    # --- AC-301 Air Compressor -------------------------------------------
    Component(tenant_id=DEMO_TENANT_ID, component_id="AC-301-AIREND", machine_id="AC-301", name="Air End (Screw Element)",
              type=ComponentType.pump, part_number="AC-GA75-AE"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="AC-301-MTR", machine_id="AC-301", name="Main Drive Motor",
              type=ComponentType.motor, part_number="ABB-M3BP-75KW"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="AC-301-MTR-BRG", machine_id="AC-301", name="Motor Bearing",
              type=ComponentType.bearing, part_number="SKF-6314-C3",
              parent_component_id="AC-301-MTR"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="AC-301-OILP", machine_id="AC-301", name="Oil Injection Pump",
              type=ComponentType.pump, part_number="AC-OP-75"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="AC-301-INVLV", machine_id="AC-301", name="Inlet Valve",
              type=ComponentType.valve, part_number="AC-IV-75"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="AC-301-SEP", machine_id="AC-301", name="Oil Separator Tank",
              type=ComponentType.other, part_number="AC-SEP-75"),

    # --- HP-150 Hydraulic Press ------------------------------------------
    Component(tenant_id=DEMO_TENANT_ID, component_id="HP-150-HPU", machine_id="HP-150", name="Hydraulic Power Unit",
              type=ComponentType.other, part_number="BW-HPU-150"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="HP-150-PUMP", machine_id="HP-150", name="Main Hydraulic Pump",
              type=ComponentType.pump, part_number="BOSCH-A10VSO-140",
              parent_component_id="HP-150-HPU"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="HP-150-PUMP-MTR", machine_id="HP-150", name="Pump Drive Motor",
              type=ComponentType.motor, part_number="SIEMENS-1LE1-45KW",
              parent_component_id="HP-150-HPU"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="HP-150-CYL", machine_id="HP-150", name="Main Ram Cylinder",
              type=ComponentType.cylinder, part_number="BW-CYL-150T"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="HP-150-VLV", machine_id="HP-150", name="Directional Control Valve",
              type=ComponentType.valve, part_number="REXROTH-4WRA-10",
              parent_component_id="HP-150-HPU"),
    Component(tenant_id=DEMO_TENANT_ID, component_id="HP-150-CYL-BRG", machine_id="HP-150", name="Ram Guide Bearing",
              type=ComponentType.bearing, part_number="INA-GE-80-DO",
              parent_component_id="HP-150-CYL"),
]


# ---------------------------------------------------------------------------
# Sensors (3-5 per machine)
# ---------------------------------------------------------------------------
SENSORS: list[Sensor] = [
    # --- CV-201 ------------------------------------------------------------
    Sensor(tenant_id=DEMO_TENANT_ID, sensor_id="CV-201-TMP-01", machine_id="CV-201", component_id="CV-201-MTR",
           type=SensorType.temperature, unit="°C",
           normal_min=15, normal_max=70, warning_threshold=80, critical_threshold=95),
    Sensor(tenant_id=DEMO_TENANT_ID, sensor_id="CV-201-VIB-01", machine_id="CV-201", component_id="CV-201-BRG-D",
           type=SensorType.vibration, unit="mm/s",
           normal_min=0.0, normal_max=2.8, warning_threshold=4.5, critical_threshold=7.1),
    Sensor(tenant_id=DEMO_TENANT_ID, sensor_id="CV-201-RPM-01", machine_id="CV-201", component_id="CV-201-DRLR",
           type=SensorType.rpm, unit="rpm",
           normal_min=40, normal_max=90, warning_threshold=100, critical_threshold=110),
    Sensor(tenant_id=DEMO_TENANT_ID, sensor_id="CV-201-PWR-01", machine_id="CV-201", component_id="CV-201-MTR",
           type=SensorType.power, unit="kW",
           normal_min=0.5, normal_max=5.0, warning_threshold=5.8, critical_threshold=6.5),

    # --- MC-110 ------------------------------------------------------------
    Sensor(tenant_id=DEMO_TENANT_ID, sensor_id="MC-110-TMP-01", machine_id="MC-110", component_id="MC-110-SPN-BRG",
           type=SensorType.temperature, unit="°C",
           normal_min=18, normal_max=55, warning_threshold=65, critical_threshold=75),
    Sensor(tenant_id=DEMO_TENANT_ID, sensor_id="MC-110-VIB-01", machine_id="MC-110", component_id="MC-110-SPN",
           type=SensorType.vibration, unit="mm/s",
           normal_min=0.0, normal_max=1.8, warning_threshold=2.8, critical_threshold=4.5),
    Sensor(tenant_id=DEMO_TENANT_ID, sensor_id="MC-110-RPM-01", machine_id="MC-110", component_id="MC-110-SPN",
           type=SensorType.rpm, unit="rpm",
           normal_min=0, normal_max=8100, warning_threshold=8300, critical_threshold=8500),
    Sensor(tenant_id=DEMO_TENANT_ID, sensor_id="MC-110-PWR-01", machine_id="MC-110", component_id="MC-110-SPN-MTR",
           type=SensorType.power, unit="kW",
           normal_min=0.0, normal_max=22.0, warning_threshold=26.0, critical_threshold=30.0),
    Sensor(tenant_id=DEMO_TENANT_ID, sensor_id="MC-110-PRS-01", machine_id="MC-110", component_id="MC-110-COOL",
           type=SensorType.pressure, unit="bar",
           normal_min=3.0, normal_max=7.0, warning_threshold=8.5, critical_threshold=10.0),

    # --- AC-301 ------------------------------------------------------------
    Sensor(tenant_id=DEMO_TENANT_ID, sensor_id="AC-301-PRS-01", machine_id="AC-301", component_id="AC-301-SEP",
           type=SensorType.pressure, unit="bar",
           normal_min=6.0, normal_max=7.5, warning_threshold=8.0, critical_threshold=8.6),
    Sensor(tenant_id=DEMO_TENANT_ID, sensor_id="AC-301-TMP-01", machine_id="AC-301", component_id="AC-301-AIREND",
           type=SensorType.temperature, unit="°C",
           normal_min=60, normal_max=95, warning_threshold=105, critical_threshold=112),
    Sensor(tenant_id=DEMO_TENANT_ID, sensor_id="AC-301-VIB-01", machine_id="AC-301", component_id="AC-301-MTR-BRG",
           type=SensorType.vibration, unit="mm/s",
           normal_min=0.0, normal_max=2.8, warning_threshold=4.5, critical_threshold=7.1),
    Sensor(tenant_id=DEMO_TENANT_ID, sensor_id="AC-301-PWR-01", machine_id="AC-301", component_id="AC-301-MTR",
           type=SensorType.power, unit="kW",
           normal_min=10.0, normal_max=75.0, warning_threshold=82.0, critical_threshold=90.0),

    # --- HP-150 ------------------------------------------------------------
    Sensor(tenant_id=DEMO_TENANT_ID, sensor_id="HP-150-PRS-01", machine_id="HP-150", component_id="HP-150-CYL",
           type=SensorType.pressure, unit="bar",
           normal_min=0.0, normal_max=210.0, warning_threshold=230.0, critical_threshold=250.0),
    Sensor(tenant_id=DEMO_TENANT_ID, sensor_id="HP-150-TMP-01", machine_id="HP-150", component_id="HP-150-HPU",
           type=SensorType.temperature, unit="°C",
           normal_min=20, normal_max=60, warning_threshold=70, critical_threshold=80),
    Sensor(tenant_id=DEMO_TENANT_ID, sensor_id="HP-150-VIB-01", machine_id="HP-150", component_id="HP-150-PUMP",
           type=SensorType.vibration, unit="mm/s",
           normal_min=0.0, normal_max=3.5, warning_threshold=5.5, critical_threshold=8.0),
    Sensor(tenant_id=DEMO_TENANT_ID, sensor_id="HP-150-PWR-01", machine_id="HP-150", component_id="HP-150-PUMP-MTR",
           type=SensorType.power, unit="kW",
           normal_min=2.0, normal_max=45.0, warning_threshold=50.0, critical_threshold=55.0),
]


# ---------------------------------------------------------------------------
# Error codes (3-4 per machine model)
# ---------------------------------------------------------------------------
ERROR_CODES: list[ErrorCode] = [
    # --- SpanTech SB-3000 (CV-201) ---------------------------------------
    ErrorCode(tenant_id=DEMO_TENANT_ID, code="E101", machine_model="SpanTech SB-3000",
              description="Belt slip detected — drive/tail roller speed mismatch",
              fault_class="mechanical"),
    ErrorCode(tenant_id=DEMO_TENANT_ID, code="E102", machine_model="SpanTech SB-3000",
              description="Drive motor overtemperature trip",
              fault_class="electrical"),
    ErrorCode(tenant_id=DEMO_TENANT_ID, code="E103", machine_model="SpanTech SB-3000",
              description="Belt tracking fault — edge sensor tripped",
              fault_class="mechanical"),
    ErrorCode(tenant_id=DEMO_TENANT_ID, code="E104", machine_model="SpanTech SB-3000",
              description="Emergency stop circuit open",
              fault_class="safety"),

    # --- Haas VF-4 (MC-110) ----------------------------------------------
    ErrorCode(tenant_id=DEMO_TENANT_ID, code="A201", machine_model="Haas VF-4",
              description="Spindle overload — commanded torque exceeded limit",
              fault_class="mechanical"),
    ErrorCode(tenant_id=DEMO_TENANT_ID, code="A202", machine_model="Haas VF-4",
              description="Spindle bearing overtemperature",
              fault_class="mechanical"),
    ErrorCode(tenant_id=DEMO_TENANT_ID, code="A203", machine_model="Haas VF-4",
              description="Low coolant pressure / coolant pump fault",
              fault_class="hydraulic"),
    ErrorCode(tenant_id=DEMO_TENANT_ID, code="A204", machine_model="Haas VF-4",
              description="X-axis servo following error / overtravel",
              fault_class="electrical"),

    # --- Atlas Copco GA-75 (AC-301) --------------------------------------
    ErrorCode(tenant_id=DEMO_TENANT_ID, code="C301", machine_model="Atlas Copco GA-75",
              description="Element outlet temperature high shutdown",
              fault_class="thermal"),
    ErrorCode(tenant_id=DEMO_TENANT_ID, code="C302", machine_model="Atlas Copco GA-75",
              description="Separator vessel overpressure",
              fault_class="pneumatic"),
    ErrorCode(tenant_id=DEMO_TENANT_ID, code="C303", machine_model="Atlas Copco GA-75",
              description="Main motor overload / phase failure",
              fault_class="electrical"),
    ErrorCode(tenant_id=DEMO_TENANT_ID, code="C304", machine_model="Atlas Copco GA-75",
              description="Low oil pressure — injection pump fault",
              fault_class="lubrication"),

    # --- Beckwood BX-150 (HP-150) ----------------------------------------
    ErrorCode(tenant_id=DEMO_TENANT_ID, code="H401", machine_model="Beckwood BX-150",
              description="System pressure below setpoint — pump/relief fault",
              fault_class="hydraulic"),
    ErrorCode(tenant_id=DEMO_TENANT_ID, code="H402", machine_model="Beckwood BX-150",
              description="Hydraulic oil overtemperature",
              fault_class="hydraulic"),
    ErrorCode(tenant_id=DEMO_TENANT_ID, code="H403", machine_model="Beckwood BX-150",
              description="Ram position sensor out of range",
              fault_class="electrical"),
    ErrorCode(tenant_id=DEMO_TENANT_ID, code="H404", machine_model="Beckwood BX-150",
              description="Light curtain / two-hand control safety violation",
              fault_class="safety"),
]


async def _upsert_tenants(tenants: list[Tenant]) -> int:
    """Upsert the tenant registry itself.

    This is the one collection that is not tenant-scoped — it defines the
    scopes, so it cannot be written through one. ``created_at`` is preserved on
    re-runs: re-seeding must not rewrite when a tenant came into existence.
    """
    database = db.get_database()
    written = 0
    for tenant in tenants:
        doc = tenant.model_dump()
        existing = await database[COLLECTIONS.tenants].find_one(
            {"tenant_id": tenant.tenant_id}
        )
        if existing and existing.get("created_at"):
            doc["created_at"] = existing["created_at"]
        result = await database[COLLECTIONS.tenants].replace_one(
            {"tenant_id": tenant.tenant_id}, doc, upsert=True
        )
        written += (result.upserted_id is not None) + result.modified_count
    return written


async def _upsert(
    tenant_id: str, collection: str, key_fields: tuple[str, ...], docs: list[dict]
) -> int:
    """Replace-upsert ``docs`` into one tenant's ``collection``.

    Goes through the tenant scope, so the upsert key is
    ``(tenant_id, *key_fields)`` — matching the compound unique indexes, and
    making it impossible for this seed to overwrite another tenant's rows.
    """
    if not docs:
        return 0
    scope = db.get_tenant_scope(tenant_id)
    result = await scope[collection].upsert_many(list(key_fields), docs, ordered=False)
    if result is None:
        return 0
    return result.upserted_count + result.modified_count


async def seed() -> dict[str, int]:
    """Run the idempotent seed and return per-collection write counts."""
    db.connect()
    await db.create_indexes()

    counts: dict[str, int] = {}
    counts[COLLECTIONS.tenants] = await _upsert_tenants(TENANTS)

    counts[COLLECTIONS.sites] = await _upsert(
        DEMO_TENANT_ID, COLLECTIONS.sites, ("site_id",), [SITE.model_dump()]
    )
    counts[COLLECTIONS.lines] = await _upsert(
        DEMO_TENANT_ID, COLLECTIONS.lines, ("line_id",), [LINE.model_dump()]
    )
    machine_docs = []
    for m in MACHINES:
        doc = m.model_dump()
        doc.update(_PRODUCTION_FIELDS.get(m.machine_id, {}))
        machine_docs.append(doc)
    counts[COLLECTIONS.machines] = await _upsert(
        DEMO_TENANT_ID,
        COLLECTIONS.machines,
        ("machine_id",),
        machine_docs,
    )
    counts[COLLECTIONS.components] = await _upsert(
        DEMO_TENANT_ID,
        COLLECTIONS.components,
        ("component_id",),
        [c.model_dump() for c in COMPONENTS],
    )
    counts[COLLECTIONS.sensors] = await _upsert(
        DEMO_TENANT_ID,
        COLLECTIONS.sensors,
        ("sensor_id",),
        [s.model_dump() for s in SENSORS],
    )
    counts[COLLECTIONS.error_codes] = await _upsert(
        DEMO_TENANT_ID,
        COLLECTIONS.error_codes,
        ("code", "machine_model"),
        [e.model_dump() for e in ERROR_CODES],
    )

    # A newly seeded tenant must not sit behind a cached "unknown tenant".
    clear_tenant_cache()

    await db.close()
    return counts


def _print_summary(counts: dict[str, int]) -> None:
    print("FactoryPilot AI — seed complete")
    print("=" * 40)
    for tenant in TENANTS:
        owned = "the fleet below" if tenant.tenant_id == DEMO_TENANT_ID else "nothing (empty by design)"
        print(f"  Tenant:          {tenant.tenant_id:<6} {tenant.name} — owns {owned}")
    print("-" * 40)
    print(f"  Site:            {SITE.site_id} ({SITE.name})")
    print(f"  Production line:  {LINE.line_id} ({LINE.name})")
    print(f"  Machines:        {len(MACHINES)}  ->  " + ", ".join(m.machine_id for m in MACHINES))
    print(f"  Components:      {len(COMPONENTS)}")
    print(f"  Sensors:         {len(SENSORS)}")
    print(f"  Error codes:     {len(ERROR_CODES)}")
    print("-" * 40)
    print("  Upserted/modified this run (idempotent):")
    for collection, n in counts.items():
        print(f"    {collection:<18} {n}")
    print("=" * 40)
    print("  Verify isolation:")
    print("    curl -H 'X-Tenant-Id: demo' localhost:8000/machines   # 4 machines")
    print("    curl -H 'X-Tenant-Id: acme' localhost:8000/machines   # []")


async def _main() -> None:
    counts = await seed()
    _print_summary(counts)


if __name__ == "__main__":
    asyncio.run(_main())
