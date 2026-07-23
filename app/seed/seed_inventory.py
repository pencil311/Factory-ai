"""Seed realistic spare parts inventory for the four demo machines.

Run with:

    python -m app.seed.seed_inventory

Includes at least one OUT_OF_STOCK part (NSK-7014A5-P4 spindle bearing pack)
with a valid alternative (SKF-7014A5-P4) — that case is what makes the
Inventory Agent visibly useful.

Idempotent via upsert on (tenant_id, part_number).
"""

from __future__ import annotations

import asyncio

from app import db
from app.models.part import Part
from app.schemas.machine import COLLECTIONS
from app.seed.seed_machines import DEMO_TENANT_ID

PARTS: list[Part] = [
    # --- CV-201 Belt Conveyor parts -----------------------------------------
    Part(
        tenant_id=DEMO_TENANT_ID,
        part_number="SKF-6310-2RS1",
        description="Deep groove ball bearing 50x110x27mm, sealed",
        category="bearing",
        compatible_components=["CV-201-BRG-D"],
        compatible_machine_models=["SpanTech SB-3000"],
        quantity_on_hand=4,
        reorder_level=2,
        warehouse_location="A-03-12",
        unit_cost=42.50,
        lead_time_days=3,
        supplier="SKF Distribution",
        alternative_part_numbers=["FAG-6310-2RSR", "NTN-6310-LLU"],
    ),
    Part(
        tenant_id=DEMO_TENANT_ID,
        part_number="SKF-6308-2RS1",
        description="Deep groove ball bearing 40x90x23mm, sealed",
        category="bearing",
        compatible_components=["CV-201-BRG-T"],
        compatible_machine_models=["SpanTech SB-3000"],
        quantity_on_hand=3,
        reorder_level=2,
        warehouse_location="A-03-11",
        unit_cost=35.00,
        lead_time_days=3,
        supplier="SKF Distribution",
        alternative_part_numbers=["FAG-6308-2RSR"],
    ),
    Part(
        tenant_id=DEMO_TENANT_ID,
        part_number="WEG-W22-5.5KW",
        description="WEG W22 three-phase motor 5.5kW 1450rpm",
        category="motor",
        compatible_components=["CV-201-MTR"],
        compatible_machine_models=["SpanTech SB-3000"],
        quantity_on_hand=1,
        reorder_level=1,
        warehouse_location="B-01-04",
        unit_cost=890.00,
        lead_time_days=14,
        supplier="WEG Electric",
    ),
    Part(
        tenant_id=DEMO_TENANT_ID,
        part_number="HAB-PVC-1200-GRN",
        description="PVC conveyor belt, 1200mm wide, green food-grade",
        category="belt",
        compatible_components=["CV-201-BELT"],
        compatible_machine_models=["SpanTech SB-3000"],
        quantity_on_hand=1,
        reorder_level=1,
        warehouse_location="C-02-01",
        unit_cost=1250.00,
        lead_time_days=10,
        supplier="Habasit",
    ),

    # --- MC-110 CNC Mill parts ----------------------------------------------
    Part(
        tenant_id=DEMO_TENANT_ID,
        part_number="NSK-7014A5-P4",
        description="Angular contact bearing pack for Haas VF-4 spindle",
        category="bearing",
        compatible_components=["MC-110-SPN-BRG"],
        compatible_machine_models=["Haas VF-4"],
        quantity_on_hand=0,  # OUT OF STOCK — the interesting case
        reorder_level=1,
        warehouse_location="A-05-02",
        unit_cost=1850.00,
        lead_time_days=21,
        supplier="NSK Americas",
        alternative_part_numbers=["SKF-7014A5-P4"],
    ),
    Part(
        tenant_id=DEMO_TENANT_ID,
        part_number="SKF-7014A5-P4",
        description="Angular contact bearing (alternative for NSK-7014A5-P4)",
        category="bearing",
        compatible_components=["MC-110-SPN-BRG"],
        compatible_machine_models=["Haas VF-4"],
        quantity_on_hand=1,
        reorder_level=1,
        warehouse_location="A-05-03",
        unit_cost=1920.00,
        lead_time_days=14,
        supplier="SKF Distribution",
    ),
    Part(
        tenant_id=DEMO_TENANT_ID,
        part_number="GRUNDFOS-CR3-11",
        description="Grundfos CR3-11 multistage coolant pump",
        category="pump",
        compatible_components=["MC-110-COOL"],
        compatible_machine_models=["Haas VF-4"],
        quantity_on_hand=1,
        reorder_level=1,
        warehouse_location="B-02-08",
        unit_cost=680.00,
        lead_time_days=7,
        supplier="Grundfos",
    ),
    Part(
        tenant_id=DEMO_TENANT_ID,
        part_number="HAAS-VMTR-30HP",
        description="Haas VF-4 spindle motor 30HP vector drive",
        category="motor",
        compatible_components=["MC-110-SPN-MTR"],
        compatible_machine_models=["Haas VF-4"],
        quantity_on_hand=0,  # Also out of stock, no alternative
        reorder_level=1,
        warehouse_location="B-01-12",
        unit_cost=4200.00,
        lead_time_days=28,
        supplier="Haas Automation",
    ),

    # --- AC-301 Air Compressor parts ----------------------------------------
    Part(
        tenant_id=DEMO_TENANT_ID,
        part_number="SKF-6314-C3",
        description="Deep groove ball bearing 70x150x35mm, C3 clearance",
        category="bearing",
        compatible_components=["AC-301-MTR-BRG"],
        compatible_machine_models=["Atlas Copco GA-75"],
        quantity_on_hand=2,
        reorder_level=1,
        warehouse_location="A-03-18",
        unit_cost=78.00,
        lead_time_days=5,
        supplier="SKF Distribution",
    ),
    Part(
        tenant_id=DEMO_TENANT_ID,
        part_number="AC-SEP-ELEM-75",
        description="Oil separator element for Atlas Copco GA-75",
        category="filter",
        compatible_components=["AC-301-SEP"],
        compatible_machine_models=["Atlas Copco GA-75"],
        quantity_on_hand=2,
        reorder_level=1,
        warehouse_location="D-01-06",
        unit_cost=320.00,
        lead_time_days=10,
        supplier="Atlas Copco",
    ),
    Part(
        tenant_id=DEMO_TENANT_ID,
        part_number="ABB-M3BP-75KW",
        description="ABB M3BP three-phase motor 75kW",
        category="motor",
        compatible_components=["AC-301-MTR"],
        compatible_machine_models=["Atlas Copco GA-75"],
        quantity_on_hand=0,
        reorder_level=1,
        warehouse_location="B-01-20",
        unit_cost=5600.00,
        lead_time_days=35,
        supplier="ABB Motors",
        alternative_part_numbers=["WEG-W22-75KW"],
    ),

    # --- HP-150 Hydraulic Press parts ---------------------------------------
    Part(
        tenant_id=DEMO_TENANT_ID,
        part_number="BOSCH-A10VSO-140",
        description="Bosch Rexroth A10VSO-140 variable displacement pump",
        category="pump",
        compatible_components=["HP-150-PUMP"],
        compatible_machine_models=["Beckwood BX-150"],
        quantity_on_hand=0,
        reorder_level=1,
        warehouse_location="B-03-01",
        unit_cost=8500.00,
        lead_time_days=42,
        supplier="Bosch Rexroth",
    ),
    Part(
        tenant_id=DEMO_TENANT_ID,
        part_number="BW-SEAL-KIT-150",
        description="Complete seal kit for BX-150 main ram cylinder",
        category="seal",
        compatible_components=["HP-150-CYL"],
        compatible_machine_models=["Beckwood BX-150"],
        quantity_on_hand=3,
        reorder_level=2,
        warehouse_location="D-02-03",
        unit_cost=280.00,
        lead_time_days=7,
        supplier="Beckwood Press",
    ),
    Part(
        tenant_id=DEMO_TENANT_ID,
        part_number="INA-GE-80-DO",
        description="INA GE80-DO spherical plain bearing 80x120x55mm",
        category="bearing",
        compatible_components=["HP-150-CYL-BRG"],
        compatible_machine_models=["Beckwood BX-150"],
        quantity_on_hand=2,
        reorder_level=1,
        warehouse_location="A-04-09",
        unit_cost=195.00,
        lead_time_days=10,
        supplier="Schaeffler",
    ),
    Part(
        tenant_id=DEMO_TENANT_ID,
        part_number="REXROTH-4WRA-10",
        description="Rexroth 4WRA-10 proportional directional valve",
        category="valve",
        compatible_components=["HP-150-VLV"],
        compatible_machine_models=["Beckwood BX-150"],
        quantity_on_hand=1,
        reorder_level=1,
        warehouse_location="D-03-02",
        unit_cost=1650.00,
        lead_time_days=14,
        supplier="Bosch Rexroth",
    ),

    # --- General consumables ------------------------------------------------
    Part(
        tenant_id=DEMO_TENANT_ID,
        part_number="LUBE-MOBIL-SHC629",
        description="Mobil SHC 629 synthetic bearing grease, 400g cartridge",
        category="lubricant",
        compatible_components=[],
        compatible_machine_models=[],
        quantity_on_hand=12,
        reorder_level=6,
        warehouse_location="D-04-01",
        unit_cost=18.50,
        lead_time_days=2,
        supplier="ExxonMobil",
    ),
]


async def seed_inventory(tenant_id: str = DEMO_TENANT_ID) -> dict[str, int]:
    """Upsert the parts inventory. Idempotent."""
    db.connect()
    await db.create_indexes()

    scope = db.get_tenant_scope(tenant_id)
    docs = [p.model_dump() for p in PARTS]
    result = await scope[COLLECTIONS.parts].upsert_many(["part_number"], docs, ordered=False)

    await db.close()
    upserted = result.upserted_count if result else 0
    modified = result.modified_count if result else 0
    return {"parts": len(PARTS), "upserted": upserted, "modified": modified}


async def _main() -> None:
    counts = await seed_inventory()
    print("FactoryPilot AI — inventory seeded")
    print("=" * 40)
    print(f"  Tenant:    {DEMO_TENANT_ID}")
    print(f"  Parts:     {counts['parts']}")
    print(f"  Upserted:  {counts['upserted']}")
    print(f"  Modified:  {counts['modified']}")
    print("=" * 40)


if __name__ == "__main__":
    asyncio.run(_main())
