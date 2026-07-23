"""Seed a small knowledge corpus for the demo tenant.

Run with:

    python -m app.seed.seed_documents

Deliberately small — this exists to demonstrate retrieval, not to benchmark
volume. Per machine: a manual excerpt (with a spec table and an error-code
table), an SOP, a maintenance guide, and two past repair records, plus two
tenant-wide general documents.

Everything goes through the REAL ingestion pipeline — parse, chunk, embed,
index — rather than being inserted as pre-made chunks. Seeding by shortcut
would mean the demo corpus never exercises the chunker, and a table-splitting
regression would go unnoticed until a user hit it.

Idempotent: ingestion keys on content hash, so re-running replaces rather than
duplicating.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app import db
from app.models.document import DocType
from app.rag.ingest import ensure_vector_index, ingest_document
from app.seed.seed_machines import DEMO_TENANT_ID, MACHINES


@dataclass(frozen=True)
class SeedDoc:
    """One document to be ingested."""

    filename: str
    title: str
    doc_type: DocType
    machine_ids: tuple[str, ...]
    machine_models: tuple[str, ...]
    body: str


# ---------------------------------------------------------------------------
# Per-machine content
# ---------------------------------------------------------------------------
#: (machine_id, model, friendly name, spec rows, error rows, symptom line)
_MACHINE_CONTENT = {
    "CV-201": {
        "specs": [
            ("Belt width", "1200 mm"),
            ("Belt speed range", "0.2 - 1.6 m/s"),
            ("Drive motor", "WEG-W22-5.5KW"),
            ("Drive roller bearing", "SKF-6310-2RS1"),
            ("Max load", "120 kg/m"),
            ("Nominal vibration (drive end)", "< 2.8 mm/s RMS"),
        ],
        "errors": [
            ("E101", "Belt slip detected", "Drive/tail roller speed mismatch > 8%. Check belt tension and drive coupling."),
            ("E102", "Drive motor overtemperature trip", "Motor winding above 95 C. Check ventilation and load."),
            ("E103", "Belt tracking fault", "Edge sensor tripped. Adjust tail roller alignment."),
            ("E104", "Emergency stop circuit open", "E-stop latched or safety relay failed. Reset all E-stops, then the relay."),
        ],
        "procedure": "Drive Roller Bearing Replacement",
        "steps": [
            "Isolate the conveyor at the local disconnect and apply your personal lock and tag.",
            "Verify zero energy: attempt a start from the HMI and confirm no motion.",
            "Release belt tension at the take-up assembly, recording the original position.",
            "Remove the drive guard and uncouple the gearbox from the drive roller shaft.",
            "Support the roller, remove the bearing housing bolts, and withdraw the bearing.",
            "Fit the replacement SKF-6310-2RS1 bearing; do not strike the outer race.",
            "Reassemble in reverse order and torque the housing bolts to 45 Nm.",
            "Restore belt tension to the recorded take-up position and re-check tracking.",
        ],
    },
    "MC-110": {
        "specs": [
            ("Spindle speed", "0 - 8100 rpm"),
            ("Spindle taper", "CT40"),
            ("Spindle motor", "30 HP vector drive"),
            ("Spindle bearing pack", "NSK-7014A5-P4"),
            ("Coolant pressure", "3.0 - 7.0 bar"),
            ("Nominal spindle vibration", "< 1.8 mm/s RMS"),
        ],
        "errors": [
            ("A201", "Spindle overload", "Commanded torque exceeded limit. Reduce feed rate or depth of cut."),
            ("A202", "Spindle bearing overtemperature", "Bearing pack above 75 C. Stop and allow to cool; check lubrication."),
            ("A203", "Low coolant pressure", "Coolant pump fault or blocked filter. Check filter and pump inlet."),
            ("A204", "X-axis servo following error", "Axis lag beyond tolerance. Check way lube and ballscrew condition."),
        ],
        "procedure": "Spindle Bearing Pack Inspection",
        "steps": [
            "Isolate the machine and apply lock and tag at the main disconnect.",
            "Allow the spindle to cool to ambient before measuring; heat masks bearing play.",
            "Mount a dial indicator against the spindle nose and check radial play; limit is 0.010 mm.",
            "Rotate the spindle by hand and listen for roughness through a stethoscope.",
            "Inspect the drawbar and check clamping force with the gauge fixture.",
            "Record vibration at the spindle housing; anything above 1.8 mm/s RMS warrants trending.",
        ],
    },
    "AC-301": {
        "specs": [
            ("Rated capacity", "13.2 m3/min"),
            ("Working pressure", "7.5 bar"),
            ("Main motor", "ABB-M3BP-75KW"),
            ("Motor bearing", "SKF-6314-C3"),
            ("Element outlet temp (normal)", "60 - 95 C"),
            ("Oil separator change interval", "4000 h"),
        ],
        "errors": [
            ("C301", "Element outlet temperature high shutdown", "Above 112 C. Check oil level, cooler fouling and ambient temperature."),
            ("C302", "Separator vessel overpressure", "Above 8.6 bar. Check minimum pressure valve and separator element."),
            ("C303", "Main motor overload", "Overload relay tripped or phase loss. Check supply and motor windings."),
            ("C304", "Low oil pressure", "Injection pump fault. Check oil level and pump inlet screen."),
        ],
        "procedure": "Oil Separator Element Replacement",
        "steps": [
            "Stop the compressor and isolate it electrically; lock and tag the disconnect.",
            "Close the service valve and vent the separator vessel to zero pressure.",
            "Confirm zero pressure on the vessel gauge before loosening any fastener.",
            "Remove the separator cover bolts in a star pattern and lift the cover clear.",
            "Withdraw the spent element and clean the sealing faces.",
            "Fit the new element with a new gasket; torque cover bolts to 55 Nm in a star pattern.",
            "Refill with the specified lubricant, run for 10 minutes and re-check for leaks.",
        ],
    },
    "HP-150": {
        "specs": [
            ("Press capacity", "150 ton"),
            ("System pressure", "0 - 210 bar"),
            ("Pump", "BOSCH-A10VSO-140"),
            ("Pump drive motor", "SIEMENS-1LE1-45KW"),
            ("Hydraulic oil temp (normal)", "20 - 60 C"),
            ("Ram guide bearing", "INA-GE-80-DO"),
        ],
        "errors": [
            ("H401", "System pressure below setpoint", "Pump wear or relief valve bypassing. Check pump case drain flow."),
            ("H402", "Hydraulic oil overtemperature", "Above 80 C. Check cooler, oil level and relief valve setting."),
            ("H403", "Ram position sensor out of range", "Transducer or wiring fault. Check the linear transducer connector."),
            ("H404", "Light curtain / two-hand control violation", "Safety device interrupted. Clear the field and reset with both hands."),
        ],
        "procedure": "Main Ram Cylinder Seal Replacement",
        "steps": [
            "Lower the ram fully onto blocks so it cannot descend under gravity.",
            "Isolate electrically, lock and tag, then bleed hydraulic pressure to zero.",
            "Verify zero pressure at the gauge and crack a fitting to confirm before disassembly.",
            "Disconnect the hydraulic lines and cap them to keep contamination out.",
            "Remove the gland nut and withdraw the piston rod assembly.",
            "Replace the rod seal, wiper and piston seals as a complete set.",
            "Reassemble, torque the gland nut to 320 Nm, and cycle five times at low pressure to bleed air.",
        ],
    },
}

_REPAIRS = {
    "CV-201": [
        (
            "2026-03-14",
            "E101 belt slip recurring on night shift",
            "Operator reported repeated E101 trips. Found drive roller lagging worn smooth and take-up "
            "at the end of its travel. Replaced lagging, reset take-up to 40 mm, re-tensioned. "
            "Vibration at drive end fell from 3.9 to 1.6 mm/s RMS. No recurrence in 3 weeks.",
        ),
        (
            "2026-05-02",
            "Drive roller bearing replacement after vibration trend",
            "Vibration trended from 1.5 to 4.2 mm/s RMS over 11 days with no temperature rise initially. "
            "Bearing SKF-6310-2RS1 showed outer race spalling on strip-down. Replaced bearing and "
            "seals; torqued housing to 45 Nm. Post-repair vibration 1.3 mm/s RMS.",
        ),
    ],
    "MC-110": [
        (
            "2026-04-08",
            "A202 spindle bearing overtemperature during long cycle",
            "A202 raised after 40 minutes of continuous cutting. Found spindle lubrication line partially "
            "blocked with swarf. Cleared line, flushed system, replaced filter. Bearing temperature "
            "stabilised at 52 C under the same program.",
        ),
        (
            "2026-06-19",
            "A203 low coolant pressure",
            "Coolant pressure dropped to 1.8 bar. Coolant pump inlet screen was blocked with fines. "
            "Cleaned screen and replaced the coolant filter. Pressure restored to 5.4 bar.",
        ),
    ],
    "AC-301": [
        (
            "2026-02-21",
            "C304 low oil pressure at start",
            "C304 on cold start. Oil injection pump inlet screen partially blocked. Cleaned screen, "
            "changed oil and filter. Oil pressure normal at 4.1 bar after restart.",
        ),
        (
            "2026-07-01",
            "C301 element outlet high temperature",
            "Element outlet reached 108 C in warm ambient. Cooler matrix was fouled with dust. "
            "Cleaned cooler externally and confirmed fan operation. Outlet temperature returned to 88 C.",
        ),
    ],
    "HP-150": [
        (
            "2026-01-30",
            "H401 pressure below setpoint under load",
            "System held only 165 bar against a 210 bar setpoint. Pump case drain flow measured at "
            "3.1 L/min, well above the 1.2 L/min limit, indicating internal wear. Replaced the "
            "BOSCH-A10VSO-140 pump. Pressure restored to 208 bar.",
        ),
        (
            "2026-05-21",
            "H402 hydraulic oil overtemperature",
            "Oil reached 84 C during a long production run. Cooler thermostatic valve stuck closed. "
            "Replaced valve and flushed the cooler circuit. Oil temperature settled at 47 C.",
        ),
    ],
}


def _manual(machine, content) -> str:
    """Manual excerpt with a spec table and an error-code table.

    Both tables are pipe-delimited so the chunker's table detector keeps them
    whole — splitting an error-code table would separate 'E104' from what it
    means, which is precisely what the chunker exists to prevent.
    """
    specs = "\n".join(f"| {k} | {v} |" for k, v in content["specs"])
    errors = "\n".join(
        f"| {code} | {name} | {action} |" for code, name, action in content["errors"]
    )
    return f"""# {machine.name} — Operation and Maintenance Manual

Model: {machine.model}
Manufacturer: {machine.manufacturer}
Asset: {machine.machine_id}

## 1. Overview

This excerpt covers routine operation, technical specifications and fault
codes for the {machine.name} ({machine.model}). Refer to the full manufacturer
manual for commissioning and warranty terms.

## 2. Technical Specifications

| Parameter | Value |
| --- | --- |
{specs}

## 3. Fault Codes

The controller reports the following codes. Codes clear on acknowledgement
once the underlying condition is resolved.

| Code | Description | Operator action |
| --- | --- | --- |
{errors}

## 4. Routine Checks

Inspect for abnormal noise, leaks and loose fasteners at the start of each
shift. Record any reading outside the normal range in the maintenance log and
raise a work order before the condition worsens.
"""


def _sop(machine, content) -> str:
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(content["steps"], start=1))
    return f"""# SOP — {content['procedure']} ({machine.machine_id})

## Scope

This procedure applies to {machine.name}, asset {machine.machine_id},
model {machine.model}.

## Required PPE

Safety glasses, cut-resistant gloves, steel-toe boots. Hearing protection is
required whenever adjacent equipment is running.

## Lockout / Tagout

Energy sources must be isolated and verified at zero before any part of this
procedure begins. Each worker applies their own lock. Locks are removed only
by the person who applied them.

## Procedure

{steps}

## Return to Service

Restore guarding before restoring power. Run the machine unloaded for five
minutes and confirm normal readings before returning it to production.
"""


def _maintenance_guide(machine, content) -> str:
    return f"""# Preventive Maintenance Guide — {machine.machine_id}

## Applicability

{machine.name} ({machine.model}), asset {machine.machine_id}.

## Interval Schedule

| Interval | Task |
| --- | --- |
| Daily | Visual inspection for leaks, damage and loose fasteners |
| Weekly | Record vibration and temperature readings; compare against baseline |
| Monthly | Check and adjust alignment; inspect wear parts |
| Quarterly | Lubricate per schedule; inspect electrical connections |
| Annually | Full teardown inspection of wear components |

## Condition Limits

Readings outside the normal band call for investigation, not immediate
shutdown, unless the critical limit is crossed. Trend direction matters more
than any single reading: a value rising steadily within the normal band is a
stronger signal than one high reading.

## Notes on {content['procedure']}

Schedule this work during planned downtime where possible. The procedure is
documented in the corresponding SOP and requires full lockout/tagout.
"""


def _repair(machine, date, summary, detail) -> str:
    return f"""# Repair Record — {machine.machine_id} — {date}

Asset: {machine.machine_id} ({machine.name})
Model: {machine.model}
Date: {date}
Summary: {summary}

## Work Performed

{detail}

## Outcome

Machine returned to service and verified under normal production load.
"""


GENERAL_DOCS = (
    SeedDoc(
        filename="lockout-tagout-policy.md",
        title="Site Lockout/Tagout Policy",
        doc_type=DocType.sop,
        machine_ids=(),
        machine_models=(),
        body="""# Site Lockout/Tagout Policy

## Purpose

This policy applies to all equipment on site. No maintenance, cleaning or
adjustment may take place on a machine capable of unexpected start-up or
stored-energy release until it has been isolated and verified.

## Rules

1. Every person working on isolated equipment applies their own lock and tag.
2. A lock is removed only by the person who applied it, without exception.
3. Isolation must be verified by attempting a start before work begins.
4. Stored energy — hydraulic pressure, compressed air, suspended loads,
   capacitors, springs — must be released or restrained before work starts.
5. Group lockout uses a lockbox; the group lead controls the box key.

## Verification

Verification is a physical test, not a paperwork step. Attempt to start the
equipment from its normal control position and confirm no motion, then return
the control to off.
""",
    ),
    SeedDoc(
        filename="vibration-severity-reference.md",
        title="Vibration Severity Reference (ISO 10816 summary)",
        doc_type=DocType.troubleshooting,
        machine_ids=(),
        machine_models=(),
        body="""# Vibration Severity Reference

## Scope

General reference for interpreting broadband vibration readings on rotating
equipment. Machine-specific limits in the equipment manual take precedence
over this table.

## Severity Bands (RMS velocity, mm/s)

| Class | Good | Satisfactory | Unsatisfactory | Unacceptable |
| --- | --- | --- | --- | --- |
| Small machines (< 15 kW) | < 0.71 | 0.71 - 1.8 | 1.8 - 4.5 | > 4.5 |
| Medium machines (15 - 75 kW) | < 1.12 | 1.12 - 2.8 | 2.8 - 7.1 | > 7.1 |
| Large machines, rigid mount | < 1.8 | 1.8 - 4.5 | 4.5 - 11.2 | > 11.2 |

## Interpreting Trends

A reading rising steadily over days or weeks is more informative than a single
elevated sample. Bearing degradation typically shows as rising vibration first,
with temperature following later once friction has developed. Temperature
rising without any vibration change usually points at lubrication or cooling
rather than at the bearing itself.
""",
    ),
)


def build_corpus() -> list[SeedDoc]:
    """Assemble the seed corpus for the four demo machines."""
    docs: list[SeedDoc] = []
    for machine in MACHINES:
        content = _MACHINE_CONTENT.get(machine.machine_id)
        if content is None:
            continue
        ids = (machine.machine_id,)
        models = (machine.model,)

        docs.append(
            SeedDoc(
                filename=f"{machine.machine_id.lower()}-manual.md",
                title=f"{machine.name} Manual ({machine.model})",
                doc_type=DocType.manual,
                machine_ids=ids,
                machine_models=models,
                body=_manual(machine, content),
            )
        )
        docs.append(
            SeedDoc(
                filename=f"{machine.machine_id.lower()}-sop.md",
                title=f"SOP — {content['procedure']} ({machine.machine_id})",
                doc_type=DocType.sop,
                machine_ids=ids,
                machine_models=models,
                body=_sop(machine, content),
            )
        )
        docs.append(
            SeedDoc(
                filename=f"{machine.machine_id.lower()}-pm-guide.md",
                title=f"Preventive Maintenance Guide — {machine.machine_id}",
                doc_type=DocType.maintenance_guide,
                machine_ids=ids,
                machine_models=models,
                body=_maintenance_guide(machine, content),
            )
        )
        for index, (date, summary, detail) in enumerate(
            _REPAIRS.get(machine.machine_id, []), start=1
        ):
            docs.append(
                SeedDoc(
                    filename=f"{machine.machine_id.lower()}-repair-{index}.md",
                    title=f"Repair Record {machine.machine_id} {date} — {summary}",
                    doc_type=DocType.repair_history,
                    machine_ids=ids,
                    machine_models=models,
                    body=_repair(machine, date, summary, detail),
                )
            )

    docs.extend(GENERAL_DOCS)
    return docs


async def seed_documents(tenant_id: str = DEMO_TENANT_ID) -> dict:
    """Ingest the corpus through the real pipeline. Idempotent."""
    db.connect()
    dimension = await ensure_vector_index()

    corpus = build_corpus()
    indexed = failed = chunks = replaced = 0

    for doc in corpus:
        result = await ingest_document(
            tenant_id=tenant_id,
            data=doc.body.encode("utf-8"),
            filename=doc.filename,
            title=doc.title,
            doc_type=doc.doc_type,
            machine_ids=list(doc.machine_ids),
            machine_models=list(doc.machine_models),
        )
        if result.succeeded:
            indexed += 1
            chunks += result.chunk_count
        else:
            failed += 1
            print(f"  FAILED {doc.filename}: {result.error}")
        replaced += int(result.replaced_existing)

    await db.close()
    return {
        "documents": len(corpus),
        "indexed": indexed,
        "failed": failed,
        "chunks": chunks,
        "replaced": replaced,
        "dimension": dimension,
    }


async def _main() -> None:
    counts = await seed_documents()
    print("FactoryPilot AI — knowledge corpus seeded")
    print("=" * 46)
    print(f"  Tenant:          {DEMO_TENANT_ID}")
    print(f"  Documents:       {counts['documents']}")
    print(f"  Indexed:         {counts['indexed']}")
    print(f"  Failed:          {counts['failed']}")
    print(f"  Chunks:          {counts['chunks']}")
    print(f"  Replaced:        {counts['replaced']} (idempotent re-ingest)")
    print(f"  Vector dim:      {counts['dimension']}")
    print("=" * 46)


if __name__ == "__main__":
    asyncio.run(_main())
