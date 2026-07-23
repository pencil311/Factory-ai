"""Safety Agent — produces a safety briefing for maintenance work.

Safety NEVER returns UNAVAILABLE for a machine it knows. If it has no
documented procedure, it returns conservative generic industrial precautions
and marks them as GENERIC rather than staying silent.

Two lists carry preconditions, deliberately kept apart:

* ``blocking_conditions`` — genuine, machine-state-specific hazards that must
  be resolved before work may begin at all. Derived here from CRITICAL-rated
  hazards only: a pressurised circuit from a seal leak, stored energy in a
  hydraulic cylinder, a spindle that is still under power. The orchestrator
  leads the operator's answer with a warning banner whenever this list is
  non-empty, so it stays reserved for conditions that are actually
  exceptional — never for standard procedure.
* ``standard_preconditions`` — the routine LOTO sequence and standard PPE that
  apply to every job on every machine, regardless of what is wrong. These are
  real requirements and still appear in the briefing; they just do not, by
  themselves, mean the machine is in an unusually dangerous state.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from app.agents.base import Agent
from app.db import get_tenant_scope, normalize_tenant_id
from app.schemas.agents import (
    AgentContext,
    AgentStatus,
    EnergySource,
    EnergySourceType,
    Hazard,
    HazardSeverity,
    LotoStep,
    SafetyBriefing,
    SafetySource,
)
from app.schemas.machine import COLLECTIONS

# ---------------------------------------------------------------------------
# Machine-type specific hazards and energy sources
# ---------------------------------------------------------------------------

# Component type -> typical hazards
_COMPONENT_HAZARDS: dict[str, list[dict]] = {
    "motor": [
        {"hazard_type": "Electrical", "description": "Risk of electrical shock from motor windings and power supply",
         "severity": HazardSeverity.high},
        {"hazard_type": "Rotation", "description": "Rotating shaft and coupling can cause entanglement",
         "severity": HazardSeverity.high},
        {"hazard_type": "Thermal", "description": "Motor surfaces may be hot during or after operation",
         "severity": HazardSeverity.medium},
    ],
    "bearing": [
        {"hazard_type": "Mechanical", "description": "Components under compression may release stored energy when removed",
         "severity": HazardSeverity.medium},
        {"hazard_type": "Pinch point", "description": "Risk of pinching during bearing removal and installation",
         "severity": HazardSeverity.medium},
    ],
    "pump": [
        {"hazard_type": "Hydraulic/Pneumatic", "description": "Residual pressure in lines and pump housing",
         "severity": HazardSeverity.high},
        {"hazard_type": "Chemical", "description": "Contact with hydraulic fluid or lubricant",
         "severity": HazardSeverity.medium},
    ],
    "cylinder": [
        {"hazard_type": "Stored energy", "description": "Hydraulic cylinder may retain pressure even when isolated",
         "severity": HazardSeverity.critical},
        {"hazard_type": "Gravity", "description": "Suspended load or ram may descend under gravity if not blocked",
         "severity": HazardSeverity.critical},
    ],
    "valve": [
        {"hazard_type": "Hydraulic/Pneumatic", "description": "Pressurized fluid behind the valve",
         "severity": HazardSeverity.high},
    ],
    "belt": [
        {"hazard_type": "Entanglement", "description": "Moving belt and rotating rollers can catch loose clothing or hands",
         "severity": HazardSeverity.high},
    ],
    "spindle": [
        {"hazard_type": "Rotation", "description": "High-speed spindle rotation with extreme entanglement risk",
         "severity": HazardSeverity.critical},
        {"hazard_type": "Sharp edges", "description": "Cutting tools and workpiece edges",
         "severity": HazardSeverity.high},
        {"hazard_type": "Coolant", "description": "Coolant spray and slippery surfaces",
         "severity": HazardSeverity.medium},
    ],
    "gearbox": [
        {"hazard_type": "Mechanical", "description": "Meshing gears present crushing and shearing hazard",
         "severity": HazardSeverity.high},
        {"hazard_type": "Chemical", "description": "Gear oil contact",
         "severity": HazardSeverity.low},
    ],
}

# Machine model -> energy sources
_MACHINE_ENERGY_SOURCES: dict[str, list[dict]] = {
    "SpanTech SB-3000": [
        {"type": EnergySourceType.electrical, "location": "Local disconnect switch on drive unit frame",
         "isolation_method": "Open disconnect, apply lock and tag"},
        {"type": EnergySourceType.mechanical, "location": "Belt tension via take-up assembly",
         "isolation_method": "Release belt tension at take-up before working on rollers"},
    ],
    "Haas VF-4": [
        {"type": EnergySourceType.electrical, "location": "Main disconnect on machine cabinet",
         "isolation_method": "Open main disconnect, apply lock and tag"},
        {"type": EnergySourceType.pneumatic, "location": "Compressed air supply to tool changer",
         "isolation_method": "Close air supply valve and bleed residual pressure"},
        {"type": EnergySourceType.mechanical, "location": "Spindle and axis drives",
         "isolation_method": "Verify axes are at rest and spindle is stopped"},
    ],
    "Atlas Copco GA-75": [
        {"type": EnergySourceType.electrical, "location": "Main disconnect on compressor panel",
         "isolation_method": "Open disconnect, apply lock and tag"},
        {"type": EnergySourceType.pneumatic, "location": "Air receiver and separator vessel",
         "isolation_method": "Close service valve and vent vessel to zero pressure"},
        {"type": EnergySourceType.thermal, "location": "Air end discharge — up to 112°C during operation",
         "isolation_method": "Allow to cool to ambient before opening"},
    ],
    "Beckwood BX-150": [
        {"type": EnergySourceType.electrical, "location": "Main disconnect on HPU control cabinet",
         "isolation_method": "Open disconnect, apply lock and tag"},
        {"type": EnergySourceType.hydraulic, "location": "Hydraulic power unit — system pressure up to 250 bar",
         "isolation_method": "Bleed pressure to zero, verify at gauge, crack fitting to confirm"},
        {"type": EnergySourceType.mechanical, "location": "Ram — 150 tons of force, may descend under gravity",
         "isolation_method": "Lower ram fully onto blocks before any work"},
    ],
}

# Generic LOTO steps (always included)
_GENERIC_LOTO: list[dict] = [
    {"order": 1, "instruction": "Notify affected personnel that lockout is being applied",
     "verification": "Verbal confirmation from operators in the area"},
    {"order": 2, "instruction": "Shut down the machine using the normal stop procedure",
     "verification": "Confirm machine has stopped and all motion has ceased"},
    {"order": 3, "instruction": "Isolate all energy sources identified in the briefing",
     "verification": "Each isolation device is in the off/closed/open position"},
    {"order": 4, "instruction": "Apply personal lock and tag to each isolation point",
     "verification": "Lock is secured and tag is dated and signed"},
    {"order": 5, "instruction": "Verify zero energy: attempt a start from the normal control position",
     "verification": "No motion, no pressure, no voltage at the work point"},
    {"order": 6, "instruction": "Return controls to the off position after verification",
     "verification": "Controls confirmed in off position"},
]

_GENERIC_PPE = [
    "Safety glasses",
    "Steel-toe boots",
    "Cut-resistant gloves",
    "Hearing protection (when adjacent equipment is running)",
]

#: Routine steps true of every job — never a blocking condition on their own.
_STANDARD_PRECONDITIONS = [
    "All energy sources must be isolated and verified at zero before work begins",
    "Each worker must apply their own personal lock — no shared locks",
    "Guarding must not be removed until lockout is complete",
]


def _severity_value(hazard: Hazard) -> str:
    """``hazard.severity`` as a plain upper-case string, enum or not."""
    severity = hazard.severity
    return (severity if isinstance(severity, str) else severity.value).upper()


def _blocking_condition_for(hazard: Hazard) -> str:
    """Phrase a CRITICAL hazard as the condition that must be resolved first."""
    return (
        f"{hazard.hazard_type} hazard must be verified resolved before work "
        f"begins: {hazard.description}"
    )


class SafetyAgent(Agent):
    """Produces a safety briefing. NEVER returns UNAVAILABLE for a known machine."""

    @property
    def name(self) -> str:
        return "safety"

    async def _run(self, context: AgentContext):
        tenant_id = normalize_tenant_id(context.tenant_id)
        scope = get_tenant_scope(tenant_id)

        machine = await scope[COLLECTIONS.machines].find_one({"machine_id": context.machine_id})
        if machine is None:
            return AgentStatus.unavailable, None, f"Machine '{context.machine_id}' not found", []

        machine_model = str(machine.get("model", ""))
        rca = context.rca_result
        component_id = None
        if rca and rca.get("primary_cause"):
            component_id = rca["primary_cause"].get("component_id")

        # Determine source (documented vs generic)
        source = SafetySource.generic
        citations: list[dict] = []

        # Check retrieved chunks for documented safety procedures
        for chunk in context.retrieved_chunks:
            text = chunk.get("text", "").lower()
            if any(kw in text for kw in ("lockout", "tagout", "ppe", "safety", "isolat")):
                source = SafetySource.documented
                citations.append({
                    "document_id": chunk.get("document_id", ""),
                    "page_number": chunk.get("page_number"),
                    "section_title": chunk.get("section_title"),
                })

        # Build hazards
        hazards: list[Hazard] = []

        # Hazards from the affected component type
        if component_id:
            comp = await scope[COLLECTIONS.components].find_one({"component_id": component_id})
            if comp:
                comp_type = str(comp.get("type", "other"))
                for h in _COMPONENT_HAZARDS.get(comp_type, []):
                    hazards.append(Hazard(
                        hazard_type=h["hazard_type"],
                        description=h["description"],
                        severity=h["severity"],
                        source_component_id=component_id,
                    ))

        # If no component-specific hazards, add generic machine hazards
        if not hazards:
            hazards.append(Hazard(
                hazard_type="General mechanical",
                description="Moving parts, pinch points, and stored energy present during maintenance",
                severity=HazardSeverity.high,
            ))
            hazards.append(Hazard(
                hazard_type="Electrical",
                description="Electrical energy from machine power supply and controls",
                severity=HazardSeverity.high,
            ))

        # RCA-specific hazards (based on fault mode)
        if rca and rca.get("primary_cause"):
            fault_mode = rca["primary_cause"].get("fault_mode", "")
            if "OVERHEAT" in fault_mode or "THERMAL" in fault_mode:
                hazards.append(Hazard(
                    hazard_type="Thermal burn",
                    description="Component may be at elevated temperature due to the fault condition",
                    severity=HazardSeverity.high,
                    source_component_id=component_id,
                ))
            if "SEAL_LEAK" in fault_mode:
                hazards.append(Hazard(
                    hazard_type="Pressurized fluid",
                    description="Hydraulic fluid under residual pressure — verify zero before disassembly",
                    severity=HazardSeverity.critical,
                    source_component_id=component_id,
                ))

        # Energy sources
        energy_sources = []
        model_sources = _MACHINE_ENERGY_SOURCES.get(machine_model, [])
        if model_sources:
            for es in model_sources:
                energy_sources.append(EnergySource(**es))
        else:
            # Generic: at minimum there is always electrical
            energy_sources.append(EnergySource(
                type=EnergySourceType.electrical,
                location="Main disconnect switch",
                isolation_method="Open disconnect, apply personal lock and tag",
            ))

        # LOTO steps
        loto_steps = [LotoStep(**s) for s in _GENERIC_LOTO]

        # PPE
        ppe = list(_GENERIC_PPE)
        # Add fault-specific PPE
        if rca and rca.get("primary_cause"):
            fault_mode = rca["primary_cause"].get("fault_mode", "")
            if "SEAL_LEAK" in fault_mode:
                ppe.append("Chemical-resistant gloves")
                ppe.append("Face shield")
            if "OVERHEAT" in fault_mode:
                ppe.append("Heat-resistant gloves")

        # Permits
        permits: list[str] = []
        for h in hazards:
            if _severity_value(h) == "CRITICAL":
                permits.append("Confined space / high-risk work permit")
                break

        # Blocking conditions: derived from the hazards actually rated
        # CRITICAL for this machine and fault — never a fixed list, so a
        # routine job with no such hazard produces none.
        blocking = [
            _blocking_condition_for(h) for h in hazards if _severity_value(h) == "CRITICAL"
        ]
        standard_preconditions = list(_STANDARD_PRECONDITIONS)

        briefing = SafetyBriefing(
            hazards=hazards,
            required_ppe=ppe,
            lockout_tagout_steps=loto_steps,
            energy_sources_to_isolate=energy_sources,
            permits_required=permits,
            blocking_conditions=blocking,
            standard_preconditions=standard_preconditions,
            citations=citations,
            source=source,
        )
        return AgentStatus.ok, briefing, None, citations
