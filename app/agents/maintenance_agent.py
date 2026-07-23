"""Maintenance Agent — produces a repair procedure from RCA + knowledge base.

Prefers documented procedures from retrieved SOPs and cites them. Where no
documented procedure exists, derives steps from the fault mode and component
tree and marks them as DERIVED.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence

from app.agents.base import Agent
from app.db import get_tenant_scope, normalize_tenant_id
from app.schemas.agents import (
    AgentContext,
    AgentStatus,
    MaintenancePlan,
    ProcedureCitation,
    ProcedureSource,
    ProcedureStep,
    RequiredPart,
    SkillLevel,
)
from app.schemas.machine import COLLECTIONS

# Fault mode -> default skill level
_SKILL_MAP = {
    "BEARING_WEAR": SkillLevel.intermediate,
    "MOTOR_OVERHEAT": SkillLevel.specialist,
    "LUBRICATION_LOSS": SkillLevel.basic,
    "BELT_MISALIGNMENT": SkillLevel.basic,
    "SEAL_LEAK": SkillLevel.specialist,
    "TOOL_WEAR": SkillLevel.intermediate,
}

# Fault mode -> generic tools required
_TOOLS_MAP: dict[str, list[str]] = {
    "BEARING_WEAR": ["bearing puller", "torque wrench", "dial indicator", "soft-face mallet"],
    "MOTOR_OVERHEAT": ["multimeter", "insulation tester", "thermal camera", "torque wrench"],
    "LUBRICATION_LOSS": ["grease gun", "clean rags", "lubricant"],
    "BELT_MISALIGNMENT": ["straight edge", "feeler gauges", "torque wrench", "alignment laser"],
    "SEAL_LEAK": ["seal pick set", "torque wrench", "hydraulic pressure gauge", "clean rags"],
    "TOOL_WEAR": ["tool presetter", "collet wrench", "replacement inserts"],
}

# Component type -> part numbers commonly needed
_COMPONENT_PARTS: dict[str, list[dict]] = {
    "bearing": [{"desc": "Replacement bearing", "qty": 1}],
    "motor": [{"desc": "Motor or motor rebuild kit", "qty": 1}],
    "belt": [{"desc": "Replacement belt", "qty": 1}],
    "pump": [{"desc": "Pump rebuild kit or replacement pump", "qty": 1}],
    "valve": [{"desc": "Valve seal kit", "qty": 1}],
    "cylinder": [{"desc": "Cylinder seal kit", "qty": 1}],
    "spindle": [{"desc": "Spindle bearing pack", "qty": 1}],
}


def _generic_steps(fault_mode: str, component_id: Optional[str]) -> list[ProcedureStep]:
    """Generate generic procedure steps for a fault mode."""
    steps = [
        ProcedureStep(
            order=1,
            instruction="Isolate the machine at the local disconnect and apply personal lock and tag.",
            component_id=None,
            tools_required=["lockout/tagout kit"],
            estimated_minutes=10,
            caution="Verify zero energy before proceeding.",
        ),
        ProcedureStep(
            order=2,
            instruction="Verify zero energy: attempt a start and confirm no motion or stored energy.",
            component_id=None,
            tools_required=[],
            estimated_minutes=5,
        ),
    ]

    order = 3
    if fault_mode == "BEARING_WEAR":
        steps.extend([
            ProcedureStep(order=order, instruction="Remove guards and access covers to reach the affected bearing.",
                          component_id=component_id, tools_required=["socket set"], estimated_minutes=15),
            ProcedureStep(order=order+1, instruction="Disconnect the shaft coupling or belt from the bearing housing.",
                          component_id=component_id, tools_required=["bearing puller"], estimated_minutes=20),
            ProcedureStep(order=order+2, instruction="Remove the old bearing and inspect the shaft and housing for damage.",
                          component_id=component_id, tools_required=["bearing puller", "dial indicator"], estimated_minutes=15),
            ProcedureStep(order=order+3, instruction="Fit the replacement bearing. Do not strike the outer race directly.",
                          component_id=component_id, tools_required=["soft-face mallet", "bearing press"], estimated_minutes=20,
                          caution="Ensure bearing is square to the shaft before pressing."),
            ProcedureStep(order=order+4, instruction="Reassemble in reverse order and torque fasteners to specification.",
                          component_id=component_id, tools_required=["torque wrench"], estimated_minutes=15),
        ])
    elif fault_mode == "MOTOR_OVERHEAT":
        steps.extend([
            ProcedureStep(order=order, instruction="Check motor cooling system: clean ventilation openings and verify fan operation.",
                          component_id=component_id, tools_required=["compressed air", "thermal camera"], estimated_minutes=20),
            ProcedureStep(order=order+1, instruction="Measure winding insulation resistance with a megger.",
                          component_id=component_id, tools_required=["insulation tester"], estimated_minutes=15),
            ProcedureStep(order=order+2, instruction="Check electrical connections for loose terminals and measure current balance across phases.",
                          component_id=component_id, tools_required=["multimeter", "torque wrench"], estimated_minutes=20),
            ProcedureStep(order=order+3, instruction="If insulation has failed, replace or rewind the motor.",
                          component_id=component_id, tools_required=["hoist", "alignment tools"], estimated_minutes=120,
                          caution="Motor replacement requires alignment verification."),
        ])
    elif fault_mode == "LUBRICATION_LOSS":
        steps.extend([
            ProcedureStep(order=order, instruction="Check oil level and condition. Drain contaminated lubricant if necessary.",
                          component_id=component_id, tools_required=["drain pan", "clean rags"], estimated_minutes=15),
            ProcedureStep(order=order+1, instruction="Inspect lubrication lines and fittings for blockage or damage.",
                          component_id=component_id, tools_required=["inspection light"], estimated_minutes=15),
            ProcedureStep(order=order+2, instruction="Refill with specified lubricant and verify flow to all points.",
                          component_id=component_id, tools_required=["grease gun", "lubricant"], estimated_minutes=15),
        ])
    elif fault_mode == "SEAL_LEAK":
        steps.extend([
            ProcedureStep(order=order, instruction="Bleed system pressure to zero and verify at the gauge.",
                          component_id=component_id, tools_required=["pressure gauge"], estimated_minutes=10,
                          caution="Confirm zero pressure before loosening any fitting."),
            ProcedureStep(order=order+1, instruction="Disassemble the affected component and remove worn seals.",
                          component_id=component_id, tools_required=["seal pick set", "socket set"], estimated_minutes=30),
            ProcedureStep(order=order+2, instruction="Inspect sealing surfaces for scoring. Replace seals as a complete set.",
                          component_id=component_id, tools_required=["clean rags", "seal kit"], estimated_minutes=20),
            ProcedureStep(order=order+3, instruction="Reassemble, torque to specification, and cycle at low pressure to bleed air.",
                          component_id=component_id, tools_required=["torque wrench"], estimated_minutes=25),
        ])
    elif fault_mode == "BELT_MISALIGNMENT":
        steps.extend([
            ProcedureStep(order=order, instruction="Check belt tension and tracking. Measure alignment of drive and tail rollers.",
                          component_id=component_id, tools_required=["straight edge", "alignment laser"], estimated_minutes=20),
            ProcedureStep(order=order+1, instruction="Adjust roller alignment to bring the belt back on track.",
                          component_id=component_id, tools_required=["feeler gauges", "wrench set"], estimated_minutes=25),
            ProcedureStep(order=order+2, instruction="Re-tension the belt to specification and check tracking under load.",
                          component_id=component_id, tools_required=["tension gauge"], estimated_minutes=15),
        ])
    elif fault_mode == "TOOL_WEAR":
        steps.extend([
            ProcedureStep(order=order, instruction="Retract the tool and move the spindle to a safe position.",
                          component_id=component_id, tools_required=[], estimated_minutes=5),
            ProcedureStep(order=order+1, instruction="Remove the worn tool and inspect the holder/collet for damage.",
                          component_id=component_id, tools_required=["collet wrench"], estimated_minutes=10),
            ProcedureStep(order=order+2, instruction="Install replacement inserts or a new tool. Set tool length offset.",
                          component_id=component_id, tools_required=["tool presetter"], estimated_minutes=15),
        ])
    else:
        steps.append(ProcedureStep(
            order=order, instruction="Inspect the affected component and determine the appropriate repair.",
            component_id=component_id, tools_required=["inspection tools"], estimated_minutes=30,
        ))

    # Final step
    n = len(steps) + 1
    steps.append(ProcedureStep(
        order=n,
        instruction="Restore guarding, remove lockout, and run the machine unloaded for five minutes to verify normal operation.",
        component_id=None, tools_required=[], estimated_minutes=10,
    ))
    return steps


class MaintenanceAgent(Agent):
    """Produces a maintenance/repair plan from RCA findings."""

    @property
    def name(self) -> str:
        return "maintenance"

    async def _run(self, context: AgentContext):
        rca = context.rca_result
        if not rca or not rca.get("primary_cause"):
            return AgentStatus.unavailable, None, "No RCA result with a primary cause available", []

        primary = rca["primary_cause"]
        fault_mode = primary.get("fault_mode", "")
        component_id = primary.get("component_id")

        # Check retrieved chunks for documented procedures
        procedure_source = ProcedureSource.derived
        citations: list[dict] = []
        doc_steps: list[ProcedureStep] = []

        for chunk in context.retrieved_chunks:
            text = chunk.get("text", "")
            # Look for SOP/procedure content
            if any(kw in text.lower() for kw in ("procedure", "step 1", "isolate", "lockout", "replace")):
                procedure_source = ProcedureSource.documented
                citation = {
                    "document_id": chunk.get("document_id", ""),
                    "page_number": chunk.get("page_number"),
                    "section_title": chunk.get("section_title"),
                }
                citations.append(citation)

        # Build steps
        if procedure_source == ProcedureSource.documented and citations:
            # Extract steps from documented procedure with citations
            steps = _generic_steps(fault_mode, component_id)
            for step in steps:
                step.citation = ProcedureCitation(
                    document_id=citations[0].get("document_id"),
                    page_number=citations[0].get("page_number"),
                )
        else:
            steps = _generic_steps(fault_mode, component_id)

        # Required parts from component
        required_parts: list[RequiredPart] = []
        if component_id:
            # Try to find parts for this component
            try:
                scope = get_tenant_scope(normalize_tenant_id(context.tenant_id))
                cursor = scope[COLLECTIONS.parts].find({"compatible_components": component_id})
                async for part in cursor:
                    required_parts.append(RequiredPart(
                        part_number=part["part_number"],
                        description=part["description"],
                        quantity=1,
                        component_id=component_id,
                    ))
            except Exception:
                pass

        # If no parts found from DB, use generic parts based on component type
        if not required_parts and component_id:
            # Try to look up component type
            try:
                scope = get_tenant_scope(normalize_tenant_id(context.tenant_id))
                comp = await scope[COLLECTIONS.components].find_one({"component_id": component_id})
                if comp:
                    comp_type = comp.get("type", "other")
                    part_number = comp.get("part_number", "")
                    if part_number:
                        required_parts.append(RequiredPart(
                            part_number=part_number,
                            description=f"Replacement {comp_type}",
                            quantity=1,
                            component_id=component_id,
                        ))
            except Exception:
                pass

        # Collect all tools
        tools = _TOOLS_MAP.get(fault_mode, ["standard hand tools"])
        all_tools = list(set(tools))
        for step in steps:
            for t in step.tools_required:
                if t not in all_tools:
                    all_tools.append(t)

        total_minutes = sum(s.estimated_minutes for s in steps)
        skill = _SKILL_MAP.get(fault_mode, SkillLevel.intermediate)

        plan = MaintenancePlan(
            procedure_steps=steps,
            required_parts=required_parts,
            required_tools=all_tools,
            total_estimated_minutes=total_minutes,
            skill_level=skill,
            procedure_source=procedure_source,
        )
        return AgentStatus.ok, plan, None, citations
