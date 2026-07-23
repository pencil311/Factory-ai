"""Production Agent — estimates downtime, cost, and production impact.

All numbers come from data (machine seed fields, RCA, inventory). The agent
states every assumption it uses. Every agent must function without an LLM.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from app.agents.base import Agent
from app.db import get_tenant_scope, normalize_tenant_id
from app.schemas.agents import (
    AgentContext,
    AgentStatus,
    CostEstimate,
    DowntimeEstimate,
    DowntimeRecommendation,
    ProductionImpact,
)
from app.schemas.machine import COLLECTIONS

# Fault mode -> typical repair time in minutes (conservative estimates)
_REPAIR_TIMES: dict[str, int] = {
    "BEARING_WEAR": 180,
    "MOTOR_OVERHEAT": 240,
    "LUBRICATION_LOSS": 60,
    "BELT_MISALIGNMENT": 90,
    "SEAL_LEAK": 210,
    "TOOL_WEAR": 30,
}

# Default unit cost when parts cost is unknown
_DEFAULT_PARTS_COST = 500.0


class ProductionAgent(Agent):
    """Estimates production impact of the identified fault."""

    @property
    def name(self) -> str:
        return "production"

    async def _run(self, context: AgentContext):
        tenant_id = normalize_tenant_id(context.tenant_id)
        scope = get_tenant_scope(tenant_id)

        machine = await scope[COLLECTIONS.machines].find_one({"machine_id": context.machine_id})
        if machine is None:
            return AgentStatus.unavailable, None, f"Machine '{context.machine_id}' not found", []

        rca = context.rca_result
        assumptions: list[str] = []

        # Read production fields from machine document
        units_per_hour = machine.get("units_per_hour", 0)
        cost_per_hour = machine.get("cost_per_hour_downtime", 0.0)
        criticality = int(machine.get("criticality", 3))
        position = int(machine.get("position_in_line", 0))
        line_id = str(machine.get("line_id", ""))

        if units_per_hour == 0:
            assumptions.append("units_per_hour not set on machine; using 0")
        if cost_per_hour == 0.0:
            assumptions.append("cost_per_hour_downtime not set on machine; using 0")

        # Repair time estimate
        fault_mode = ""
        if rca and rca.get("primary_cause"):
            fault_mode = rca["primary_cause"].get("fault_mode", "")

        repair_minutes = _REPAIR_TIMES.get(fault_mode, 120)
        assumptions.append(f"Repair time estimated at {repair_minutes} minutes based on fault mode '{fault_mode or 'unknown'}'")

        # Parts wait time (from inventory agent if available, else estimate)
        parts_wait_minutes = 0
        # Check if there are blocking parts in the RCA context
        # The orchestrator would pass this, but we can also check inventory directly
        try:
            # Find parts for the affected component
            component_id = rca["primary_cause"].get("component_id") if rca and rca.get("primary_cause") else None
            if component_id:
                cursor = scope[COLLECTIONS.parts].find({"compatible_components": component_id})
                parts_cost = 0.0
                async for part in cursor:
                    qty = int(part.get("quantity_on_hand", 0))
                    parts_cost += float(part.get("unit_cost", 0))
                    if qty <= 0:
                        lead_days = int(part.get("lead_time_days", 7))
                        # Check alternatives
                        alt_available = False
                        for alt_pn in part.get("alternative_part_numbers", []):
                            alt = await scope[COLLECTIONS.parts].find_one({"part_number": alt_pn})
                            if alt and int(alt.get("quantity_on_hand", 0)) > 0:
                                alt_available = True
                                parts_cost = max(parts_cost, float(alt.get("unit_cost", 0)))
                                break
                        if not alt_available:
                            parts_wait_minutes = max(parts_wait_minutes, lead_days * 24 * 60)
                            assumptions.append(f"Part '{part['part_number']}' out of stock; lead time {lead_days} days")
                if parts_cost == 0:
                    parts_cost = _DEFAULT_PARTS_COST
                    assumptions.append(f"Parts cost unknown; using default estimate of ${_DEFAULT_PARTS_COST:.0f}")
            else:
                parts_cost = _DEFAULT_PARTS_COST
                assumptions.append("No component identified; using default parts cost estimate")
        except Exception:
            parts_cost = _DEFAULT_PARTS_COST
            assumptions.append("Could not check inventory; using default parts cost estimate")

        total_downtime = repair_minutes + parts_wait_minutes

        # Units lost
        units_lost = int(units_per_hour * (total_downtime / 60.0))

        # Is this a bottleneck?
        is_bottleneck = criticality >= 4 and position > 0

        # Downstream machines affected
        downstream: list[str] = []
        if is_bottleneck and line_id:
            cursor = scope[COLLECTIONS.machines].find(
                {"line_id": line_id, "position_in_line": {"$gt": position}}
            )
            async for m in cursor:
                downstream.append(m["machine_id"])
            if downstream:
                assumptions.append(f"Downstream machines on {line_id} after position {position} will be starved")

        # Cost estimate
        downtime_hours = total_downtime / 60.0
        downtime_cost = round(cost_per_hour * downtime_hours, 2)
        total_cost = round(downtime_cost + parts_cost, 2)

        cost = CostEstimate(
            downtime_cost=downtime_cost,
            parts_cost=round(parts_cost, 2),
            total=total_cost,
            currency="USD",
        )

        # Recommendation
        if rca and rca.get("primary_cause"):
            confidence = rca.get("confidence", 0)
            failure_prob = 0.0
            if context.pdm_result:
                failure_prob = context.pdm_result.get("failure_probability", 0.0)

            if failure_prob > 0.7 or (criticality >= 5 and confidence > 0.4):
                recommendation = DowntimeRecommendation.repair_now
                rationale = (
                    f"High failure probability ({failure_prob:.0%}) on criticality-{criticality} "
                    f"machine. Unplanned failure will cost more than planned repair."
                )
            elif failure_prob > 0.4 or criticality >= 4:
                recommendation = DowntimeRecommendation.schedule_next_window
                rationale = (
                    f"Moderate failure probability ({failure_prob:.0%}). Schedule repair during "
                    f"the next planned maintenance window to avoid unplanned downtime."
                )
            else:
                recommendation = DowntimeRecommendation.monitor
                rationale = (
                    f"Low failure probability ({failure_prob:.0%}). Continue monitoring sensor "
                    f"trends and re-evaluate if the condition worsens."
                )
        else:
            recommendation = DowntimeRecommendation.monitor
            rationale = "No confirmed root cause. Monitor and re-evaluate."

        impact = ProductionImpact(
            downtime_estimate_minutes=DowntimeEstimate(
                repair_time=repair_minutes,
                total_including_parts_wait=total_downtime,
            ),
            units_lost_estimate=units_lost,
            is_bottleneck=is_bottleneck,
            downstream_machines_affected=downstream,
            cost_estimate=cost,
            recommendation=recommendation,
            recommendation_rationale=rationale,
            assumptions=assumptions,
        )
        return AgentStatus.ok, impact, None, []
