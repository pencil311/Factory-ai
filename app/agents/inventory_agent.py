"""Inventory Agent — checks parts availability for a repair.

Looks up required parts from the maintenance plan or RCA component,
checks stock levels, finds alternatives for out-of-stock parts, and
reports blocking parts and estimated availability.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from app.agents.base import Agent
from app.db import get_tenant_scope, normalize_tenant_id
from app.schemas.agents import (
    AgentContext,
    AgentStatus,
    InventoryItem,
    InventoryStatus,
    StockStatus,
)
from app.schemas.machine import COLLECTIONS


def _stock_status(qty: int, reorder_level: int) -> StockStatus:
    if qty <= 0:
        return StockStatus.out_of_stock
    if qty <= reorder_level:
        return StockStatus.low_stock
    return StockStatus.in_stock


class InventoryAgent(Agent):
    """Checks parts availability against the inventory collection."""

    @property
    def name(self) -> str:
        return "inventory"

    async def _run(self, context: AgentContext):
        rca = context.rca_result
        if not rca or not rca.get("primary_cause"):
            return AgentStatus.unavailable, None, "No RCA result to determine required parts", []

        tenant_id = normalize_tenant_id(context.tenant_id)
        scope = get_tenant_scope(tenant_id)

        primary = rca["primary_cause"]
        component_id = primary.get("component_id")

        # Find parts for the affected component
        parts_query: dict[str, Any] = {}
        if component_id:
            parts_query["compatible_components"] = component_id
        else:
            # Fall back to machine model
            machine = await scope[COLLECTIONS.machines].find_one({"machine_id": context.machine_id})
            if machine:
                parts_query["compatible_machine_models"] = machine.get("model", "")

        cursor = scope[COLLECTIONS.parts].find(parts_query)
        part_docs = [doc async for doc in cursor]

        if not part_docs:
            return (
                AgentStatus.partial,
                InventoryStatus(items=[], all_parts_available=False, blocking_parts=[], earliest_full_availability_days=0),
                f"No parts found in inventory for component '{component_id or context.machine_id}'",
                [],
            )

        items: list[InventoryItem] = []
        blocking: list[str] = []
        max_lead_time = 0

        for part in part_docs:
            pn = part["part_number"]
            qty = int(part.get("quantity_on_hand", 0))
            reorder = int(part.get("reorder_level", 1))
            status = _stock_status(qty, reorder)
            lead_time = int(part.get("lead_time_days", 0))

            # Find alternatives for out-of-stock parts
            alternatives: list[str] = []
            alt_pns = part.get("alternative_part_numbers", [])
            if status == StockStatus.out_of_stock and alt_pns:
                # Check if alternatives are in stock
                for alt_pn in alt_pns:
                    alt_doc = await scope[COLLECTIONS.parts].find_one({"part_number": alt_pn})
                    if alt_doc and int(alt_doc.get("quantity_on_hand", 0)) > 0:
                        alternatives.append(alt_pn)

            location = part.get("warehouse_location")

            if status == StockStatus.out_of_stock and not alternatives:
                blocking.append(pn)
                max_lead_time = max(max_lead_time, lead_time)

            items.append(InventoryItem(
                part_number=pn,
                description=part.get("description", ""),
                required_qty=1,
                available_qty=qty,
                status=status,
                location=location,
                alternatives=alternatives,
                lead_time_days=lead_time,
            ))

        all_available = len(blocking) == 0

        inv_status = InventoryStatus(
            items=items,
            all_parts_available=all_available,
            blocking_parts=blocking,
            earliest_full_availability_days=max_lead_time if not all_available else 0,
        )
        return AgentStatus.ok, inv_status, None, []
