"""Agent endpoints.

POST /agents/maintenance, /agents/inventory, /agents/safety, /agents/production
Each takes {machine_id, rca_result?} and returns its AgentResult.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.agents.inventory_agent import InventoryAgent
from app.agents.maintenance_agent import MaintenanceAgent
from app.agents.production_agent import ProductionAgent
from app.agents.safety_agent import SafetyAgent
from app.core.tenant_context import get_current_tenant
from app.schemas.agents import AgentContext, AgentResult

router = APIRouter(prefix="/agents", tags=["agents"])


class AgentRequest(BaseModel):
    machine_id: str
    rca_result: Optional[dict[str, Any]] = None
    pdm_result: Optional[dict[str, Any]] = None
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)


def _context(req: AgentRequest, tenant_id: str) -> AgentContext:
    return AgentContext(
        tenant_id=tenant_id,
        machine_id=req.machine_id,
        rca_result=req.rca_result,
        pdm_result=req.pdm_result,
        retrieved_chunks=req.retrieved_chunks,
    )


@router.post("/maintenance", response_model=AgentResult, summary="Maintenance agent")
async def maintenance(req: AgentRequest, tenant_id: str = Depends(get_current_tenant)):
    agent = MaintenanceAgent()
    return await agent.run(_context(req, tenant_id))


@router.post("/inventory", response_model=AgentResult, summary="Inventory agent")
async def inventory(req: AgentRequest, tenant_id: str = Depends(get_current_tenant)):
    agent = InventoryAgent()
    return await agent.run(_context(req, tenant_id))


@router.post("/safety", response_model=AgentResult, summary="Safety agent")
async def safety(req: AgentRequest, tenant_id: str = Depends(get_current_tenant)):
    agent = SafetyAgent()
    return await agent.run(_context(req, tenant_id))


@router.post("/production", response_model=AgentResult, summary="Production agent")
async def production(req: AgentRequest, tenant_id: str = Depends(get_current_tenant)):
    agent = ProductionAgent()
    return await agent.run(_context(req, tenant_id))
