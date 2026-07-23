"""Machine-centric read endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.tenant_context import get_current_tenant
from app.db import get_tenant_scope
from app.schemas.machine import (
    COLLECTIONS,
    ComponentOut,
    MachineOut,
    SensorOut,
    strip_mongo_id,
)

router = APIRouter(prefix="/machines", tags=["machines"])


async def _get_machine_or_404(tenant_id: str, machine_id: str) -> dict:
    """Fetch one tenant's machine by canonical id or raise 404.

    Falls back to matching an alias so ERP/floor names resolve too. The scope
    confines both lookups to ``tenant_id``, so another tenant's machine reads
    as absent rather than as a hit.
    """
    db = get_tenant_scope(tenant_id)
    doc = await db[COLLECTIONS.machines].find_one({"machine_id": machine_id})
    if doc is None:
        # allow lookup by alias (floor / ERP / drawing name)
        doc = await db[COLLECTIONS.machines].find_one({"aliases": machine_id})
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Machine '{machine_id}' not found",
        )
    return doc


@router.get("", response_model=list[MachineOut], summary="List machines")
async def list_machines(
    site_id: Optional[str] = Query(default=None, description="Filter by site id"),
    line_id: Optional[str] = Query(default=None, description="Filter by production line id"),
    status_filter: Optional[str] = Query(
        default=None, alias="status", description="Filter by operational status"
    ),
    tenant_id: str = Depends(get_current_tenant),
) -> list[dict]:
    """Return the tenant's machines, optionally filtered by site, line, or status."""
    db = get_tenant_scope(tenant_id)
    query: dict = {}
    if site_id:
        query["site_id"] = site_id
    if line_id:
        query["line_id"] = line_id
    if status_filter:
        query["status"] = status_filter

    cursor = db[COLLECTIONS.machines].find(query).sort(
        [("line_id", 1), ("position_in_line", 1)]
    )
    return [strip_mongo_id(doc) async for doc in cursor]


@router.get("/{machine_id}", response_model=MachineOut, summary="Get one machine")
async def get_machine(
    machine_id: str, tenant_id: str = Depends(get_current_tenant)
) -> dict:
    """Return a single machine by canonical id or alias."""
    doc = await _get_machine_or_404(tenant_id, machine_id)
    return strip_mongo_id(doc)


@router.get(
    "/{machine_id}/components",
    response_model=list[ComponentOut],
    summary="List a machine's components",
)
async def get_machine_components(
    machine_id: str, tenant_id: str = Depends(get_current_tenant)
) -> list[dict]:
    """Return the component tree (flat list) for a machine."""
    doc = await _get_machine_or_404(tenant_id, machine_id)
    db = get_tenant_scope(tenant_id)
    cursor = db[COLLECTIONS.components].find({"machine_id": doc["machine_id"]}).sort(
        [("parent_component_id", 1), ("component_id", 1)]
    )
    return [strip_mongo_id(c) async for c in cursor]


@router.get(
    "/{machine_id}/sensors",
    response_model=list[SensorOut],
    summary="List a machine's sensors",
)
async def get_machine_sensors(
    machine_id: str, tenant_id: str = Depends(get_current_tenant)
) -> list[dict]:
    """Return all sensors attached to a machine."""
    doc = await _get_machine_or_404(tenant_id, machine_id)
    db = get_tenant_scope(tenant_id)
    cursor = db[COLLECTIONS.sensors].find({"machine_id": doc["machine_id"]}).sort(
        [("sensor_id", 1)]
    )
    return [strip_mongo_id(s) async for s in cursor]
