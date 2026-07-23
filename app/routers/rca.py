"""RCA endpoints.

POST /rca/analyze  — full root-cause analysis for a machine
GET  /rca/{machine_id}/latest — most recent stored result (placeholder for persistence)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.tenant_context import get_current_tenant
from app.schemas.rca import RCARequest, RCAResult
from app.services.rca import get_rca_service

router = APIRouter(prefix="/rca", tags=["rca"])


@router.post("/analyze", response_model=RCAResult, summary="Run root-cause analysis")
async def analyze(
    body: RCARequest,
    tenant_id: str = Depends(get_current_tenant),
) -> RCAResult:
    """Analyze a machine's condition and identify the most likely root cause.

    Combines deterministic signal analysis, fault-signature matching, knowledge
    retrieval, and optional LLM narrative composition.
    """
    service = get_rca_service()
    try:
        return await service.analyze(
            tenant_id=tenant_id,
            machine_id=body.machine_id,
            include_narrative=body.include_narrative,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.get(
    "/{machine_id}/latest",
    response_model=RCAResult,
    summary="Latest RCA result for a machine",
)
async def latest(
    machine_id: str,
    tenant_id: str = Depends(get_current_tenant),
) -> RCAResult:
    """Return the most recent RCA result.

    Currently runs a fresh analysis. A future version will cache results and
    return the stored one, running fresh only on demand.
    """
    service = get_rca_service()
    try:
        return await service.analyze(
            tenant_id=tenant_id,
            machine_id=machine_id,
            include_narrative=False,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
