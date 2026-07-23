"""Predictive-maintenance endpoints.

Every endpoint depends on the trained artifacts being loaded. When they are
not, the response is a 503 carrying the exact training commands — loud and
actionable, never a silently degraded heuristic answer.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.tenant_context import get_current_tenant
from app.schemas.pdm import FleetEntryOut, ModelInfoOut, PdmPredictionOut, TrendOut
from app.services.pdm import (
    InsufficientDataError,
    PdmArtifactsMissingError,
    PdmService,
    get_pdm_service,
)

router = APIRouter(prefix="/pdm", tags=["predictive-maintenance"])


def _service() -> PdmService:
    try:
        return get_pdm_service()
    except PdmArtifactsMissingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


# Fixed paths are declared before parameterised ones so /pdm/fleet can never
# be captured as machine_id="fleet".
@router.get("/fleet", response_model=list[FleetEntryOut], summary="Fleet risk ranking")
async def fleet(
    tenant_id: str = Depends(get_current_tenant),
) -> list[FleetEntryOut]:
    """Predictions for every machine in the tenant's fleet, most-at-risk first.

    Machines without enough data appear at the bottom with an ``error`` field
    rather than being silently dropped — an invisible machine is how a failing
    one gets missed.
    """
    return await _service().fleet(tenant_id)


@router.get("/model-info", response_model=ModelInfoOut, summary="Loaded models & metrics")
async def model_info() -> ModelInfoOut:
    """Model versions and their honest held-out metrics."""
    return _service().model_info()


@router.get(
    "/{machine_id}/prediction",
    response_model=PdmPredictionOut,
    summary="Prediction for one machine",
)
async def prediction(
    machine_id: str, tenant_id: str = Depends(get_current_tenant)
) -> PdmPredictionOut:
    """Failure probability, RUL, health, mode, trend and contributing features."""
    try:
        return await _service().predict(tenant_id, machine_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except InsufficientDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.get(
    "/{machine_id}/trend",
    response_model=TrendOut,
    summary="Health trajectory over a window",
)
async def trend(
    machine_id: str,
    hours: int = Query(default=168, ge=1, le=8760, description="Look-back window"),
    tenant_id: str = Depends(get_current_tenant),
) -> TrendOut:
    """Bucketed health/deviation series suitable for plotting."""
    try:
        return await _service().trend(tenant_id, machine_id, hours=hours)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except InsufficientDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
