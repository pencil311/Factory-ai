"""Entity-resolution endpoint.

Note that AMBIGUOUS and NOT_FOUND are returned as HTTP 200 with a blocking
``status``, not as 4xx errors. They are valid, expected answers that carry the
candidates and the clarification question the caller needs in order to ask the
operator — an error status would throw that payload away.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.tenant_context import get_current_tenant
from app.schemas.resolution import ResolutionRequest, ResolutionResult
from app.services.resolver import resolve_machine

router = APIRouter(prefix="/resolve", tags=["resolution"])


@router.post(
    "",
    response_model=ResolutionResult,
    summary="Resolve free text to exactly one machine",
    response_description=(
        "RESOLVED with a single machine, or AMBIGUOUS/NOT_FOUND — both of which "
        "block: the caller must ask the operator rather than proceed."
    ),
)
async def resolve(
    payload: ResolutionRequest, tenant_id: str = Depends(get_current_tenant)
) -> ResolutionResult:
    """Resolve messy operator input to a canonical machine within the tenant.

    Never guesses. If the input maps to more than one plausible machine, the
    response carries ranked candidates and a question naming the specific
    options. Candidates come only from the requesting tenant's fleet.
    """
    return await resolve_machine(
        text=payload.text, tenant_id=tenant_id, context=payload.context
    )
