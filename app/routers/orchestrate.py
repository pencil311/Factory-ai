"""Orchestration endpoints.

POST /orchestrate                        — one question in, one answer out
GET  /orchestrate/sessions/{session_id}  — turn history for a conversation

A degraded answer is still an answer: a request where modules failed comes back
200 with ``status=PARTIAL`` and a ledger saying which ones and why. The same
goes for the blocking resolution states — ``CLARIFICATION_NEEDED`` and
``NOT_FOUND`` carry the candidates and the question the caller needs to put to
the operator, and an error status would throw that payload away.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.tenant_context import get_current_tenant
from app.orchestrator.orchestrator import get_orchestrator
from app.schemas.orchestration import (
    OrchestrateRequest,
    OrchestrationResult,
    SessionOut,
    SessionTurn,
    UserRole,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orchestrate", tags=["orchestration"])


@router.post(
    "",
    response_model=OrchestrationResult,
    summary="Answer an operator request end to end",
    response_description=(
        "COMPLETE or PARTIAL with the composed answer, or "
        "CLARIFICATION_NEEDED/NOT_FOUND when the machine could not be confirmed "
        "— in which case no module was run."
    ),
)
async def orchestrate(
    body: OrchestrateRequest,
    tenant_id: str = Depends(get_current_tenant),
) -> OrchestrationResult:
    """Route, resolve, run the module graph, and compose one answer."""
    orchestrator = get_orchestrator()
    return await orchestrator.handle(
        tenant_id=tenant_id,
        message=body.message,
        user_role=UserRole(body.user_role),
        session_id=body.session_id,
        machine_id=body.machine_id,
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionOut,
    summary="Conversation history for a session",
)
async def get_session(
    session_id: str,
    tenant_id: str = Depends(get_current_tenant),
) -> SessionOut:
    """Return a session's turns and the machine it is currently about.

    Scoped to the calling tenant: a session id belonging to another tenant is
    indistinguishable from one that does not exist.
    """
    orchestrator = get_orchestrator()
    document = await orchestrator.sessions.load(tenant_id, session_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No conversation '{session_id}' for this tenant.",
        )

    cache = document.get("cache") or {}
    cached_modules = sorted((cache.get("modules") or {}).keys())
    _, age = orchestrator.sessions.fresh_cache(document, cache.get("machine_id", ""))

    turns: list[SessionTurn] = []
    for raw in document.get("turns") or []:
        try:
            turns.append(SessionTurn.model_validate(raw))
        except Exception:
            logger.warning("Skipping malformed turn in session '%s'", session_id)

    return SessionOut(
        session_id=session_id,
        tenant_id=tenant_id,
        last_machine_id=document.get("last_machine_id"),
        turns=turns,
        created_at=document.get("created_at"),
        updated_at=document.get("updated_at"),
        cached_modules=cached_modules,
        cache_age_seconds=age,
    )
