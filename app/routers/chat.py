"""Chat endpoints.

POST   /chat/stream                — Server-Sent Events, the orchestration live
POST   /chat                       — the same answer, in one piece
GET    /chat/sessions/{session_id} — turn history for a conversation
DELETE /chat/sessions/{session_id} — forget a conversation

The two POST routes take the same body and produce the same information; they
differ only in delivery. A client that cannot or does not want to consume SSE
uses ``/chat`` and gets the complete ``OrchestrationResult``; everything the
stream emits along the way is derivable from that object, which is why the
non-streaming route is a thin call rather than a second implementation.

As with ``/orchestrate``, a degraded answer is still an answer: module
failures come back 200 with ``status=PARTIAL``, and the blocking resolution
states carry the candidates and the question the caller needs to put to the
operator. On the streaming route this extends to errors — the response has
already begun with a 200 by the time anything can go wrong, so failures are
delivered as ``error`` events rather than as status codes.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

from app.core.tenant_context import get_current_tenant
from app.orchestrator.orchestrator import get_orchestrator
from app.schemas.orchestration import (
    OrchestrationResult,
    SessionOut,
    SessionTurn,
    UserRole,
)
from app.schemas.stream import ChatRequest
from app.services.chat import ChatService, get_chat_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

#: Sent with every stream. ``no-cache`` and ``no-transform`` keep intermediaries
#: from buffering or rewriting frames, and ``X-Accel-Buffering`` turns off nginx
#: proxy buffering specifically — without it nginx holds the whole response and
#: the stream arrives as one block at the end, which defeats the entire feature.
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post(
    "/stream",
    summary="Answer an operator request as a live event stream",
    response_class=StreamingResponse,
    response_description=(
        "text/event-stream carrying session, routing, resolution, module_start, "
        "module_finish, narrative_delta, citation, conflict, result, error and "
        "done events, in the order they occurred."
    ),
)
async def chat_stream(
    body: ChatRequest,
    request: Request,
    tenant_id: str = Depends(get_current_tenant),
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """Stream one orchestration as Server-Sent Events."""

    async def frames():
        async for message in service.stream(
            tenant_id=tenant_id,
            message=body.message,
            user_role=UserRole(body.user_role),
            session_id=body.session_id,
            machine_id=body.machine_id,
            language=body.language,
            is_disconnected=request.is_disconnected,
        ):
            yield message.to_sse()

    return StreamingResponse(
        frames(), media_type="text/event-stream", headers=SSE_HEADERS
    )


@router.post(
    "",
    response_model=OrchestrationResult,
    summary="Answer an operator request in one piece",
    response_description=(
        "COMPLETE or PARTIAL with the composed answer, or "
        "CLARIFICATION_NEEDED/NOT_FOUND when the machine could not be "
        "confirmed — in which case no module was run."
    ),
)
async def chat(
    body: ChatRequest,
    tenant_id: str = Depends(get_current_tenant),
    service: ChatService = Depends(get_chat_service),
) -> OrchestrationResult:
    """The non-streaming path, for clients that do not want SSE."""
    return await service.handle(
        tenant_id=tenant_id,
        message=body.message,
        user_role=UserRole(body.user_role),
        session_id=body.session_id,
        machine_id=body.machine_id,
        language=body.language,
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


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation",
)
async def delete_session(
    session_id: str,
    tenant_id: str = Depends(get_current_tenant),
) -> Response:
    """Forget a conversation: its turns and its cached module output.

    404s for a session this tenant does not own, for the same reason the read
    does — "not yours" and "not there" must look identical from outside.
    """
    orchestrator = get_orchestrator()
    if not await orchestrator.sessions.delete(tenant_id, session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No conversation '{session_id}' for this tenant.",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
