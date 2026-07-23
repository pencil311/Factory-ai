"""Abstract agent base class.

Every agent owns exactly one domain, receives structured input, and returns
structured output. Agents do not call each other. An agent with nothing to
contribute returns UNAVAILABLE with a reason — it must never fabricate.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from app.core.errors import describe_failure
from app.schemas.agents import AgentContext, AgentResult, AgentStatus


class Agent(ABC):
    """Base for all domain agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name, e.g. 'maintenance'."""
        ...

    @abstractmethod
    async def _run(self, context: AgentContext) -> tuple[AgentStatus, Any, str | None, list[dict]]:
        """Subclass implementation. Returns (status, data, reason, citations)."""
        ...

    async def run(self, context: AgentContext) -> AgentResult:
        """Execute the agent and wrap the result."""
        start = time.monotonic()
        try:
            status, data, reason, citations = await self._run(context)
        except Exception as exc:
            # The exception's own message is exactly where a raw driver or
            # server error would leak into a narrative; reason stays short and
            # safe, error_detail keeps the full text for debugging only.
            reason, detail = describe_failure(exc, label=f"{self.name.title()} agent")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.unavailable,
                reason=reason,
                error_detail=detail,
                elapsed_ms=int((time.monotonic() - start) * 1000),
            )
        return AgentResult(
            agent_name=self.name,
            status=status,
            data=data.model_dump() if hasattr(data, "model_dump") else data,
            reason=reason,
            citations=citations,
            elapsed_ms=int((time.monotonic() - start) * 1000),
        )
