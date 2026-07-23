"""Module selection: an LLM decides WHICH modules run, never what they say.

The router is the only place an LLM touches control flow, so its output is
treated as untrusted input:

* Module names are validated against :class:`ModuleName`. Anything else is
  dropped and recorded in ``dropped_modules`` — a hallucinated "WEATHER"
  module cannot cause a request to fail, and cannot cause one to run.
* If the LLM is absent, slow, or returns something that is not the expected
  JSON, routing falls back to a safe default set rather than erroring. A
  degraded route runs too much; a failed route runs nothing.

Every decision is logged with its reasoning, because the UI shows it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Awaitable, Callable, Optional

from app.config import get_settings
from app.schemas.orchestration import (
    Intent,
    ModuleName,
    RoutingDecision,
    RoutingSource,
    Urgency,
)

logger = logging.getLogger(__name__)

#: Used whenever the LLM cannot be trusted or reached: everything except the
#: resolver, which the graph adds anyway. Broad on purpose — an over-full run
#: costs latency, an under-full one costs an answer.
DEFAULT_MODULES: tuple[ModuleName, ...] = (
    ModuleName.rag,
    ModuleName.pdm,
    ModuleName.rca,
    ModuleName.maintenance,
    ModuleName.inventory,
    ModuleName.safety,
    ModuleName.production,
)

DEFAULT_ROUTER_MODEL = get_settings().anthropic_model
DEFAULT_ROUTER_TIMEOUT_SECONDS = 8.0

#: Machine-designator shapes: "CV-201", "MC110", "line A press".
_MACHINE_REF_RE = re.compile(r"\b[A-Za-z]{2,4}[- ]?\d{2,4}\b")
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_URGENCY_WORDS: tuple[tuple[Urgency, tuple[str, ...]], ...] = (
    (
        Urgency.critical,
        ("smoke", "fire", "burning", "injury", "injured", "emergency", "sparking"),
    ),
    (
        Urgency.high,
        ("stopped", "down", "urgent", "immediately", "asap", "alarm", "leaking",
         "grinding", "shutdown", "failed", "failure", "critical", "overheat"),
    ),
    (Urgency.low, ("routine", "whenever", "planning", "next month", "no rush")),
)

_INTENT_WORDS: tuple[tuple[Intent, tuple[str, ...]], ...] = (
    (
        Intent.request_procedure,
        ("how do i", "how to", "procedure", "steps", "walk me through", "sop",
         "instructions", "replace the", "lockout"),
    ),
    (
        Intent.assess_impact,
        ("impact", "cost", "downtime", "how long", "lose", "production hit",
         "schedule", "worth it"),
    ),
    (
        Intent.check_status,
        ("status", "how is", "how's", "health", "still ok", "condition of",
         "what's the state"),
    ),
    (
        Intent.report_fault,
        ("noise", "vibrat", "overheat", "leak", "smell", "fault", "alarm",
         "error", "not working", "stopped", "grinding", "smoke"),
    ),
)

#: Modules worth running per intent when the LLM does answer sensibly but the
#: request is narrow. Also used by the heuristic fallback's reasoning text.
_INTENT_HINTS: dict[Intent, tuple[ModuleName, ...]] = {
    Intent.report_fault: DEFAULT_MODULES,
    Intent.request_procedure: (
        ModuleName.rag,
        ModuleName.rca,
        ModuleName.maintenance,
        ModuleName.inventory,
        ModuleName.safety,
    ),
    Intent.assess_impact: (
        ModuleName.pdm,
        ModuleName.rca,
        ModuleName.inventory,
        ModuleName.production,
        ModuleName.safety,
    ),
    Intent.check_status: (ModuleName.pdm, ModuleName.rca, ModuleName.safety),
    Intent.ask_question: (ModuleName.rag, ModuleName.rca, ModuleName.safety),
}

_SYSTEM_PROMPT = (
    "You are the router for an industrial maintenance assistant. You do not "
    "answer the operator. You decide which analysis modules should run.\n\n"
    "Modules:\n"
    "  RAG         - retrieves manuals, SOPs and error-code documentation\n"
    "  PDM         - predictive model: failure probability, remaining useful life\n"
    "  RCA         - root-cause analysis from sensor signals and evidence\n"
    "  MAINTENANCE - repair procedure and required parts\n"
    "  INVENTORY   - spare-part availability and lead times\n"
    "  SAFETY      - hazards, PPE, lockout/tagout, permits\n"
    "  PRODUCTION  - downtime, units lost, cost, repair timing recommendation\n\n"
    "Rules:\n"
    "  - Return ONLY module names from that list.\n"
    "  - Include SAFETY whenever any physical work on the machine is implied.\n"
    "  - Prefer running too many modules over too few.\n"
    "Respond with a single JSON object and nothing else:\n"
    '{"modules": ["RCA", "SAFETY"], "intent": "REPORT_FAULT", '
    '"urgency": "HIGH", "machine_reference": "CV-201", '
    '"reasoning": "one sentence"}\n'
    "intent is one of REPORT_FAULT, ASK_QUESTION, CHECK_STATUS, "
    "REQUEST_PROCEDURE, ASSESS_IMPACT. urgency is one of LOW, NORMAL, HIGH, "
    "CRITICAL."
)

#: A callable that takes the prompt and returns the raw model text. Injectable
#: so tests can drive the router without a network or an API key.
LLMCall = Callable[[str], Awaitable[str]]


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------
def extract_machine_reference(message: str) -> Optional[str]:
    """Pull a machine-designator-shaped token out of free text, if there is one."""
    match = _MACHINE_REF_RE.search(message or "")
    return match.group(0) if match else None


def detect_urgency(message: str) -> Urgency:
    """Keyword urgency. Deliberately conservative: unknown means NORMAL."""
    lowered = (message or "").lower()
    for urgency, words in _URGENCY_WORDS:
        if any(word in lowered for word in words):
            return urgency
    return Urgency.normal


def detect_intent(message: str) -> Intent:
    """Keyword intent classification used when no LLM answered."""
    lowered = (message or "").lower()
    for intent, words in _INTENT_WORDS:
        if any(word in lowered for word in words):
            return intent
    return Intent.ask_question


def fallback_decision(message: str, reason: str, source: RoutingSource) -> RoutingDecision:
    """The safe default route: every module, plus a stated reason why."""
    return RoutingDecision(
        selected_modules=list(DEFAULT_MODULES),
        reasoning=(
            f"{reason} Falling back to the full module set so the request is "
            f"answered with everything available."
        ),
        source=source,
        intent=detect_intent(message),
        urgency=detect_urgency(message),
        machine_reference=extract_machine_reference(message),
    )


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------
def _coerce_modules(raw: Any) -> tuple[list[ModuleName], list[str]]:
    """Keep the names that are real modules; report the rest.

    RESOLVER is accepted but not meaningful — the graph always adds it.
    """
    if not isinstance(raw, (list, tuple)):
        return [], [] if raw is None else [str(raw)]

    known: list[ModuleName] = []
    dropped: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            dropped.append(str(item))
            continue
        try:
            known.append(ModuleName(item.strip().upper()))
        except ValueError:
            dropped.append(item)
    return known, dropped


def _coerce_enum(raw: Any, enum_cls, default):
    if isinstance(raw, str):
        try:
            return enum_cls(raw.strip().upper())
        except ValueError:
            return default
    return default


def parse_routing_response(payload: str, message: str) -> Optional[RoutingDecision]:
    """Parse and validate the router LLM's reply.

    Returns ``None`` when the reply is not usable — no JSON object in it, or no
    recognisable module names — which the caller turns into a fallback route.
    """
    if not payload or not payload.strip():
        return None

    match = _JSON_OBJECT_RE.search(payload)
    if match is None:
        return None

    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    modules, dropped = _coerce_modules(data.get("modules"))
    real_modules = [m for m in modules if m != ModuleName.resolver]
    if not real_modules:
        # Every name was junk, or the list was empty. Either way the LLM has
        # not made a decision we can act on.
        return None

    reasoning = data.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        reasoning = "Router returned a module selection with no stated reasoning."

    machine_reference = data.get("machine_reference")
    if not isinstance(machine_reference, str) or not machine_reference.strip():
        machine_reference = extract_machine_reference(message)

    return RoutingDecision(
        selected_modules=real_modules,
        reasoning=reasoning.strip(),
        source=RoutingSource.llm,
        intent=_coerce_enum(data.get("intent"), Intent, detect_intent(message)),
        urgency=_coerce_enum(data.get("urgency"), Urgency, detect_urgency(message)),
        machine_reference=machine_reference,
        dropped_modules=dropped,
    )


# ---------------------------------------------------------------------------
# The Anthropic-backed call
# ---------------------------------------------------------------------------
async def anthropic_router_call(prompt: str, *, model: str = DEFAULT_ROUTER_MODEL) -> str:
    """Ask Claude to route. Raises if no key or no SDK — the caller falls back.

    The SDK client is synchronous, so it runs in a worker thread; the caller
    puts the timeout around this coroutine.
    """
    api_key = get_settings().anthropic_api_key
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    import anthropic  # imported lazily: the package is optional

    client = anthropic.Anthropic(api_key=api_key)

    def _call() -> str:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

    return await asyncio.to_thread(_call)


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------
class LLMRouter:
    """Chooses the module set for a request."""

    def __init__(
        self,
        call_llm: Optional[LLMCall] = None,
        *,
        timeout_seconds: float = DEFAULT_ROUTER_TIMEOUT_SECONDS,
        model: str = DEFAULT_ROUTER_MODEL,
    ) -> None:
        self._call_llm = call_llm
        self._timeout = timeout_seconds
        self._model = model

    def _prompt(
        self,
        message: str,
        *,
        user_role: Optional[str],
        last_machine_id: Optional[str],
    ) -> str:
        lines = [f"Operator message: {message}"]
        if user_role:
            lines.append(f"Asking as: {user_role}")
        if last_machine_id:
            lines.append(
                f"Previous turn in this conversation was about machine "
                f"{last_machine_id}; the message may be a follow-up about it."
            )
        return "\n".join(lines)

    async def route(
        self,
        message: str,
        *,
        user_role: Optional[str] = None,
        last_machine_id: Optional[str] = None,
    ) -> RoutingDecision:
        """Return the modules to run, the intent, the urgency and the reasoning.

        Never raises: every failure path produces the default module set.
        """
        decision = await self._route_uncaught(
            message, user_role=user_role, last_machine_id=last_machine_id
        )

        logger.info(
            "Routing decision [%s]: modules=%s intent=%s urgency=%s machine=%s :: %s",
            decision.source,
            ",".join(str(ModuleName(m).value) for m in decision.selected_modules),
            decision.intent,
            decision.urgency,
            decision.machine_reference,
            decision.reasoning,
        )
        if decision.dropped_modules:
            logger.warning(
                "Router returned %d unknown module name(s), dropped: %s",
                len(decision.dropped_modules),
                ", ".join(decision.dropped_modules),
            )
        return decision

    async def _route_uncaught(
        self,
        message: str,
        *,
        user_role: Optional[str],
        last_machine_id: Optional[str],
    ) -> RoutingDecision:
        call = self._call_llm
        if call is None:
            if not get_settings().anthropic_api_key:
                return fallback_decision(
                    message,
                    "No routing LLM is configured (ANTHROPIC_API_KEY unset).",
                    RoutingSource.fallback_no_client,
                )

            async def call(prompt: str) -> str:  # noqa: F811 - bound per request
                return await anthropic_router_call(prompt, model=self._model)

        prompt = self._prompt(
            message, user_role=user_role, last_machine_id=last_machine_id
        )

        try:
            raw = await asyncio.wait_for(call(prompt), timeout=self._timeout)
        except asyncio.TimeoutError:
            return fallback_decision(
                message,
                f"Routing LLM did not respond within {self._timeout:g}s.",
                RoutingSource.fallback_timeout,
            )
        except Exception as exc:  # any SDK/network/import failure
            logger.warning("Routing LLM call failed: %s: %s", type(exc).__name__, exc)
            return fallback_decision(
                message,
                f"Routing LLM call failed ({type(exc).__name__}).",
                RoutingSource.fallback_error,
            )

        decision = parse_routing_response(raw if isinstance(raw, str) else "", message)
        if decision is None:
            return fallback_decision(
                message,
                "Routing LLM returned no usable module selection.",
                RoutingSource.fallback_invalid,
            )
        return decision


def explicit_decision(
    modules: list[ModuleName], reasoning: str, message: str = ""
) -> RoutingDecision:
    """A route chosen by the caller rather than by a model. Used by tests and
    by callers that already know exactly what they want run."""
    return RoutingDecision(
        selected_modules=list(modules),
        reasoning=reasoning,
        source=RoutingSource.explicit,
        intent=detect_intent(message),
        urgency=detect_urgency(message),
        machine_reference=extract_machine_reference(message),
    )
