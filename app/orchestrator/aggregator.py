"""Aggregation: structured results in, one answer out.

Three things happen here, in this order, and the order matters.

1. **Hydration.** Each module's raw output is revalidated into its typed
   schema. A module whose data no longer fits its contract is treated as
   unavailable rather than passed through.

2. **Conflict rules.** These are code, not prompt. An LLM is never asked to
   decide whether safety outranks schedule, or whether a downtime estimate
   should include the parts wait — those answers are fixed:

   * Safety always wins. A blocking condition, or a CRITICAL/HIGH hazard,
     leads the response, and a REPAIR_NOW recommendation is annotated as
     subject to safety clearance. Routine preconditions (LOTO, PPE) do not
     trigger this — they apply to every job. Safety text is never shortened
     or dropped, for any role.
   * Blocking parts block. Maintenance steps needing an unavailable part are
     marked blocked, and Production's downtime becomes repair time *plus* the
     parts wait.
   * Thin evidence is stated as thin. If RCA reported ``insufficient_data``,
     everything downstream is labelled provisional.
   * Disagreements are surfaced, never silently reconciled.

3. **Role scoping, then composition.** Sections outside a role's remit are
   dropped — except safety-critical content, which every role receives — and
   only then is prose written, once, from what survived.

The composer LLM sees a *brief*: a flat, already-computed set of facts. It may
rephrase them and nothing else. Every number in the prose is checked against
the brief afterwards; one that is not there means the model invented something,
and the deterministic template is used instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Iterable, Mapping, Optional

from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.orchestrator.executor import ModuleOutcome
from app.schemas.agents import (
    InventoryStatus,
    MaintenancePlan,
    ProductionImpact,
    SafetyBriefing,
)
from app.schemas.orchestration import (
    BlockedStep,
    ModuleName,
    ModuleRun,
    ModuleStatus,
    NarrativeSource,
    OrchestrationCitation,
    OrchestrationResult,
    OrchestrationStatus,
    UserRole,
)
from app.schemas.pdm import PdmPredictionOut
from app.schemas.rca import RCAResult
from app.services.language import composer_language_instruction, language_name

logger = logging.getLogger(__name__)

DEFAULT_COMPOSER_MODEL = get_settings().anthropic_model
DEFAULT_COMPOSER_TIMEOUT_SECONDS = 25.0

REPAIR_NOW = "REPAIR_NOW"
MONITOR = "MONITOR"

#: Sections each role receives. Safety is in every set and is never removed by
#: scoping — the "always receives safety-critical warnings" rule is enforced by
#: construction, not by a check that could be forgotten.
ROLE_SECTIONS: dict[UserRole, frozenset[str]] = {
    UserRole.technician: frozenset({"safety", "maintenance", "inventory", "cause"}),
    UserRole.engineer: frozenset(
        {"safety", "maintenance", "inventory", "cause", "pdm", "production"}
    ),
    UserRole.manager: frozenset({"safety", "production", "inventory", "cause", "pdm"}),
    UserRole.safety_officer: frozenset(
        {"safety", "cause", "maintenance", "inventory"}
    ),
}

#: Order sections are narrated in, per role. Safety-critical alerts are hoisted
#: above all of this regardless.
SECTION_ORDER: dict[UserRole, tuple[str, ...]] = {
    UserRole.technician: ("safety", "cause", "maintenance", "inventory"),
    UserRole.engineer: (
        "cause",
        "pdm",
        "safety",
        "maintenance",
        "inventory",
        "production",
    ),
    UserRole.manager: ("safety", "production", "cause", "inventory", "pdm"),
    UserRole.safety_officer: ("safety", "cause", "maintenance", "inventory"),
}

#: Result attribute backing each section name.
_SECTION_FIELD = {
    "safety": "safety",
    "maintenance": "maintenance",
    "inventory": "inventory",
    "production": "production",
    "cause": "rca",
    "pdm": "pdm",
}

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
#: Steps that put a part into the machine — used to attribute a blocking part
#: to a step when the plan gives no component to key off.
_FITTING_VERBS = ("install", "replace", "fit", "refit", "reassemble", "mount", "seat")


# ---------------------------------------------------------------------------
# Hydration
# ---------------------------------------------------------------------------
def _as_model(model_cls: type[BaseModel], data: Any) -> Optional[BaseModel]:
    """Revalidate module output into its declared schema, or drop it."""
    if data is None:
        return None
    if isinstance(data, model_cls):
        return data
    try:
        if isinstance(data, BaseModel):
            return model_cls.model_validate(data.model_dump())
        return model_cls.model_validate(data)
    except ValidationError as exc:
        logger.warning("Discarding %s output that failed validation: %s", model_cls.__name__, exc)
        return None


def build_module_ledger(
    outcomes: Mapping[ModuleName, ModuleOutcome],
    selected: Iterable[ModuleName],
) -> list[ModuleRun]:
    """One row per selected module, in canonical order."""
    from app.orchestrator.graph import sort_modules

    rows: list[ModuleRun] = []
    for module in sort_modules(selected):
        outcome = outcomes.get(module)
        if outcome is None:
            rows.append(
                ModuleRun(
                    name=module,
                    status=ModuleStatus.skipped,
                    reason="Selected but never reached.",
                )
            )
            continue
        rows.append(
            ModuleRun(
                name=module,
                status=ModuleStatus(outcome.status),
                elapsed_ms=outcome.elapsed_ms,
                reason=outcome.reason,
                error_detail=outcome.error_detail,
                degraded_inputs=list(outcome.degraded_inputs),
                reused=outcome.reused,
            )
        )
    return rows


def _collect_citations(
    outcomes: Mapping[ModuleName, ModuleOutcome],
) -> list[OrchestrationCitation]:
    """Deduplicated citations from every module that produced any."""
    seen: set[tuple] = set()
    citations: list[OrchestrationCitation] = []
    for outcome in outcomes.values():
        for raw in outcome.citations or []:
            if not isinstance(raw, Mapping):
                continue
            document_id = raw.get("document_id")
            if not document_id:
                continue
            citation = OrchestrationCitation(
                document_id=str(document_id),
                title=raw.get("title") or raw.get("document_title"),
                page_number=raw.get("page_number"),
                section_title=raw.get("section_title"),
            )
            key = (
                citation.document_id,
                citation.page_number,
                citation.section_title,
            )
            if key in seen:
                continue
            seen.add(key)
            citations.append(citation)
    return citations


def hydrate(result: OrchestrationResult, outcomes: Mapping[ModuleName, ModuleOutcome]) -> None:
    """Fill the typed module fields on ``result`` from raw outcomes."""
    def data_for(module: ModuleName) -> Any:
        outcome = outcomes.get(module)
        if outcome is None or not outcome.usable:
            return None
        return outcome.data

    result.rca = _as_model(RCAResult, data_for(ModuleName.rca))
    result.pdm = _as_model(PdmPredictionOut, data_for(ModuleName.pdm))
    result.maintenance = _as_model(MaintenancePlan, data_for(ModuleName.maintenance))
    result.inventory = _as_model(InventoryStatus, data_for(ModuleName.inventory))
    result.safety = _as_model(SafetyBriefing, data_for(ModuleName.safety))
    result.production = _as_model(ProductionImpact, data_for(ModuleName.production))
    result.citations = _collect_citations(outcomes)


# ---------------------------------------------------------------------------
# Conflict rules
# ---------------------------------------------------------------------------
#: Severities serious enough to lead the response with a banner. Routine
#: preconditions (LOTO, PPE) never qualify no matter how they are phrased —
#: only the hazards the safety agent actually rated this severely do.
_SEVERE_SEVERITIES = frozenset({"CRITICAL", "HIGH"})


def _severe_hazards(safety: SafetyBriefing) -> list[str]:
    return [
        f"{h.hazard_type}: {h.description}"
        for h in safety.hazards
        if str(h.severity).upper() in _SEVERE_SEVERITIES
    ]


def _apply_safety_precedence(result: OrchestrationResult) -> list[str]:
    """Safety leads, and no recommendation may quietly outrank it.

    The banner is driven by ``blocking_conditions`` — genuine, machine-state
    blockers the safety agent identified — and by CRITICAL/HIGH hazards.
    ``standard_preconditions`` (routine LOTO steps, standard PPE) never
    trigger it: those apply to every job and would make the banner meaningless
    if they counted.
    """
    conflicts: list[str] = []
    safety = result.safety
    if safety is None:
        return conflicts

    severe = _severe_hazards(safety)
    blocking = list(safety.blocking_conditions)
    result.safety_critical = bool(severe or blocking)
    if not result.safety_critical:
        return conflicts

    production = result.production
    if production is not None and str(production.recommendation) == REPAIR_NOW:
        result.safety_clearance_required = True
        lead = blocking[0] if blocking else severe[0]
        note = (
            "Subject to safety clearance: this repair may not begin until the "
            f"safety condition is cleared and verified ({lead})."
        )
        if note not in production.recommendation_rationale:
            production.recommendation_rationale = (
                f"{production.recommendation_rationale} {note}".strip()
            )
        conflicts.append(
            "Production recommends REPAIR_NOW while Safety reports a "
            "blocking condition or a CRITICAL/HIGH hazard. The repair is "
            "subject to safety clearance; safety takes precedence over the "
            "schedule."
        )
    return conflicts


def _blocking_component_ids(
    plan: Optional[MaintenancePlan], blocking_parts: Iterable[str]
) -> dict[str, Optional[str]]:
    """Map each blocking part number to the component it belongs to, if known."""
    mapping: dict[str, Optional[str]] = {pn: None for pn in blocking_parts}
    if plan is None:
        return mapping
    for part in plan.required_parts:
        if part.part_number in mapping and part.component_id:
            mapping[part.part_number] = part.component_id
    return mapping


def _apply_inventory_blocking(result: OrchestrationResult) -> list[str]:
    """Blocking parts block steps, and lengthen the downtime estimate."""
    conflicts: list[str] = []
    inventory = result.inventory
    if inventory is None or not inventory.blocking_parts:
        return conflicts

    blocking = list(inventory.blocking_parts)
    wait_days = int(inventory.earliest_full_availability_days or 0)
    plan = result.maintenance

    # 1. Mark the steps that cannot proceed.
    if plan is not None:
        component_of = _blocking_component_ids(plan, blocking)
        unmapped = [pn for pn, cid in component_of.items() if cid is None]
        for step in plan.procedure_steps:
            named = [pn for pn in blocking if pn in (step.instruction or "")]
            by_component = [
                pn
                for pn, cid in component_of.items()
                if cid is not None and step.component_id == cid
            ]
            by_verb: list[str] = []
            if unmapped and any(
                verb in (step.instruction or "").lower() for verb in _FITTING_VERBS
            ):
                # No component mapping to key off: a step that fits a part is
                # the one the missing part blocks.
                by_verb = list(unmapped)

            blocked_by = sorted({*named, *by_component, *by_verb})
            if blocked_by:
                result.blocked_steps.append(
                    BlockedStep(
                        order=step.order,
                        instruction=step.instruction,
                        blocked_by_parts=blocked_by,
                    )
                )

        if result.blocked_steps:
            conflicts.append(
                "Maintenance requires parts that Inventory reports as "
                f"unavailable ({', '.join(blocking)}). "
                f"{len(result.blocked_steps)} procedure step(s) are blocked "
                f"until the parts arrive."
            )
        else:
            conflicts.append(
                "Inventory reports blocking parts "
                f"({', '.join(blocking)}) that could not be matched to a "
                "specific procedure step. Confirm parts on hand before starting."
            )

    # 2. Downtime must reflect the wait, not the bare repair time.
    production = result.production
    if production is not None:
        downtime = production.downtime_estimate_minutes
        wait_minutes = wait_days * 24 * 60
        expected_total = downtime.repair_time + wait_minutes
        if downtime.total_including_parts_wait < expected_total:
            previous = downtime.total_including_parts_wait
            # Recover the machine's hourly downtime cost from the estimate the
            # production agent already produced, so the corrected figure stays
            # consistent with its own assumptions.
            hourly = 0.0
            if previous > 0 and production.cost_estimate.downtime_cost > 0:
                hourly = production.cost_estimate.downtime_cost / (previous / 60.0)

            downtime.total_including_parts_wait = expected_total
            if hourly > 0:
                production.cost_estimate.downtime_cost = round(
                    hourly * expected_total / 60.0, 2
                )
                production.cost_estimate.total = round(
                    production.cost_estimate.downtime_cost
                    + production.cost_estimate.parts_cost,
                    2,
                )
            if previous > 0 and production.units_lost_estimate > 0:
                units_per_minute = production.units_lost_estimate / previous
                production.units_lost_estimate = int(units_per_minute * expected_total)

            production.assumptions.append(
                f"Downtime raised to {expected_total} minutes: repair time "
                f"{downtime.repair_time} minutes plus a {wait_days}-day wait "
                f"for blocking parts ({', '.join(blocking)})."
            )
            conflicts.append(
                "Production's downtime estimate excluded the parts wait. It now "
                f"uses {expected_total} minutes (repair {downtime.repair_time} "
                f"minutes + {wait_days} days waiting on {', '.join(blocking)})."
            )

    return conflicts


def _apply_evidence_quality(result: OrchestrationResult) -> list[str]:
    """Thin evidence is labelled, not hidden behind a confident tone."""
    conflicts: list[str] = []
    rca = result.rca
    if rca is None or not rca.insufficient_data:
        return conflicts

    result.provisional = True
    missing = ", ".join(rca.missing_data) if rca.missing_data else "not specified"
    conflicts.append(
        "RCA reported insufficient data "
        f"(confidence {round(rca.confidence * 100)}%; missing: {missing}). "
        "All recommendations below are provisional and must be confirmed "
        "before acting."
    )
    return conflicts


def _apply_contradictions(result: OrchestrationResult) -> list[str]:
    """Cross-module disagreements the reader should judge for themselves."""
    conflicts: list[str] = []
    rca, pdm, production, inventory, plan = (
        result.rca,
        result.pdm,
        result.production,
        result.inventory,
        result.maintenance,
    )

    if pdm is not None and production is not None:
        if pdm.failure_probability >= 0.7 and str(production.recommendation) == MONITOR:
            conflicts.append(
                f"PdM puts failure probability at "
                f"{round(pdm.failure_probability * 100)}% while Production "
                "recommends MONITOR. The two disagree on urgency."
            )

    if pdm is not None and rca is not None and rca.primary_cause is not None:
        predicted = (pdm.predicted_failure_mode or "").upper()
        identified = (rca.primary_cause.fault_mode or "").upper()
        if predicted and identified and predicted != identified:
            conflicts.append(
                f"PdM predicts fault mode {predicted} but RCA identifies "
                f"{identified}. The two models point at different failures."
            )

    if inventory is not None and plan is not None and inventory.all_parts_available:
        stocked = {item.part_number for item in inventory.items}
        unknown = sorted(
            part.part_number
            for part in plan.required_parts
            if part.part_number not in stocked
        )
        if unknown:
            conflicts.append(
                "Inventory reports all parts available, but Maintenance requires "
                f"part(s) it did not check: {', '.join(unknown)}."
            )

    return conflicts


def apply_conflict_rules(result: OrchestrationResult) -> None:
    """Run every conflict rule and record what they found, in priority order."""
    conflicts: list[str] = []
    conflicts.extend(_apply_safety_precedence(result))
    conflicts.extend(_apply_inventory_blocking(result))
    conflicts.extend(_apply_evidence_quality(result))
    conflicts.extend(_apply_contradictions(result))

    for conflict in conflicts:
        if conflict not in result.conflicts_surfaced:
            result.conflicts_surfaced.append(conflict)


# ---------------------------------------------------------------------------
# Role scoping
# ---------------------------------------------------------------------------
def apply_role_scope(result: OrchestrationResult) -> None:
    """Drop sections outside the role's remit. Safety is never dropped."""
    role = UserRole(result.user_role)
    allowed = ROLE_SECTIONS.get(role, ROLE_SECTIONS[UserRole.engineer])

    for section, attribute in _SECTION_FIELD.items():
        if section == "safety":
            continue  # every role gets safety, in full
        if section in allowed:
            continue
        if getattr(result, attribute, None) is None:
            continue
        setattr(result, attribute, None)
        if section not in result.omitted_for_role:
            result.omitted_for_role.append(section)


# ---------------------------------------------------------------------------
# The narrative brief
# ---------------------------------------------------------------------------
def _pct(value: Optional[float]) -> Optional[int]:
    return None if value is None else int(round(float(value) * 100))


def build_brief(result: OrchestrationResult) -> dict[str, Any]:
    """Flatten the scoped result into the facts prose may be written from.

    Every number the answer is allowed to contain is computed here — including
    derived ones such as hours — so that the template and the LLM draw from the
    same fixed pool and validation has something exact to check against.
    """
    role = UserRole(result.user_role)
    allowed = ROLE_SECTIONS.get(role, ROLE_SECTIONS[UserRole.engineer])

    brief: dict[str, Any] = {
        "role": role.value,
        "intent": str(result.intent),
        "urgency": str(result.urgency),
        "status": str(result.status),
        "safety_critical": result.safety_critical,
        "safety_clearance_required": result.safety_clearance_required,
        "provisional": result.provisional,
        "machine": None,
        "safety_alerts": [],
        "safety": None,
        "cause": None,
        "pdm": None,
        "maintenance": None,
        "inventory": None,
        "production": None,
        "conflicts": list(result.conflicts_surfaced),
        "unavailable_modules": [],
        "degraded_modules": [],
        "citations": [],
    }

    if result.machine is not None:
        brief["machine"] = result.machine.model_dump()

    if result.clarification is not None:
        brief["clarification"] = {
            "question": result.clarification.question,
            "candidates": [c.model_dump() for c in result.clarification.candidates],
        }

    for row in result.modules_run:
        if ModuleStatus(row.status) in (ModuleStatus.unavailable, ModuleStatus.skipped):
            brief["unavailable_modules"].append(
                {
                    "name": str(ModuleName(row.name).value),
                    "status": str(ModuleStatus(row.status).value),
                    "reason": row.reason or "no reason recorded",
                }
            )
        if row.degraded_inputs:
            brief["degraded_modules"].append(
                {
                    "name": str(ModuleName(row.name).value),
                    "missing": list(row.degraded_inputs),
                }
            )

    safety = result.safety
    if safety is not None:
        brief["safety_alerts"] = _severe_hazards(safety) + [
            f"Blocking condition: {c}" for c in safety.blocking_conditions
        ]
        brief["safety"] = {
            "hazards": [
                {
                    "type": h.hazard_type,
                    "severity": str(h.severity),
                    "description": h.description,
                }
                for h in safety.hazards
            ],
            "ppe": list(safety.required_ppe),
            "loto": [
                {
                    "order": s.order,
                    "instruction": s.instruction,
                    "verification": s.verification,
                }
                for s in safety.lockout_tagout_steps
            ],
            "energy_sources": [
                {
                    "type": str(e.type),
                    "location": e.location,
                    "isolation_method": e.isolation_method,
                }
                for e in safety.energy_sources_to_isolate
            ],
            "permits": list(safety.permits_required),
            "blocking_conditions": list(safety.blocking_conditions),
            "standard_preconditions": list(safety.standard_preconditions),
            "source": str(safety.source),
        }

    rca = result.rca
    if rca is not None and "cause" in allowed:
        cause: dict[str, Any] = {
            "confidence_pct": _pct(rca.confidence),
            "confidence_basis": rca.confidence_basis,
            "insufficient_data": rca.insufficient_data,
            "missing_data": list(rca.missing_data),
            "evidence_count": len(rca.evidence),
        }
        if rca.primary_cause is not None:
            cause.update(
                {
                    "description": rca.primary_cause.description,
                    "fault_mode": rca.primary_cause.fault_mode,
                    "component_id": rca.primary_cause.component_id,
                    "probability_pct": _pct(rca.primary_cause.probability),
                }
            )
        else:
            cause["description"] = "No single root cause could be identified."
        if role == UserRole.engineer:
            cause["causal_chain"] = [
                {"order": s.order, "description": s.description, "mechanism": s.mechanism}
                for s in rca.causal_chain
            ]
            cause["alternatives"] = [
                {
                    "description": alt.description,
                    "fault_mode": alt.fault_mode,
                    "probability_pct": _pct(alt.probability),
                }
                for alt in rca.alternative_causes
            ]
        brief["cause"] = cause

    pdm = result.pdm
    if pdm is not None and "pdm" in allowed:
        brief["pdm"] = {
            "failure_probability_pct": _pct(pdm.failure_probability),
            "remaining_useful_life_hours": round(pdm.remaining_useful_life_hours, 1),
            "health_pct": _pct(pdm.health_score),
            "trend": str(pdm.trend_direction),
            "predicted_failure_mode": pdm.predicted_failure_mode,
            "confidence_pct": _pct(pdm.confidence),
            "readings_used": pdm.readings_used,
        }

    plan = result.maintenance
    if plan is not None and "maintenance" in allowed:
        blocked_by_order = {b.order: b.blocked_by_parts for b in result.blocked_steps}
        brief["maintenance"] = {
            "steps": [
                {
                    "order": step.order,
                    "instruction": step.instruction,
                    "minutes": step.estimated_minutes,
                    "caution": step.caution,
                    "blocked": step.order in blocked_by_order,
                    "blocked_by_parts": blocked_by_order.get(step.order, []),
                }
                for step in plan.procedure_steps
            ],
            "step_count": len(plan.procedure_steps),
            "total_minutes": plan.total_estimated_minutes,
            "total_hours": round(plan.total_estimated_minutes / 60.0, 1),
            "skill_level": str(plan.skill_level),
            "procedure_source": str(plan.procedure_source),
            "tools": list(plan.required_tools),
            "parts": [
                {
                    "part_number": part.part_number,
                    "description": part.description,
                    "quantity": part.quantity,
                }
                for part in plan.required_parts
            ],
        }

    inventory = result.inventory
    if inventory is not None and "inventory" in allowed:
        include_location = role in (UserRole.technician, UserRole.engineer)
        brief["inventory"] = {
            "items": [
                {
                    "part_number": item.part_number,
                    "description": item.description,
                    "status": str(item.status),
                    "available_qty": item.available_qty,
                    "required_qty": item.required_qty,
                    "location": item.location if include_location else None,
                    "lead_time_days": item.lead_time_days,
                    "alternatives": list(item.alternatives),
                }
                for item in inventory.items
            ],
            "all_parts_available": inventory.all_parts_available,
            "blocking_parts": list(inventory.blocking_parts),
            "earliest_availability_days": inventory.earliest_full_availability_days,
        }

    production = result.production
    if production is not None and "production" in allowed:
        downtime = production.downtime_estimate_minutes
        cost = production.cost_estimate
        brief["production"] = {
            "repair_minutes": downtime.repair_time,
            "total_downtime_minutes": downtime.total_including_parts_wait,
            "total_downtime_hours": round(downtime.total_including_parts_wait / 60.0, 1),
            "units_lost": production.units_lost_estimate,
            "is_bottleneck": production.is_bottleneck,
            "downstream_machines": list(production.downstream_machines_affected),
            "recommendation": str(production.recommendation),
            "rationale": production.recommendation_rationale,
            "assumptions": list(production.assumptions),
            "downtime_cost": cost.downtime_cost,
            "parts_cost": cost.parts_cost,
            "total_cost": cost.total,
            "currency": cost.currency,
        }

    brief["citations"] = [c.model_dump() for c in result.citations]
    return brief


# ---------------------------------------------------------------------------
# Deterministic rendering
# ---------------------------------------------------------------------------
def _render_safety(brief: Mapping[str, Any]) -> list[str]:
    safety = brief.get("safety")
    if not safety:
        return []
    lines = ["SAFETY BRIEFING"]
    if safety["hazards"]:
        lines.append("Hazards:")
        lines += [
            f"  - [{h['severity']}] {h['type']}: {h['description']}"
            for h in safety["hazards"]
        ]
    if safety["blocking_conditions"]:
        lines.append("Do not start work while any of these hold:")
        lines += [f"  - {c}" for c in safety["blocking_conditions"]]
    if safety["standard_preconditions"]:
        lines.append("Standard preconditions:")
        lines += [f"  - {c}" for c in safety["standard_preconditions"]]
    if safety["ppe"]:
        lines.append("Required PPE: " + ", ".join(safety["ppe"]))
    if safety["energy_sources"]:
        lines.append("Energy sources to isolate:")
        lines += [
            f"  - {e['type']} at {e['location']} — {e['isolation_method']}"
            for e in safety["energy_sources"]
        ]
    if safety["loto"]:
        lines.append("Lockout/tagout:")
        lines += [
            f"  {s['order']}. {s['instruction']} (verify: {s['verification']})"
            for s in safety["loto"]
        ]
    if safety["permits"]:
        lines.append("Permits required: " + ", ".join(safety["permits"]))
    if safety["source"] == "GENERIC":
        lines.append(
            "This briefing is generic industrial guidance — no machine-specific "
            "documented procedure was found. Verify against site policy."
        )
    return lines


def _render_cause(brief: Mapping[str, Any]) -> list[str]:
    cause = brief.get("cause")
    if not cause:
        return []
    lines = ["ROOT CAUSE"]
    headline = cause.get("description", "")
    if cause.get("fault_mode"):
        headline = f"{headline} (fault mode {cause['fault_mode']})"
    if cause.get("component_id"):
        headline = f"{headline}, localised to component {cause['component_id']}"
    lines.append(headline.strip())
    confidence = cause.get("confidence_pct")
    if confidence is not None:
        note = f"Confidence: {confidence}%. {cause.get('confidence_basis', '')}".strip()
        lines.append(note)
    if cause.get("insufficient_data"):
        lines.append(
            "Evidence is insufficient for a firm conclusion — treat this cause "
            "and everything derived from it as provisional."
        )
    for step in cause.get("causal_chain", []):
        lines.append(f"  {step['order']}. {step['description']} — {step['mechanism']}")
    for alt in cause.get("alternatives", []):
        lines.append(
            f"  Alternative: {alt['description']} "
            f"({alt['fault_mode']}, {alt['probability_pct']}%)"
        )
    return lines


def _render_pdm(brief: Mapping[str, Any]) -> list[str]:
    pdm = brief.get("pdm")
    if not pdm:
        return []
    return [
        "PREDICTIVE MODEL",
        (
            f"Failure probability {pdm['failure_probability_pct']}%, health score "
            f"{pdm['health_pct']}%, remaining useful life "
            f"{pdm['remaining_useful_life_hours']} hours, trend {pdm['trend']}."
        ),
    ]


def _render_maintenance(brief: Mapping[str, Any]) -> list[str]:
    plan = brief.get("maintenance")
    if not plan:
        return []
    lines = [
        "REPAIR PROCEDURE",
        (
            f"{plan['step_count']} steps, about {plan['total_minutes']} minutes "
            f"({plan['total_hours']} hours), {plan['skill_level']} skill level, "
            f"procedure {plan['procedure_source']}."
        ),
    ]
    for step in plan["steps"]:
        suffix = ""
        if step["caution"]:
            suffix += f" CAUTION: {step['caution']}"
        if step["blocked"]:
            suffix += (
                " BLOCKED — waiting on part(s) "
                + ", ".join(step["blocked_by_parts"])
                + "."
            )
        lines.append(f"  {step['order']}. {step['instruction']}{suffix}")
    if plan["parts"]:
        lines.append(
            "Parts needed: "
            + ", ".join(
                f"{p['part_number']} x{p['quantity']} ({p['description']})"
                for p in plan["parts"]
            )
        )
    if plan["tools"]:
        lines.append("Tools: " + ", ".join(plan["tools"]))
    return lines


def _render_inventory(brief: Mapping[str, Any]) -> list[str]:
    inventory = brief.get("inventory")
    if not inventory:
        return []
    lines = ["PARTS"]
    for item in inventory["items"]:
        detail = (
            f"  - {item['part_number']} ({item['description']}): {item['status']}, "
            f"{item['available_qty']} on hand"
        )
        if item["location"]:
            detail += f", location {item['location']}"
        if item["lead_time_days"]:
            detail += f", lead time {item['lead_time_days']} days"
        if item["alternatives"]:
            detail += ", alternatives in stock: " + ", ".join(item["alternatives"])
        lines.append(detail)
    if inventory["blocking_parts"]:
        lines.append(
            "Blocking: "
            + ", ".join(inventory["blocking_parts"])
            + f". Earliest full availability {inventory['earliest_availability_days']} days."
        )
    elif inventory["items"]:
        lines.append("All required parts are available.")
    return lines


def _render_production(brief: Mapping[str, Any]) -> list[str]:
    production = brief.get("production")
    if not production:
        return []
    lines = [
        "PRODUCTION IMPACT",
        (
            f"Downtime {production['total_downtime_minutes']} minutes "
            f"({production['total_downtime_hours']} hours), of which "
            f"{production['repair_minutes']} minutes is repair work. "
            f"Estimated {production['units_lost']} units lost."
        ),
        (
            f"Cost: {production['currency']} {production['downtime_cost']} downtime "
            f"+ {production['parts_cost']} parts = {production['total_cost']} total."
        ),
        f"Recommendation: {production['recommendation']}. {production['rationale']}",
    ]
    if production["is_bottleneck"]:
        downstream = ", ".join(production["downstream_machines"])
        lines.append(
            "This machine is a line bottleneck"
            + (f"; downstream machines affected: {downstream}." if downstream else ".")
        )
    return lines


_RENDERERS = {
    "safety": _render_safety,
    "cause": _render_cause,
    "pdm": _render_pdm,
    "maintenance": _render_maintenance,
    "inventory": _render_inventory,
    "production": _render_production,
}


def render_template(brief: Mapping[str, Any]) -> str:
    """Deterministic prose. Always available; used whenever the LLM is not.

    Uses only values present in ``brief``, which is what makes it a safe
    fallback when an LLM narrative fails validation.
    """
    role = UserRole(brief["role"])
    blocks: list[str] = []

    machine = brief.get("machine")
    if machine:
        blocks.append(
            f"{machine['name']} ({machine['machine_id']}, {machine['model']}) "
            f"on line {machine['line_id']} — status {machine['status']}."
        )

    clarification = brief.get("clarification")
    if clarification:
        lines = [clarification["question"]]
        lines += [
            f"  - {c['machine_id']}: {c['name']} ({c['model']}) on line {c['line_id']}"
            for c in clarification["candidates"]
        ]
        blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    if brief["safety_critical"] and brief["safety_alerts"]:
        alerts = "\n".join(f"  - {a}" for a in brief["safety_alerts"])
        blocks.append(
            "SAFETY CRITICAL — read before anything else:\n"
            + alerts
            + (
                "\nAny recommendation to repair now is subject to safety clearance."
                if brief["safety_clearance_required"]
                else ""
            )
        )

    if brief["provisional"]:
        blocks.append(
            "PROVISIONAL: root-cause evidence was insufficient, so the "
            "recommendations below are not confirmed."
        )

    for section in SECTION_ORDER.get(role, SECTION_ORDER[UserRole.engineer]):
        lines = _RENDERERS[section](brief)
        if lines:
            blocks.append("\n".join(lines))

    if brief["conflicts"]:
        blocks.append(
            "UNRESOLVED CONFLICTS\n"
            + "\n".join(f"  - {c}" for c in brief["conflicts"])
        )

    if brief.get("degraded_modules"):
        blocks.append(
            "PARTIAL INPUT\n"
            + "\n".join(
                f"  - {m['name']} proceeded without: {', '.join(m['missing'])}."
                for m in brief["degraded_modules"]
            )
        )

    if brief["unavailable_modules"]:
        blocks.append(
            "NOT AVAILABLE FOR THIS ANSWER\n"
            + "\n".join(
                f"  - {m['name']} ({m['status']}): {m['reason']}"
                for m in brief["unavailable_modules"]
            )
        )

    if brief["citations"]:
        blocks.append(
            "SOURCES\n"
            + "\n".join(
                "  - "
                + ", ".join(
                    str(part)
                    for part in (
                        c.get("title") or c.get("document_id"),
                        f"p.{c['page_number']}" if c.get("page_number") else None,
                        c.get("section_title"),
                    )
                    if part
                )
                for c in brief["citations"]
            )
        )

    if not blocks:
        blocks.append("No module produced a result for this request.")

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Narrative validation
# ---------------------------------------------------------------------------
def _normalize_number(token: str) -> str:
    cleaned = token.replace(",", "").rstrip(".")
    return cleaned or token


def numbers_in_text(text: str) -> list[str]:
    """Every number-shaped token in ``text``, normalised."""
    return [_normalize_number(m.group(0)) for m in _NUMBER_RE.finditer(text or "")]


def _variants(value: float) -> set[str]:
    """Every written form a number from the brief may legitimately take."""
    out: set[str] = set()
    as_float = float(value)
    rounded = round(as_float)
    out.update({str(rounded), f"{rounded}.0", f"{rounded}.00"})
    out.update({f"{as_float:.1f}", f"{as_float:.2f}", f"{as_float:g}"})
    out.add(str(int(as_float)))
    # A ratio in the brief may be written as a percentage in the prose.
    if 0.0 <= as_float <= 1.0:
        pct = round(as_float * 100)
        out.update({str(pct), f"{pct}.0"})
    return {_normalize_number(v) for v in out}


def allowed_numbers(brief: Any) -> set[str]:
    """The pool of numbers the prose may contain.

    Anything numeric in the brief, anything numeric *inside* a brief string
    (part numbers, machine ids, quoted figures), and the index/length of every
    list — prose is allowed to say "5 steps" about a five-element list.
    """
    allowed: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            allowed.update(_variants(node))
            return
        if isinstance(node, str):
            allowed.update(numbers_in_text(node))
            return
        if isinstance(node, Mapping):
            for value in node.values():
                walk(value)
            return
        if isinstance(node, (list, tuple)):
            allowed.update(_variants(len(node)))
            allowed.update(str(i) for i in range(1, len(node) + 1))
            for value in node:
                walk(value)

    walk(brief)
    return allowed


def validate_narrative(text: str, brief: Any) -> list[str]:
    """Numbers in ``text`` that do not exist in ``brief``. Empty means clean."""
    permitted = allowed_numbers(brief)
    return sorted({n for n in numbers_in_text(text) if n not in permitted})


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
_COMPOSER_SYSTEM = (
    "You write the final answer for an industrial maintenance assistant.\n\n"
    "You are given a JSON brief of findings that other systems produced. Turn "
    "it into clear prose for the stated role.\n\n"
    "Absolute rules:\n"
    "  - Use ONLY facts present in the brief. Do not add causes, numbers, "
    "part availability, costs, timings or safety steps that are not there.\n"
    "  - Every number you write must appear in the brief. Do not compute new "
    "ones, do not round differently, do not estimate.\n"
    "  - If safety_critical is true, lead with the safety alerts and state "
    "them in full. Never shorten, soften or omit a safety requirement, for "
    "any reason including time or cost.\n"
    "  - If provisional is true, say plainly that the findings are provisional.\n"
    "  - Report every entry in conflicts. Do not resolve or reconcile them.\n"
    "  - If a module is listed in unavailable_modules, say what is missing.\n"
    "  - If a module is listed in degraded_modules, add one brief note that it "
    "proceeded without that input. Do not speculate about what the missing "
    "input would have shown.\n"
    "Write plain prose with short headed sections. No preamble."
)


def composer_system(language: Optional[str] = None) -> str:
    """The composer system prompt, with a language directive when needed.

    English adds nothing — the base prompt already produces English. Any other
    language appends the directive *and* the rule that identifiers are not
    words and must survive translation untouched.
    """
    directive = composer_language_instruction(language)
    if not directive:
        return _COMPOSER_SYSTEM
    return f"{_COMPOSER_SYSTEM}\n\n{directive}"


async def anthropic_compose_call(
    prompt: str,
    *,
    model: str = DEFAULT_COMPOSER_MODEL,
    language: Optional[str] = None,
) -> str:
    """Ask Claude to write the answer. Raises without a key or SDK."""
    api_key = get_settings().anthropic_api_key
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    import anthropic  # optional dependency

    client = anthropic.Anthropic(api_key=api_key)
    system = composer_system(language)

    def _call() -> str:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text
            for block in response.content
            if getattr(block, "type", "") == "text"
        )

    return await asyncio.to_thread(_call)


def build_composer_prompt(
    brief: Mapping[str, Any], language: Optional[str] = None
) -> str:
    """The user-turn prompt handed to the composer, streaming or not.

    Shared so a streamed composition and a blocking one are given exactly the
    same instructions, and validation therefore judges both by the same rules.
    """
    lines = [
        f"Role reading this answer: {brief['role']}",
        f"Operator intent: {brief['intent']} (urgency {brief['urgency']})",
    ]
    directive = composer_language_instruction(language)
    if directive:
        lines.append(f"Answer language: {language_name(language)} ({language})")
    return "\n".join(lines) + f"\n\nBrief:\n{json.dumps(brief, indent=2, default=str)}"


async def compose_narrative(
    result: OrchestrationResult,
    *,
    call_llm=None,
    timeout_seconds: float = DEFAULT_COMPOSER_TIMEOUT_SECONDS,
    model: str = DEFAULT_COMPOSER_MODEL,
    language: Optional[str] = None,
) -> tuple[str, NarrativeSource]:
    """Write the answer once, from the structured result.

    Falls back to the template when there is no LLM, when the call fails, or
    when the prose contains a number the structured data does not — a
    fabricated figure is treated as a failed generation, not a stylistic issue.

    ``language`` only reaches the LLM. The template is English and stays
    English; the caller marks that with ``language_fallback``.
    """
    brief = build_brief(result)
    template = render_template(brief)

    if call_llm is None:
        if not get_settings().anthropic_api_key:
            return template, NarrativeSource.template

        async def call_llm(prompt: str) -> str:  # noqa: F811 - bound per request
            return await anthropic_compose_call(prompt, model=model, language=language)

    prompt = build_composer_prompt(brief, language)

    try:
        text = await asyncio.wait_for(call_llm(prompt), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning("Narrative composer timed out; using template rendering")
        return template, NarrativeSource.template
    except Exception as exc:
        logger.warning(
            "Narrative composer failed (%s: %s); using template rendering",
            type(exc).__name__,
            exc,
        )
        return template, NarrativeSource.template

    if not isinstance(text, str) or not text.strip():
        return template, NarrativeSource.template

    fabricated = validate_narrative(text, brief)
    if fabricated:
        logger.warning(
            "Composed narrative contained %d number(s) absent from the "
            "structured data (%s); falling back to template rendering",
            len(fabricated),
            ", ".join(fabricated),
        )
        return template, NarrativeSource.template

    return text.strip(), NarrativeSource.llm


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def aggregate(
    result: OrchestrationResult,
    outcomes: Mapping[ModuleName, ModuleOutcome],
    selected: Iterable[ModuleName],
) -> OrchestrationResult:
    """Hydrate, resolve conflicts, scope to the role, and set the final status.

    Narrative composition is deliberately not done here — it is async, and it
    is the last step, after everything structural has settled.
    """
    hydrate(result, outcomes)
    result.modules_run = build_module_ledger(outcomes, selected)
    apply_conflict_rules(result)
    apply_role_scope(result)

    degraded = any(
        ModuleStatus(row.status)
        in (ModuleStatus.unavailable, ModuleStatus.skipped, ModuleStatus.partial)
        for row in result.modules_run
    )
    result.status = (
        OrchestrationStatus.partial if degraded else OrchestrationStatus.complete
    )
    return result
