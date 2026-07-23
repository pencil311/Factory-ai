"""Root-cause analysis service.

Hybrid approach: deterministic signal analysis first, then fault-signature
matching, then RAG retrieval for documented causes, and LLM synthesis last
(only for narrative composition, never for evidence selection).

RCA explains WHY. It does not prescribe repairs.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np

from app.config import get_settings
from app.db import get_tenant_scope, normalize_tenant_id
from app.schemas.machine import COLLECTIONS
from app.schemas.rca import (
    CausalHypothesis,
    CausalStep,
    Citation,
    Evidence,
    EvidenceSource,
    EvidenceStrength,
    RCAResult,
)
from app.sensors.simulator import FAULT_MODELS, FaultType, SensorEffect

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fault signatures — derived from the simulator's fault taxonomy
# ---------------------------------------------------------------------------

# Each fault type maps to expected sensor patterns: which sensors move, in what
# order (by lag), and what direction. This is the same knowledge encoded in the
# simulator, extracted here for matching against observed signals.

_FAULT_COMPONENT_TYPES: dict[str, list[str]] = {
    ft.value: [ct.value for ct in model.component_types]
    for ft, model in FAULT_MODELS.items()
}

_FAULT_DESCRIPTIONS: dict[str, str] = {
    ft.value: model.description for ft, model in FAULT_MODELS.items()
}

#: Trend classification, in span-fractions per HOUR — not per reading.
#:
#: A slope measured "per reading" quietly shrinks toward zero as more
#: readings accumulate in the lookback window, independent of how fast the
#: sensor is actually moving: the same physical rate of change (mm/s per
#: hour) produces an ever-smaller number the longer a machine has been
#: monitored, or the more densely it is sampled, purely because there are
#: more index steps to spread the same total rise across. That is backwards —
#: it means degradation that is rising in perfectly real terms can permanently
#: fail to cross a fixed "per reading" threshold once a deployment has been
#: running a while. Measuring against elapsed wall-clock time instead keeps
#: the number meaningful regardless of sample count or density.
#:
#: 0.4 spans/hour comfortably clears the worst-case healthy load-cycle
#: wobble (~0.03 spans/hour, from LOAD_SENSITIVITY's slow ~3.75h cycle in the
#: simulator) with a wide margin, while 0.08 is loose enough to flag a
#: meaningfully accelerating trend before it reaches "strong".
TREND_RISING_PER_HOUR = 0.08
TREND_STRONG_PER_HOUR = 0.4


def _signature_for(fault_type: FaultType) -> dict[str, dict[str, Any]]:
    """Extract the expected sensor pattern for a fault type."""
    model = FAULT_MODELS[fault_type]
    sig: dict[str, dict[str, Any]] = {}
    for sensor_type, effect in model.effects.items():
        sig[sensor_type.value] = {
            "direction": "up" if effect.coefficient > 0 else "down",
            "lag": effect.lag,
            "coefficient": abs(effect.coefficient),
            "curve": effect.curve,
        }
    return sig


FAULT_SIGNATURES: dict[str, dict[str, dict[str, Any]]] = {
    ft.value: _signature_for(ft) for ft in FaultType
}


# ---------------------------------------------------------------------------
# Data access port
# ---------------------------------------------------------------------------
class RCARepository(Protocol):
    """Persistence surface for RCA."""

    async def fetch_machine(
        self, tenant_id: str, machine_id: str
    ) -> Optional[Mapping[str, Any]]: ...

    async def fetch_sensors(
        self, tenant_id: str, machine_id: str
    ) -> list[Mapping[str, Any]]: ...

    async def fetch_components(
        self, tenant_id: str, machine_id: str
    ) -> list[Mapping[str, Any]]: ...

    async def fetch_readings(
        self, tenant_id: str, machine_id: str, since: datetime, limit: int
    ) -> list[Mapping[str, Any]]: ...

    async def fetch_past_failures(
        self, tenant_id: str, machine_id: Optional[str], machine_model: Optional[str]
    ) -> list[Mapping[str, Any]]: ...


class MongoRCARepository:
    """RCA repository over live MongoDB collections."""

    def _scope(self, tenant_id: str):
        return get_tenant_scope(tenant_id)

    async def fetch_machine(
        self, tenant_id: str, machine_id: str
    ) -> Optional[Mapping[str, Any]]:
        return await self._scope(tenant_id)[COLLECTIONS.machines].find_one(
            {"machine_id": machine_id}
        )

    async def fetch_sensors(
        self, tenant_id: str, machine_id: str
    ) -> list[Mapping[str, Any]]:
        cursor = self._scope(tenant_id)[COLLECTIONS.sensors].find(
            {"machine_id": machine_id}
        )
        return [doc async for doc in cursor]

    async def fetch_components(
        self, tenant_id: str, machine_id: str
    ) -> list[Mapping[str, Any]]:
        cursor = self._scope(tenant_id)[COLLECTIONS.components].find(
            {"machine_id": machine_id}
        )
        return [doc async for doc in cursor]

    async def fetch_readings(
        self, tenant_id: str, machine_id: str, since: datetime, limit: int
    ) -> list[Mapping[str, Any]]:
        from app.models.reading import flatten_reading_document

        query: dict = {"meta.machine_id": machine_id, "timestamp": {"$gte": since}}
        cursor = (
            self._scope(tenant_id)[COLLECTIONS.sensor_readings]
            .find(query)
            .sort([("timestamp", -1)])
            .limit(limit)
        )
        docs = [flatten_reading_document(doc) async for doc in cursor]
        return list(reversed(docs))

    async def fetch_past_failures(
        self, tenant_id: str, machine_id: Optional[str], machine_model: Optional[str]
    ) -> list[Mapping[str, Any]]:
        # Search documents collection for repair_history type
        query: dict[str, Any] = {"doc_type": "repair_history"}
        if machine_id:
            query["machine_ids"] = machine_id
        cursor = self._scope(tenant_id)[COLLECTIONS.documents].find(query)
        return [doc async for doc in cursor]


class InMemoryRCARepository:
    """In-memory RCA repository for tests."""

    def __init__(
        self,
        machines: Sequence[Mapping[str, Any]] = (),
        sensors: Sequence[Mapping[str, Any]] = (),
        components: Sequence[Mapping[str, Any]] = (),
        readings: Sequence[Mapping[str, Any]] = (),
        past_failures: Sequence[Mapping[str, Any]] = (),
    ):
        self._machines = [dict(m) for m in machines]
        self._sensors = [dict(s) for s in sensors]
        self._components = [dict(c) for c in components]
        self._readings = sorted(
            (dict(r) for r in readings), key=lambda r: r["timestamp"]
        )
        self._past_failures = [dict(f) for f in past_failures]

    @staticmethod
    def _owned(rows, tenant_id):
        return [r for r in rows if r.get("tenant_id") == tenant_id]

    async def fetch_machine(self, tenant_id, machine_id):
        return next(
            (m for m in self._owned(self._machines, tenant_id) if m["machine_id"] == machine_id),
            None,
        )

    async def fetch_sensors(self, tenant_id, machine_id):
        return [s for s in self._owned(self._sensors, tenant_id) if s["machine_id"] == machine_id]

    async def fetch_components(self, tenant_id, machine_id):
        return [c for c in self._owned(self._components, tenant_id) if c["machine_id"] == machine_id]

    async def fetch_readings(self, tenant_id, machine_id, since, limit):
        rows = [
            r for r in self._owned(self._readings, tenant_id)
            if r["machine_id"] == machine_id and r["timestamp"] >= since
        ]
        return rows[-limit:]

    async def fetch_past_failures(self, tenant_id, machine_id, machine_model):
        rows = self._owned(self._past_failures, tenant_id)
        if machine_id:
            rows = [r for r in rows if machine_id in r.get("machine_ids", [])]
        return rows


# ---------------------------------------------------------------------------
# Step 1: Deterministic signal analysis
# ---------------------------------------------------------------------------

def _analyze_signals(
    readings: Sequence[Mapping[str, Any]],
    sensors: Sequence[Mapping[str, Any]],
) -> tuple[list[Evidence], dict[str, dict[str, Any]]]:
    """Analyze which sensors are out of band and their trends.

    Returns evidence list and a signal summary keyed by sensor_type.
    """
    evidence: list[Evidence] = []
    # Build sensor lookup: sensor_id -> sensor doc
    sensor_by_id: dict[str, Mapping[str, Any]] = {s["sensor_id"]: s for s in sensors}
    # Also build sensor_type -> sensor doc (use first match)
    sensor_by_type: dict[str, Mapping[str, Any]] = {}
    for s in sensors:
        stype = str(s.get("type", ""))
        if stype not in sensor_by_type:
            sensor_by_type[stype] = s

    # Group readings by sensor_type
    by_type: dict[str, list[tuple[datetime, float]]] = {}
    for r in readings:
        stype = str(r.get("sensor_type", ""))
        by_type.setdefault(stype, []).append((r["timestamp"], float(r["value"])))

    signal_summary: dict[str, dict[str, Any]] = {}

    for stype, values in by_type.items():
        if len(values) < 3:
            continue

        values.sort(key=lambda x: x[0])
        vals = [v for _, v in values]
        timestamps = [t for t, _ in values]

        current = vals[-1]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals)) if len(vals) > 1 else 0.0

        # Get thresholds from sensor spec
        spec = sensor_by_type.get(stype)
        if not spec:
            continue

        normal_min = float(spec.get("normal_min", 0))
        normal_max = float(spec.get("normal_max", 0))
        warning = float(spec.get("warning_threshold", normal_max))
        critical = float(spec.get("critical_threshold", warning))
        component_id = spec.get("component_id")

        # Compute trend slope against elapsed WALL-CLOCK TIME, not sample
        # index — see TREND_RISING_PER_HOUR for why per-index would silently
        # go blind as the lookback window fills up with more readings.
        span = max(abs(normal_max - normal_min), 1e-6)
        elapsed_hours = np.array(
            [(t - timestamps[0]).total_seconds() / 3600.0 for t in timestamps]
        )
        total_hours = float(elapsed_hours[-1])
        if len(vals) >= 2 and total_hours > 1e-6:
            slope = float(np.polyfit(elapsed_hours, vals, 1)[0])  # value units / hour
        else:
            slope = 0.0
        slope_per_span = slope / span  # spans per hour

        # Determine out-of-band status
        # Check both high and low deviations. Sensors like pressure in a
        # seal leak drop rather than rise, and the thresholds only cover the
        # high side. A value far below the midpoint is just as abnormal.
        midpoint = (normal_min + normal_max) / 2.0
        out_of_band = False
        band_status = "normal"
        if current >= critical:
            band_status = "critical"
            out_of_band = True
        elif current >= warning:
            band_status = "warning"
            out_of_band = True
        elif current < normal_min - 0.1 * span:
            band_status = "below_normal"
            out_of_band = True
        elif current < midpoint - 0.5 * span:
            # Significantly below the operating midpoint
            band_status = "below_normal"
            out_of_band = True

        # Determine direction
        if slope_per_span > TREND_RISING_PER_HOUR:
            direction = "rising"
        elif slope_per_span < -TREND_RISING_PER_HOUR:
            direction = "falling"
        else:
            direction = "stable"

        signal_summary[stype] = {
            "current": current,
            "mean": mean_val,
            "std": std_val,
            "slope": slope,
            "slope_per_span": slope_per_span,
            "direction": direction,
            "band_status": band_status,
            "out_of_band": out_of_band,
            "component_id": component_id,
            "normal_min": normal_min,
            "normal_max": normal_max,
            "warning": warning,
            "critical": critical,
            "span": span,
            "n_readings": len(vals),
            "first_out_of_band_idx": None,
        }

        # Find when this sensor first went out of band
        for i, v in enumerate(vals):
            if v >= warning or v < normal_min - 0.1 * span:
                signal_summary[stype]["first_out_of_band_idx"] = i
                break

        if out_of_band:
            strength = EvidenceStrength.strong if band_status == "critical" else EvidenceStrength.moderate
            eid = f"SIG-{stype.upper()}-BAND"
            evidence.append(Evidence(
                evidence_id=eid,
                source=EvidenceSource.sensor,
                description=f"{stype} is {band_status}: current value {current:.2f} "
                            f"(normal range {normal_min:.1f}-{normal_max:.1f}, "
                            f"warning at {warning:.1f}, critical at {critical:.1f})",
                strength=strength,
                value=round(current, 4),
            ))

        # Threshold evidence (separate from sensor signal)
        if out_of_band and band_status == "critical":
            evidence.append(Evidence(
                evidence_id=f"THR-{stype.upper()}-CRITICAL",
                source=EvidenceSource.threshold,
                description=f"{stype} has crossed the critical threshold of {critical:.1f}",
                strength=EvidenceStrength.strong,
                value=round(current, 4),
            ))

        if direction != "stable":
            strength = (
                EvidenceStrength.strong
                if abs(slope_per_span) > TREND_STRONG_PER_HOUR
                else EvidenceStrength.moderate
            )
            evidence.append(Evidence(
                evidence_id=f"SIG-{stype.upper()}-TREND",
                source=EvidenceSource.sensor,
                description=f"{stype} is {direction} at {slope:.4f}/hour "
                            f"({slope_per_span:.3f} spans/hour)",
                strength=strength,
                value=round(slope_per_span, 4),
            ))

    return evidence, signal_summary


def _determine_deviation_order(signal_summary: dict[str, dict[str, Any]]) -> list[str]:
    """Order sensor types by when they first deviated from normal."""
    ordered = []
    for stype, info in signal_summary.items():
        idx = info.get("first_out_of_band_idx")
        if idx is not None:
            ordered.append((idx, stype))
    ordered.sort()
    return [stype for _, stype in ordered]


# ---------------------------------------------------------------------------
# Step 2: Fault signature matching
# ---------------------------------------------------------------------------

def _match_fault_signatures(
    signal_summary: dict[str, dict[str, Any]],
    components: Sequence[Mapping[str, Any]],
    pdm_result: Optional[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[Evidence]]:
    """Match observed signals against known fault signatures.

    Returns scored fault candidates and PdM evidence.
    """
    evidence: list[Evidence] = []
    candidates: list[dict[str, Any]] = []

    # Build component type lookup
    comp_by_id: dict[str, Mapping[str, Any]] = {c["component_id"]: c for c in components}
    comp_by_type: dict[str, list[str]] = {}
    for c in components:
        ctype = str(c.get("type", ""))
        comp_by_type.setdefault(ctype, []).append(c["component_id"])

    # PdM evidence
    pdm_mode = None
    pdm_features: dict[str, float] = {}
    if pdm_result:
        pdm_mode = pdm_result.get("predicted_failure_mode")
        pdm_prob = pdm_result.get("failure_probability", 0.0)
        if pdm_prob > 0.3:
            strength = EvidenceStrength.strong if pdm_prob > 0.6 else EvidenceStrength.moderate
            evidence.append(Evidence(
                evidence_id="PDM-FAILURE-PROB",
                source=EvidenceSource.pdm_model,
                description=f"PdM model predicts {pdm_prob:.0%} failure probability"
                            + (f", mode: {pdm_mode}" if pdm_mode else ""),
                strength=strength,
                value=round(pdm_prob, 4),
            ))
        for feat in pdm_result.get("contributing_features", []):
            if isinstance(feat, dict):
                pdm_features[feat["name"]] = feat.get("importance", 0.0)
            else:
                pdm_features[feat.name] = feat.importance

    # Score each fault signature
    for fault_name, signature in FAULT_SIGNATURES.items():
        score = 0.0
        matched_signals = 0
        total_signals = len(signature)
        contradictions = 0

        for sensor_type, expected in signature.items():
            observed = signal_summary.get(sensor_type)
            if observed is None:
                continue

            expected_dir = expected["direction"]
            observed_dir = observed["direction"]

            # Accept signal if out of band OR if there is a meaningful trend —
            # a sensor still inside its normal band but clearly rising is
            # exactly the case trend detection exists to catch ahead of a
            # threshold crossing (see TREND_RISING_PER_HOUR).
            has_signal = observed["out_of_band"] or observed["direction"] != "stable"
            if has_signal:
                if (expected_dir == "up" and observed_dir == "rising") or \
                   (expected_dir == "down" and observed_dir == "falling"):
                    # Direction matches
                    weight = expected["coefficient"]
                    # Bonus weight for strong trends
                    if abs(observed["slope_per_span"]) > TREND_STRONG_PER_HOUR:
                        weight *= 1.3
                    score += weight
                    matched_signals += 1
                elif (expected_dir == "up" and observed_dir == "falling") or \
                     (expected_dir == "down" and observed_dir == "rising"):
                    # Direction contradicts
                    contradictions += 1
                    score -= expected["coefficient"] * 0.5

        if total_signals > 0 and matched_signals > 0:
            match_ratio = matched_signals / total_signals
            # Boost if PdM agrees
            pdm_boost = 0.0
            if pdm_mode and pdm_mode == fault_name:
                pdm_boost = 1.5
                score += pdm_boost

            # Find matching component
            fault_comp_types = _FAULT_COMPONENT_TYPES.get(fault_name, [])
            matching_components = []
            for ct in fault_comp_types:
                matching_components.extend(comp_by_type.get(ct, []))

            # Pick the component whose sensor is most out of band
            best_component = None
            if matching_components:
                best_score = -1.0
                for comp_id in matching_components:
                    for stype, info in signal_summary.items():
                        if info.get("component_id") == comp_id and info["out_of_band"]:
                            deviation = abs(info["current"] - (info["normal_min"] + info["normal_max"]) / 2) / info["span"]
                            if deviation > best_score:
                                best_score = deviation
                                best_component = comp_id
                if best_component is None:
                    best_component = matching_components[0]

            candidates.append({
                "fault_type": fault_name,
                "score": score,
                "match_ratio": match_ratio,
                "matched_signals": matched_signals,
                "total_signals": total_signals,
                "contradictions": contradictions,
                "pdm_boost": pdm_boost,
                "component_id": best_component,
                "component_types": fault_comp_types,
                "description": _FAULT_DESCRIPTIONS.get(fault_name, fault_name),
            })

    # Sort by score descending
    candidates.sort(key=lambda c: -c["score"])
    return candidates, evidence


# ---------------------------------------------------------------------------
# Step 3: RAG retrieval (thin wrapper)
# ---------------------------------------------------------------------------

async def _retrieve_knowledge(
    tenant_id: str,
    machine_id: str,
    signal_summary: dict[str, dict[str, Any]],
    top_candidate: Optional[dict[str, Any]],
) -> tuple[list[Evidence], list[Any]]:
    """Retrieve knowledge passages related to observed symptoms."""
    evidence: list[Evidence] = []
    chunks: list[Any] = []

    try:
        from app.rag.retriever import retrieve

        # Build query from observed symptoms
        symptoms = []
        for stype, info in signal_summary.items():
            if info["out_of_band"]:
                symptoms.append(f"{stype} {info['band_status']}")
        if top_candidate:
            symptoms.append(top_candidate["fault_type"].lower().replace("_", " "))

        if not symptoms:
            return evidence, chunks

        query = f"{machine_id} " + " ".join(symptoms)
        result = await retrieve(tenant_id, query, machine_id=machine_id, min_score=0.1)

        for chunk in result.chunks:
            chunks.append(chunk)
            # Strong evidence if it directly discusses the fault mode
            strength = EvidenceStrength.moderate
            if top_candidate and top_candidate["fault_type"].lower().replace("_", " ") in chunk.text.lower():
                strength = EvidenceStrength.strong

            eid = f"DOC-{chunk.chunk_id[:12]}"
            citation = Citation(
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
            )
            evidence.append(Evidence(
                evidence_id=eid,
                source=EvidenceSource.document,
                description=f"Retrieved from '{chunk.document_title or 'unknown'}'"
                            + (f", section '{chunk.section_title}'" if chunk.section_title else ""),
                strength=strength,
                value=chunk.text[:200] if chunk.text else None,
                citation=citation,
            ))

    except Exception as exc:
        logger.warning("RAG retrieval failed during RCA (non-fatal): %s", exc)

    return evidence, chunks


# ---------------------------------------------------------------------------
# History evidence
# ---------------------------------------------------------------------------

def _history_evidence(
    past_failures: Sequence[Mapping[str, Any]],
    top_candidate: Optional[dict[str, Any]],
) -> list[Evidence]:
    """Check if this fault mode has occurred before on this machine/model."""
    evidence: list[Evidence] = []
    if not past_failures:
        return evidence

    # Past repair records are documents; their titles/content may mention fault modes
    for doc in past_failures[:5]:  # Limit
        title = str(doc.get("title", ""))
        eid = f"HIST-{doc.get('document_id', uuid.uuid4().hex[:8])[:12]}"
        evidence.append(Evidence(
            evidence_id=eid,
            source=EvidenceSource.history,
            description=f"Past repair record: {title}",
            strength=EvidenceStrength.moderate,
            value=title,
        ))

    return evidence


# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------

def _compute_confidence(
    evidence: list[Evidence],
    primary: Optional[dict[str, Any]],
    signal_summary: dict[str, dict[str, Any]],
    pdm_agrees: bool,
    has_history: bool,
) -> tuple[float, str, bool, list[str]]:
    """Derive confidence from evidence, not self-report.

    Returns (confidence, basis, insufficient_data, missing_data).
    """
    if primary is None:
        return 0.0, "No fault signature matched the observed signals", True, ["sensor readings with clear deviation"]

    strong_sources: set[str] = set()
    moderate_sources: set[str] = set()
    contradicting = 0

    for ev in evidence:
        src = ev.source if isinstance(ev.source, str) else ev.source.value
        strength = ev.strength if isinstance(ev.strength, str) else ev.strength.value
        if strength == "STRONG":
            strong_sources.add(src)
        elif strength == "MODERATE":
            moderate_sources.add(src)

    contradicting = primary.get("contradictions", 0)
    all_sources = strong_sources | moderate_sources
    independent_strong = len(strong_sources)

    # Base confidence from signal match ratio
    confidence = primary.get("match_ratio", 0.0) * 0.4

    # Boost for number of independent strong evidence sources
    confidence += min(independent_strong * 0.15, 0.3)

    # Boost if PdM agrees
    if pdm_agrees:
        confidence += 0.15

    # Boost if history supports
    if has_history:
        confidence += 0.05

    # Penalty for contradictions
    confidence -= contradicting * 0.1

    confidence = max(0.0, min(1.0, confidence))

    # Insufficient data check
    insufficient = independent_strong < 2
    missing_data: list[str] = []

    if "SENSOR" not in all_sources:
        missing_data.append("sensor readings showing clear deviation")
    if "PDM_MODEL" not in all_sources:
        missing_data.append("PdM model prediction for this machine")
    if "HISTORY" not in all_sources:
        missing_data.append("historical failure records for this machine or model")
    if "DOCUMENT" not in all_sources:
        missing_data.append("documented troubleshooting procedures")

    # A sensor trending clearly, even while still inside its normal band, is
    # a real deviation — that is the whole point of trend detection existing
    # alongside threshold crossing. Requiring out_of_band specifically here
    # would silently discard exactly the evidence this system is meant to act
    # on before a threshold ever fires.
    deviating_count = sum(
        1 for s in signal_summary.values() if s["out_of_band"] or s["direction"] != "stable"
    )
    if deviating_count == 0:
        insufficient = True
        missing_data.insert(0, "at least one sensor reading outside its normal band or trending")

    # Cap at 0.5 when insufficient
    if insufficient and confidence > 0.5:
        confidence = 0.5

    basis_parts = []
    basis_parts.append(f"{independent_strong} independent strong evidence source(s)")
    if pdm_agrees:
        basis_parts.append("PdM model agrees with primary cause")
    if contradicting:
        basis_parts.append(f"{contradicting} contradicting signal(s)")
    basis_parts.append(f"{primary['matched_signals']}/{primary['total_signals']} signature signals matched")
    basis = "; ".join(basis_parts)

    return round(confidence, 4), basis, insufficient, missing_data


# ---------------------------------------------------------------------------
# Step 4: Causal chain composition
# ---------------------------------------------------------------------------

def _build_causal_chain(
    primary: dict[str, Any],
    signal_summary: dict[str, dict[str, Any]],
    evidence: list[Evidence],
    deviation_order: list[str],
) -> list[CausalStep]:
    """Build an ordered causal chain from evidence already gathered."""
    steps: list[CausalStep] = []
    fault_type = primary["fault_type"]
    signature = FAULT_SIGNATURES.get(fault_type, {})

    # Order by lag (causal ordering from the signature)
    ordered_effects = sorted(signature.items(), key=lambda x: x[1]["lag"])

    order = 1
    for sensor_type, effect in ordered_effects:
        observed = signal_summary.get(sensor_type)
        if observed is None:
            continue

        direction_word = "increased" if effect["direction"] == "up" else "decreased"
        mechanism = _mechanism_text(fault_type, sensor_type, effect)

        # Find evidence IDs that reference this sensor type
        ev_ids = [
            e.evidence_id for e in evidence
            if sensor_type.upper() in e.evidence_id
        ]

        steps.append(CausalStep(
            order=order,
            description=f"{sensor_type.capitalize()} {direction_word}"
                        + (f" (currently {observed['band_status']})" if observed["out_of_band"] else " (within normal band)"),
            mechanism=mechanism,
            evidence_ids=ev_ids,
            sensor_signals=[sensor_type],
        ))
        order += 1

    return steps


def _mechanism_text(fault_type: str, sensor_type: str, effect: dict) -> str:
    """Generate mechanism text describing the physical why."""
    mechanisms: dict[str, dict[str, str]] = {
        "BEARING_WEAR": {
            "vibration": "Rolling-element surface fatigue creates spalling, producing broadband vibration that accelerates as the damaged area spreads",
            "temperature": "Friction from damaged bearing surfaces generates heat that conducts through the housing",
            "power": "Increased mechanical drag from bearing damage raises the motor's electrical draw",
            "rpm": "Bearing drag applies a braking torque that slightly reduces shaft speed under load",
        },
        "MOTOR_OVERHEAT": {
            "temperature": "Winding insulation degradation or cooling system failure causes resistive heating in the motor windings",
            "power": "Higher winding resistance and reduced efficiency increase electrical power consumption",
            "rpm": "Thermal derating or increased slip reduces motor output speed under load",
            "vibration": "Thermal expansion of rotor/stator components introduces mechanical unbalance",
        },
        "LUBRICATION_LOSS": {
            "temperature": "Metal-to-metal contact from oil film breakdown generates friction heat",
            "vibration": "Loss of the damping oil film allows metal-to-metal contact vibration",
            "power": "Increased friction from dry running raises the power needed to maintain speed",
        },
        "BELT_MISALIGNMENT": {
            "vibration": "Off-track belt creates lateral oscillation and uneven loading on rollers",
            "power": "Belt drag from misalignment increases the power needed to drive the conveyor",
            "rpm": "Belt hunting and slip cause speed instability rather than steady reduction",
        },
        "SEAL_LEAK": {
            "pressure": "Worn seal allows fluid to bypass, reducing system pressure below setpoint",
            "power": "Pump works harder to compensate for internal leakage, drawing more power",
            "temperature": "Bypassing fluid generates heat from throttling through the worn seal gap",
        },
        "TOOL_WEAR": {
            "vibration": "Dulled cutting edge produces chatter as it plows rather than shears material",
            "power": "Increased cutting forces from a dull edge raise spindle power consumption",
            "temperature": "Friction from the dull edge generates heat at the tool-workpiece interface",
            "rpm": "Spindle servo compensates for increased torque, causing minor speed variations",
        },
    }
    return mechanisms.get(fault_type, {}).get(
        sensor_type,
        f"{sensor_type} deviation consistent with {fault_type.lower().replace('_', ' ')} mechanism",
    )


async def _compose_narrative(
    causal_chain: list[CausalStep],
    primary: dict[str, Any],
    evidence: list[Evidence],
) -> Optional[str]:
    """Use Anthropic API to compose a narrative from evidence already gathered.

    The LLM must not invent evidence or select the primary cause.
    Returns None if ANTHROPIC_API_KEY is not set.
    """
    api_key = get_settings().anthropic_api_key
    if not api_key:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        chain_text = "\n".join(
            f"  {s.order}. {s.description} — {s.mechanism}" for s in causal_chain
        )
        evidence_text = "\n".join(
            f"  - [{e.source}] {e.description}" for e in evidence[:10]
        )

        prompt = (
            "You are an industrial maintenance analyst. Based ONLY on the evidence "
            "and causal chain provided below, compose a concise narrative explaining "
            "the root cause. Do NOT invent evidence or speculate beyond what is given.\n\n"
            f"Primary cause: {primary['fault_type']} — {primary['description']}\n\n"
            f"Causal chain:\n{chain_text}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            "Write a 2-3 paragraph explanation suitable for a maintenance supervisor."
        )

        response = client.messages.create(
            model=get_settings().anthropic_model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    except Exception as exc:
        logger.warning("LLM narrative composition failed (non-fatal): %s", exc)
        return None


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------

class RCAService:
    """Root-cause analysis from sensor data, PdM output, and knowledge base."""

    def __init__(
        self,
        repository: Optional[RCARepository] = None,
        history_hours: float = 24.0,
    ):
        self.repository = repository or MongoRCARepository()
        self.history_hours = history_hours

    async def analyze(
        self,
        tenant_id: str,
        machine_id: str,
        pdm_result: Optional[Mapping[str, Any]] = None,
        include_narrative: bool = True,
    ) -> RCAResult:
        """Full RCA for one machine."""
        tenant_id = normalize_tenant_id(tenant_id)

        # Fetch machine data
        machine = await self.repository.fetch_machine(tenant_id, machine_id)
        if machine is None:
            raise KeyError(f"Machine '{machine_id}' not found")

        machine_model = str(machine.get("model", ""))

        sensors = await self.repository.fetch_sensors(tenant_id, machine_id)
        components = await self.repository.fetch_components(tenant_id, machine_id)

        now = datetime.now(timezone.utc)
        readings = await self.repository.fetch_readings(
            tenant_id, machine_id,
            since=now - timedelta(hours=self.history_hours),
            limit=10000,
        )

        # Try to get PdM result if not provided
        if pdm_result is None:
            try:
                from app.services.pdm import get_pdm_service
                service = get_pdm_service()
                prediction = await service.predict(tenant_id, machine_id)
                pdm_result = prediction.model_dump()
            except Exception:
                pass  # PdM unavailable is not fatal for RCA

        # Step 1: Deterministic signal analysis
        signal_evidence, signal_summary = _analyze_signals(readings, sensors)

        # Determine deviation order
        deviation_order = _determine_deviation_order(signal_summary)

        # Step 2: Fault signature matching
        fault_candidates, pdm_evidence = _match_fault_signatures(
            signal_summary, components, pdm_result,
        )

        # Step 3: RAG retrieval
        top_candidate = fault_candidates[0] if fault_candidates else None
        doc_evidence, retrieved_chunks = await _retrieve_knowledge(
            tenant_id, machine_id, signal_summary, top_candidate,
        )

        # History evidence
        past_failures = await self.repository.fetch_past_failures(
            tenant_id, machine_id, machine_model,
        )
        hist_evidence = _history_evidence(past_failures, top_candidate)

        # Combine all evidence
        all_evidence = signal_evidence + pdm_evidence + doc_evidence + hist_evidence

        # Build hypotheses
        pdm_mode = pdm_result.get("predicted_failure_mode") if pdm_result else None

        if not fault_candidates or fault_candidates[0]["score"] <= 0:
            # No viable fault candidate
            pdm_agrees = False
            confidence, basis, insufficient, missing = _compute_confidence(
                all_evidence, None, signal_summary, False, bool(hist_evidence),
            )
            return RCAResult(
                machine_id=machine_id,
                tenant_id=tenant_id,
                primary_cause=None,
                alternative_causes=[],
                causal_chain=[],
                evidence=all_evidence,
                confidence=confidence,
                confidence_basis=basis,
                insufficient_data=insufficient,
                missing_data=missing,
                narrative_generated=False,
            )

        # Build hypotheses from top candidates
        hypotheses: list[CausalHypothesis] = []
        for i, cand in enumerate(fault_candidates[:4]):
            if cand["score"] <= 0 and i > 0:
                break

            # Find supporting and contradicting evidence IDs
            supporting = []
            contradicting_ids = []
            for ev in all_evidence:
                # Evidence that mentions this fault type's sensors supports it
                fault_sig = FAULT_SIGNATURES.get(cand["fault_type"], {})
                ev_supports = False
                for st in fault_sig:
                    if st.upper() in ev.evidence_id:
                        obs = signal_summary.get(st)
                        expected_dir = fault_sig[st]["direction"]
                        if obs:
                            if (expected_dir == "up" and obs["direction"] == "rising") or \
                               (expected_dir == "down" and obs["direction"] == "falling"):
                                supporting.append(ev.evidence_id)
                                ev_supports = True
                            elif obs["out_of_band"] and (
                                (expected_dir == "up" and obs["direction"] == "falling") or
                                (expected_dir == "down" and obs["direction"] == "rising")
                            ):
                                contradicting_ids.append(ev.evidence_id)

                if not ev_supports and ev.source in (EvidenceSource.pdm_model, "PDM_MODEL"):
                    if pdm_mode and pdm_mode == cand["fault_type"]:
                        supporting.append(ev.evidence_id)

                if not ev_supports and ev.source in (EvidenceSource.document, "DOCUMENT"):
                    supporting.append(ev.evidence_id)
                if not ev_supports and ev.source in (EvidenceSource.history, "HISTORY"):
                    supporting.append(ev.evidence_id)

            # Compute probability from score relative to total
            total_score = sum(max(0, c["score"]) for c in fault_candidates[:4])
            probability = cand["score"] / total_score if total_score > 0 else 0.0
            probability = max(0.0, min(1.0, probability))

            hypotheses.append(CausalHypothesis(
                cause_id=f"RCA-{cand['fault_type']}-{uuid.uuid4().hex[:6]}",
                description=cand["description"],
                component_id=cand.get("component_id"),
                fault_mode=cand["fault_type"],
                probability=round(probability, 4),
                supporting_evidence_ids=list(set(supporting)),
                contradicting_evidence_ids=list(set(contradicting_ids)),
            ))

        primary = hypotheses[0] if hypotheses else None
        alternatives = hypotheses[1:3]  # Up to 3 alternatives

        # Build causal chain for primary cause
        causal_chain = []
        if top_candidate and top_candidate["score"] > 0:
            causal_chain = _build_causal_chain(
                top_candidate, signal_summary, all_evidence, deviation_order,
            )

        # Confidence computation
        pdm_agrees = pdm_mode is not None and top_candidate is not None and pdm_mode == top_candidate["fault_type"]
        confidence, basis, insufficient, missing = _compute_confidence(
            all_evidence, top_candidate, signal_summary, pdm_agrees, bool(hist_evidence),
        )

        # Step 4: LLM narrative (optional)
        narrative_generated = False
        if include_narrative and top_candidate and causal_chain:
            narrative = await _compose_narrative(causal_chain, top_candidate, all_evidence)
            if narrative:
                narrative_generated = True
                # Enrich the causal chain descriptions with narrative
                # (but don't replace the mechanically-composed descriptions)

        return RCAResult(
            machine_id=machine_id,
            tenant_id=tenant_id,
            primary_cause=primary,
            alternative_causes=alternatives,
            causal_chain=causal_chain,
            evidence=all_evidence,
            confidence=confidence,
            confidence_basis=basis,
            insufficient_data=insufficient,
            missing_data=missing,
            narrative_generated=narrative_generated,
        )


# ---------------------------------------------------------------------------
# Process-wide instance
# ---------------------------------------------------------------------------
_service: Optional[RCAService] = None


def get_rca_service() -> RCAService:
    """Return the process-wide RCA service, creating it lazily."""
    global _service
    if _service is None:
        _service = RCAService()
    return _service


def set_rca_service(service: Optional[RCAService]) -> None:
    """Test hook."""
    global _service
    _service = service
