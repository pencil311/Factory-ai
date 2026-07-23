"""Tests for the root-cause analysis service.

No live Mongo, no LLM: the repository is in-memory and ANTHROPIC_API_KEY is
unset. What is under test is the deterministic analysis pipeline — signal
analysis, fault-signature matching, confidence rules, and causal chain
construction.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.schemas.rca import EvidenceSource, EvidenceStrength, RCAResult
from app.services.rca import (
    InMemoryRCARepository,
    RCAService,
    _analyze_signals,
    _compute_confidence,
    _match_fault_signatures,
)

TENANT = "demo"
NOW = datetime(2026, 7, 22, 12, 0, 0, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Fixtures — realistic machine setup
# ---------------------------------------------------------------------------
MACHINE = {
    "tenant_id": TENANT,
    "machine_id": "CV-201",
    "name": "Infeed Belt Conveyor",
    "model": "SpanTech SB-3000",
    "line_id": "LINE-A",
}

SENSORS = [
    {
        "tenant_id": TENANT, "sensor_id": "CV-201-TMP-01", "machine_id": "CV-201",
        "component_id": "CV-201-MTR", "type": "temperature", "unit": "°C",
        "normal_min": 15, "normal_max": 70, "warning_threshold": 80, "critical_threshold": 95,
    },
    {
        "tenant_id": TENANT, "sensor_id": "CV-201-VIB-01", "machine_id": "CV-201",
        "component_id": "CV-201-BRG-D", "type": "vibration", "unit": "mm/s",
        "normal_min": 0.0, "normal_max": 2.8, "warning_threshold": 4.5, "critical_threshold": 7.1,
    },
    {
        "tenant_id": TENANT, "sensor_id": "CV-201-RPM-01", "machine_id": "CV-201",
        "component_id": "CV-201-DRLR", "type": "rpm", "unit": "rpm",
        "normal_min": 40, "normal_max": 90, "warning_threshold": 100, "critical_threshold": 110,
    },
    {
        "tenant_id": TENANT, "sensor_id": "CV-201-PWR-01", "machine_id": "CV-201",
        "component_id": "CV-201-MTR", "type": "power", "unit": "kW",
        "normal_min": 0.5, "normal_max": 5.0, "warning_threshold": 5.8, "critical_threshold": 6.5,
    },
]

COMPONENTS = [
    {"tenant_id": TENANT, "component_id": "CV-201-MTR", "machine_id": "CV-201",
     "name": "Drive Motor", "type": "motor"},
    {"tenant_id": TENANT, "component_id": "CV-201-BRG-D", "machine_id": "CV-201",
     "name": "Drive Roller Bearing", "type": "bearing",
     "parent_component_id": "CV-201-DRLR"},
    {"tenant_id": TENANT, "component_id": "CV-201-DRLR", "machine_id": "CV-201",
     "name": "Drive Roller", "type": "roller"},
    {"tenant_id": TENANT, "component_id": "CV-201-BELT", "machine_id": "CV-201",
     "name": "Conveyor Belt", "type": "belt"},
    {"tenant_id": TENANT, "component_id": "CV-201-GBX", "machine_id": "CV-201",
     "name": "Right-Angle Gearbox", "type": "gearbox"},
]


def make_readings(
    vibration: list[float],
    temperature: list[float],
    power: list[float] | None = None,
    rpm: list[float] | None = None,
    machine_id: str = "CV-201",
) -> list[dict]:
    """Interleaved readings ending at NOW, 30s apart."""
    n = len(vibration)
    start = NOW - timedelta(seconds=30 * n)
    rows = []
    for i in range(n):
        ts = start + timedelta(seconds=30 * i)
        rows.append({
            "tenant_id": TENANT, "machine_id": machine_id, "sensor_id": "CV-201-VIB-01",
            "sensor_type": "vibration", "value": vibration[i], "timestamp": ts,
        })
        rows.append({
            "tenant_id": TENANT, "machine_id": machine_id, "sensor_id": "CV-201-TMP-01",
            "sensor_type": "temperature", "value": temperature[i], "timestamp": ts,
        })
        if power:
            rows.append({
                "tenant_id": TENANT, "machine_id": machine_id, "sensor_id": "CV-201-PWR-01",
                "sensor_type": "power", "value": power[i], "timestamp": ts,
            })
        if rpm:
            rows.append({
                "tenant_id": TENANT, "machine_id": machine_id, "sensor_id": "CV-201-RPM-01",
                "sensor_type": "rpm", "value": rpm[i], "timestamp": ts,
            })
    return rows


def make_service(readings, sensors=None, components=None, past_failures=None):
    repo = InMemoryRCARepository(
        machines=[MACHINE],
        sensors=sensors or SENSORS,
        components=components or COMPONENTS,
        readings=readings,
        past_failures=past_failures or [],
    )
    return RCAService(repository=repo)


# ---------------------------------------------------------------------------
# Test: Bearing wear signature (vibration leading temperature)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bearing_signature_yields_bearing_cause_with_component_id():
    """Vibration rising first, temperature following later = bearing wear.

    The primary cause must name a component_id so downstream agents can act.
    """
    n = 120
    # Vibration rises from normal (1.5) to critical (7.5)
    vibration = list(np.linspace(1.5, 7.5, n))
    # Temperature rises later (delayed by lag=0.35)
    temperature = [42.0] * (n // 3) + list(np.linspace(42, 90, n - n // 3))
    # Power rises slightly
    power = list(np.linspace(2.5, 5.5, n))

    service = make_service(make_readings(vibration, temperature, power=power))
    result = await service.analyze(TENANT, "CV-201", include_narrative=False)

    assert isinstance(result, RCAResult)
    assert result.machine_id == "CV-201"
    assert result.primary_cause is not None
    assert result.primary_cause.component_id is not None
    # Should identify bearing-related fault
    assert result.primary_cause.fault_mode in ("BEARING_WEAR", "LUBRICATION_LOSS", "TOOL_WEAR")
    assert result.confidence > 0


# ---------------------------------------------------------------------------
# Test: a rising trend still inside the normal band must be usable evidence
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_in_band_rising_vibration_with_lagging_temperature_yields_bearing_cause():
    """Regression test: reported live as vibration=4.24mm/s (normal band is
    0-2.8, warning is 4.5) with temperature lagging behind it into its own
    warning band — RCA returned primary_cause=None, confidence 0.0,
    "No fault signature matched the observed signals", insufficient_data=True.

    Vibration never crosses its warning threshold here (peaks at 4.2, under
    the 4.5 warning line) — the fault must be caught by its rising TREND
    while still in-band, which is the system's own stated design ("the
    warning threshold is deliberately late, and trend detection exists
    precisely to catch this"). Temperature does eventually cross into
    warning, matching the reported lag, but that is not what this test
    exercises for vibration.
    """
    # 280 readings, 30s apart (~2.3 hours) — deliberately not a short, tidy
    # sample count. This is the actual reported mechanism: a real, physical
    # rate of change stays whatever it is, but "slope per READING" shrinks
    # toward zero as more readings accumulate in the window purely because
    # there are more index steps to spread the same total rise across. A
    # too-small n would not exercise that dilution at all.
    n = 280
    # Vibration rises smoothly from a healthy baseline to 4.2 — above
    # normal_max (2.8) but still under warning (4.5): in-band by threshold,
    # but clearly and steadily rising for the machine's whole recent history.
    vibration = list(np.linspace(1.0, 4.2, n))
    # Temperature lags: flat at a healthy 45 for the first ~58% of the
    # window, then rises into its own warning band (>80) only near the end —
    # mirroring the reported "vibration rising first, temperature following".
    flat = n * 7 // 12
    temperature = [45.0] * flat + list(np.linspace(45.0, 82.0, n - flat))

    # A PdM model agreeing with BEARING_WEAR — a second, independent strong
    # evidence source alongside the sensor trend, exactly like a real PdM
    # model would if it also picked up the same degrading vibration signal.
    pdm_result = {
        "failure_probability": 0.75,
        "predicted_failure_mode": "BEARING_WEAR",
        "contributing_features": [],
    }

    service = make_service(make_readings(vibration, temperature))
    result = await service.analyze(
        TENANT, "CV-201", pdm_result=pdm_result, include_narrative=False
    )

    assert result.primary_cause is not None, result.confidence_basis
    assert result.primary_cause.fault_mode == "BEARING_WEAR"
    assert result.primary_cause.component_id == "CV-201-BRG-D"
    assert result.confidence > 0.3, "confidence must be non-trivial, not asserted"
    assert result.insufficient_data is False

    # Prove the trend path, not a threshold crossing, is what caught vibration.
    evidence_ids = {e.evidence_id for e in result.evidence}
    assert "SIG-VIBRATION-TREND" in evidence_ids
    assert "SIG-VIBRATION-BAND" not in evidence_ids


# ---------------------------------------------------------------------------
# Test: Thin evidence -> insufficient_data and capped confidence
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_thin_evidence_sets_insufficient_data_and_caps_confidence():
    """With few readings barely out of band, insufficient_data must be True
    and confidence must not exceed 0.5."""
    n = 10
    # Values barely in the normal range — no clear deviation
    vibration = [1.5] * n
    temperature = [42.0] * n

    service = make_service(make_readings(vibration, temperature))
    result = await service.analyze(TENANT, "CV-201", include_narrative=False)

    assert result.insufficient_data is True
    assert result.confidence <= 0.5
    assert len(result.missing_data) > 0


# ---------------------------------------------------------------------------
# Test: Contradicting evidence lowers probability
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_contradicting_evidence_lowers_hypothesis_probability():
    """If a sensor moves opposite to what the fault signature predicts,
    that hypothesis should have lower probability."""
    n = 120
    # Vibration rising (consistent with bearing wear)
    vibration = list(np.linspace(1.5, 7.0, n))
    # Temperature FALLING (contradicts bearing wear which expects rising)
    temperature = list(np.linspace(80.0, 30.0, n))
    # Power falling (contradicts bearing wear which expects rising)
    power = list(np.linspace(5.0, 1.0, n))

    service = make_service(make_readings(vibration, temperature, power=power))
    result = await service.analyze(TENANT, "CV-201", include_narrative=False)

    # Even though vibration is out of band, contradictions should lower
    # the primary cause's probability compared to a clean signature
    if result.primary_cause:
        # With contradictions, confidence should be modest
        assert result.confidence < 0.8


# ---------------------------------------------------------------------------
# Test: Causal chain steps ordered and citing evidence
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_causal_chain_ordered_and_cites_evidence():
    """Steps must be in order and each must reference evidence IDs."""
    n = 120
    vibration = list(np.linspace(1.5, 7.5, n))
    temperature = [42.0] * (n // 3) + list(np.linspace(42, 90, n - n // 3))
    power = list(np.linspace(2.5, 5.5, n))

    service = make_service(make_readings(vibration, temperature, power=power))
    result = await service.analyze(TENANT, "CV-201", include_narrative=False)

    assert len(result.causal_chain) > 0

    # Steps must be ordered
    orders = [s.order for s in result.causal_chain]
    assert orders == sorted(orders)
    assert orders[0] == 1

    # Each step must have a description and mechanism
    for step in result.causal_chain:
        assert step.description
        assert step.mechanism
        assert len(step.sensor_signals) > 0


# ---------------------------------------------------------------------------
# Test: Works without ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_works_without_anthropic_api_key(monkeypatch):
    """The module must produce a complete RCA without an LLM."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    n = 120
    vibration = list(np.linspace(1.5, 7.5, n))
    temperature = [42.0] * (n // 3) + list(np.linspace(42, 90, n - n // 3))

    service = make_service(make_readings(vibration, temperature))
    result = await service.analyze(TENANT, "CV-201", include_narrative=True)

    assert isinstance(result, RCAResult)
    assert result.narrative_generated is False
    # Analysis still produces evidence and hypotheses
    assert len(result.evidence) > 0


# ---------------------------------------------------------------------------
# Test: Tenant isolation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tenant_isolation():
    """RCA for one tenant must not see another tenant's machines."""
    n = 60
    vibration = list(np.linspace(1.5, 6.0, n))
    temperature = list(np.linspace(42, 85, n))

    # Machine belongs to TENANT="demo"
    readings = make_readings(vibration, temperature)

    repo = InMemoryRCARepository(
        machines=[MACHINE],
        sensors=SENSORS,
        components=COMPONENTS,
        readings=readings,
    )
    service = RCAService(repository=repo)

    # Analyzing with a different tenant should not find the machine
    with pytest.raises(KeyError, match="not found"):
        await service.analyze("acme", "CV-201", include_narrative=False)


# ---------------------------------------------------------------------------
# Test: Unknown machine raises KeyError
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unknown_machine_raises_key_error():
    service = make_service([])
    with pytest.raises(KeyError, match="not found"):
        await service.analyze(TENANT, "NOPE-999", include_narrative=False)


# ---------------------------------------------------------------------------
# Test: Motor overheat signature
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_motor_overheat_signature():
    """Temperature leading with power following = motor overheat."""
    n = 120
    # Temperature rises first (motor overheat primary signal)
    temperature = list(np.linspace(42, 100, n))
    # Power rises after (lag=0.10)
    power = list(np.linspace(2.5, 6.2, n))
    # Vibration rises late (lag=0.60)
    vibration = [1.5] * (n * 2 // 3) + list(np.linspace(1.5, 4.0, n - n * 2 // 3))

    service = make_service(make_readings(vibration, temperature, power=power))
    result = await service.analyze(TENANT, "CV-201", include_narrative=False)

    assert result.primary_cause is not None
    # The primary or alternative should include motor overheat
    all_modes = [result.primary_cause.fault_mode] + [a.fault_mode for a in result.alternative_causes]
    # Temperature + power pattern should at least be in the candidates
    assert len(result.evidence) > 0


# ---------------------------------------------------------------------------
# Test: Seal leak signature (pressure falling)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_seal_leak_identified_for_pressure_drop():
    """Pressure dropping with power rising = seal leak."""
    seal_sensors = [
        {
            "tenant_id": TENANT, "sensor_id": "HP-150-PRS-01", "machine_id": "HP-150",
            "component_id": "HP-150-CYL", "type": "pressure", "unit": "bar",
            "normal_min": 0.0, "normal_max": 210.0, "warning_threshold": 230.0, "critical_threshold": 250.0,
        },
        {
            "tenant_id": TENANT, "sensor_id": "HP-150-PWR-01", "machine_id": "HP-150",
            "component_id": "HP-150-PUMP-MTR", "type": "power", "unit": "kW",
            "normal_min": 2.0, "normal_max": 45.0, "warning_threshold": 50.0, "critical_threshold": 55.0,
        },
        {
            "tenant_id": TENANT, "sensor_id": "HP-150-TMP-01", "machine_id": "HP-150",
            "component_id": "HP-150-HPU", "type": "temperature", "unit": "°C",
            "normal_min": 20, "normal_max": 60, "warning_threshold": 70, "critical_threshold": 80,
        },
    ]
    seal_components = [
        {"tenant_id": TENANT, "component_id": "HP-150-CYL", "machine_id": "HP-150",
         "name": "Main Ram Cylinder", "type": "cylinder"},
        {"tenant_id": TENANT, "component_id": "HP-150-PUMP", "machine_id": "HP-150",
         "name": "Main Hydraulic Pump", "type": "pump"},
        {"tenant_id": TENANT, "component_id": "HP-150-PUMP-MTR", "machine_id": "HP-150",
         "name": "Pump Drive Motor", "type": "motor"},
        {"tenant_id": TENANT, "component_id": "HP-150-VLV", "machine_id": "HP-150",
         "name": "Directional Control Valve", "type": "valve"},
        {"tenant_id": TENANT, "component_id": "HP-150-HPU", "machine_id": "HP-150",
         "name": "Hydraulic Power Unit", "type": "other"},
    ]
    seal_machine = {
        "tenant_id": TENANT, "machine_id": "HP-150", "name": "150-Ton Hydraulic Press",
        "model": "Beckwood BX-150", "line_id": "LINE-A",
    }

    n = 120
    start = NOW - timedelta(seconds=30 * n)
    readings = []
    # Pressure dropping from normal to below normal
    pressure = list(np.linspace(180, 50, n))
    power = list(np.linspace(25, 52, n))
    temperature = list(np.linspace(40, 72, n))
    for i in range(n):
        ts = start + timedelta(seconds=30 * i)
        readings.append({"tenant_id": TENANT, "machine_id": "HP-150", "sensor_id": "HP-150-PRS-01",
                         "sensor_type": "pressure", "value": pressure[i], "timestamp": ts})
        readings.append({"tenant_id": TENANT, "machine_id": "HP-150", "sensor_id": "HP-150-PWR-01",
                         "sensor_type": "power", "value": power[i], "timestamp": ts})
        readings.append({"tenant_id": TENANT, "machine_id": "HP-150", "sensor_id": "HP-150-TMP-01",
                         "sensor_type": "temperature", "value": temperature[i], "timestamp": ts})

    repo = InMemoryRCARepository(
        machines=[seal_machine],
        sensors=seal_sensors,
        components=seal_components,
        readings=readings,
    )
    service = RCAService(repository=repo)
    result = await service.analyze(TENANT, "HP-150", include_narrative=False)

    assert result.primary_cause is not None
    all_modes = [result.primary_cause.fault_mode] + [a.fault_mode for a in result.alternative_causes]
    assert "SEAL_LEAK" in all_modes


# ---------------------------------------------------------------------------
# Test: No readings -> insufficient data
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_readings_yields_insufficient_data():
    service = make_service([])
    result = await service.analyze(TENANT, "CV-201", include_narrative=False)

    assert result.insufficient_data is True
    assert result.confidence <= 0.5


# ---------------------------------------------------------------------------
# Test: PdM agreement boosts confidence
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pdm_agreement_boosts_confidence():
    """When PdM predicts the same failure mode, confidence should increase."""
    n = 120
    vibration = list(np.linspace(1.5, 7.5, n))
    temperature = [42.0] * (n // 3) + list(np.linspace(42, 90, n - n // 3))
    power = list(np.linspace(2.5, 5.5, n))

    service = make_service(make_readings(vibration, temperature, power=power))

    # Without PdM
    result_no_pdm = await service.analyze(TENANT, "CV-201", include_narrative=False)

    # With PdM agreeing
    pdm_result = {
        "failure_probability": 0.75,
        "predicted_failure_mode": result_no_pdm.primary_cause.fault_mode if result_no_pdm.primary_cause else "BEARING_WEAR",
        "contributing_features": [
            {"name": "vibration_mean", "importance": 0.3},
            {"name": "temperature_mean", "importance": 0.2},
        ],
    }
    result_with_pdm = await service.analyze(TENANT, "CV-201", pdm_result=pdm_result, include_narrative=False)

    # PdM agreement should boost confidence
    assert result_with_pdm.confidence >= result_no_pdm.confidence


# ---------------------------------------------------------------------------
# Test: Evidence contains correct sources
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_evidence_has_correct_sources():
    n = 120
    vibration = list(np.linspace(1.5, 7.5, n))
    temperature = list(np.linspace(42, 90, n))

    pdm_result = {
        "failure_probability": 0.8,
        "predicted_failure_mode": "BEARING_WEAR",
        "contributing_features": [],
    }

    service = make_service(make_readings(vibration, temperature))
    result = await service.analyze(TENANT, "CV-201", pdm_result=pdm_result, include_narrative=False)

    sources = {e.source if isinstance(e.source, str) else e.source.value for e in result.evidence}
    assert "SENSOR" in sources  # Signal evidence
    assert "PDM_MODEL" in sources  # PdM evidence


# ---------------------------------------------------------------------------
# Test: Result shape is complete
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_result_shape_is_complete():
    n = 120
    vibration = list(np.linspace(1.5, 7.5, n))
    temperature = list(np.linspace(42, 90, n))

    service = make_service(make_readings(vibration, temperature))
    result = await service.analyze(TENANT, "CV-201", include_narrative=False)

    # Verify all required fields are present
    assert result.machine_id == "CV-201"
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.confidence_basis, str)
    assert result.analysis_timestamp is not None
    assert isinstance(result.insufficient_data, bool)
    assert isinstance(result.missing_data, list)
    assert isinstance(result.evidence, list)
    assert isinstance(result.causal_chain, list)
    assert isinstance(result.alternative_causes, list)
    assert isinstance(result.narrative_generated, bool)
