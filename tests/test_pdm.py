"""Tests for the PdM feature pipeline and inference service.

No live Mongo, no training run: the repository is in-memory and the models are
minimal fakes exposing the same ``predict`` / ``predict_proba`` /
``feature_importances_`` surface. What is under test is everything around the
models — the shared feature pipeline, the parity guard, clamping, trend logic
and explanation assembly — which is where serving bugs actually live.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from ml.preprocess import (
    CHANNEL_FEATURES,
    CHANNEL_ROLES,
    GLOBAL_FEATURES,
    FeatureParityError,
    FeatureSchema,
    build_feature_vector,
    default_schema,
    feature_names,
)
from app.schemas.pdm import TrendDirection
from app.services.pdm import (
    DEFAULT_ARTIFACTS_DIR,
    InMemoryPdmRepository,
    InsufficientDataError,
    PdmArtifactsMissingError,
    PdmModels,
    PdmService,
    load_artifacts,
)

TENANT = "demo"
NOW = datetime(2026, 7, 22, 12, 0, 0, tzinfo=timezone.utc)
WINDOW = 30
RUL_CAP = 125.0
CYCLE_HOURS = 10.0


# ---------------------------------------------------------------------------
# Feature schema & ordering
# ---------------------------------------------------------------------------
def test_feature_names_have_the_expected_structure_and_order():
    names = feature_names()

    # Channel blocks in role order, then the two globals, nothing else.
    expected_len = len(CHANNEL_ROLES) * (len(CHANNEL_FEATURES) + 1) + len(GLOBAL_FEATURES)
    assert len(names) == expected_len
    assert names[0] == "temperature_mean"
    assert names[len(CHANNEL_FEATURES)] == "temperature_present"
    assert names[-2:] == ("time_since_maintenance_h", "cumulative_runtime_h")
    assert len(set(names)) == len(names), "duplicate feature names"


def test_build_feature_vector_matches_the_schema_exactly():
    channels = {role: list(np.linspace(0, 1, WINDOW)) for role in CHANNEL_ROLES}
    names, values = build_feature_vector(channels, 12.0, 400.0, window=WINDOW)

    assert names == default_schema(WINDOW).names
    assert values.shape == (len(names),)
    assert values[names.index("time_since_maintenance_h")] == pytest.approx(12.0)
    assert values[names.index("cumulative_runtime_h")] == pytest.approx(400.0)


def test_absent_channel_yields_zero_block_and_present_flag_zero():
    channels = {"temperature": list(np.ones(WINDOW))}  # everything else absent
    names, values = build_feature_vector(channels, 0.0, 0.0, window=WINDOW)
    by_name = dict(zip(names, values))

    assert by_name["temperature_present"] == 1.0
    assert by_name["vibration_present"] == 0.0
    assert all(by_name[f"vibration_{f}"] == 0.0 for f in CHANNEL_FEATURES)


def test_short_series_is_treated_as_absent_not_windowed():
    names, values = build_feature_vector({"rpm": [1.0, 2.0]}, 0.0, 0.0, window=WINDOW)
    by_name = dict(zip(names, values))

    assert by_name["rpm_present"] == 0.0


# ---------------------------------------------------------------------------
# Train/serve parity guard
# ---------------------------------------------------------------------------
def test_parity_guard_fires_on_reordered_features():
    schema = default_schema(WINDOW)
    swapped = list(schema.names)
    swapped[0], swapped[1] = swapped[1], swapped[0]

    with pytest.raises(FeatureParityError, match="order mismatch at index 0"):
        schema.assert_parity(swapped)


def test_parity_guard_fires_on_count_mismatch():
    schema = default_schema(WINDOW)

    with pytest.raises(FeatureParityError, match="count mismatch"):
        schema.assert_parity(list(schema.names)[:-1])


def test_parity_guard_passes_on_the_real_pipeline_output():
    channels = {role: list(np.zeros(WINDOW)) for role in CHANNEL_ROLES}
    names, _ = build_feature_vector(channels, 0.0, 0.0, window=WINDOW)
    default_schema(WINDOW).assert_parity(names)  # must not raise


def test_schema_round_trips_through_json(tmp_path: Path):
    schema = default_schema(WINDOW)
    schema.to_json(tmp_path / "schema.json")

    assert FeatureSchema.from_json(tmp_path / "schema.json") == schema


# ---------------------------------------------------------------------------
# Fakes: the model surface the service depends on, nothing more
# ---------------------------------------------------------------------------
class FakeRulModel:
    def __init__(self, cycles: float, n_features: int):
        self.cycles = cycles
        self.feature_importances_ = np.linspace(1.0, 0.0, n_features)

    def predict(self, X):
        return np.full(len(X), self.cycles)


class FakeClassifier:
    """predict_proba over the AI4I class order used in training."""

    def __init__(self, probs: list[float], n_features: int):
        self.probs = probs
        self.feature_importances_ = np.linspace(0.0, 1.0, n_features)

    def predict_proba(self, X):
        return np.tile(np.asarray(self.probs), (len(X), 1))


CLASSES = ["NO_FAILURE", "TOOL_WEAR", "HEAT_DISSIPATION", "POWER", "OVERSTRAIN", "RANDOM"]


def make_models(
    rul_cycles: float = 60.0,
    clf_probs: list[float] | None = None,
) -> PdmModels:
    schema = default_schema(WINDOW)
    n = len(schema.names)
    wide_lo, wide_hi = [-1e9] * n, [1e9] * n
    return PdmModels(
        rul_model=FakeRulModel(rul_cycles, n),
        rul_schema=schema,
        rul_meta={
            "rul_cap_cycles": RUL_CAP,
            "cycle_hours": CYCLE_HOURS,
            "feature_p1": wide_lo,
            "feature_p99": wide_hi,
            "metrics": {"test_rmse_cycles": 15.0},
            "model": "FakeRul",
            "dataset": "fake",
        },
        rul_importances={name: float(v) for name, v in zip(schema.names, np.linspace(1, 0, n))},
        clf_model=FakeClassifier(clf_probs or [0.9, 0.02, 0.02, 0.02, 0.02, 0.02], n),
        clf_schema=schema,
        clf_meta={
            "classes": CLASSES,
            "feature_p1": wide_lo,
            "feature_p99": wide_hi,
            "model": "FakeClf",
            "dataset": "fake",
        },
        clf_importances={name: 0.0 for name in schema.names},
        artifacts_dir=Path("fake-artifacts"),
        loaded_at=NOW,
    )


MACHINE = {
    "tenant_id": TENANT,
    "machine_id": "CV-201",
    "name": "Infeed Belt Conveyor",
    "line_id": "LINE-A",
    "last_maintenance_at": NOW - timedelta(days=30),
    "installed_at": NOW - timedelta(days=2000),
}


def make_readings(
    vibration: list[float],
    temperature: list[float],
    machine_id: str = "CV-201",
) -> list[dict]:
    """Interleaved vib+temp readings ending at NOW, 30s apart."""
    n = len(vibration)
    start = NOW - timedelta(seconds=30 * n)
    rows = []
    for i, (v, t) in enumerate(zip(vibration, temperature)):
        ts = start + timedelta(seconds=30 * i)
        rows.append(
            {"tenant_id": TENANT, "machine_id": machine_id, "sensor_id": "VIB-01",
             "sensor_type": "vibration", "value": v, "timestamp": ts}
        )
        rows.append(
            {"tenant_id": TENANT, "machine_id": machine_id, "sensor_id": "TMP-01",
             "sensor_type": "temperature", "value": t, "timestamp": ts}
        )
    return rows


#: Catalogue entries carrying the normal bands the healthy reference comes from.
SENSOR_SPECS = [
    {"tenant_id": TENANT, "sensor_id": "VIB-01", "machine_id": "CV-201",
     "type": "vibration", "normal_min": 0.0, "normal_max": 2.8},
    {"tenant_id": TENANT, "sensor_id": "TMP-01", "machine_id": "CV-201",
     "type": "temperature", "normal_min": 15.0, "normal_max": 70.0},
]


def make_service(readings: list[dict], **model_kwargs) -> PdmService:
    repo = InMemoryPdmRepository(
        machines=[MACHINE], readings=readings, sensors=SENSOR_SPECS
    )
    return PdmService(models=make_models(**model_kwargs), repository=repo)


# ---------------------------------------------------------------------------
# Trend detection
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_degrading_input_yields_degrading_trend():
    """Vibration and temperature climbing away from baseline -> DEGRADING."""
    n = 120
    vib = list(1.0 + np.linspace(0, 4.0, n))       # 1.0 -> 5.0 mm/s
    tmp = list(45.0 + np.linspace(0, 30.0, n))     # 45 -> 75 C
    service = make_service(make_readings(vib, tmp))

    prediction = await service.predict(TENANT, "CV-201")

    assert prediction.trend_direction == TrendDirection.degrading
    assert set(prediction.channels_present) == {"vibration", "temperature"}
    assert prediction.readings_used == 2 * n


@pytest.mark.asyncio
async def test_steady_input_yields_stable_trend():
    n = 120
    vib = [1.5] * n
    tmp = [50.0] * n
    service = make_service(make_readings(vib, tmp))

    prediction = await service.predict(TENANT, "CV-201")

    assert prediction.trend_direction == TrendDirection.stable


@pytest.mark.asyncio
async def test_recovering_input_yields_improving_trend():
    """A repaired machine returning toward baseline -> IMPROVING."""
    n = 120
    vib = list(6.0 - np.linspace(0, 4.5, n))
    tmp = list(80.0 - np.linspace(0, 28.0, n))
    service = make_service(make_readings(vib, tmp))

    prediction = await service.predict(TENANT, "CV-201")

    assert prediction.trend_direction == TrendDirection.improving


# ---------------------------------------------------------------------------
# Missing artifacts
# ---------------------------------------------------------------------------
def test_missing_artifacts_raise_a_clear_startup_error(tmp_path: Path):
    with pytest.raises(PdmArtifactsMissingError) as excinfo:
        load_artifacts(tmp_path)

    message = str(excinfo.value)
    assert "rul_model.joblib" in message
    assert "python ml/train_rul.py" in message
    assert "no heuristic fallback" in message.lower()


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_contributing_features_are_populated_and_sorted_by_importance():
    n = 120
    service = make_service(make_readings([2.0] * n, [50.0] * n))

    prediction = await service.predict(TENANT, "CV-201")

    features = prediction.contributing_features
    assert len(features) > 0
    importances = [f.importance for f in features]
    assert importances == sorted(importances, reverse=True)
    assert all(f.name in default_schema(WINDOW).names for f in features)
    # The fake RUL model puts maximum importance on the first schema feature.
    assert features[0].name == "temperature_mean"


# ---------------------------------------------------------------------------
# Output bounds
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_cycles,expected_hours",
    [
        (-50.0, 0.0),                      # negative regression output clamps to 0
        (1e6, RUL_CAP * CYCLE_HOURS),      # runaway output clamps to the cap
        (60.0, 600.0),                     # in-range passes through, in hours
    ],
)
async def test_rul_output_is_non_negative_and_bounded(model_cycles, expected_hours):
    n = 120
    service = make_service(
        make_readings([2.0] * n, [50.0] * n), rul_cycles=model_cycles
    )

    prediction = await service.predict(TENANT, "CV-201")

    assert prediction.remaining_useful_life_hours == pytest.approx(expected_hours)
    assert prediction.remaining_useful_life_hours >= 0.0
    assert 0.0 <= prediction.health_score <= 1.0
    assert 0.0 <= prediction.failure_probability <= 1.0
    assert 0.0 <= prediction.confidence <= 1.0


@pytest.mark.asyncio
async def test_capped_rul_reports_no_failure_time():
    """At the cap the RUL is censored — 'at least this much', not a date."""
    n = 120
    service = make_service(make_readings([2.0] * n, [50.0] * n), rul_cycles=RUL_CAP)

    prediction = await service.predict(TENANT, "CV-201")

    assert prediction.predicted_failure_time is None


@pytest.mark.asyncio
async def test_low_rul_reports_a_concrete_failure_time():
    n = 120
    service = make_service(make_readings([2.0] * n, [50.0] * n), rul_cycles=10.0)

    prediction = await service.predict(TENANT, "CV-201")

    assert prediction.predicted_failure_time is not None
    assert prediction.predicted_failure_time > prediction.generated_at


# ---------------------------------------------------------------------------
# Failure mode
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_confident_mode_is_named_and_probability_reflects_it():
    n = 120
    service = make_service(
        make_readings([2.0] * n, [50.0] * n),
        clf_probs=[0.3, 0.05, 0.5, 0.05, 0.05, 0.05],
    )

    prediction = await service.predict(TENANT, "CV-201")

    assert prediction.predicted_failure_mode == "HEAT_DISSIPATION"
    assert prediction.failure_probability == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_diffuse_mode_probabilities_name_no_mode():
    """No single mode above threshold -> mode is None, not a weak guess."""
    n = 120
    service = make_service(
        make_readings([2.0] * n, [50.0] * n),
        clf_probs=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
    )

    prediction = await service.predict(TENANT, "CV-201")

    assert prediction.predicted_failure_mode is None
    assert prediction.failure_probability == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unknown_machine_raises_key_error():
    service = make_service(make_readings([2.0] * 120, [50.0] * 120))

    with pytest.raises(KeyError):
        await service.predict(TENANT, "NOPE-999")


@pytest.mark.asyncio
async def test_no_readings_raises_insufficient_data():
    service = make_service([])

    with pytest.raises(InsufficientDataError, match="No readings"):
        await service.predict(TENANT, "CV-201")


@pytest.mark.asyncio
async def test_fleet_ranks_by_risk_and_keeps_unpredictable_machines_visible():
    n = 120
    readings = make_readings([2.0] * n, [50.0] * n)
    silent = {
        "tenant_id": TENANT,
        "machine_id": "MC-110",
        "name": "CNC Mill",
        "line_id": "LINE-A",
    }
    repo = InMemoryPdmRepository(machines=[MACHINE, silent], readings=readings)
    service = PdmService(models=make_models(rul_cycles=20.0), repository=repo)

    fleet = await service.fleet(TENANT)

    assert [e.machine_id for e in fleet] == ["CV-201", "MC-110"]
    assert fleet[0].prediction is not None
    assert fleet[1].prediction is None
    assert fleet[1].error is not None  # visible with a reason, not dropped


# ---------------------------------------------------------------------------
# Real artifacts smoke (skipped when not trained — CI without artifacts)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (DEFAULT_ARTIFACTS_DIR / "rul_model.joblib").exists(),
    reason="trained artifacts not present",
)
def test_real_artifacts_load_and_agree_with_the_serving_schema():
    models = load_artifacts(DEFAULT_ARTIFACTS_DIR)

    channels = {role: list(np.zeros(models.window)) for role in CHANNEL_ROLES}
    names, _ = build_feature_vector(channels, 0.0, 0.0, window=models.window)
    models.rul_schema.assert_parity(names)
    models.clf_schema.assert_parity(names)
    assert models.rul_cap_cycles > 0
    assert models.cycle_hours > 0
