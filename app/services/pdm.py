"""Predictive-maintenance inference service.

Loads the trained artifacts from ``ml/artifacts/`` and, per machine, turns
recent ``sensor_readings`` into a prediction: failure probability, RUL, health,
failure mode, trend, and the contributing features RCA consumes downstream.

Train/serve parity: features are built by the SAME ``ml.preprocess`` functions
the training scripts used, and every inference runs
``FeatureSchema.assert_parity`` against the schema frozen into the artifacts.
If the serving feature builder ever drifts from what the model was trained on,
inference refuses to run rather than returning confident nonsense.

If artifacts are missing this module raises :class:`PdmArtifactsMissingError`
with the exact commands to produce them. There is NO heuristic fallback — a
made-up RUL number presented as a prediction is worse than an honest 503.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

import numpy as np

from ml.preprocess import (
    CHANNEL_ROLES,
    MIN_CHANNEL_POINTS,
    FeatureSchema,
    baseline_stats,
    build_feature_vector,
    normalize_series,
)
from app.db import normalize_tenant_id
from app.models.reading import flatten_reading_document
from app.schemas.machine import COLLECTIONS
from app.schemas.pdm import (
    ContributingFeature,
    FleetEntryOut,
    ModelInfoOut,
    PdmPredictionOut,
    TrendDirection,
    TrendOut,
    TrendPointOut,
)

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "ml" / "artifacts"

#: Slope of the aggregate deviation series (z-units per reading) beyond which
#: the condition is moving rather than holding.
TREND_SLOPE_THRESHOLD = 0.01
#: A failure mode is only named when the model puts real mass on it.
FAILURE_MODE_MIN_PROBABILITY = 0.15
#: Portion of the fetched history treated as the machine's healthy baseline.
BASELINE_FRACTION = 0.25
MIN_BASELINE_POINTS = 10
#: How many contributing features to expose.
TOP_FEATURES = 8


class PdmArtifactsMissingError(RuntimeError):
    """Raised at startup when trained models are absent. Never worked around."""


class InsufficientDataError(RuntimeError):
    """Raised when a machine has too little history to predict on."""


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
REQUIRED_ARTIFACTS = (
    "rul_model.joblib",
    "rul_schema.json",
    "rul_metrics.json",
    "rul_feature_importances.json",
    "failure_classifier.joblib",
    "clf_schema.json",
    "clf_metrics.json",
    "clf_feature_importances.json",
)


@dataclass
class PdmModels:
    """Everything the service needs from ml/artifacts, loaded once."""

    rul_model: Any
    rul_schema: FeatureSchema
    rul_meta: dict
    rul_importances: dict[str, float]
    clf_model: Any
    clf_schema: FeatureSchema
    clf_meta: dict
    clf_importances: dict[str, float]
    artifacts_dir: Path
    loaded_at: datetime

    @property
    def classes(self) -> list[str]:
        return list(self.clf_meta["classes"])

    @property
    def rul_cap_cycles(self) -> float:
        return float(self.rul_meta["rul_cap_cycles"])

    @property
    def cycle_hours(self) -> float:
        return float(self.rul_meta["cycle_hours"])

    @property
    def window(self) -> int:
        return int(self.rul_schema.window)


def load_artifacts(artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR) -> PdmModels:
    """Load trained models or fail with instructions. No silent fallback."""
    artifacts_dir = Path(artifacts_dir)
    missing = [f for f in REQUIRED_ARTIFACTS if not (artifacts_dir / f).exists()]
    if missing:
        raise PdmArtifactsMissingError(
            f"Predictive-maintenance artifacts missing from {artifacts_dir}: "
            f"{', '.join(missing)}. Train them first:\n"
            "  python ml/download_datasets.py\n"
            "  python ml/train_rul.py\n"
            "  python ml/train_failure_classifier.py\n"
            "There is no heuristic fallback: predictions come from models "
            "trained on real run-to-failure data or not at all."
        )

    import joblib  # deferred so importing this module never needs it

    def read_json(name: str) -> dict:
        return json.loads((artifacts_dir / name).read_text(encoding="utf-8"))

    models = PdmModels(
        rul_model=joblib.load(artifacts_dir / "rul_model.joblib"),
        rul_schema=FeatureSchema.from_json(artifacts_dir / "rul_schema.json"),
        rul_meta=read_json("rul_metrics.json"),
        rul_importances=read_json("rul_feature_importances.json"),
        clf_model=joblib.load(artifacts_dir / "failure_classifier.joblib"),
        clf_schema=FeatureSchema.from_json(artifacts_dir / "clf_schema.json"),
        clf_meta=read_json("clf_metrics.json"),
        clf_importances=read_json("clf_feature_importances.json"),
        artifacts_dir=artifacts_dir,
        loaded_at=datetime.now(timezone.utc),
    )
    logger.info(
        "PdM artifacts loaded from %s (%d features, window=%d)",
        artifacts_dir,
        len(models.rul_schema.names),
        models.window,
    )
    return models


# ---------------------------------------------------------------------------
# Data access port
# ---------------------------------------------------------------------------
class PdmRepository(Protocol):
    """The only persistence surface the PdM service needs.

    Every method takes ``tenant_id`` first. The service is a process-wide
    singleton holding loaded models, so the tenant cannot be bound at
    construction — it has to travel with each call, and making it a required
    positional argument means it cannot be left out by accident.
    """

    async def fetch_machine(
        self, tenant_id: str, machine_id: str
    ) -> Optional[Mapping[str, Any]]: ...

    async def fetch_machines(self, tenant_id: str) -> list[Mapping[str, Any]]: ...

    async def fetch_sensors(
        self, tenant_id: str, machine_id: str
    ) -> list[Mapping[str, Any]]:
        """Sensor catalogue entries (with normal bands) for one machine."""
        ...

    async def fetch_readings(
        self, tenant_id: str, machine_id: str, since: Optional[datetime], limit: int
    ) -> list[Mapping[str, Any]]:
        """Readings for one machine, ascending by timestamp."""
        ...


class MongoPdmRepository:
    """:class:`PdmRepository` over the live sensor_readings collection."""

    def __init__(self, database: Any = None) -> None:
        self._db = database

    def _scope(self, tenant_id: str) -> Any:
        from app.db import get_tenant_scope

        return get_tenant_scope(tenant_id, self._db)

    async def fetch_machine(
        self, tenant_id: str, machine_id: str
    ) -> Optional[Mapping[str, Any]]:
        return await self._scope(tenant_id)[COLLECTIONS.machines].find_one(
            {"machine_id": machine_id}
        )

    async def fetch_machines(self, tenant_id: str) -> list[Mapping[str, Any]]:
        cursor = (
            self._scope(tenant_id)[COLLECTIONS.machines]
            .find({})
            .sort([("machine_id", 1)])
        )
        return [doc async for doc in cursor]

    async def fetch_sensors(
        self, tenant_id: str, machine_id: str
    ) -> list[Mapping[str, Any]]:
        cursor = self._scope(tenant_id)[COLLECTIONS.sensors].find(
            {"machine_id": machine_id}
        )
        return [doc async for doc in cursor]

    async def fetch_readings(
        self, tenant_id: str, machine_id: str, since: Optional[datetime], limit: int
    ) -> list[Mapping[str, Any]]:
        # In the time-series collection the identifiers live under `meta`;
        # documents are flattened on the way out so everything downstream sees
        # one reading shape regardless of storage layout.
        query: dict = {"meta.machine_id": machine_id}
        if since is not None:
            query["timestamp"] = {"$gte": since}
        # Newest N, then flipped ascending: bounds work without scanning history.
        cursor = (
            self._scope(tenant_id)[COLLECTIONS.sensor_readings]
            .find(query)
            .sort([("timestamp", -1)])
            .limit(limit)
        )
        docs = [flatten_reading_document(doc) async for doc in cursor]
        return list(reversed(docs))


class InMemoryPdmRepository:
    """:class:`PdmRepository` over plain dicts — for tests and dry runs."""

    def __init__(
        self,
        machines: Iterable[Mapping[str, Any]],
        readings: Iterable[Mapping[str, Any]],
        sensors: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        self._machines = [dict(m) for m in machines]
        self._sensors = [dict(s) for s in sensors]
        self._readings = sorted(
            (dict(r) for r in readings), key=lambda r: r["timestamp"]
        )

    @staticmethod
    def _owned(
        rows: Iterable[Mapping[str, Any]], tenant_id: str
    ) -> list[Mapping[str, Any]]:
        return [r for r in rows if r.get("tenant_id") == tenant_id]

    async def fetch_machine(
        self, tenant_id: str, machine_id: str
    ) -> Optional[Mapping[str, Any]]:
        return next(
            (
                m
                for m in self._owned(self._machines, tenant_id)
                if m["machine_id"] == machine_id
            ),
            None,
        )

    async def fetch_machines(self, tenant_id: str) -> list[Mapping[str, Any]]:
        return self._owned(self._machines, tenant_id)

    async def fetch_sensors(
        self, tenant_id: str, machine_id: str
    ) -> list[Mapping[str, Any]]:
        return [
            s
            for s in self._owned(self._sensors, tenant_id)
            if s["machine_id"] == machine_id
        ]

    async def fetch_readings(
        self, tenant_id: str, machine_id: str, since: Optional[datetime], limit: int
    ) -> list[Mapping[str, Any]]:
        rows = [
            r
            for r in self._owned(self._readings, tenant_id)
            if r["machine_id"] == machine_id
            and (since is None or r["timestamp"] >= since)
        ]
        return rows[-limit:]


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------
class PdmService:
    """Per-machine predictions from recent readings + trained models."""

    def __init__(
        self,
        models: PdmModels,
        repository: Optional[PdmRepository] = None,
        history_hours: float = 24.0,
    ) -> None:
        self.models = models
        self.repository = repository or MongoPdmRepository()
        self.history_hours = history_hours

    # -- channel assembly --------------------------------------------------
    @staticmethod
    def _series_by_role(
        readings: Sequence[Mapping[str, Any]]
    ) -> dict[str, list[float]]:
        """Group readings into per-role series, averaging same-type sensors.

        Readings are already time-ascending; same-timestamp values from two
        sensors of one type (e.g. two temperature probes) are averaged so one
        machine contributes one series per role.
        """
        buckets: dict[str, dict[Any, list[float]]] = {}
        for reading in readings:
            role = str(reading["sensor_type"])
            if role not in CHANNEL_ROLES:
                continue
            buckets.setdefault(role, {}).setdefault(reading["timestamp"], []).append(
                float(reading["value"])
            )
        return {
            role: [float(np.mean(vals)) for _, vals in sorted(stamps.items())]
            for role, stamps in buckets.items()
        }

    @staticmethod
    def _healthy_reference(
        sensors: Sequence[Mapping[str, Any]]
    ) -> dict[str, tuple[float, float]]:
        """Per-role healthy (mean, std) from the sensor catalogue's normal bands.

        Baseline = band midpoint, spread = band width / 4 (a normal band read
        as a ±2-sigma envelope). This is the engineering definition of "healthy
        self" and — unlike any window-derived baseline — it does not depend on
        the machine having been healthy when the fetch window opened. A
        window-start baseline reads recovery as degradation, because returning
        to normal moves *away* from the sick starting point.
        """
        by_role: dict[str, list[tuple[float, float]]] = {}
        for sensor in sensors:
            role = str(sensor.get("type", ""))
            try:
                lo, hi = float(sensor["normal_min"]), float(sensor["normal_max"])
            except (KeyError, TypeError, ValueError):
                continue
            by_role.setdefault(role, []).append(
                ((lo + hi) / 2.0, max((hi - lo) / 4.0, 1e-6))
            )
        return {
            role: (
                float(np.mean([m for m, _ in entries])),
                float(np.mean([s for _, s in entries])),
            )
            for role, entries in by_role.items()
        }

    def _normalized_channels(
        self,
        series: Mapping[str, list[float]],
        healthy_ref: Mapping[str, tuple[float, float]],
    ) -> tuple[dict[str, Optional[np.ndarray]], int]:
        """Normalise each role against the machine's healthy reference.

        Prefers the catalogue-derived reference; falls back to early-window
        statistics only for roles with no configured normal band, where the
        recovery-reads-as-degradation caveat applies and is accepted.
        """
        channels: dict[str, Optional[np.ndarray]] = {}
        usable = 0
        for role in CHANNEL_ROLES:
            values = series.get(role)
            if not values or len(values) < MIN_CHANNEL_POINTS:
                channels[role] = None
                continue
            if role in healthy_ref:
                mean, std = healthy_ref[role]
            else:
                n_baseline = max(MIN_BASELINE_POINTS, int(len(values) * BASELINE_FRACTION))
                mean, std = baseline_stats(values[: min(n_baseline, len(values))])
            channels[role] = normalize_series(values, mean, std)
            usable += 1
        if usable == 0:
            raise InsufficientDataError(
                "No channel has enough readings to predict on "
                f"(need >= {MIN_CHANNEL_POINTS} per sensor type)."
            )
        return channels, usable

    def _clip_to_training_envelope(self, vec: np.ndarray, meta: dict) -> np.ndarray:
        """Clamp features into the p1..p99 envelope seen in training.

        Outside that envelope tree models silently saturate; clipping makes the
        out-of-distribution behaviour explicit and bounded.
        """
        low = np.asarray(meta["feature_p1"], dtype=float)
        high = np.asarray(meta["feature_p99"], dtype=float)
        return np.clip(vec, low, high)

    # -- feature vector ----------------------------------------------------
    def _features(
        self,
        channels: Mapping[str, Optional[np.ndarray]],
        machine: Mapping[str, Any],
        now: datetime,
    ) -> tuple[np.ndarray, dict[str, float]]:
        tsm_h = _hours_since(machine.get("last_maintenance_at"), now)
        runtime_h = _hours_since(machine.get("installed_at"), now)

        names, vec = build_feature_vector(
            channels,
            time_since_maintenance_h=tsm_h,
            cumulative_runtime_h=runtime_h,
            window=self.models.window,
        )
        # The parity guard — runs on EVERY inference, against both schemas.
        self.models.rul_schema.assert_parity(names)
        self.models.clf_schema.assert_parity(names)
        return vec, dict(zip(names, vec.tolist()))

    # -- model heads -------------------------------------------------------
    def _predict_rul_hours(self, vec: np.ndarray) -> float:
        clipped = self._clip_to_training_envelope(vec, self.models.rul_meta)
        cycles = float(self.models.rul_model.predict(clipped.reshape(1, -1))[0])
        cycles = float(np.clip(cycles, 0.0, self.models.rul_cap_cycles))
        return cycles * self.models.cycle_hours

    def _predict_failure(self, vec: np.ndarray) -> tuple[float, Optional[str], dict[str, float]]:
        clipped = self._clip_to_training_envelope(vec, self.models.clf_meta)
        probs = np.asarray(
            self.models.clf_model.predict_proba(clipped.reshape(1, -1))[0], dtype=float
        )
        by_class = dict(zip(self.models.classes, probs.tolist()))
        failure_probability = float(np.clip(1.0 - by_class.get("NO_FAILURE", 0.0), 0.0, 1.0))

        mode: Optional[str] = None
        fault_probs = {c: p for c, p in by_class.items() if c != "NO_FAILURE"}
        if fault_probs:
            best = max(fault_probs, key=fault_probs.get)
            if fault_probs[best] >= FAILURE_MODE_MIN_PROBABILITY:
                mode = best
        return failure_probability, mode, by_class

    # -- trend -------------------------------------------------------------
    @staticmethod
    def _deviation_series(channels: Mapping[str, Optional[np.ndarray]]) -> np.ndarray:
        """Aggregate |deviation from healthy| per time step, across channels.

        In baseline-normalised units every channel speaks the same language, so
        the mean absolute value is a single 'how far from healthy' series —
        rising means degrading regardless of which mechanism is at work.
        """
        present = [c for c in channels.values() if c is not None]
        length = min(len(c) for c in present)
        stacked = np.vstack([np.abs(c[-length:]) for c in present])
        return stacked.mean(axis=0)

    @classmethod
    def _trend_direction(cls, channels: Mapping[str, Optional[np.ndarray]]) -> TrendDirection:
        deviation = cls._deviation_series(channels)
        if len(deviation) < 2:
            return TrendDirection.stable
        slope = float(np.polyfit(np.arange(len(deviation)), deviation, 1)[0])
        if slope > TREND_SLOPE_THRESHOLD:
            return TrendDirection.degrading
        if slope < -TREND_SLOPE_THRESHOLD:
            return TrendDirection.improving
        return TrendDirection.stable

    # -- explanation -------------------------------------------------------
    def _contributing_features(
        self, values_by_name: Mapping[str, float]
    ) -> list[ContributingFeature]:
        """Top features by model importance, with their current values.

        Importance is the max of the two models' normalised importances — a
        feature that drives either the RUL estimate or the mode classification
        is worth surfacing to RCA.
        """
        def normalized(imp: Mapping[str, float]) -> dict[str, float]:
            total = sum(imp.values()) or 1.0
            return {k: v / total for k, v in imp.items()}

        rul_imp = normalized(self.models.rul_importances)
        clf_imp = normalized(self.models.clf_importances)
        merged = {
            name: max(rul_imp.get(name, 0.0), clf_imp.get(name, 0.0))
            for name in values_by_name
        }
        top = sorted(merged.items(), key=lambda kv: -kv[1])[:TOP_FEATURES]
        return [
            ContributingFeature(
                name=name,
                value=round(values_by_name[name], 4),
                importance=round(importance, 4),
            )
            for name, importance in top
        ]

    # -- confidence --------------------------------------------------------
    def _confidence(self, usable_channels: int, n_readings: int) -> float:
        """Honest composite: channel coverage x data sufficiency x model skill.

        Model skill is derived from the held-out RMSE frozen into the
        artifacts, so confidence degrades if a worse model is deployed.
        """
        coverage = usable_channels / len(CHANNEL_ROLES)
        needed = self.models.window * 2
        sufficiency = min(1.0, n_readings / needed)
        rmse = float(self.models.rul_meta["metrics"]["test_rmse_cycles"])
        model_skill = 1.0 - min(1.0, rmse / self.models.rul_cap_cycles)
        return round(float(np.clip(coverage * sufficiency * model_skill, 0.0, 1.0)), 4)

    # -- public API --------------------------------------------------------
    async def predict(self, tenant_id: str, machine_id: str) -> PdmPredictionOut:
        """Full prediction for one machine from its recent readings."""
        tenant_id = normalize_tenant_id(tenant_id)
        machine = await self.repository.fetch_machine(tenant_id, machine_id)
        if machine is None:
            raise KeyError(f"Machine '{machine_id}' not found")

        now = datetime.now(timezone.utc)
        readings = await self.repository.fetch_readings(
            tenant_id,
            machine_id,
            since=now - timedelta(hours=self.history_hours),
            limit=self.models.window * 20,
        )
        if not readings:
            raise InsufficientDataError(
                f"No readings for '{machine_id}' in the last "
                f"{self.history_hours:g}h — is ingestion running?"
            )

        sensors = await self.repository.fetch_sensors(tenant_id, machine_id)
        series = self._series_by_role(readings)
        channels, usable = self._normalized_channels(
            series, self._healthy_reference(sensors)
        )
        vec, values_by_name = self._features(channels, machine, now)

        rul_hours = self._predict_rul_hours(vec)
        failure_probability, mode, _ = self._predict_failure(vec)
        trend = self._trend_direction(channels)

        # Health blends the two heads: how much life remains, and how loudly
        # the failure classifier is objecting to the current operating point.
        rul_fraction = rul_hours / (self.models.rul_cap_cycles * self.models.cycle_hours)
        health = float(np.clip(0.6 * rul_fraction + 0.4 * (1.0 - failure_probability), 0.0, 1.0))

        # A capped RUL is censored ("at least this much"), not a forecast date.
        censored = rul_hours >= 0.95 * self.models.rul_cap_cycles * self.models.cycle_hours
        predicted_failure_time = None if censored else now + timedelta(hours=rul_hours)

        return PdmPredictionOut(
            machine_id=machine_id,
            failure_probability=round(failure_probability, 4),
            remaining_useful_life_hours=round(rul_hours, 2),
            health_score=round(health, 4),
            predicted_failure_time=predicted_failure_time,
            predicted_failure_mode=mode,
            confidence=self._confidence(usable, len(readings)),
            contributing_features=self._contributing_features(values_by_name),
            trend_direction=trend,
            readings_used=len(readings),
            channels_present=[r for r in CHANNEL_ROLES if channels.get(r) is not None],
            generated_at=now,
        )

    async def trend(
        self, tenant_id: str, machine_id: str, hours: int = 168
    ) -> TrendOut:
        """Health trajectory over ``hours``, bucketed for plotting."""
        tenant_id = normalize_tenant_id(tenant_id)
        machine = await self.repository.fetch_machine(tenant_id, machine_id)
        if machine is None:
            raise KeyError(f"Machine '{machine_id}' not found")

        now = datetime.now(timezone.utc)
        readings = await self.repository.fetch_readings(
            tenant_id, machine_id, since=now - timedelta(hours=hours), limit=50_000
        )
        if not readings:
            return TrendOut(
                machine_id=machine_id, hours=hours,
                direction=TrendDirection.stable, points=[],
            )

        sensors = await self.repository.fetch_sensors(tenant_id, machine_id)
        series = self._series_by_role(readings)
        channels, _ = self._normalized_channels(
            series, self._healthy_reference(sensors)
        )
        deviation = self._deviation_series(channels)

        n_buckets = min(24, max(2, len(deviation) // MIN_CHANNEL_POINTS))
        bucket_edges = np.linspace(0, len(deviation), n_buckets + 1, dtype=int)
        first_ts = readings[0]["timestamp"]
        last_ts = readings[-1]["timestamp"]
        span = (last_ts - first_ts) or timedelta(seconds=1)

        points: list[TrendPointOut] = []
        for i in range(n_buckets):
            lo, hi = bucket_edges[i], bucket_edges[i + 1]
            if hi <= lo:
                continue
            bucket_dev = float(deviation[lo:hi].mean())
            # Deviation -> health via a soft knee: 0 z -> 1.0, ~3 z -> ~0.25.
            health = float(1.0 / (1.0 + (bucket_dev / 1.5) ** 2))
            points.append(
                TrendPointOut(
                    timestamp=first_ts + span * ((lo + hi) / (2 * len(deviation))),
                    health_score=round(health, 4),
                    deviation=round(bucket_dev, 4),
                    readings=int(hi - lo),
                )
            )

        return TrendOut(
            machine_id=machine_id,
            hours=hours,
            direction=self._trend_direction(channels),
            points=points,
        )

    async def fleet(self, tenant_id: str) -> list[FleetEntryOut]:
        """Predictions for every machine in one tenant, most-at-risk first."""
        tenant_id = normalize_tenant_id(tenant_id)
        entries: list[FleetEntryOut] = []
        for machine in await self.repository.fetch_machines(tenant_id):
            machine_id = str(machine["machine_id"])
            try:
                prediction = await self.predict(tenant_id, machine_id)
                entries.append(
                    FleetEntryOut(
                        machine_id=machine_id,
                        name=str(machine.get("name", "")),
                        line_id=str(machine.get("line_id", "")),
                        prediction=prediction,
                    )
                )
            except (InsufficientDataError, KeyError) as exc:
                entries.append(
                    FleetEntryOut(
                        machine_id=machine_id,
                        name=str(machine.get("name", "")),
                        line_id=str(machine.get("line_id", "")),
                        error=str(exc),
                    )
                )

        def risk(entry: FleetEntryOut) -> tuple:
            if entry.prediction is None:
                return (1, 0.0, 0.0)  # unpredictable machines sink to the bottom
            p = entry.prediction
            return (0, -p.failure_probability, p.remaining_useful_life_hours)

        return sorted(entries, key=risk)

    def model_info(self) -> ModelInfoOut:
        """Loaded versions and honest held-out metrics."""
        m = self.models
        return ModelInfoOut(
            artifacts_dir=str(m.artifacts_dir),
            loaded_at=m.loaded_at,
            schema_version=m.rul_schema.version,
            feature_count=len(m.rul_schema.names),
            window=m.window,
            rul_model={
                "algorithm": m.rul_meta.get("model"),
                "dataset": m.rul_meta.get("dataset"),
                "trained_at": m.rul_meta.get("trained_at"),
                "rul_cap_cycles": m.rul_cap_cycles,
                "cycle_hours": m.cycle_hours,
                "metrics": m.rul_meta.get("metrics", {}),
            },
            failure_classifier={
                "algorithm": m.clf_meta.get("model"),
                "dataset": m.clf_meta.get("dataset"),
                "trained_at": m.clf_meta.get("trained_at"),
                "classes": m.classes,
                "macro_f1": m.clf_meta.get("classification_report", {})
                .get("macro avg", {})
                .get("f1-score"),
            },
        )


def _hours_since(value: Any, now: datetime) -> float:
    """Hours from a stored datetime to now; 0.0 when unknown."""
    if value is None:
        return 0.0
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return max(0.0, (now - value).total_seconds() / 3600.0)


# ---------------------------------------------------------------------------
# Process-wide instance, wired by the FastAPI lifespan
# ---------------------------------------------------------------------------
_service: Optional[PdmService] = None
_load_error: Optional[str] = None


def init_pdm_service(
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    repository: Optional[PdmRepository] = None,
) -> PdmService:
    """Load artifacts and install the process-wide service. Raises loudly."""
    global _service, _load_error
    try:
        models = load_artifacts(artifacts_dir)
    except PdmArtifactsMissingError as exc:
        _service, _load_error = None, str(exc)
        raise
    _service = PdmService(models=models, repository=repository)
    _load_error = None
    return _service


def get_pdm_service() -> PdmService:
    """The active service, or the load error explaining why there is none."""
    if _service is None:
        raise PdmArtifactsMissingError(
            _load_error
            or "PdM service was never initialised — startup did not run init_pdm_service()."
        )
    return _service


def set_pdm_service(service: Optional[PdmService]) -> None:
    """Test hook."""
    global _service, _load_error
    _service = service
    _load_error = None
