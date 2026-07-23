"""Windowed feature extraction shared by training and inference.

THE contract of this module: ``build_feature_vector`` is the ONE function that
turns sensor series into model input, and both the training scripts and
``app/services/pdm.py`` call it. Feature-order mismatch between train and serve
is the classic silent killer in deployed ML — the model happily consumes a
permuted vector and returns confident nonsense. Two guards close that door:

* the schema (names, order, window, version) is frozen into the artifacts at
  training time, and
* :func:`FeatureSchema.assert_parity` is called on every inference, raising
  :class:`FeatureParityError` the moment produced names differ from trained
  names in content or order.

Channel abstraction
-------------------
Features are computed per *channel role* — temperature, vibration, pressure,
rpm, power — not per concrete sensor. Training data (turbofan channels, milling
snapshots) and our plant sensors are both mapped onto these roles, which is what
lets one model serve both. A role a dataset or machine lacks contributes a zero
block plus a ``present`` flag, so the model can learn that absence is not the
same as zero.

Normalisation
-------------
Series are expected to be normalised against a *healthy baseline* before
windowing (:func:`normalize_series`). That is deliberate: it converts absolute
turbofan magnitudes (T50 around 1400 °R) and our conveyor magnitudes (70 °C)
into the same currency — "how far from this asset's healthy self, in units of
its healthy variability" — which is the quantity that transfers across domains.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy import stats as scipy_stats

#: Order matters everywhere in this file. Never reorder without bumping VERSION.
CHANNEL_ROLES: tuple[str, ...] = ("temperature", "vibration", "pressure", "rpm", "power")

#: Per-channel statistics, in emission order.
CHANNEL_FEATURES: tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "max",
    "rms",
    "slope",
    "roc",
    "ema",
    "crest",
    "kurtosis",
)

#: Global (non-channel) features appended after all channel blocks.
GLOBAL_FEATURES: tuple[str, ...] = (
    "time_since_maintenance_h",
    "cumulative_runtime_h",
)

DEFAULT_WINDOW = 30
#: Below this many points a channel is treated as absent rather than windowed.
MIN_CHANNEL_POINTS = 5

SCHEMA_VERSION = "1.0.0"


class FeatureParityError(RuntimeError):
    """Raised when produced features do not match the trained schema exactly."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSchema:
    """The frozen contract between a trained model and the serving path."""

    names: tuple[str, ...]
    window: int = DEFAULT_WINDOW
    version: str = SCHEMA_VERSION
    channels: tuple[str, ...] = CHANNEL_ROLES

    def assert_parity(self, produced_names: Sequence[str]) -> None:
        """Fail loudly if ``produced_names`` differ from the trained schema.

        Checks content AND order — a permutation is exactly the bug this guard
        exists to catch, and it is invisible to a length check.
        """
        produced = tuple(produced_names)
        if produced == self.names:
            return
        if len(produced) != len(self.names):
            raise FeatureParityError(
                f"Feature count mismatch: model was trained on "
                f"{len(self.names)} features, serving produced {len(produced)}. "
                f"Retrain or fix the serving feature builder."
            )
        for i, (trained, got) in enumerate(zip(self.names, produced)):
            if trained != got:
                raise FeatureParityError(
                    f"Feature order mismatch at index {i}: model was trained "
                    f"with '{trained}' here, serving produced '{got}'. "
                    f"A permuted feature vector produces silently wrong "
                    f"predictions — refusing to serve."
                )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "names": list(self.names),
                    "window": self.window,
                    "version": self.version,
                    "channels": list(self.channels),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "FeatureSchema":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            names=tuple(raw["names"]),
            window=int(raw["window"]),
            version=str(raw["version"]),
            channels=tuple(raw["channels"]),
        )


def feature_names(window: int = DEFAULT_WINDOW) -> tuple[str, ...]:
    """The canonical feature name order. Single source of truth."""
    names: list[str] = []
    for role in CHANNEL_ROLES:
        names.extend(f"{role}_{feat}" for feat in CHANNEL_FEATURES)
        names.append(f"{role}_present")
    names.extend(GLOBAL_FEATURES)
    return tuple(names)


def default_schema(window: int = DEFAULT_WINDOW) -> FeatureSchema:
    return FeatureSchema(names=feature_names(window), window=window)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def baseline_stats(series: Sequence[float]) -> tuple[float, float]:
    """Mean and std of a healthy-baseline slice; std floored to stay divisible."""
    arr = np.asarray(series, dtype=float)
    return float(arr.mean()), float(max(arr.std(), 1e-6))


def normalize_series(
    series: Sequence[float], baseline_mean: float, baseline_std: float
) -> np.ndarray:
    """Express a series as deviation from its asset's healthy self (z-units)."""
    arr = np.asarray(series, dtype=float)
    return (arr - baseline_mean) / max(baseline_std, 1e-6)


# ---------------------------------------------------------------------------
# Per-window statistics
# ---------------------------------------------------------------------------
def window_features(values: Sequence[float]) -> dict[str, float]:
    """Compute the per-channel statistics for one window of one channel.

    Every guard here (std floor, rms floor) exists so a perfectly flat window —
    common on a stopped machine — yields zeros rather than NaNs.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std())
    rms = float(np.sqrt(np.mean(arr**2)))

    if n >= 2:
        # Least-squares slope over the window index — the trend feature.
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, arr, 1)[0])
        roc = float((arr[-1] - arr[0]) / n)
    else:
        slope, roc = 0.0, 0.0

    # Exponential moving average, terminal value; alpha follows span=n.
    alpha = 2.0 / (n + 1.0)
    ema = arr[0]
    for v in arr[1:]:
        ema = alpha * v + (1 - alpha) * ema

    # Crest factor and kurtosis: the bearing-fault features. Impulsive spalling
    # spikes both long before the RMS energy rises.
    crest = float(np.max(np.abs(arr)) / rms) if rms > 1e-9 else 0.0
    kurt = float(scipy_stats.kurtosis(arr, fisher=True, bias=False)) if std > 1e-9 and n >= 4 else 0.0

    return {
        "mean": mean,
        "std": std,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "rms": rms,
        "slope": slope,
        "roc": roc,
        "ema": float(ema),
        "crest": crest,
        "kurtosis": kurt,
    }


# ---------------------------------------------------------------------------
# The shared entry point
# ---------------------------------------------------------------------------
def build_feature_vector(
    channels: Mapping[str, Optional[Sequence[float]]],
    time_since_maintenance_h: float = 0.0,
    cumulative_runtime_h: float = 0.0,
    window: int = DEFAULT_WINDOW,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Turn per-role series into the model's feature vector.

    ``channels`` maps role -> normalised series (or None/short = absent).
    Only the trailing ``window`` points of each series are used.

    Returns ``(names, values)`` — names are always emitted so the caller can
    (and must) run :meth:`FeatureSchema.assert_parity` against them.
    """
    names: list[str] = []
    values: list[float] = []

    for role in CHANNEL_ROLES:
        series = channels.get(role)
        present = series is not None and len(series) >= MIN_CHANNEL_POINTS
        if present:
            feats = window_features(list(series)[-window:])
        else:
            feats = {feat: 0.0 for feat in CHANNEL_FEATURES}
        for feat in CHANNEL_FEATURES:
            names.append(f"{role}_{feat}")
            values.append(feats[feat])
        names.append(f"{role}_present")
        values.append(1.0 if present else 0.0)

    names.append("time_since_maintenance_h")
    values.append(float(time_since_maintenance_h))
    names.append("cumulative_runtime_h")
    values.append(float(cumulative_runtime_h))

    return tuple(names), np.asarray(values, dtype=float)


def build_feature_matrix(
    windows: Sequence[Mapping[str, Optional[Sequence[float]]]],
    time_since_maintenance_h: Sequence[float],
    cumulative_runtime_h: Sequence[float],
    window: int = DEFAULT_WINDOW,
    schema: Optional[FeatureSchema] = None,
) -> np.ndarray:
    """Vectorise many windows, parity-checking every row against ``schema``."""
    schema = schema or default_schema(window)
    rows = []
    for chans, tsm, run in zip(windows, time_since_maintenance_h, cumulative_runtime_h):
        names, vec = build_feature_vector(chans, tsm, run, window=window)
        schema.assert_parity(names)
        rows.append(vec)
    return np.vstack(rows) if rows else np.empty((0, len(schema.names)))
