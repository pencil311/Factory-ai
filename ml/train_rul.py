"""Train the RUL regressor on NASA C-MAPSS FD001 run-to-failure data.

Run with:

    python ml/download_datasets.py   # once
    python ml/train_rul.py

Why this data: C-MAPSS is real *run-to-failure* trajectories with ground-truth
RUL — the thing our simulator can never provide, because a model trained on
degradation rules we wrote ourselves would just be reading our own constants
back to us. The domain transfer (turbofan -> industrial rotating equipment) is
made explicit and defensible by the feature design: every channel is normalised
against its own asset's healthy baseline, so the model learns *degradation
trajectory shapes* in deviation units, not turbofan magnitudes. See
ml/README.md for the full mapping and its limits.

Outputs to ml/artifacts/:
    rul_model.joblib            trained XGBRegressor
    rul_schema.json             frozen feature schema (names, order, window)
    rul_metrics.json            honest held-out metrics + training metadata
    rul_feature_importances.json
    rul_val_predictions.csv     per-engine predictions on the official test set
"""

from __future__ import annotations

import sys
from pathlib import Path

# Runnable as `python ml/<script>.py` from the backend root: put the backend
# root (parent of ml/) on sys.path so `ml.preprocess` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ml.preprocess import (
    DEFAULT_WINDOW,
    FeatureSchema,
    baseline_stats,
    build_feature_vector,
    default_schema,
    normalize_series,
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "cmapss"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

#: RUL cap in cycles — the standard piecewise-linear target. Early in life the
#: true RUL is unknowable from condition data (nothing has degraded yet), so
#: everything above the cap is treated as "healthy, >=125 cycles remain".
RUL_CAP_CYCLES = 125.0
#: Hours per C-MAPSS cycle when mapping to our domain. One cycle is one flight;
#: we equate it to ~10 machine-hours of duty. This is a unit convention, not a
#: physical claim — it scales the output, never the model. Documented in README.
CYCLE_HOURS = 10.0

#: How many early cycles define an engine's healthy baseline.
BASELINE_CYCLES = 15

#: C-MAPSS column layout: 0=unit, 1=cycle, 2-4=op settings, 5..25 = s1..s21.
#: Channel-role mapping (see ml/README.md for the reasoning):
#:   temperature <- s4  (T50, LPT outlet temp — strongest thermal degradation signal)
#:   pressure    <- s11 (Ps30, static HPC outlet pressure)
#:   rpm         <- s9  (Nc, physical core speed)
#:   power       <- s12 (phi, fuel-flow ratio — power demand proxy)
#:   vibration   <- absent in C-MAPSS (zero block + present=0; see README)
CHANNEL_COLUMNS: dict[str, int] = {
    "temperature": 5 + 3,   # s4
    "pressure": 5 + 10,     # s11
    "rpm": 5 + 8,           # s9
    "power": 5 + 11,        # s12
}


def load_fd001(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run `python ml/download_datasets.py` first — "
            "training will not proceed on substitute data."
        )
    return pd.read_csv(path, sep=r"\s+", header=None)


def engine_windows(
    engine: pd.DataFrame, window: int, is_training: bool
) -> tuple[list[dict], list[float], list[float], list[float]]:
    """Slide windows over one engine's life, in baseline-normalised units.

    Returns (channel_windows, rul_labels, tsm_hours, runtime_hours) — for the
    test set (``is_training=False``) only the final window is emitted, matching
    the official evaluation protocol (predict RUL at the end of each history).
    """
    cycles = engine[1].to_numpy()
    n = len(cycles)
    max_cycle = cycles.max()

    normalized: dict[str, np.ndarray] = {}
    for role, col in CHANNEL_COLUMNS.items():
        series = engine[col].to_numpy(dtype=float)
        mean, std = baseline_stats(series[: min(BASELINE_CYCLES, n)])
        normalized[role] = normalize_series(series, mean, std)

    positions = range(window, n + 1) if is_training else [n]
    windows, labels, tsm, runtime = [], [], [], []
    for end in positions:
        start = max(0, end - window)
        chans: dict[str, np.ndarray | None] = {
            role: normalized[role][start:end] for role in CHANNEL_COLUMNS
        }
        chans["vibration"] = None  # honest absence, not fabricated zeros-as-data
        windows.append(chans)
        labels.append(min(float(max_cycle - cycles[end - 1]), RUL_CAP_CYCLES))
        hours = float(cycles[end - 1]) * CYCLE_HOURS
        tsm.append(hours)
        runtime.append(hours)
    return windows, labels, tsm, runtime


def build_dataset(
    df: pd.DataFrame, window: int, schema: FeatureSchema, is_training: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature matrix + labels + engine ids, parity-checked on every row."""
    X_rows, y_rows, unit_rows = [], [], []
    for unit, engine in df.groupby(0):
        windows, labels, tsm, runtime = engine_windows(engine, window, is_training)
        for chans, label, t, r in zip(windows, labels, tsm, runtime):
            names, vec = build_feature_vector(chans, t, r, window=window)
            schema.assert_parity(names)
            X_rows.append(vec)
            y_rows.append(label)
            unit_rows.append(unit)
    return np.vstack(X_rows), np.asarray(y_rows), np.asarray(unit_rows)


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The official asymmetric C-MAPSS score. Late predictions cost more.

    d > 0 means we predicted MORE life than the engine had — the maintenance
    visit gets scheduled after the failure. That is why exp(d/10) grows faster
    than the early-side exp(-d/13).
    """
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def main(window: int = DEFAULT_WINDOW) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    schema = default_schema(window)

    print("Loading C-MAPSS FD001 ...")
    train_df = load_fd001("train_FD001.txt")
    test_df = load_fd001("test_FD001.txt")
    true_test_rul = np.minimum(
        load_fd001("RUL_FD001.txt")[0].to_numpy(dtype=float), RUL_CAP_CYCLES
    )

    # Hold out whole engines, never windows: windows within one engine are
    # heavily correlated, and splitting them across train/val would leak.
    rng = np.random.default_rng(42)
    units = train_df[0].unique()
    val_units = set(rng.choice(units, size=max(1, len(units) // 5), replace=False))

    print(f"Extracting windowed features (window={window}) ...")
    X_all, y_all, unit_ids = build_dataset(train_df, window, schema, is_training=True)
    val_mask = np.isin(unit_ids, list(val_units))
    X_train, y_train = X_all[~val_mask], y_all[~val_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    print(
        f"  {X_train.shape[0]} train windows / {X_val.shape[0]} val windows "
        f"({len(units) - len(val_units)}/{len(val_units)} engines), "
        f"{X_train.shape[1]} features"
    )

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    print("Training XGBRegressor ...")
    model.fit(X_train, y_train)

    # --- honest metrics ---------------------------------------------------
    val_pred = np.clip(model.predict(X_val), 0.0, RUL_CAP_CYCLES)
    X_test, _, _ = build_dataset(test_df, window, schema, is_training=False)
    test_pred = np.clip(model.predict(X_test), 0.0, RUL_CAP_CYCLES)

    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    metrics = {
        "val_rmse_cycles": rmse(y_val, val_pred),
        "val_mae_cycles": mae(y_val, val_pred),
        "test_rmse_cycles": rmse(true_test_rul, test_pred),
        "test_mae_cycles": mae(true_test_rul, test_pred),
        "test_nasa_score": nasa_score(true_test_rul, test_pred),
        "n_test_engines": int(len(true_test_rul)),
    }

    # --- artifacts --------------------------------------------------------
    joblib.dump(model, ARTIFACTS_DIR / "rul_model.joblib")
    schema.to_json(ARTIFACTS_DIR / "rul_schema.json")

    meta = {
        "model": "XGBRegressor",
        "dataset": "NASA C-MAPSS FD001",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "window": window,
        "rul_cap_cycles": RUL_CAP_CYCLES,
        "cycle_hours": CYCLE_HOURS,
        "baseline_cycles": BASELINE_CYCLES,
        "channel_columns": CHANNEL_COLUMNS,
        "n_train_windows": int(X_train.shape[0]),
        "n_val_windows": int(X_val.shape[0]),
        "train_target_mean_cycles": float(y_train.mean()),
        # Serving clips features into this envelope: outside it the model is
        # extrapolating and tree predictions silently saturate.
        "feature_p1": np.percentile(X_train, 1, axis=0).tolist(),
        "feature_p99": np.percentile(X_train, 99, axis=0).tolist(),
        "metrics": metrics,
    }
    (ARTIFACTS_DIR / "rul_metrics.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    importances = {
        name: float(v)
        for name, v in zip(schema.names, model.feature_importances_)
    }
    (ARTIFACTS_DIR / "rul_feature_importances.json").write_text(
        json.dumps(importances, indent=2), encoding="utf-8"
    )

    pd.DataFrame({"y_true": true_test_rul, "y_pred": test_pred}).to_csv(
        ARTIFACTS_DIR / "rul_val_predictions.csv", index=False
    )

    print("\nRUL model — held-out results")
    print("=" * 48)
    print(f"  val   RMSE {metrics['val_rmse_cycles']:6.2f} cycles   MAE {metrics['val_mae_cycles']:6.2f}")
    print(f"  test  RMSE {metrics['test_rmse_cycles']:6.2f} cycles   MAE {metrics['test_mae_cycles']:6.2f}")
    print(f"  test  NASA score {metrics['test_nasa_score']:.1f}  ({metrics['n_test_engines']} engines)")
    top = sorted(importances.items(), key=lambda kv: -kv[1])[:5]
    print("  top features: " + ", ".join(f"{n} ({v:.3f})" for n, v in top))
    print(f"\nArtifacts written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
