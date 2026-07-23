"""Train the failure-mode classifier on the UCI AI4I 2020 dataset.

Run with:

    python ml/download_datasets.py   # once
    python ml/train_failure_classifier.py

AI4I 2020 is 10,000 labelled snapshots from a milling process with five failure
modes. It is severely imbalanced (~3.4% failures overall; some modes have a few
dozen positives), which is handled explicitly with per-class sample weights —
an unweighted model here scores 96% accuracy by predicting NO_FAILURE forever,
which is exactly the useless model the metrics below would expose.

Feature parity: rows are fed through the SAME ``build_feature_vector`` as
serving, as constant windows. That is honest, not lazy — AI4I rows are
instantaneous snapshots, so the classifier learns operating-point failure
signatures (temperature deviation x load x wear), and at serve time those same
mean-level features are populated from real windows.

Outputs to ml/artifacts/:
    failure_classifier.joblib
    clf_schema.json
    clf_metrics.json            per-class P/R/F1, confusion matrix, meta
    clf_feature_importances.json
    clf_val_predictions.csv
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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ml.preprocess import DEFAULT_WINDOW, build_feature_vector, default_schema

DATA_PATH = Path(__file__).resolve().parent / "data" / "ai4i2020" / "ai4i2020.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

#: Class order is frozen here; predict_proba columns follow the encoded order.
CLASSES = (
    "NO_FAILURE",
    "TOOL_WEAR",
    "HEAT_DISSIPATION",
    "POWER",
    "OVERSTRAIN",
    "RANDOM",
)

#: Flag -> class, in priority order for the rare multi-flag rows.
FLAG_PRIORITY = (
    ("TWF", "TOOL_WEAR"),
    ("HDF", "HEAT_DISSIPATION"),
    ("PWF", "POWER"),
    ("OSF", "OVERSTRAIN"),
    ("RNF", "RANDOM"),
)


def label_rows(df: pd.DataFrame) -> pd.Series:
    """Derive one failure-mode label per row from the AI4I flag columns.

    Rows flagged as failed but with no mode flag set are dropped (they carry no
    learnable signal about *which* mode), and stray RNF flags on rows where the
    machine did not fail are treated as NO_FAILURE — the machine kept running.
    """
    labels = pd.Series("NO_FAILURE", index=df.index)
    failed = df["Machine failure"] == 1
    assigned = pd.Series(False, index=df.index)
    for flag, cls in FLAG_PRIORITY:
        pick = failed & (df[flag] == 1) & ~assigned
        labels[pick] = cls
        assigned |= pick
    labels[failed & ~assigned] = "DROP"
    return labels


def row_to_channels(
    row: pd.Series, healthy: dict[str, tuple[float, float]], window: int
) -> dict[str, list[float] | None]:
    """Map one AI4I snapshot onto our channel roles, healthy-normalised.

    Mapping (full discussion in ml/README.md):
        temperature <- Process temperature [K]
        rpm         <- Rotational speed [rpm]
        power       <- Torque x angular velocity, in kW
        vibration   <- absent in AI4I
        pressure    <- absent in AI4I
    Normalisation uses the healthy-population stats, the training-side analogue
    of the per-asset baseline used at serve time: both express "deviation from
    healthy" in z-units.
    """
    power_kw = row["Torque [Nm]"] * row["Rotational speed [rpm]"] * 2 * np.pi / 60 / 1000
    raw = {
        "temperature": row["Process temperature [K]"],
        "rpm": row["Rotational speed [rpm]"],
        "power": power_kw,
    }
    channels: dict[str, list[float] | None] = {"vibration": None, "pressure": None}
    for role, value in raw.items():
        mean, std = healthy[role]
        channels[role] = [(value - mean) / std] * window
    return channels


def main(window: int = DEFAULT_WINDOW) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    schema = default_schema(window)

    if not DATA_PATH.exists():
        raise SystemExit(
            f"Missing {DATA_PATH}. Run `python ml/download_datasets.py` first — "
            "training will not proceed on substitute data."
        )
    df = pd.read_csv(DATA_PATH)
    labels = label_rows(df)
    keep = labels != "DROP"
    dropped = int((~keep).sum())
    df, labels = df[keep].reset_index(drop=True), labels[keep].reset_index(drop=True)
    print(f"Loaded AI4I 2020: {len(df)} rows ({dropped} unlabelable failures dropped)")
    print(labels.value_counts().to_string())

    # Healthy-population stats = the training-side "baseline".
    healthy_rows = df[labels == "NO_FAILURE"]
    power_kw = (
        healthy_rows["Torque [Nm]"]
        * healthy_rows["Rotational speed [rpm]"] * 2 * np.pi / 60 / 1000
    )
    healthy = {
        "temperature": (
            float(healthy_rows["Process temperature [K]"].mean()),
            float(max(healthy_rows["Process temperature [K]"].std(), 1e-6)),
        ),
        "rpm": (
            float(healthy_rows["Rotational speed [rpm]"].mean()),
            float(max(healthy_rows["Rotational speed [rpm]"].std(), 1e-6)),
        ),
        "power": (float(power_kw.mean()), float(max(power_kw.std(), 1e-6))),
    }

    print(f"Building features through the shared pipeline (window={window}) ...")
    X_rows = []
    for _, row in df.iterrows():
        wear_h = float(row["Tool wear [min]"]) / 60.0
        names, vec = build_feature_vector(
            row_to_channels(row, healthy, window),
            time_since_maintenance_h=wear_h,
            cumulative_runtime_h=wear_h,
            window=window,
        )
        schema.assert_parity(names)
        X_rows.append(vec)
    X = np.vstack(X_rows)
    y = np.array([CLASSES.index(l) for l in labels])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # Explicit imbalance handling: weight each sample inversely to its class
    # frequency so the ~40-positive classes actually shape the trees.
    counts = np.bincount(y_train, minlength=len(CLASSES)).astype(float)
    class_weight = np.where(counts > 0, len(y_train) / (len(CLASSES) * np.maximum(counts, 1)), 0.0)
    sample_weight = class_weight[y_train]

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASSES),
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    print("Training XGBClassifier with class-balanced sample weights ...")
    model.fit(X_train, y_train, sample_weight=sample_weight)

    # --- honest metrics ---------------------------------------------------
    y_pred = model.predict(X_val)
    present = sorted(set(y_val) | set(y_pred))
    report = classification_report(
        y_val,
        y_pred,
        labels=present,
        target_names=[CLASSES[i] for i in present],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_val, y_pred, labels=list(range(len(CLASSES))))

    print("\nFailure classifier — held-out results")
    print("=" * 60)
    print(
        classification_report(
            y_val,
            y_pred,
            labels=present,
            target_names=[CLASSES[i] for i in present],
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=true, cols=pred, order=" + ",".join(CLASSES) + "):")
    print(matrix)

    # --- artifacts --------------------------------------------------------
    joblib.dump(model, ARTIFACTS_DIR / "failure_classifier.joblib")
    schema.to_json(ARTIFACTS_DIR / "clf_schema.json")

    meta = {
        "model": "XGBClassifier",
        "dataset": "UCI AI4I 2020",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "window": window,
        "classes": list(CLASSES),
        "class_counts": {CLASSES[i]: int(c) for i, c in enumerate(np.bincount(y, minlength=len(CLASSES)))},
        "healthy_stats": healthy,
        "dropped_unlabelable_failures": dropped,
        "feature_p1": np.percentile(X_train, 1, axis=0).tolist(),
        "feature_p99": np.percentile(X_train, 99, axis=0).tolist(),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }
    (ARTIFACTS_DIR / "clf_metrics.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    importances = {
        name: float(v) for name, v in zip(schema.names, model.feature_importances_)
    }
    (ARTIFACTS_DIR / "clf_feature_importances.json").write_text(
        json.dumps(importances, indent=2), encoding="utf-8"
    )

    pd.DataFrame(
        {
            "y_true": [CLASSES[i] for i in y_val],
            "y_pred": [CLASSES[i] for i in y_pred],
        }
    ).to_csv(ARTIFACTS_DIR / "clf_val_predictions.csv", index=False)

    print(f"\nArtifacts written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
