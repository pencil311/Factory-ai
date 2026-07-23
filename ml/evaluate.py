"""Evaluation report: quantify what the models buy us over doing nothing.

Run with (after both trainers):

    python ml/evaluate.py

Writes ml/artifacts/metrics.json and prints a human-readable summary. Every
model metric is paired with a do-nothing baseline — predict-the-mean for RUL,
predict-the-majority-class for the classifier — so the models' value is
quantified, not asserted. A model that does not beat its baseline should not be
deployed, and this report is where that would show.

Also runs a feature-sanity check against the CWRU bearing data: our
bearing-relevant features (kurtosis, crest factor) must separate seeded-fault
vibration from healthy vibration. That is what CWRU is *for* in this pipeline —
it validates the feature design against real bearing physics without training
on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Runnable as `python ml/<script>.py` from the backend root: put the backend
# root (parent of ml/) on sys.path so `ml.preprocess` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.preprocess import window_features

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
CWRU_DIR = Path(__file__).resolve().parent / "data" / "cwru"


def _require(path: Path, producer: str) -> Path:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run `python {producer}` first.")
    return path


# ---------------------------------------------------------------------------
# RUL vs predict-the-mean
# ---------------------------------------------------------------------------
def evaluate_rul() -> dict:
    preds = pd.read_csv(_require(ARTIFACTS_DIR / "rul_val_predictions.csv", "ml/train_rul.py"))
    meta = json.loads((ARTIFACTS_DIR / "rul_metrics.json").read_text(encoding="utf-8"))

    y_true = preds["y_true"].to_numpy()
    y_pred = preds["y_pred"].to_numpy()
    baseline_pred = np.full_like(y_true, meta["train_target_mean_cycles"])

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    def nasa(a, b):
        d = b - a
        return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))

    model_rmse, base_rmse = rmse(y_true, y_pred), rmse(y_true, baseline_pred)
    return {
        "dataset": meta["dataset"],
        "n_test_engines": int(len(y_true)),
        "model": {
            "rmse_cycles": model_rmse,
            "mae_cycles": mae(y_true, y_pred),
            "nasa_score": nasa(y_true, y_pred),
        },
        "baseline_predict_mean": {
            "rmse_cycles": base_rmse,
            "mae_cycles": mae(y_true, baseline_pred),
            "nasa_score": nasa(y_true, baseline_pred),
        },
        # Fraction of the baseline's error the model eliminates. 0 = worthless.
        "skill_vs_baseline": float(1.0 - model_rmse / base_rmse),
    }


# ---------------------------------------------------------------------------
# Classifier vs predict-the-majority
# ---------------------------------------------------------------------------
def evaluate_classifier() -> dict:
    preds = pd.read_csv(
        _require(ARTIFACTS_DIR / "clf_val_predictions.csv", "ml/train_failure_classifier.py")
    )
    meta = json.loads((ARTIFACTS_DIR / "clf_metrics.json").read_text(encoding="utf-8"))

    y_true, y_pred = preds["y_true"], preds["y_pred"]
    majority = y_true.mode()[0]
    classes = sorted(y_true.unique())

    def macro_f1(pred: pd.Series) -> float:
        scores = []
        for cls in classes:
            tp = int(((y_true == cls) & (pred == cls)).sum())
            fp = int(((y_true != cls) & (pred == cls)).sum())
            fn = int(((y_true == cls) & (pred != cls)).sum())
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            scores.append(2 * p * r / (p + r) if p + r else 0.0)
        return float(np.mean(scores))

    # Failure detection collapsed to binary: did we notice something is wrong?
    true_fail = y_true != "NO_FAILURE"
    pred_fail = y_pred != "NO_FAILURE"
    detected = int((true_fail & pred_fail).sum())

    return {
        "dataset": meta["dataset"],
        "n_val_rows": int(len(y_true)),
        "model": {
            "accuracy": float((y_true == y_pred).mean()),
            "macro_f1": macro_f1(y_pred),
            "failure_recall": float(detected / max(1, int(true_fail.sum()))),
        },
        "baseline_predict_majority": {
            "accuracy": float((y_true == majority).mean()),
            "macro_f1": macro_f1(pd.Series([majority] * len(y_true), index=y_true.index)),
            "failure_recall": 0.0,  # the majority class is NO_FAILURE
        },
        "per_class": meta["classification_report"],
        "confusion_matrix": meta["confusion_matrix"],
        "classes": meta["classes"],
    }


# ---------------------------------------------------------------------------
# CWRU feature sanity — do the bearing features see real bearing faults?
# ---------------------------------------------------------------------------
def evaluate_cwru_features(window_samples: int = 2048) -> dict:
    from scipy.io import loadmat

    files = {
        "normal": "97_normal.mat",
        "inner_race": "105_inner_race.mat",
        "ball": "118_ball.mat",
        "outer_race": "130_outer_race.mat",
    }
    missing = [f for f in files.values() if not (CWRU_DIR / f).exists()]
    if missing:
        # Loud skip, never silent: the report says exactly what is absent.
        return {
            "skipped": True,
            "reason": f"CWRU files missing: {missing}. Run `python ml/download_datasets.py`.",
        }

    def stats(name: str) -> dict[str, float]:
        mat = loadmat(CWRU_DIR / name)
        key = next(k for k in mat if k.endswith("DE_time"))
        signal = mat[key].ravel()
        kurts, crests = [], []
        for start in range(0, len(signal) - window_samples, window_samples):
            feats = window_features(signal[start : start + window_samples])
            kurts.append(feats["kurtosis"])
            crests.append(feats["crest"])
        return {"kurtosis": float(np.median(kurts)), "crest": float(np.median(crests))}

    result = {condition: stats(fname) for condition, fname in files.items()}
    normal_kurt = result["normal"]["kurtosis"]
    separations = {
        cond: result[cond]["kurtosis"] - normal_kurt
        for cond in ("inner_race", "ball", "outer_race")
    }
    return {
        "skipped": False,
        "median_features_per_condition": result,
        "kurtosis_separation_vs_normal": separations,
        # The claim the feature design actually rests on: race spalls are
        # impulsive and time-domain kurtosis must see them. Small ball defects
        # are the documented exception — rolling-element impulses are smeared
        # by slip and need envelope analysis, which we do not claim to do — so
        # the ball fault is reported but not required to separate.
        "features_separate_race_faults": bool(
            separations["inner_race"] > 0.5 and separations["outer_race"] > 0.5
        ),
        "ball_fault_note": (
            f"ball-fault kurtosis separation is {separations['ball']:+.2f} — "
            "expected to be weak at 0.007\" severity; detecting rolling-element "
            "defects reliably requires envelope analysis, which these "
            "time-domain features intentionally do not attempt."
        ),
    }


def main() -> None:
    rul = evaluate_rul()
    clf = evaluate_classifier()
    cwru = evaluate_cwru_features()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rul": rul,
        "failure_classifier": clf,
        "cwru_feature_sanity": cwru,
    }
    out = ARTIFACTS_DIR / "metrics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("FactoryPilot PdM — evaluation report")
    print("=" * 64)
    print(f"RUL ({rul['dataset']}, {rul['n_test_engines']} held-out engines)")
    print(f"  model     RMSE {rul['model']['rmse_cycles']:6.2f}  MAE {rul['model']['mae_cycles']:6.2f}  NASA {rul['model']['nasa_score']:9.1f}")
    b = rul["baseline_predict_mean"]
    print(f"  baseline  RMSE {b['rmse_cycles']:6.2f}  MAE {b['mae_cycles']:6.2f}  NASA {b['nasa_score']:9.1f}")
    print(f"  skill vs predict-the-mean: {rul['skill_vs_baseline']:.1%}")
    print()
    print(f"Failure classifier ({clf['dataset']}, {clf['n_val_rows']} held-out rows)")
    m, bl = clf["model"], clf["baseline_predict_majority"]
    print(f"  model     acc {m['accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}  failure recall {m['failure_recall']:.3f}")
    print(f"  baseline  acc {bl['accuracy']:.3f}  macro-F1 {bl['macro_f1']:.3f}  failure recall {bl['failure_recall']:.3f}")
    print()
    if cwru.get("skipped"):
        print(f"CWRU feature sanity: SKIPPED — {cwru['reason']}")
    else:
        print("CWRU feature sanity (median kurtosis, healthy vs seeded faults)")
        for cond, feats in cwru["median_features_per_condition"].items():
            print(f"  {cond:<12} kurtosis {feats['kurtosis']:7.2f}   crest {feats['crest']:6.2f}")
        verdict = (
            "separate" if cwru["features_separate_race_faults"] else "DO NOT separate"
        )
        print(f"  -> bearing features {verdict} real race-fault signatures")
        print(f"  -> {cwru['ball_fault_note']}")
    print("=" * 64)
    print(f"Report written to {out}")

    if not cwru.get("skipped") and not cwru["features_separate_race_faults"]:
        print(
            "WARNING: bearing features failed the CWRU race-fault sanity check.",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
