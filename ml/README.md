# FactoryPilot AI — Predictive Maintenance ML

Training pipeline and artifacts for the PdM engine served by
`app/services/pdm.py`.

## The one rule

**The models are never trained on simulator output.** The simulator's
degradation curves are constants we wrote; a model trained on them would just
read our own assumptions back to us with decimal places attached, and its RUL
numbers would be meaningless. Training data is real run-to-failure and
real labelled-failure data. The simulator is *runtime input only* — it feeds
readings through the same pipeline real hardware would.

## Quick start

```bash
python ml/download_datasets.py          # fetch + verify all three datasets
python ml/train_rul.py                  # RUL regressor      -> ml/artifacts/
python ml/train_failure_classifier.py   # failure classifier -> ml/artifacts/
python ml/evaluate.py                   # metrics.json + baseline comparison
```

If a dataset source is unreachable the download script fails loudly and prints
manual-download instructions. It never substitutes generated data.

## Datasets

| Dataset | What it is | What it trains |
|---|---|---|
| **NASA C-MAPSS FD001** | 100 turbofan engines run to failure in simulation-of-record, with ground-truth RUL. 21 sensor channels. | The RUL regressor (`rul_model.joblib`). |
| **UCI AI4I 2020** | 10,000 milling-process snapshots, 3.4% failures, labelled with five failure modes. | The failure-mode classifier (`failure_classifier.joblib`). |
| **CWRU bearing data** | 12 kHz accelerometer captures of healthy and seeded-fault bearings. | Nothing — it *validates* that our bearing features (kurtosis, crest factor) separate real fault signatures. We check the feature design against physics without fitting to it. |

## The domain transfer, stated honestly

C-MAPSS is turbofan data. AI4I is milling-machine data. We apply both to
industrial rotating equipment — conveyors, compressors, presses. That transfer
is real and it has limits; here is exactly what carries and what does not.

**What makes it defensible:** the models never see raw magnitudes. Every
channel is normalised to *deviation from the asset's own healthy self, in
units of its healthy variability* (z-units) before features are computed:

- C-MAPSS training: baseline = each engine's first 15 cycles.
- AI4I training: baseline = the healthy (non-failure) population statistics.
- Serving: baseline = each sensor's configured normal band
  (midpoint, width/4), the engineering definition of healthy.

In that currency, "temperature is drifting 3σ above healthy and accelerating"
means the same thing on a turbofan and a gearbox. What the RUL model learns —
and all it is claimed to learn — is the *shape of degradation trajectories*:
how deviation growth across correlated channels maps to remaining life
fraction. The `cycle_hours = 10` constant that converts C-MAPSS cycles to
machine-hours is a unit convention, not a physical claim; treat relative RUL
(shrinking vs stable) as the trustworthy signal and absolute hours as
calibration-dependent until validated against your own failure history.

**Channel mapping:**

| Our role | C-MAPSS (RUL model) | AI4I (classifier) | Our fleet |
|---|---|---|---|
| temperature | s4 / T50, LPT outlet temp | Process temperature (K; serving converts °C) | motor/bearing/oil temp |
| pressure | s11 / Ps30, HPC static pressure | — absent | pneumatic/hydraulic pressure |
| rpm | s9 / Nc, core speed | Rotational speed | roller/spindle/motor rpm |
| power | s12 / phi, fuel-flow ratio (power proxy) | Torque × angular velocity (kW) | drive power draw |
| vibration | — absent | — absent | accelerometers |

**Known limits — read these before trusting a number:**

1. **Vibration is invisible to both models.** Neither training set has a
   vibration channel, so the models carry a zero block with `present=0` for
   it. Vibration still drives the *trend* computation (signal-level, not
   model-level), and the CWRU check validates those features — but an RUL
   estimate cannot shorten because vibration alone is rising. This is the
   single biggest gap; closing it needs run-to-failure vibration data
   (e.g. IMS/FEMTO bearing datasets) in a future iteration.
2. **HEAT_DISSIPATION recall is poor (~0.14) by construction.** In AI4I that
   mode is defined by the air-minus-process temperature *difference*, and our
   plants have no ambient sensor — so the discriminative feature cannot exist
   at serve time and was excluded from training rather than trained-on and
   silently absent later. POWER (F1 0.96) and OVERSTRAIN (F1 0.70) transfer
   well; TOOL_WEAR is weak in the source data itself (failures are
   near-random within the high-wear band).
3. **Out-of-distribution inputs are clipped**, not extrapolated: serving
   clamps every feature into the p1–p99 envelope seen in training, so an
   operating point far outside anything trained-on produces a bounded, stated
   answer instead of tree-model saturation nonsense.
4. **The ball-bearing caveat:** CWRU validation shows kurtosis/crest separate
   race faults strongly (+2.7, +4.9 kurtosis vs healthy) but small
   rolling-element faults barely (+0.2). That matches the literature —
   detecting ball defects reliably needs envelope analysis, which this
   feature set does not attempt.

## Train/serve parity

`ml/preprocess.py` is the single source of feature truth. The training scripts
and `app/services/pdm.py` import the same `build_feature_vector`; the schema
(names, order, window) is frozen into the artifacts at training time and
`FeatureSchema.assert_parity` runs on **every inference**, raising
`FeatureParityError` on any drift in feature content or order. A permuted
feature vector is the classic silent serving bug — here it is a hard error.

## Artifacts (`ml/artifacts/`)

| File | Contents |
|---|---|
| `rul_model.joblib` | XGBRegressor, RUL in capped cycles |
| `rul_schema.json` / `clf_schema.json` | frozen feature schemas |
| `rul_metrics.json` | held-out metrics, cycle_hours, RUL cap, feature envelope |
| `failure_classifier.joblib` | XGBClassifier, 6 classes |
| `clf_metrics.json` | per-class P/R/F1, confusion matrix, healthy stats |
| `*_feature_importances.json` | per-feature importance (feeds `contributing_features`) |
| `*_val_predictions.csv` | held-out predictions (feed `evaluate.py` baselines) |
| `metrics.json` | the full evaluation report with baseline comparisons |

If artifacts are missing the API starts, logs the exact commands above, and
every `/pdm/*` endpoint returns 503 with the same message. There is no
heuristic fallback.

## Honest numbers (this training run)

- RUL, official FD001 test set (100 engines): **RMSE 15.4 cycles, MAE 11.8,
  NASA score 377** — vs predict-the-mean baseline RMSE 40.5 (skill +62%).
- Classifier, held-out 2,498 rows: **macro-F1 0.59 vs 0.20 baseline;
  binary failure recall 0.63 vs 0.00** at a 1.9% false-alarm rate.
- Every number above comes from held-out engines/rows, never training data.
