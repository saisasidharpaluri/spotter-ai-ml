# Spotter Freight Rate Prediction

This repository contains a reproducible solution for the Spotter freight-rate prediction assessment. It trains on `data/train_test.csv`, predicts all 12,000 rows in `data/validation.csv`, and predicts the supplied fixed December scenario.

## Setup and run

Use Python 3.10 or later. From the repository root:

```bash
python -m pip install -r requirements.txt
python train_model.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python make_report.py
```

`train_model.py` reports its time-based validation metrics, saves them to `reports/validation_metrics.csv`, selects a model using September and October forward holdouts, trains it on all labeled rows, and writes `validation_predictions.csv` plus the completed `data/december_chart_inputs.csv`. The scorer verifies the submission files and creates `scorer_results/candidate_december.png`. `make_report.py` assembles the validation results and that chart into `reports/freight_rate_assessment.pdf`.

To use a different training duration, pass `--iterations N` to `train_model.py`. Model selection and final training use the same iteration count. The default is 300 iterations per CatBoost candidate. `--reuse-validation-metrics` skips the benchmark and reuses `reports/validation_metrics.csv`; use it only when those metrics were produced with the same model settings.

## Approach

- The split respects time: train through August and validate on September; then train through September and validate on October. October is the closest labeled proxy for the November–December prediction period.
- A rate-per-mile baseline is compared with CatBoost trained with MAE and RMSE losses. Selection averages relative MAE and RMSE against the baseline over both holdouts.
- Pickup and delivery categories, equipment, coordinates, distance, weight, market and quote signals, and calendar features are used. IDs are excluded. Negative weights are treated as missing; CatBoost handles other missing numeric values natively. High target values are retained.
- After selecting the loss, the chosen model is refit on all development rows and applied to both requested prediction sets.

## Validation results

The selected model was CatBoost with MAE loss. CatBoost models trained with MAE and RMSE losses were compared with an equipment-level median rate-per-mile baseline using equal-weighted relative MAE and RMSE across two forward-in-time holdouts:

| Holdout | Model | MAE | RMSE | Baseline MAE | Baseline RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| September 2025 | CatBoost / MAE | $113.44 | $619.96 | $227.32 | $657.07 |
| October 2025 | CatBoost / MAE | $109.71 | $646.49 | $231.30 | $683.87 |

These are local holdout estimates; the assessment's final evaluation metric is hidden.

## Data checks

The development file has 48,000 rows from January through October 2025. It contains 300 missing weights, 292 negative weight values, and 374 missing market-index values. Negative weights are treated as missing, and numeric missing values are handled natively by CatBoost. High posted-rate values are retained. Eight location names appear in validation but not in development; geographic coordinates and route geometry help the model handle unfamiliar locations. The December chart input does not include coordinates or market signals, so those optional numeric features are treated as missing for that scenario.

## Deliverables

- `validation_predictions.csv`: exactly `load_id,predicted_rate` for the 12,000 validation loads.
- `data/december_chart_inputs.csv`: the original fixed inputs plus model predictions.
- `reports/freight_rate_assessment.pdf`: validation approach, results, data checks, model rationale, and the scorer's fixed December chart.
- `reports/loom_walkthrough.md`: a timed outline for the requested 2–3 minute recording.

The final evaluation metric is not disclosed by the supplied scorer; it checks file structure and generates the December chart. Do not interpret local holdout metrics as the final Spotter score.
