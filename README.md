# Spotter Freight Rate Prediction

This repository contains a reproducible model workflow for the Machine Learning Engineer assessment. It uses the labeled development data in `data/train_test.csv`, predicts every row in the November–December validation file, and predicts the fixed December scenario used by the supplied scorer.

## Setup and run

Use Python 3.10 or later. From the repository root:

```bash
python -m pip install -r requirements.txt
python train_model.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python make_report.py
```

`train_model.py` reports its time-based validation metrics, saves them to `reports/validation_metrics.csv`, selects a model using September and October forward holdouts, trains it on all labeled rows, and writes `validation_predictions.csv` plus the completed `data/december_chart_inputs.csv`. The scorer verifies the submission files and creates `scorer_results/candidate_december.png`. `make_report.py` assembles the validation results and that chart into `reports/freight_rate_assessment.pdf`.

To use a different training duration, pass `--iterations N` to `train_model.py`. Model selection and final training use the same iteration count. The default is 300 iterations per CatBoost candidate.

## Approach

- The split respects time: train through August and validate on September; then train through September and validate on October. October is the closest labeled proxy for the November–December prediction period.
- A rate-per-mile baseline is compared with CatBoost trained with MAE and RMSE losses. Selection averages relative MAE and RMSE against the baseline over both holdouts.
- Pickup and delivery categories, equipment, coordinates, distance, weight, market and quote signals, and calendar features are used. IDs are excluded. Negative weights are treated as missing; CatBoost handles other missing numeric values natively. High target values are retained.
- After selecting the loss, the chosen model is refit on all development rows and applied to both requested prediction sets.

## Deliverables

- `validation_predictions.csv`: exactly `load_id,predicted_rate` for the 12,000 validation loads.
- `data/december_chart_inputs.csv`: the original fixed inputs plus model predictions.
- `reports/freight_rate_assessment.pdf`: validation approach, results, data checks, model rationale, and the scorer's fixed December chart.
- `reports/loom_walkthrough.md`: a timed outline for the requested 2–3 minute recording.

The final evaluation metric is not disclosed by the supplied scorer; it checks file structure and generates the December chart. Do not interpret local holdout metrics as the final Spotter score.
