# Loom walkthrough outline (about 2 minutes 30 seconds)

## 0:00–0:25 — Problem and data

“This project predicts posted freight rates for 12,000 loads. The labeled file has 48,000 loads dated January through October 2025. The final validation rows are in November and December, so I used chronological validation rather than a random split.”

## 0:25–0:55 — Findings and data quality

“Distance has the strongest simple relationship with price, and the data also includes route coordinates, equipment, market and quote signals, and date. I found missing weight and market-index values and some negative weights. I kept those rows, treated negative weights as missing, and used CatBoost’s native handling for missing numeric values. I kept the high-rate target tail so validation reflects the provided labels. Some validation locations are new, so the model also uses coordinates and route geometry.”

## 0:55–1:25 — Model and validation

“I compared an equipment-level median rate-per-mile baseline with CatBoost models trained using MAE and RMSE losses. I trained on January through August to validate on September, then through September to validate on October. The selection score averages MAE and RMSE relative to the baseline across both holdouts. October is the most relevant local check because it is nearest to the November and December prediction period.”

## 1:25–2:05 — Code walkthrough

Show `train_model.py`: point out `engineer_features`, `validate_models`, `choose_loss`, and the final refit. Mention `load_id` is used only to align the output template, not as a model feature. Then show the exact two-column `validation_predictions.csv` and the completed December inputs.

## 2:05–2:30 — Results and submission

Show the generated PDF and December chart. Mention the supplied scorer validates file structure and creates the chart, while Spotter calculates the hidden final evaluation score after submission. Close by noting the results are reproducible from the README commands.
