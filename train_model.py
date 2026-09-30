"""Validate and train a reproducible freight-rate prediction model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FEATURES = [
    "pickup", "delivery", "equipment", "distance", "weight",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "market_index", "quote_signal", "geo_distance", "distance_to_geo_ratio",
    "latitude_delta", "longitude_delta", "year", "month", "day",
    "day_of_week", "day_of_year", "date_ordinal", "month_sin", "month_cos",
]
CATEGORICAL = ["pickup", "delivery", "equipment"]
TARGET = "posted_rate"


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create model inputs without using IDs or target-derived information."""
    result = frame.copy()
    dates = pd.to_datetime(result["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("date contains unparseable values")

    # A negative load weight is physically invalid. Treat it as missing so the
    # tree model can learn a missing-value branch instead of a false quantity.
    result["weight"] = pd.to_numeric(result["weight"], errors="coerce")
    result.loc[result["weight"] < 0, "weight"] = np.nan
    for column in ("market_index", "quote_signal", "distance", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")

    lat1 = np.radians(result["pickup_lat"])
    lat2 = np.radians(result["delivery_lat"])
    dlat = lat2 - lat1
    dlon = np.radians(result["delivery_lon"] - result["pickup_lon"])
    hav = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    result["geo_distance"] = 3958.7613 * 2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
    result["distance_to_geo_ratio"] = result["distance"] / result["geo_distance"].replace(0, np.nan)
    result["latitude_delta"] = result["delivery_lat"] - result["pickup_lat"]
    result["longitude_delta"] = result["delivery_lon"] - result["pickup_lon"]
    result["year"] = dates.dt.year
    result["month"] = dates.dt.month
    result["day"] = dates.dt.day
    result["day_of_week"] = dates.dt.dayofweek
    result["day_of_year"] = dates.dt.dayofyear
    result["date_ordinal"] = dates.map(pd.Timestamp.toordinal)
    result["month_sin"] = np.sin(2 * np.pi * (result["month"] - 1) / 12)
    result["month_cos"] = np.cos(2 * np.pi * (result["month"] - 1) / 12)
    for column in CATEGORICAL:
        result[column] = result[column].fillna("Unknown").astype(str)
    return result[FEATURES]


def new_model(loss: str, iterations: int = 300) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function=loss,
        iterations=iterations,
        depth=6,
        learning_rate=0.045,
        l2_leaf_reg=5.0,
        random_seed=20260930,
        verbose=False,
        allow_writing_files=False,
        # A modest fixed count avoids oversubscribing laptop/CI environments.
        thread_count=4,
    )


def rate_per_mile_baseline(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    rates = train.assign(rate_per_mile=train[TARGET] / train["distance"].clip(lower=1))
    global_rate = float(rates["rate_per_mile"].median())
    by_equipment = rates.groupby("equipment")["rate_per_mile"].median()
    per_mile = valid["equipment"].map(by_equipment).fillna(global_rate)
    return np.maximum(1.0, per_mile.to_numpy() * valid["distance"].to_numpy())


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error": float(np.median(np.abs(error))),
    }


def validate_models(data: pd.DataFrame, output: Path, iterations: int) -> pd.DataFrame:
    """Compare baseline and CatBoost losses on September and October holdouts."""
    dates = pd.to_datetime(data["date"])
    folds = [("September", "2025-08-31", "2025-09-01", "2025-09-30"),
             ("October", "2025-09-30", "2025-10-01", "2025-10-31")]
    rows: list[dict[str, object]] = []
    for fold, train_through, valid_from, valid_through in folds:
        train_mask = dates <= train_through
        valid_mask = (dates >= valid_from) & (dates <= valid_through)
        train, valid = data.loc[train_mask].copy(), data.loc[valid_mask].copy()
        if train.empty or valid.empty:
            raise ValueError(f"empty chronological fold: {fold}")
        actual = valid[TARGET].to_numpy(dtype=float)
        base = rate_per_mile_baseline(train, valid)
        rows.append({"fold": fold, "model": "equipment median rate-per-mile", "train_rows": len(train), "validation_rows": len(valid), **metrics(actual, base)})
        train_x = engineer_features(train)
        valid_x = engineer_features(valid)
        for loss in ("MAE", "RMSE"):
            model = new_model(loss, iterations)
            model.fit(train_x, train[TARGET], cat_features=CATEGORICAL)
            pred = np.maximum(1.0, model.predict(valid_x))
            rows.append({"fold": fold, "model": f"CatBoost {loss}", "train_rows": len(train), "validation_rows": len(valid), **metrics(actual, pred)})
            print(f"{fold:10s} CatBoost {loss:4s}: MAE {rows[-1]['mae']:.2f}, RMSE {rows[-1]['rmse']:.2f}", flush=True)
        print(f"{fold:10s} baseline: MAE {rows[-3]['mae']:.2f}, RMSE {rows[-3]['rmse']:.2f}", flush=True)

    result = pd.DataFrame(rows)
    # Equal weight to the two forward folds and to MAE/RMSE after normalizing
    # by each metric's fold-level baseline; the candidate must beat the baseline
    # in both error views rather than winning on one large scale alone.
    baseline = result[result.model == "equipment median rate-per-mile"].set_index("fold")
    for metric in ("mae", "rmse"):
        result[f"relative_{metric}"] = result.apply(lambda r: r[metric] / baseline.loc[r.fold, metric], axis=1)
    result["selection_score"] = (result["relative_mae"] + result["relative_rmse"]) / 2
    result.to_csv(output, index=False)
    return result


def choose_loss(results: pd.DataFrame) -> str:
    candidates = results[results.model.isin(["CatBoost MAE", "CatBoost RMSE"])]
    averages = candidates.groupby("model")["selection_score"].mean()
    return str(averages.idxmin()).replace("CatBoost ", "")


def predict_files(model: CatBoostRegressor, data: pd.DataFrame, template_path: Path, validation_path: Path, december_path: Path, output_path: Path) -> None:
    template = pd.read_csv(template_path, dtype={"load_id": str})
    validation = pd.read_csv(validation_path, dtype={"load_id": str})
    if validation["load_id"].duplicated().any() or template["load_id"].duplicated().any():
        raise ValueError("load_id values must be unique")
    if set(validation.load_id) != set(template.load_id):
        raise ValueError("validation load IDs do not exactly match the prediction template")
    prediction = pd.Series(np.maximum(1.0, model.predict(engineer_features(validation))), index=validation.load_id)
    template["predicted_rate"] = template.load_id.map(prediction)
    if template.predicted_rate.isna().any() or not np.isfinite(template.predicted_rate).all():
        raise ValueError("validation predictions contain missing or non-finite values")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template[["load_id", "predicted_rate"]].to_csv(output_path, index=False, float_format="%.2f")

    december = pd.read_csv(december_path)
    december["predicted_rate"] = np.maximum(1.0, model.predict(engineer_features(december)))
    december.to_csv(december_path, index=False, float_format="%.2f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "validation_predictions.csv")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--reuse-validation-metrics", action="store_true", help="Reuse reports/validation_metrics.csv instead of rerunning the time-split benchmark")
    args = parser.parse_args()
    training_path = args.data_dir / "train_test.csv"
    train = pd.read_csv(training_path)
    required = set(FEATURES) | {TARGET, "date", "load_id"}
    missing_columns = required - set(train.columns) - {"route", "geo_distance", "distance_to_geo_ratio", "latitude_delta", "longitude_delta", "year", "month", "day", "day_of_week", "day_of_year", "date_ordinal", "month_sin", "month_cos"}
    if missing_columns:
        raise ValueError(f"training data is missing columns: {sorted(missing_columns)}")
    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
    if train[TARGET].isna().any() or (train[TARGET] <= 0).any():
        raise ValueError("posted_rate must contain finite positive labels")
    if (train.distance <= 0).any():
        raise ValueError("distance must be positive")
    print(f"Development rows: {len(train):,}; date range: {train.date.min().date()} to {train.date.max().date()}")
    print(f"Missing weight: {train.weight.isna().sum():,}; negative weight: {(train.weight < 0).sum():,}; missing market_index: {train.market_index.isna().sum():,}; largest rate: ${train[TARGET].max():,.2f}")

    metrics_path = ROOT / "reports" / "validation_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if args.reuse_validation_metrics and metrics_path.is_file():
        results = pd.read_csv(metrics_path)
        print(f"Reusing existing holdout metrics from {metrics_path}")
    else:
        results = validate_models(train, metrics_path, args.iterations)
    loss = choose_loss(results)
    print(f"Selected model by mean normalized MAE/RMSE over September and October: CatBoost {loss}")
    final = new_model(loss, args.iterations)
    final.fit(engineer_features(train), train[TARGET], cat_features=CATEGORICAL)
    predict_files(final, train, args.data_dir / "validation_predictions_template.csv", args.data_dir / "validation.csv", args.data_dir / "december_chart_inputs.csv", args.output)
    print(f"Saved {len(pd.read_csv(args.output)):,} validation predictions to {args.output}")
    print(f"Saved December predictions to {args.data_dir / 'december_chart_inputs.csv'}")


if __name__ == "__main__":
    main()
