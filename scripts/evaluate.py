#!/usr/bin/env python3
"""Evaluate frozen seven-horizon Ridge forecasts against seasonal naive.

Keep train.py beside this script. Requires numpy, pandas, scikit-learn, joblib.
Example (validation dates; use the cutoff recorded in your model metadata):
    python scripts/evaluate.py --model models/ridge_validation/direct_ridge.joblib --data data/processed/product_22041_daily_sales.csv --product-id 22041 --start 2011-05-10 --end 2011-08-24 --output results/ridge_validation

--start and --end are inclusive TARGET dates. The first origin is start minus
one day, after that day's actual sales are available. Only complete seven-day
forecast windows inside the evaluation period are scored, matching the existing
direct_ridge_forecast.py backtest. Models remain frozen; later origins use newly
observed sales. Overlapping forecasts are descriptive, not independent samples.

Default input columns are forecast_date and units_sold. For the previous feature
CSV, pass --sales-col target_sales and preserve enough pre-evaluation history.
This evaluates forecast accuracy only. Inventory cost evaluation remains in the
existing simulator workflow. Load only trusted joblib packages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

# Reuse exactly the feature definitions used by the training entry point.
import train as training


def daily_date(value: str, label: str) -> pd.Timestamp:
    """Require a timezone-naive calendar date."""
    date = pd.Timestamp(value)
    if pd.isna(date) or date.tzinfo is not None or date != date.normalize():
        raise ValueError(f"{label} must be a timezone-naive calendar date.")
    return date


def score(actual, predicted) -> dict:
    """Positive bias is overprediction; WAPE is a percentage, null at zero demand."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or not len(actual):
        raise ValueError("Metrics require equal nonempty one-dimensional arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite actual and predicted values.")
    error = predicted - actual
    return {"n": len(actual), "MAE": float(np.abs(error).mean()),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "WAPE_pct": float(100 * np.abs(error).sum() / actual.sum()) if actual.sum() else None,
            "bias_units": float(error.mean())}


def backtest(models: dict, sales: pd.Series, start: pd.Timestamp,
             end: pd.Timestamp) -> pd.DataFrame:
    """Issue all horizons from each origin's observed information set only."""
    batches = []
    first_origin = start - pd.Timedelta(days=1)
    last_origin = end - pd.Timedelta(days=7)
    for origin in pd.date_range(first_origin, last_origin, freq="D"):
        history = training.historical_features(sales.loc[:origin]).tail(1)
        if history.empty or history.isna().any().any():
            raise ValueError(f"Insufficient observed history at origin {origin.date()}.")
        rows = []
        for horizon in training.HORIZONS:
            target = origin + pd.Timedelta(days=horizon)
            raw = float(np.asarray(models[horizon].predict(
                training.horizon_features(history, horizon))).ravel()[0])
            if not np.isfinite(raw):
                raise ValueError("Model returned a non-finite prediction.")
            rows.append({
                "origin_date": origin, "target_date": target, "horizon": horizon,
                "actual": float(sales.loc[target]), "prediction_raw": raw,
                "prediction": max(0.0, raw),
                "seasonal_naive": float(sales.loc[target - pd.Timedelta(days=7)]),
            })
        batches.append(pd.DataFrame(rows))
    if not batches:
        raise ValueError("Evaluation requires at least seven target days.")
    return pd.concat(batches, ignore_index=True)


def summarise(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match the existing per-horizon, pooled, and seven-day-total summaries."""
    totals = predictions.groupby("origin_date", as_index=False)[
        ["actual", "prediction", "seasonal_naive"]].sum()
    groups = [(f"day_{h}", part) for h, part in predictions.groupby("horizon")]
    groups += [("all_daily_forecasts", predictions), ("seven_day_total", totals)]
    rows = []
    for scope, part in groups:
        for column, model in (("prediction", "direct_ridge"), ("seasonal_naive", "seasonal_naive")):
            rows.append({"scope": scope, "model": model, **score(part.actual, part[column])})
    return pd.DataFrame(rows), totals


def evaluate(model_path: Path, data_path: Path, output_dir: Path, product_id: str,
             start: str, end: str, date_col: str = "forecast_date",
             sales_col: str = "units_sold") -> dict:
    """Validate package, backtest frozen models, and write reviewable evidence."""
    start_date, end_date = daily_date(start, "Start"), daily_date(end, "End")
    if (end_date - start_date).days < 6:
        raise ValueError("Require at least seven evaluation target dates.")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or not {"models", "metadata"}.issubset(bundle):
        raise ValueError("Expected a model package with models and metadata.")
    models, metadata = bundle["models"], bundle["metadata"]
    if set(models) != set(training.HORIZONS):
        raise ValueError("Expected seven models with integer horizon keys 1 through 7.")
    if metadata.get("strategy") != "direct" or metadata.get("features") != training.FEATURES:
        raise ValueError("Model strategy or feature definitions differ from train.py.")
    if metadata.get("sklearn_version") != sklearn.__version__:
        raise ValueError(f"Use the training scikit-learn version: {metadata.get('sklearn_version')}")
    if metadata.get("product_id", product_id) != product_id:
        raise ValueError("Saved product_id differs from --product-id.")
    cutoff = daily_date(metadata["train_end"], "Training cutoff")
    if start_date <= cutoff:
        raise ValueError("Evaluation target dates must be strictly after the training cutoff.")
    # Reuse input validation, truncating at evaluation end rather than training end.
    sales = training.load_training_sales(data_path, date_col, sales_col, product_id, end_date)
    first_origin = start_date - pd.Timedelta(days=1)
    if first_origin not in sales.index or len(sales.loc[:first_origin]) < 28:
        raise ValueError("Provide at least 28 consecutive sales observations ending at the first origin.")
    # If the new training package includes its history hash, confirm evaluation
    # has the same pre-cutoff observations rather than silently revised history.
    history_verified = False
    if "training_history_sha256" in metadata:
        history = sales.loc[:cutoff].rename(metadata["sales_column"]).rename_axis(metadata["date_column"]).to_frame()
        digest = hashlib.sha256(history.to_csv(date_format="%Y-%m-%d").encode()).hexdigest()
        if digest != metadata["training_history_sha256"]:
            raise ValueError("Training history differs from the saved snapshot; use the same history and date range as training.")
        history_verified = True
    predictions = backtest(models, sales, start_date, end_date)
    metrics, totals = summarise(predictions)
    comparison = metrics.pivot(index="scope", columns="model", values="MAE")
    comparison["MAE_reduction_pct"] = np.where(
        comparison.seasonal_naive.ne(0),
        100 * (comparison.seasonal_naive - comparison.direct_ridge) / comparison.seasonal_naive,
        np.nan,
    )
    paths = [output_dir / name for name in (
        "forecast_predictions.csv", "forecast_metrics.csv", "seven_day_totals.csv",
        "mae_comparison.csv", "evaluation_manifest.json")]
    if any(p.resolve() in {model_path.resolve(), data_path.resolve()} for p in paths):
        raise ValueError("An output path would overwrite an input.")
    report = {
        "product_id": product_id, "evaluation_start": str(start_date.date()),
        "evaluation_end": str(end_date.date()), "train_end": str(cutoff.date()),
        "first_origin": str(first_origin.date()),
        "last_origin": str(predictions.origin_date.max().date()),
        "origin_count": int(predictions.origin_date.nunique()),
        "forecast_rows": len(predictions), "horizons": list(training.HORIZONS),
        "refitted": False, "training_history_verified": history_verified,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "evaluation_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "feature_code_sha256": hashlib.sha256(Path(training.__file__).read_bytes()).hexdigest(),
        "model_metadata": metadata,
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__,
                    "numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "notes": [
            "Only complete seven-day target windows within the requested interval are scored.",
            "At each origin, only actual sales on or before that origin form historical features.",
            "Negative Ridge predictions are clipped to zero before scoring; raw values are retained.",
            "Overlapping windows repeat target dates at different horizons; observations are not independent.",
            "WAPE_pct is undefined when actual totals are zero; CSV leaves those cells blank.",
            "Positive bias_units means overprediction; positive MAE_reduction_pct favours Ridge.",
            "Forecast accuracy does not establish inventory cost savings; run the separate simulator.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(paths[0], index=False, date_format="%Y-%m-%d")
    metrics.to_csv(paths[1], index=False)
    totals.to_csv(paths[2], index=False, date_format="%Y-%m-%d")
    comparison.to_csv(paths[3])
    paths[4].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--start", required=True, help="Inclusive first evaluation target date.")
    parser.add_argument("--end", required=True, help="Inclusive last evaluation target date.")
    parser.add_argument("--date-col", default="forecast_date")
    parser.add_argument("--sales-col", default="units_sold")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate(args.model, args.data, args.output, args.product_id,
                          args.start, args.end, args.date_col, args.sales_col)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    print(pd.read_csv(args.output / "forecast_metrics.csv").to_string(index=False))
    print(f"\nEvaluated {report['origin_count']} origins without retraining. Saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
