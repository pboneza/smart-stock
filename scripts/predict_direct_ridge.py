"""Generate seven direct forecasts from an existing model package; never fit.

Keep direct_ridge_forecast.py beside this file to reuse the EXACT feature logic
used in training. Only load joblib packages you created or otherwise trust.
Origin is the END of the specified day, after its actual sales are available.
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

import direct_ridge_forecast as training


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_saved(model_path: str, data_path: str, origin_date: str,
                  product_id: str, output_dir: str,
                  date_col: str = "forecast_date", sales_col: str = "target_sales"):
    """Load frozen models, restrict observed history, predict and record inputs.

    Repeating the same inputs in the same environment gives the same forecasts.
    The output directory is scoped by model version, product and origin. Repeating
    that combination replaces its outputs; use a separate root to retain revisions.
    No actual values for target dates are required, even for future forecasts.
    """
    model_path, data_path = Path(model_path), Path(data_path)
    bundle = joblib.load(model_path)
    models, metadata = bundle["models"], bundle["metadata"]
    if set(models) != set(range(1, 8)):
        raise ValueError("Expected seven saved models with integer keys 1 through 7.")
    if metadata["sklearn_version"] != sklearn.__version__:
        raise ValueError("Use the scikit-learn version that trained this package: "
                         + metadata["sklearn_version"])
    if metadata["features"] != training.FEATURES:
        raise ValueError("Saved feature configuration differs from this feature code.")

    origin = pd.Timestamp(origin_date)
    train_end = pd.Timestamp(metadata["train_end"])
    if origin < train_end:
        raise ValueError("Origin precedes training cutoff; forecasts would use future training data.")
    sales = training.load_sales(str(data_path), date_col, sales_col)
    observed = sales.loc[:origin]
    if observed.empty or observed.index[-1] != origin:
        raise ValueError("The origin date must have an observed actual sales value.")

    # One-product input is a contract: if StockCode exists, enforce that contract.
    frame = pd.read_csv(data_path, dtype={"StockCode": str})
    if "StockCode" in frame:
        codes = frame["StockCode"].dropna().unique()
        if len(codes) != 1 or codes[0] != product_id or frame["StockCode"].isna().any():
            raise ValueError("StockCode must match --product-id for every input row.")

    model_hash = sha256(model_path)
    model_version = "ridge-" + model_hash[:16]
    prediction = training.forecast(models, observed, origin)
    prediction.insert(0, "product_id", product_id)
    prediction["model_version"] = model_version
    # Do not round forecasts: these are expected quantities. Round order
    # quantities later in the inventory policy, according to pack-size rules.
    destination = Path(output_dir) / model_version / product_id / str(origin.date())
    destination.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(destination / "forecasts.csv", index=False, float_format="%.12g")

    # Snapshot the actual feature row: target calendar values differ by horizon.
    history = training.historical_features(observed).tail(1)
    feature_rows = []
    for h in training.HORIZONS:
        row = training.horizon_features(history, h).reset_index(drop=True)
        row.insert(0, "horizon", h)
        feature_rows.append(row)
    pd.concat(feature_rows, ignore_index=True).to_csv(destination / "features.csv", index=False)
    # Persist the 28 actual values used, so later changes to the CSV do not
    # prevent reconstruction of this run's input features.
    used_history = observed.tail(28).rename("observed_sales").rename_axis("date")
    used_history.to_csv(destination / "observed_history.csv")
    manifest = {
        "product_id": product_id, "origin_date": str(origin.date()),
        "origin_timing": "end_of_day_after_actual_sales_are_available",
        "model_version": model_version, "model_sha256": model_hash,
        "input_file_sha256": sha256(data_path),
        "observed_history_sha256": sha256(destination / "observed_history.csv"),
        "feature_code_sha256": sha256(Path(training.__file__)),
        "prediction_code_sha256": sha256(Path(__file__)),
        "training_metadata": metadata, "horizons": list(training.HORIZONS),
        "seven_day_forecast_total": float(prediction.prediction.sum()),
        "refitted": False, "date_column": date_col, "sales_column": sales_col,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                    "joblib": joblib.__version__},
    }
    (destination / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return prediction, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Existing direct_ridge.joblib")
    parser.add_argument("--data", required=True, help="One product's observed daily sales CSV")
    parser.add_argument("--origin", required=True, help="Last observed day, YYYY-MM-DD")
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--output", default="results/predictions")
    parser.add_argument("--date-col", default="forecast_date")
    parser.add_argument("--sales-col", default="target_sales")
    args = parser.parse_args()
    # Restrict IDs to safe directory names on Windows and Linux.
    if not args.product_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.product_id):
        parser.error("Product ID must contain only letters, numbers, underscores or hyphens.")
    predictions, destination = predict_saved(
        args.model, args.data, args.origin, args.product_id,
        args.output, args.date_col, args.sales_col)
    print(predictions.to_string(index=False))
    print(f"\nSeven-day total: {predictions.prediction.sum():.2f}")
    print(f"Saved to: {destination.resolve()}")


if __name__ == "__main__":
    main()
