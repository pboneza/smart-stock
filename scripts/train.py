#!/usr/bin/env python3
"""Train seven direct Ridge models for the existing inventory forecast workflow.

Requires: numpy, pandas, scikit-learn, joblib.
Example, from the project root (replace cutoff with your approved boundary):
    python scripts/train.py --data data/processed/product_22041_daily_sales.csv --product-id 22041 --train-end 2011-05-09 --alpha 0.01 --output models/ridge_validation

Default input columns: forecast_date, units_sold (prepare_data.py output).
For an existing feature CSV, supply --sales-col target_sales. Features are rebuilt
from its sales column to preserve origin alignment. This drops a NEW warm-up of
27 origin rows, so changing from the old feature CSV to full daily history changes
training samples. Compare results before replacing your validated model.

Origin t is the END of the day: observations through t predict sales on t+h.
Seven independent models cover h=1..7; targets after --train-end are excluded.
This script reproduces the core training functions from direct_ridge_forecast.py.
The one-day notebook feature table is retained for one-day experiments; it cannot
be used unchanged as a seven-horizon supervised target table.

Outputs: direct_ridge.joblib, metadata.json, training_audit.csv,
training_history.csv. The package matches predict_direct_ridge.py's interface.
No evaluation, tuning, or automatic train+validation refit occurs. Specify a later
cutoff explicitly for a final refit, after decisions are frozen. Keep joblib files
trusted; loading one can execute code.
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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HORIZONS = range(1, 8)

NUMERIC = ["sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
           "sales_mean_7", "sales_std_7", "sales_mean_28", "sales_std_28"]

CALENDAR = ["target_weekday", "target_month"]

FEATURES = NUMERIC + CALENDAR

def historical_features(sales: pd.Series) -> pd.DataFrame:
    """Features indexed by origin t and available AFTER observing day t.

    lag_1 means the latest observation for tomorrow's forecast: y[t].
    lag_7 is y[t-6], lag_28 is y[t-27]. Rolling windows include day t.
    This matches shift(1) features indexed by tomorrow in the original notebook.
    """
    frame = pd.DataFrame(index=sales.index)
    for lag in (1, 7, 14, 28):
        frame[f"sales_lag_{lag}"] = sales.shift(lag - 1)
    for window in (7, 28):
        frame[f"sales_mean_{window}"] = sales.rolling(window).mean()
        frame[f"sales_std_{window}"] = sales.rolling(window).std(ddof=1)
    return frame

def horizon_features(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add known calendar information for t+h, never future sales features."""
    frame = history.copy()
    target_dates = frame.index + pd.Timedelta(days=horizon)
    frame["target_weekday"] = target_dates.dayofweek
    frame["target_month"] = target_dates.month
    return frame[FEATURES]

def make_model(alpha: float) -> Pipeline:
    """Fit scaling and calendar encoding inside each training-only pipeline."""
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("calendar", OneHotEncoder(categories=[list(range(7)), list(range(1, 13))],
                                   handle_unknown="ignore", sparse_output=False), CALENDAR),
    ])
    return Pipeline([("preprocess", preprocess), ("ridge", Ridge(alpha=alpha))])

def fit_models(sales: pd.Series, train_end: pd.Timestamp, alpha: float) -> tuple[dict, pd.DataFrame]:
    """Fit each horizon using only labels observed by train_end.

    Horizon-specific label cutoffs are essential: a row whose origin precedes
    the cutoff can still have a target AFTER the cutoff. Such rows are excluded.
    Training sample counts differ slightly across horizons for this reason.
    """
    observed = sales.loc[:train_end]
    history = historical_features(observed)
    models, audit = {}, []
    for h in HORIZONS:
        X = horizon_features(history, h)
        y = observed.shift(-h)
        valid = X.notna().all(axis=1) & y.notna()
        if valid.sum() < 30:
            raise ValueError(f"Horizon {h}: fewer than 30 usable training examples.")
        models[h] = make_model(alpha).fit(X.loc[valid], y.loc[valid])
        origins = X.index[valid]
        audit.append({"horizon": h, "training_rows": int(valid.sum()),
                      "first_origin": origins.min(), "last_origin": origins.max(),
                      "last_training_target": origins.max() + pd.Timedelta(days=h)})
    return models, pd.DataFrame(audit)


def load_training_sales(path: Path, date_col: str, sales_col: str,
                        product_id: str, train_end: pd.Timestamp) -> pd.Series:
    """Validate identifiers and the observed training calendar, then freeze history.

    Later input rows are not used for fitting or preprocessing. Their quantities
    need not be available. Input dates must be valid to establish the cutoff.
    """
    frame = pd.read_csv(path, dtype={"StockCode": "string"})
    missing = {date_col, sales_col}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    dates = pd.to_datetime(frame[date_col], errors="raise")
    if dates.isna().any() or dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Dates must be complete timezone-naive daily dates at midnight.")
    if "StockCode" in frame:
        codes = frame["StockCode"].str.strip()
        if codes.isna().any() or not codes.eq(product_id).all():
            raise ValueError("Every StockCode must match --product-id.")
    frame = frame.loc[dates.le(train_end)].copy()
    dates = dates.loc[dates.le(train_end)]
    values = pd.to_numeric(frame[sales_col], errors="raise")
    sales = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(dates), name="sales").sort_index()
    if sales.empty or sales.index.has_duplicates:
        raise ValueError("Training requires nonempty single-product history without duplicate dates.")
    if not np.isfinite(sales.to_numpy()).all() or sales.lt(0).any():
        raise ValueError("Training sales must be finite and nonnegative.")
    if train_end not in sales.index:
        raise ValueError("--train-end must be an observed date in the input.")
    if not sales.index.equals(pd.date_range(sales.index.min(), train_end, freq="D")):
        raise ValueError("Training history has missing calendar dates.")
    return sales


def train(data_path: Path, output_dir: Path, product_id: str,
          train_end: str, alpha: float = 0.01,
          date_col: str = "forecast_date", sales_col: str = "units_sold") -> dict:
    """Fit frozen horizon-specific pipelines and save their audit and metadata."""
    if not product_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in product_id):
        raise ValueError("Product ID must contain letters, digits, underscores or hyphens.")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Alpha must be finite and positive.")
    cutoff = pd.Timestamp(train_end)
    if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize():
        raise ValueError("--train-end must be a timezone-naive calendar date.")
    sales = load_training_sales(data_path, date_col, sales_col, product_id, cutoff)
    models, audit = fit_models(sales, cutoff, alpha)
    # Each horizon may have different sample counts, but every label ends by cutoff.
    if not audit.last_training_target.le(cutoff).all():
        raise ValueError("A training label exceeds the cutoff.")
    history = sales.rename(sales_col).rename_axis(date_col).to_frame()
    metadata = {
        "strategy": "direct", "product_id": product_id,
        "horizons": list(HORIZONS), "alpha": alpha,
        "train_end": cutoff.date().isoformat(),
        "training_history_start": sales.index.min().date().isoformat(),
        "observed_training_days": len(sales),
        "forecast_origin": "end_of_day", "update_frequency": "daily",
        "retrain_during_evaluation": False, "clip_negative_predictions": True,
        "features": FEATURES, "sklearn_version": sklearn.__version__,
        "status": "trained_candidate_requires_separate_evaluation",
        "input_path": str(data_path.resolve()), "date_column": date_col,
        "sales_column": sales_col,
        "input_file_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "training_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "joblib": joblib.__version__},
        "notes": [
            "Historical feature definitions exactly preserve the existing direct training module.",
            "Calendar encoding uses fixed weekday/month categories; scalers fit on training rows only.",
            "Historical lag_1 at origin t is sales on t, available at end of day.",
            "No metrics or model selection are performed in this script.",
            "Using full daily history instead of a feature CSV can change the validated training sample.",
        ],
    }
    destinations = [output_dir / f for f in (
        "direct_ridge.joblib", "metadata.json", "training_audit.csv", "training_history.csv")]
    if data_path.resolve() in [p.resolve() for p in destinations]:
        raise ValueError("An output path would overwrite the input.")
    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_dir / "training_history.csv", date_format="%Y-%m-%d")
    metadata["training_history_sha256"] = hashlib.sha256((output_dir / "training_history.csv").read_bytes()).hexdigest()
    joblib.dump({"models": models, "metadata": metadata}, output_dir / "direct_ridge.joblib")
    audit.to_csv(output_dir / "training_audit.csv", index=False, date_format="%Y-%m-%d")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--train-end", required=True, help="Inclusive last training target date, YYYY-MM-DD.")
    parser.add_argument("--alpha", type=float, default=0.01, help="Previously selected alpha; no parameter search.")
    parser.add_argument("--date-col", default="forecast_date")
    parser.add_argument("--sales-col", default="units_sold")
    parser.add_argument("--output", type=Path, required=True, help="Use a separate directory for each cutoff or experiment.")
    args = parser.parse_args()
    try:
        metadata = train(args.data, args.output, args.product_id, args.train_end,
                         args.alpha, args.date_col, args.sales_col)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Training failed: {error}\n")
    print(json.dumps(metadata, indent=2))
    print(f"Saved seven Ridge models to {args.output.resolve() / 'direct_ridge.joblib'}")


if __name__ == "__main__":
    main()
