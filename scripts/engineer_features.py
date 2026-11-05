#!/usr/bin/env python3
"""Build notebook-compatible feature tables from prepare_data.py daily sales.

Requires pandas. Run from the project root:
    python scripts/engineer_features.py --input data/processed/product_22041_daily_sales.csv --output-dir data/processed --product-code 22041

Both feature tables reproduce cells 11–12 of 02_feature_engineering.ipynb.
The notebook's v1 CSV includes optional columns, although its initial predictor
list contains only ten features. This script preserves that distinction in its
JSON report. Downstream training must select predictors explicitly.

Each row predicts sales ON forecast_date using sales strictly BEFORE that date.
This is the one-day modelling table; multi-horizon forecasting must retain its
own origin/target alignment. No model fitting, scaling or date splitting occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd

INITIAL_FEATURES = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_mean_7", "sales_std_7", "sales_mean_28", "sales_std_28",
    "forecast_weekday", "forecast_month",
]
ADDITIONAL_FEATURE_GROUPS = {
    "zero_sales": ["zero_sales_days_28"],
    "change": ["sales_trend_7_vs_28"],
    "high_sales": ["high_sales_days_28", "high_sales_lag_1"],
}
EXTENDED_FEATURES = INITIAL_FEATURES + [
    feature for group in ADDITIONAL_FEATURE_GROUPS.values() for feature in group
]
BASELINE_COLUMNS = {
    "previous_day": "sales_lag_1",
    "same_weekday_last_week": "sales_lag_7",
    "recent_average": "sales_mean_7",
}


def load_daily_sales(path: Path, product_code: str) -> pd.Series:
    """Validate a single-product daily calendar; never silently fill missing dates.

    prepare_data.py already inserts zero-sales days. A gap here signals an
    incomplete input. Unsorted complete input is sorted before feature creation.
    If StockCode is supplied, it must identify the requested product on every row.
    """
    frame = pd.read_csv(path, dtype={"StockCode": "string"})
    missing = {"forecast_date", "units_sold"}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Daily sales input is empty.")
    if "StockCode" in frame:
        codes = frame["StockCode"].str.strip().str.upper()
        if codes.isna().any() or not codes.eq(product_code).all():
            raise ValueError("Every StockCode must match --product-code.")
    dates = pd.to_datetime(frame["forecast_date"], errors="raise")
    if dates.isna().any():
        raise ValueError("Missing forecast dates.")
    if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Dates must be timezone-naive daily dates at midnight.")
    if dates.duplicated().any():
        raise ValueError("Duplicate forecast dates found.")
    values = pd.to_numeric(frame["units_sold"], errors="coerce")
    if values.isna().any() or values.isin([float("inf"), -float("inf")]).any() or values.lt(0).any():
        raise ValueError("units_sold must be finite, non-negative and complete.")
    sales = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name="units_sold").sort_index()
    sales.index.name = "forecast_date"
    if not sales.index.equals(pd.date_range(sales.index.min(), sales.index.max(), freq="D", name="forecast_date")):
        raise ValueError("Missing daily dates. Use the complete daily output from prepare_data.py.")
    if len(sales) < 29:
        raise ValueError("At least 29 daily observations are needed for a target with 28 days of history.")
    return sales


def build_features(sales: pd.Series, high_sales_threshold: float = 50) -> pd.DataFrame:
    """Construct notebook features; all sales-derived predictors exclude the target.

    Standard deviations use pandas' sample convention (ddof=1).
    Calendar features refer to the target date and are known in advance.
    The high-sales threshold is fixed, never estimated from the full dataset.
    """
    if not math.isfinite(high_sales_threshold) or high_sales_threshold <= 0:
        raise ValueError("The high-sales threshold must be finite and positive.")
    features = sales.to_frame(name="target_sales")
    for lag in (1, 7, 14, 28):
        features[f"sales_lag_{lag}"] = sales.shift(lag)
    past = sales.shift(1)
    for window in (7, 28):
        rolling = past.rolling(window, min_periods=window)
        features[f"sales_mean_{window}"] = rolling.mean()
        features[f"sales_std_{window}"] = rolling.std(ddof=1)
    features["forecast_weekday"] = sales.index.dayofweek
    features["forecast_month"] = sales.index.month
    past_zero = past.eq(0).astype(float).where(past.notna())
    features["zero_sales_days_28"] = past_zero.rolling(28, min_periods=28).sum()
    features["sales_trend_7_vs_28"] = features.sales_mean_7 - features.sales_mean_28
    past_high = past.ge(high_sales_threshold).astype(float).where(past.notna())
    features["high_sales_days_28"] = past_high.rolling(28, min_periods=28).sum()
    features["high_sales_lag_1"] = past_high
    return features


def engineer_features(input_path: Path, output_dir: Path, product_code: str = "22041",
                      high_sales_threshold: float = 50) -> dict:
    """Save both notebook feature versions and their explicit predictor contracts."""
    product_code = product_code.strip().upper()
    if not re.fullmatch(r"[A-Z0-9_-]+", product_code):
        raise ValueError("Product code must contain letters, digits, underscores or hyphens.")
    sales = load_daily_sales(input_path, product_code)
    features = build_features(sales, high_sales_threshold)
    tables = {}
    for version, predictors in (("v1", INITIAL_FEATURES), ("v2", EXTENDED_FEATURES)):
        # Preserve optional columns in v1 to reproduce the notebook's saved CSV.
        table = features.dropna(subset=["target_sales"] + predictors).copy()
        table.insert(0, "StockCode", product_code)
        if table.empty or not table.index.is_unique or not table.index.is_monotonic_increasing:
            raise ValueError(f"Invalid or empty {version} modelling table.")
        if table[EXTENDED_FEATURES + ["target_sales"]].isna().any().any():
            raise ValueError(f"Missing values remain in {version}.")
        tables[version] = table
    paths = {v: output_dir / f"product_{product_code}_features_{v}.csv" for v in tables}
    report_path = output_dir / f"product_{product_code}_feature_report.json"
    if input_path.resolve() in [p.resolve() for p in [*paths.values(), report_path]]:
        raise ValueError("An output would overwrite the input file.")
    report = {
        "product_code": product_code,
        "input_path": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "input_daily_rows": len(sales),
        "high_sales_threshold": high_sales_threshold,
        "baseline_columns": BASELINE_COLUMNS,
        "additional_feature_groups": ADDITIONAL_FEATURE_GROUPS,
        "pandas_version": pd.__version__,
        "alignment": "Target is sales on forecast_date; historical predictors use dates strictly before forecast_date.",
        "notes": [
            "First 28 days are excluded from feature tables; original daily input retains them.",
            "v1 and v2 contain the same columns and values, matching this notebook; predictor lists differ.",
            "StockCode and target_sales are not predictors.",
            "No scaling, fitting, policy calibration or chronological splitting is performed.",
            "Multi-horizon training must apply its own horizon alignment; use full daily history where needed.",
        ],
        "versions": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for version, table in tables.items():
        table.reset_index().to_csv(paths[version], index=False, date_format="%Y-%m-%d")
        report["versions"][version] = {
            "output_path": str(paths[version].resolve()),
            "rows": len(table), "warmup_rows_removed": len(sales) - len(table),
            "first_date": table.index.min().date().isoformat(),
            "last_date": table.index.max().date().isoformat(),
            "predictors": INITIAL_FEATURES if version == "v1" else EXTENDED_FEATURES,
            "stored_columns": ["forecast_date"] + table.columns.tolist(),
        }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Daily sales CSV from prepare_data.py.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--product-code", default="22041")
    parser.add_argument("--high-sales-threshold", type=float, default=50, help="Fixed optional-feature threshold, matching notebook default 50.")
    args = parser.parse_args()
    try:
        report = engineer_features(args.input, args.output_dir, args.product_code, args.high_sales_threshold)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Feature engineering failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
