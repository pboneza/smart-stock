#!/usr/bin/env python3
"""Prepare cleaned transactions and a single-product daily sales series.

Source: 01_data_exploration.ipynb and 02_feature_engineering.ipynb,
especially the final cleaning rules in the second notebook (cells 7–10).
Feature engineering, date splits, fitting, and policy selection are later steps.

Run from the project root (requires pandas):
    python scripts/prepare_data.py --input data/raw/online_retail_II.csv \
        --output-dir data/processed --product-code 22041

Outputs:
    cleaned_transactions.csv: all products retained after cleaning.
    product_<code>_daily_sales.csv: forecast_date, units_sold.
    product_<code>_preparation_report.json: rules and sequential row counts.

No-sales days between the product's first and last retained transaction are
filled with zero, matching notebook resampling. These are recorded sales,
not observed unconstrained demand. No dates outside that range are inferred.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


NON_PRODUCT_CODES = {
    "ADJUST", "ADJUST2", "AMAZONFEE", "B", "BANK CHARGES", "C2",
    "CRUK", "D", "DOT", "M", "POST", "S",
}
KNOWN_VOIDED_SALES = (("581483", "23843"), ("541431", "23166"))
REQUIRED_COLUMNS = {"Invoice", "StockCode", "InvoiceDate", "Quantity", "Price"}


def read_transactions(path: Path) -> pd.DataFrame:
    """Read original values so duplicate detection precedes numeric conversion."""
    frame = pd.read_csv(path, dtype={"Invoice": "string", "StockCode": "string"})
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("The input file contains no transactions.")
    return frame


def clean_transactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the audited exclusions without statistical outlier trimming.

    Missing Customer ID or Description does not itself remove a sale.
    Exact duplicates use all input columns, before identifier normalisation.
    Each exclusion count applies only to rows remaining after the preceding rule.
    """
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    counts = {"input_rows": len(frame)}
    clean = frame.drop_duplicates().copy()
    counts["exact_duplicates_removed"] = len(frame) - len(clean)

    def remove(mask: pd.Series, name: str) -> None:
        nonlocal clean
        counts[name] = int(mask.sum())
        clean = clean.loc[~mask].copy()

    # Use the same case-normalised identifiers consistently for every filter.
    clean["Invoice"] = clean["Invoice"].astype("string").str.strip().str.upper()
    clean["StockCode"] = clean["StockCode"].astype("string").str.strip().str.upper()
    invalid_identifier = (
        clean["Invoice"].fillna("").eq("") | clean["StockCode"].fillna("").eq("")
    )
    remove(invalid_identifier, "invalid_identifier_rows_removed")
    # Mixed valid timestamp formats are allowed; malformed dates are audited exclusions.
    clean["InvoiceDate"] = pd.to_datetime(clean["InvoiceDate"], format="mixed", errors="coerce")
    remove(clean["InvoiceDate"].isna(), "invalid_date_rows_removed")
    for column in ("Quantity", "Price"):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    invalid_numeric = (
        clean[["Quantity", "Price"]].isna().any(axis=1)
        | clean[["Quantity", "Price"]].isin([float("inf"), -float("inf")]).any(axis=1)
    )
    remove(invalid_numeric, "invalid_numeric_rows_removed")
    codes = clean["StockCode"]
    non_product = (
        codes.isin(NON_PRODUCT_CODES)
        | codes.str.startswith("GIFT_0001_", na=False)
        | codes.str.fullmatch(r"TEST\d+", na=False)
    )
    excluded_codes = sorted(codes.loc[non_product].unique().tolist())

    remove(non_product, "non_product_rows_removed")
    remove(clean["Invoice"].str.startswith("C", na=False), "cancellation_rows_removed")
    remove(clean["Quantity"].lt(0), "other_negative_quantity_rows_removed")
    remove(clean["Quantity"].eq(0), "zero_quantity_rows_removed")
    remove(clean["Price"].le(0), "non_positive_price_rows_removed")
    voided = pd.Series(False, index=clean.index)
    for invoice, code in KNOWN_VOIDED_SALES:
        voided |= clean["Invoice"].eq(invoice) & clean["StockCode"].eq(code)
    remove(voided, "known_voided_sales_removed")

    if clean.empty:
        raise ValueError("No transactions remain after cleaning.")
    clean = clean.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)
    counts["retained_rows"] = len(clean)
    return clean, {"row_counts": counts, "excluded_non_product_codes_present": excluded_codes}


def aggregate_daily_sales(clean: pd.DataFrame, product_code: str) -> pd.DataFrame:
    """Sum positive retained quantities on a continuous daily calendar."""
    product = clean.loc[clean["StockCode"].eq(product_code)]
    if product.empty:
        raise ValueError(f"No retained sales found for product {product_code}.")
    daily = (
        product.set_index("InvoiceDate")["Quantity"]
        .sort_index().resample("D").sum().rename("units_sold")
    )
    daily.index.name = "forecast_date"
    if daily.isna().any() or daily.lt(0).any():
        raise ValueError("Daily sales must be non-negative and complete.")
    return daily.reset_index()


def prepare_data(input_path: Path, output_dir: Path, product_code: str) -> dict:
    """Run preparation; return the same audit information written to JSON."""
    product_code = product_code.strip().upper()
    if not re.fullmatch(r"[A-Z0-9_-]+", product_code):
        raise ValueError("Product code must contain letters, digits, underscores or hyphens.")
    clean, report = clean_transactions(read_transactions(input_path))
    daily = aggregate_daily_sales(clean, product_code)
    outputs = {
        "cleaned_transactions": output_dir / "cleaned_transactions.csv",
        "daily_sales": output_dir / f"product_{product_code}_daily_sales.csv",
        "report": output_dir / f"product_{product_code}_preparation_report.json",
    }
    if input_path.resolve() in [path.resolve() for path in outputs.values()]:
        raise ValueError("An output path would overwrite the raw input.")
    report.update({
        "input_path": str(input_path.resolve()),
        "product_code": product_code,
        "product_transaction_rows": int(clean["StockCode"].eq(product_code).sum()),
        "daily_rows": len(daily),
        "first_date": daily["forecast_date"].min().date().isoformat(),
        "last_date": daily["forecast_date"].max().date().isoformat(),
        "zero_sales_days": int(daily["units_sold"].eq(0).sum()),
        "total_units_sold": float(daily["units_sold"].sum()),
        "non_product_codes": sorted(NON_PRODUCT_CODES),
        "non_product_patterns": ["GIFT_0001_*", r"TEST\d+ (full match)"],
        "known_voided_sales": [list(pair) for pair in KNOWN_VOIDED_SALES],
        "assumptions": [
            "Observed sales are a proxy for demand; stockouts are not observed.",
            "Days without retained sales within the product date range have zero sales.",
            "Missing customer identifiers and descriptions are retained.",
            "No quantile-based price or quantity outlier trimming is applied.",
        ],
        "implementation_notes": [
            "Identifiers are stripped and uppercased consistently (a notebook case-handling correction).",
            "Invalid identifiers, dates and numeric values are excluded with sequential audit counts.",
        ],
        "pandas_version": pd.__version__,
        "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(outputs["cleaned_transactions"], index=False)
    daily.to_csv(outputs["daily_sales"], index=False, date_format="%Y-%m-%d")
    outputs["report"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Raw Online Retail II CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--product-code", default="22041")
    args = parser.parse_args()
    try:
        report = prepare_data(args.input, args.output_dir, args.product_code)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Data preparation failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
