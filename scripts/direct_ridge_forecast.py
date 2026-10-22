"""Review-stage direct forecasting pipeline for Smart Stock (one product).

Install: pip install numpy pandas scikit-learn joblib
Example (replace the cutoff with your actual training/validation boundary):
  python direct_ridge_forecast.py --data product_22041_features_v1.csv \
      --train-end YYYY-MM-DD --eval-end YYYY-MM-DD --alpha 1.0 --output results

Input: forecast_date,target_sales, one row per calendar day, one product only.
Other columns are ignored: historical features are rebuilt to control timing.
Prefer the complete daily sales series, before dropping feature warm-up rows.
If using the existing feature CSV, its first 28 days become a new warm-up.

Forecast origin = END of day t, when actual sales through t are available.
Seven separate models predict y[t+1], ..., y[t+7]. Every horizon uses the
same observed history through t; no prediction is fed into another model.
Forecasts are refreshed DAILY, with model parameters frozen in evaluation.
This is a daily rolling seven-day horizon, not a fixed weekly ordering cycle.

Use validation dates for reviewing this new strategy. Keep the final test
period untouched until strategy/alpha decisions are complete. Alpha=1 is an
explicit starting value, not a claim to reproduce the previously tuned model.
"""

from __future__ import annotations

import argparse
import json
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


def load_sales(path: str, date_col: str, sales_col: str) -> pd.Series:
    """Reject missing days rather than silently treating unknown demand as zero."""
    frame = pd.read_csv(path)
    dates = pd.to_datetime(frame[date_col], errors="raise")
    if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Dates must be timezone-naive calendar dates at midnight.")
    values = pd.to_numeric(frame[sales_col], errors="raise")
    sales = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(dates),
                      name="sales").sort_index()
    if sales.empty or sales.index.has_duplicates:
        raise ValueError("Supply nonempty daily data for exactly one product; no duplicate dates.")
    if not np.isfinite(sales.to_numpy()).all() or (sales < 0).any():
        raise ValueError("Daily sales must be finite and nonnegative.")
    if not sales.index.equals(pd.date_range(sales.index.min(), sales.index.max(), freq="D")):
        raise ValueError("Missing calendar days. Resolve them explicitly before modelling.")
    return sales


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


def forecast(models: dict, sales: pd.Series, origin: pd.Timestamp) -> pd.DataFrame:
    """Issue all seven forecasts from one information set; works beyond data end."""
    if origin not in sales.index:
        raise ValueError("Origin must have an observed daily sales value.")
    # Truncate BEFORE feature construction to enforce the information boundary.
    history = historical_features(sales.loc[:origin]).tail(1)
    if history.isna().any().any():
        raise ValueError("At least 28 consecutive observed days are required.")
    rows = []
    for h in HORIZONS:
        raw = float(models[h].predict(horizon_features(history, h))[0])
        # Demand cannot be negative. Retain raw values for review; evaluate the
        # clipped forecasts that would actually enter the inventory procedure.
        rows.append({"origin_date": origin, "target_date": origin + pd.Timedelta(days=h),
                     "horizon": h, "prediction_raw": raw, "prediction": max(0.0, raw)})
    return pd.DataFrame(rows)


def backtest(models: dict, sales: pd.Series, train_end: pd.Timestamp,
             eval_end: pd.Timestamp) -> pd.DataFrame:
    """Daily origins; score only complete seven-day windows inside evaluation.

    First origin is train_end, so first target is train_end+1. Each subsequent
    origin incorporates newly observed actual sales, without retraining models.
    Overlapping windows repeat target dates at different horizons; results are
    descriptive and should not be treated as independent statistical samples.
    """
    last_origin = eval_end - pd.Timedelta(days=7)
    if last_origin < train_end:
        raise ValueError("Evaluation requires at least seven days after train_end.")
    batches = []
    for origin in pd.date_range(train_end, last_origin, freq="D"):
        batch = forecast(models, sales, origin)
        batch["actual"] = sales.reindex(pd.DatetimeIndex(batch.target_date)).to_numpy()
        # For horizons 1..7 the same weekday last week is already observed.
        baseline_dates = pd.DatetimeIndex(batch.target_date) - pd.Timedelta(days=7)
        batch["seasonal_naive"] = sales.reindex(baseline_dates).to_numpy()
        batches.append(batch)
    return pd.concat(batches, ignore_index=True)


def metrics(actual, predicted) -> dict:
    """Positive bias means overprediction; WAPE is a percentage (undefined at zero)."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    error = predicted - actual
    return {"n": len(error), "MAE": float(np.abs(error).mean()),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "WAPE_pct": float(100 * np.abs(error).sum() / actual.sum()) if actual.sum() else None,
            "bias_units": float(error.mean())}


def summarise(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report per-horizon, pooled daily, and seven-day-total performance."""
    totals = predictions.groupby("origin_date", as_index=False)[
        ["actual", "prediction", "seasonal_naive"]].sum()
    rows = []
    groups = [(f"day_{h}", part) for h, part in predictions.groupby("horizon")]
    groups += [("all_daily_forecasts", predictions), ("seven_day_total", totals)]
    for label, part in groups:
        for column in ("prediction", "seasonal_naive"):
            rows.append({"scope": label, "model": "direct_ridge" if column == "prediction"
                         else column, **metrics(part.actual, part[column])})
    return pd.DataFrame(rows), totals


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--date-col", default="forecast_date")
    parser.add_argument("--sales-col", default="target_sales")
    parser.add_argument("--train-end", required=True, help="Last observed training day, YYYY-MM-DD")
    parser.add_argument("--eval-end", required=True, help="Last evaluation target day, YYYY-MM-DD")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output", default="direct_ridge_results")
    args = parser.parse_args()
    if not np.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be finite and positive.")
    sales = load_sales(args.data, args.date_col, args.sales_col)
    train_end, eval_end = pd.Timestamp(args.train_end), pd.Timestamp(args.eval_end)
    if train_end not in sales.index or eval_end not in sales.index or train_end >= eval_end:
        parser.error("Cutoffs must be observed dates with train_end < eval_end.")
    models, audit = fit_models(sales, train_end, args.alpha)
    predictions = backtest(models, sales, train_end, eval_end)
    scores, totals = summarise(predictions)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    totals.to_csv(output / "validation_seven_day_totals.csv", index=False)
    scores.to_csv(output / "validation_metrics.csv", index=False)
    audit.to_csv(output / "training_audit.csv", index=False)
    metadata = {"strategy": "direct", "horizons": list(HORIZONS), "alpha": args.alpha,
                "train_end": str(train_end.date()), "eval_end": str(eval_end.date()),
                "forecast_origin": "end_of_day", "update_frequency": "daily",
                "retrain_during_evaluation": False, "clip_negative_predictions": True,
                "features": FEATURES, "sklearn_version": sklearn.__version__,
                "status": "review_candidate_not_final_production_model"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Deliberately preserve the training-cutoff models. No automatic refit on
    # validation or test data; approve the strategy before deciding final refit.
    joblib.dump({"models": models, "metadata": metadata}, output / "direct_ridge.joblib")
    print(scores.to_string(index=False))
    print(f"\nSaved review outputs to {output.resolve()}")


if __name__ == "__main__":
    main()
