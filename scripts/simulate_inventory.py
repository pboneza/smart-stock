"""Daily lost-sales inventory simulation for one product; no model retraining.

Timing: receive -> review/order -> demand. An order on day t arrives at the
START of t+L. Daily review gives protection horizon L+1, including today.
Forecasts at review t use actual sales only through END of t-1.
Historical recorded sales are treated as exogenous demand, observed even when
simulated stock is exhausted (an explicit full-demand-information assumption).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

import direct_ridge_forecast as forecasting


def simulate(demand, forecasts, initial_stock, lead_time, safety_stock,
             pack_size=1, holding_cost=0.0, lost_unit_cost=0.0, order_cost=0.0):
    """Apply order-up-to with inventory position = on-hand + outstanding orders.

    forecasts: array with one row per demand date and L+1 columns, predicting
    today through t+L from yesterday's information. Orders are rounded UP to
    pack multiples; unfulfilled demand is lost and never backlogged.
    Initial outstanding orders are empty. Start-up effects remain in results.
    """
    values = np.asarray(forecasts, dtype=float)
    if values.shape != (len(demand), lead_time + 1):
        raise ValueError("Forecast shape must be (simulation days, lead_time+1).")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Forecasts must be finite and nonnegative.")
    on_hand, outstanding, rows, orders = float(initial_stock), [], [], []
    for position, (day, requested) in enumerate(demand.items()):
        opening = on_hand
        receipts = sum(o["quantity"] for o in outstanding if o["arrival_date"] == day)
        outstanding = [o for o in outstanding if o["arrival_date"] > day]
        on_hand += receipts
        available = on_hand
        pipeline_before = sum(o["quantity"] for o in outstanding)
        inventory_position = on_hand + pipeline_before
        expected = float(values[position].sum())
        target = expected + safety_stock
        quantity = pack_size * math.ceil(max(0.0, target - inventory_position) / pack_size)
        if quantity:
            order = {"order_date": day, "arrival_date": day + pd.Timedelta(days=lead_time),
                     "quantity": quantity}
            outstanding.append(order)
            orders.append(order)
        fulfilled = min(on_hand, float(requested))
        lost = float(requested) - fulfilled
        on_hand -= fulfilled
        # Holding cost uses end-of-day stock, not an estimate of intraday average.
        rows.append({"date": day, "opening_stock": opening, "receipts": receipts,
                     "available_stock": available, "demand": requested,
                     "fulfilled": fulfilled, "lost_sales": lost, "closing_stock": on_hand,
                     "pipeline_before_order": pipeline_before,
                     "inventory_position_before_order": inventory_position,
                     "forecast_protection_demand": expected, "order_up_to_target": target,
                     "order_quantity": quantity,
                     "pipeline_after_order": pipeline_before + quantity,
                     "holding_cost": on_hand * holding_cost,
                     "lost_sales_cost": lost * lost_unit_cost,
                     "ordering_cost": order_cost if quantity else 0.0})
    daily = pd.DataFrame(rows)
    order_table = pd.DataFrame(orders, columns=["order_date", "arrival_date", "quantity"])
    total_demand = float(daily.demand.sum())
    summary = {"days": len(daily), "initial_stock": initial_stock,
               "total_demand": total_demand, "fulfilled_units": float(daily.fulfilled.sum()),
               "lost_units": float(daily.lost_sales.sum()),
               "fill_rate_pct": 100 * float(daily.fulfilled.sum()) / total_demand if total_demand else None,
               "stockout_days": int((daily.lost_sales > 0).sum()),
               "average_closing_stock": float(daily.closing_stock.mean()),
               "final_stock": float(on_hand),
               "final_pipeline": float(sum(o["quantity"] for o in outstanding)),
               "order_count": len(orders), "units_ordered": int(daily.order_quantity.sum()),
               "holding_cost": float(daily.holding_cost.sum()),
               "lost_sales_cost": float(daily.lost_sales_cost.sum()),
               "ordering_cost": float(daily.ordering_cost.sum())}
    summary["total_cost"] = sum(summary[k] for k in ("holding_cost", "lost_sales_cost", "ordering_cost"))
    return daily, order_table, summary


def make_forecasts(models, sales, start, end, lead_time):
    """Build forecasts BEFORE demand processing; no simulated-stock feedback."""
    ridge, naive, audit = [], [], []
    for day in pd.date_range(start, end):
        origin = day - pd.Timedelta(days=1)
        prediction = forecasting.forecast(models, sales, origin).iloc[:lead_time+1]
        previous_week = sales.reindex(pd.DatetimeIndex(prediction.target_date) - pd.Timedelta(days=7))
        if previous_week.isna().any():
            raise ValueError("Insufficient history for seasonal-naive forecasts.")
        ridge.append(prediction.prediction.to_numpy())
        naive.append(previous_week.to_numpy())
        part = prediction.copy()
        part["review_date"] = day
        part["seasonal_naive"] = previous_week.to_numpy()
        audit.append(part)
    return np.array(ridge), np.array(naive), pd.concat(audit, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--start", required=True, help="First simulated demand day")
    parser.add_argument("--end", required=True, help="Last simulated demand day")
    parser.add_argument("--product-id", default="22041")
    parser.add_argument("--date-col", default="forecast_date")
    parser.add_argument("--sales-col", default="target_sales")
    parser.add_argument("--lead-time", type=int, default=3, help="Calendar days: Monday order arrives Thursday for L=3")
    parser.add_argument("--safety-stock", default="0", help="Comma-separated fixed-unit scenarios, e.g. 0,20,40")
    parser.add_argument("--initial-stock", type=float, help="Same starting stock for every scenario; default pre-start mean times L+1, rounded up")
    parser.add_argument("--pack-size", type=int, default=1)
    parser.add_argument("--holding-cost", type=float, default=0, help="Currency per closing unit per day")
    parser.add_argument("--lost-unit-cost", type=float, default=0, help="Currency penalty per unmet unit")
    parser.add_argument("--order-cost", type=float, default=0, help="Fixed currency cost per nonzero order")
    parser.add_argument("--output", default="results/inventory")
    args = parser.parse_args()
    if not 1 <= args.lead_time <= 6:
        parser.error("Lead time must be 1..6 so L+1 fits within seven forecast days.")
    buffers = sorted(set(float(v) for v in args.safety_stock.split(",")))
    nonnegative = buffers + [args.holding_cost, args.lost_unit_cost, args.order_cost]
    if args.initial_stock is not None:
        nonnegative.append(args.initial_stock)
    if any(not np.isfinite(v) or v < 0 for v in nonnegative) or args.pack_size < 1:
        parser.error("Stocks and costs must be finite/nonnegative; pack size must be positive.")
    sales = forecasting.load_sales(args.data, args.date_col, args.sales_col)
    raw = pd.read_csv(args.data, dtype={"StockCode": str})
    if "StockCode" in raw and not raw.StockCode.eq(args.product_id).all():
        parser.error("Input must contain only the specified product.")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if start not in sales.index or end not in sales.index or end < start:
        parser.error("Start/end must be observed dates in chronological order.")
    bundle = joblib.load(args.model)  # Load only a package you trust.
    metadata, models = bundle["metadata"], bundle["models"]
    if set(models) != set(range(1, 8)) or metadata["features"] != forecasting.FEATURES:
        parser.error("Model horizons/features do not match this forecasting implementation.")
    if metadata["sklearn_version"] != sklearn.__version__:
        parser.error("Use training scikit-learn version " + metadata["sklearn_version"])
    if start - pd.Timedelta(days=1) < pd.Timestamp(metadata["train_end"]):
        parser.error("Simulation must start AFTER the saved model's training cutoff.")
    before = sales.loc[sales.index < start].tail(28)
    if len(before) < 28:
        parser.error("At least 28 days of observed history before start are required.")
    initial = args.initial_stock if args.initial_stock is not None else float(math.ceil(before.mean() * (args.lead_time+1)))
    ridge, naive, audit = make_forecasts(models, sales, start, end, args.lead_time)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "forecast_inputs.csv", index=False)
    summaries = []
    for buffer in buffers:
        for policy, forecasts in (("ridge", ridge), ("seasonal_naive", naive)):
            daily, orders, summary = simulate(sales.loc[start:end], forecasts, initial,
                args.lead_time, buffer, args.pack_size, args.holding_cost, args.lost_unit_cost, args.order_cost)
            name = f"{policy}_safety_{buffer:g}"
            daily.to_csv(output / f"{name}_daily.csv", index=False)
            orders.to_csv(output / f"{name}_orders.csv", index=False)
            summaries.append({"policy": policy, "safety_stock": buffer, **summary})
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(output / "summary.csv", index=False)
    manifest = {"configuration": vars(args), "resolved_initial_stock": initial,
                "model_metadata": metadata,
                "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "forecast_code_sha256": hashlib.sha256(Path(forecasting.__file__).read_bytes()).hexdigest(),
                "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__},
                "demand_assumption": "Recorded sales proxy; full demand remains observed despite simulated stockouts",
                "timing": "Receive, order from yesterday's information, satisfy demand; arrival t+L",
                "initial_pipeline": "empty", "terminal_pipeline": "reported, not liquidated",
                "costs": "Holding on closing stock, lost-unit penalty, fixed order cost; excludes procurement and terminal value",
                "cost_comparison_enabled": any(v > 0 for v in [args.holding_cost,args.lost_unit_cost,args.order_cost])}
    (output / "simulation_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    columns = ["policy", "safety_stock", "fill_rate_pct", "lost_units", "stockout_days", "average_closing_stock", "order_count"]
    if manifest["cost_comparison_enabled"]:
        columns += ["total_cost"]
    print(summary_table[columns].to_string(index=False))
    print(f"\nSaved to {output.resolve()}")


if __name__ == "__main__":
    main()
