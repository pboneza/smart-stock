"""Compare direct/recursive Ridge and seasonal naive under identical inventory rules.

Requires the supplied direct_ridge_forecast.py and inventory_simulator.py.
Loads a trusted existing direct_ridge.joblib; never trains or tunes models.
Recursive forecasts feed clipped predictions into lag/rolling features, then
reset to actual observed history at every new daily origin. No stock feedback.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
import direct_ridge_forecast as forecasting
from inventory_simulator import simulate


def recursive_forecast(model, sales, origin, steps):
    """One next-day model repeatedly applied, never accessing future actuals."""
    history = sales.loc[:origin].copy()
    rows = []
    for h in range(1, steps + 1):
        step_origin = history.index[-1]
        features = forecasting.historical_features(history).tail(1)
        if features.isna().any().any():
            raise ValueError('At least 28 consecutive historical days required.')
        raw = float(model.predict(forecasting.horizon_features(features, 1))[0])
        value = max(0.0, raw)
        target = step_origin + pd.Timedelta(days=1)
        rows.append(dict(origin_date=origin, target_date=target, horizon=h,
                         prediction_raw=raw, prediction=value))
        history.loc[target] = value  # Fractional expected demand; do not round.
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Trusted direct_ridge.joblib bundle')
    p.add_argument('--data', required=True)
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--date-col', default='forecast_date')
    p.add_argument('--sales-col', default='target_sales')
    p.add_argument('--lead-time', type=int, default=3)
    p.add_argument('--safety-stock', default='0,5,10,20,40')
    p.add_argument('--initial-stock', type=float)
    p.add_argument('--pack-size', type=int, default=1)
    p.add_argument('--holding-cost', type=float, default=0)
    p.add_argument('--lost-unit-cost', type=float, default=0)
    p.add_argument('--order-cost', type=float, default=0)
    p.add_argument('--output', default='results/recursive_comparison')
    a = p.parse_args()
    if not 1 <= a.lead_time <= 6 or a.pack_size < 1:
        p.error('Lead time must be 1..6; pack size must be positive.')
    buffers = sorted(set(float(v) for v in a.safety_stock.split(',')))
    numbers = buffers + [a.holding_cost, a.lost_unit_cost, a.order_cost]
    if a.initial_stock is not None:
        numbers.append(a.initial_stock)
    if any(not np.isfinite(v) or v < 0 for v in numbers):
        p.error('Stocks and costs must be finite and nonnegative.')
    sales = forecasting.load_sales(a.data, a.date_col, a.sales_col)
    bundle = joblib.load(a.model)
    models, metadata = bundle['models'], bundle['metadata']
    if metadata['features'] != forecasting.FEATURES or set(models) != set(range(1, 8)):
        p.error('Expected a compatible seven-horizon direct Ridge bundle.')
    if metadata['sklearn_version'] != sklearn.__version__:
        p.error('Use training scikit-learn version: ' + metadata['sklearn_version'])
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    if start not in sales.index or end not in sales.index or end < start:
        p.error('Start/end must be observed dates in chronological order.')
    if start <= pd.Timestamp(metadata['train_end']):
        p.error('Evaluation must begin after the saved training cutoff.')
    before = sales.loc[sales.index < start].tail(28)
    if len(before) < 28:
        p.error('28 observed days before start are required.')
    initial = a.initial_stock if a.initial_stock is not None else float(np.ceil(before.mean() * (a.lead_time + 1)))
    steps = a.lead_time + 1
    matrices = {k: [] for k in ['direct_ridge', 'recursive_ridge', 'seasonal_naive']}
    audit = []
    for day in pd.date_range(start, end):
        origin = day - pd.Timedelta(days=1)
        direct = forecasting.forecast(models, sales, origin).iloc[:steps].copy()
        recursive = recursive_forecast(models[1], sales, origin, steps)
        if not np.isclose(direct.iloc[0].prediction, recursive.iloc[0].prediction, rtol=1e-10, atol=1e-10):
            raise AssertionError('Direct and recursive horizon 1 must match.')
        naive = sales.reindex(pd.DatetimeIndex(direct.target_date) - pd.Timedelta(days=7)).to_numpy()
        if not np.isfinite(naive).all():
            raise ValueError('Insufficient seasonal-naive history.')
        for policy, batch in [('direct_ridge', direct), ('recursive_ridge', recursive)]:
            matrices[policy].append(batch.prediction.to_numpy())
            batch['policy'] = policy
            batch['review_date'] = day
            batch['actual'] = sales.reindex(pd.DatetimeIndex(batch.target_date)).to_numpy()
            batch['seasonal_naive'] = naive
            audit.append(batch)
        matrices['seasonal_naive'].append(naive)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    forecasts = pd.concat(audit, ignore_index=True)
    forecasts.to_csv(output / 'forecast_inputs.csv', index=False)
    scores = []
    for policy, part in forecasts.groupby('policy'):
        for h, group in part.groupby('horizon'):
            # Evaluate targets within the requested interval only.
            valid = group.loc[group.target_date.between(start, end) & group.actual.notna()]
            for name, column in [(policy, 'prediction'), ('seasonal_naive', 'seasonal_naive')]:
                if name == 'seasonal_naive' and policy != 'direct_ridge':
                    continue
                scores.append(dict(policy=name, scope=f'day_{h}', **forecasting.metrics(valid.actual, valid[column])))
        # Complete protection windows ending within evaluation; do not silently
        # sum partial windows or read beyond the evaluation boundary for scores.
        complete = part.groupby('origin_date').filter(lambda g: len(g) == steps and g.actual.notna().all() and g.target_date.max() <= end)
        totals = complete.groupby('origin_date')[['actual', 'prediction', 'seasonal_naive']].sum()
        if not totals.empty:
            for name, column in [(policy, 'prediction'), ('seasonal_naive', 'seasonal_naive')]:
                if name == 'seasonal_naive' and policy != 'direct_ridge':
                    continue
                scores.append(dict(policy=name, scope=f'{steps}_day_total', **forecasting.metrics(totals.actual, totals[column])))
    pd.DataFrame(scores).to_csv(output / 'forecast_metrics.csv', index=False)
    summaries = []
    for buffer in buffers:
        for policy, values in matrices.items():
            daily, orders, summary = simulate(sales.loc[start:end], np.asarray(values), initial,
                a.lead_time, buffer, a.pack_size, a.holding_cost, a.lost_unit_cost, a.order_cost)
            name = f'{policy}_safety_{buffer:g}'
            daily.to_csv(output / f'{name}_daily.csv', index=False)
            orders.to_csv(output / f'{name}_orders.csv', index=False)
            summaries.append(dict(policy=policy, safety_stock=buffer, **summary))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / 'summary.csv', index=False)
    paths = {'model': Path(a.model), 'data': Path(a.data), 'script': Path(__file__),
             'forecast_code': Path(forecasting.__file__), 'inventory_code': Path(__file__).with_name('inventory_simulator.py')}
    manifest = dict(configuration=vars(a), model_metadata=metadata, resolved_initial_stock=initial,
                    sha256={k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()},
                    runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__),
                    assumptions=['Recorded sales proxy; full demand observed despite stockouts',
                                 'Fixed lead time; receive then order then demand; lost sales, no backorders',
                                 'Initial pipeline empty; startup included; terminal stock/pipeline not valued',
                                 'Recursive history resets to actual observations daily; no retraining',
                                 'Costs exclude procurement, revenue and terminal value'])
    (output / 'simulation_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(pd.DataFrame(scores).round(3).to_string(index=False))
    print(summary[['policy', 'safety_stock', 'fill_rate_pct', 'lost_units', 'average_closing_stock', 'total_cost']].round(3).to_string(index=False))
    print('Saved to', output.resolve())


if __name__ == '__main__':
    main()
