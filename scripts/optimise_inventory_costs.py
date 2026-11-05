"""Select safety stock by validation cost, then freeze it for held-out testing.

No model retraining. Reuses the existing comparison and inventory simulator.
Default costs are hypothetical monetary units, not estimated retailer costs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from inventory_simulator import simulate

POLICIES = ['direct_ridge', 'recursive_ridge', 'seasonal_naive']


def cost_summary(daily, orders, summary, scenario):
    holding = float(daily.closing_stock.sum()) * scenario['holding_cost']
    lost = float(daily.lost_sales.sum()) * scenario['lost_unit_cost']
    ordering = len(orders) * scenario['order_cost']
    return {**summary, 'holding_cost': holding, 'lost_sales_cost': lost,
            'ordering_cost': ordering, 'total_cost': holding + lost + ordering}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--validation-start', required=True)
    p.add_argument('--validation-end', required=True)
    p.add_argument('--test-start', required=True)
    p.add_argument('--test-end', required=True)
    p.add_argument('--date-col', default='forecast_date')
    p.add_argument('--sales-col', default='target_sales')
    p.add_argument('--lead-time', type=int, default=3)
    p.add_argument('--pack-size', type=int, default=1)
    p.add_argument('--initial-stock', type=float, default=None,
                   help='Same starting quantity for every method and both periods; otherwise each period uses its own pre-start mean')
    p.add_argument('--safety-stock', default='0,5,10,20,30,40,60,80,100,150,200')
    p.add_argument('--scenarios', help='Optional JSON list: name, holding_cost, lost_unit_cost, order_cost')
    p.add_argument('--output', default='results/cost_optimisation')
    a = p.parse_args()
    vs, ve, ts, te = map(pd.Timestamp, [a.validation_start, a.validation_end, a.test_start, a.test_end])
    if not vs <= ve < ts <= te:
        p.error('Require validation-start <= validation-end < test-start <= test-end.')
    # Identical forecast inputs across cost scenarios; costs do not affect state transitions.
    buffers = sorted(set(float(x) for x in a.safety_stock.split(',')))
    if not buffers or any(not np.isfinite(x) or x < 0 for x in buffers):
        p.error('Safety stocks must be finite and nonnegative.')
    scenarios = json.loads(Path(a.scenarios).read_text()) if a.scenarios else [
        dict(name='lower_shortage', holding_cost=0.1, lost_unit_cost=1., order_cost=0.),
        dict(name='medium_shortage', holding_cost=0.1, lost_unit_cost=5., order_cost=0.),
        dict(name='higher_shortage', holding_cost=0.1, lost_unit_cost=10., order_cost=0.)]
    if not isinstance(scenarios, list) or not scenarios:
        p.error('Scenarios must be a nonempty JSON list.')
    names = []
    for s in scenarios:
        if not isinstance(s.get('name'), str) or not s['name'].strip():
            p.error('Every scenario needs a nonempty name.')
        names.append(s['name'])
        for k in ['holding_cost', 'lost_unit_cost', 'order_cost']:
            if k not in s or not np.isfinite(s[k]) or s[k] < 0:
                p.error('Every scenario needs finite nonnegative holding_cost, lost_unit_cost, order_cost.')
    if len(names) != len(set(names)):
        p.error('Scenario names must be unique.')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name('recursive_ridge_compare.py')
    common = [sys.executable, str(runner), '--model', a.model, '--data', a.data,
              '--date-col', a.date_col, '--sales-col', a.sales_col,
              '--lead-time', str(a.lead_time), '--pack-size', str(a.pack_size)]
    if a.initial_stock is not None:
        common += ['--initial-stock', str(a.initial_stock)]
    def run_period(start, end, stocks, destination):
        subprocess.run(common + ['--start', str(start.date()), '--end', str(end.date()),
            '--safety-stock', ','.join(f'{b:g}' for b in stocks), '--output', str(destination)], check=True,
            stdout=subprocess.PIPE, text=True)
    validation_dir = out / 'validation_runs'
    run_period(vs, ve, buffers, validation_dir)
    base = pd.read_csv(validation_dir / 'summary.csv')
    candidates = []
    for scenario in scenarios:
        for row in base.to_dict('records'):
            name = f"{row['policy']}_safety_{row['safety_stock']:g}"
            daily = pd.read_csv(validation_dir / f'{name}_daily.csv')
            orders = pd.read_csv(validation_dir / f'{name}_orders.csv')
            candidates.append({**scenario, **cost_summary(daily, orders, row, scenario)})
    candidates = pd.DataFrame(candidates)
    candidates.to_csv(out / 'validation_candidates.csv', index=False)
    # Deterministic ties: lower cost, then fewer lost units, then lower buffer.
    selected = candidates.sort_values(['total_cost', 'lost_units', 'safety_stock']).groupby(
        ['name', 'policy'], sort=True).head(1).copy()
    selected['upper_grid_boundary'] = selected.safety_stock.eq(max(buffers))
    selected.to_csv(out / 'selected_policies.csv', index=False)
    if selected.upper_grid_boundary.any():
        warnings.warn('Some validation optima are at the largest tested buffer. Consider extending the validation grid before interpreting test results; do not tune from test performance.')
    # Selection is complete before any test demand is simulated.
    test_dir = out / 'test_runs'
    run_period(ts, te, sorted(selected.safety_stock.unique()), test_dir)
    test_base = pd.read_csv(test_dir / 'summary.csv')
    results = []
    for choice in selected.to_dict('records'):
        row = test_base.loc[(test_base.policy == choice['policy']) &
                            (test_base.safety_stock == choice['safety_stock'])].iloc[0].to_dict()
        name = f"{row['policy']}_safety_{row['safety_stock']:g}"
        daily = pd.read_csv(test_dir / f'{name}_daily.csv')
        orders = pd.read_csv(test_dir / f'{name}_orders.csv')
        scenario = next(s for s in scenarios if s['name'] == choice['name'])
        result = {**scenario, **cost_summary(daily, orders, row, scenario),
                  'validation_total_cost': choice['total_cost'],
                  'upper_grid_boundary': choice['upper_grid_boundary']}
        results.append(result)
    results = pd.DataFrame(results)
    naive = results.loc[results.policy.eq('seasonal_naive')].set_index('name').total_cost
    results['cost_difference_vs_naive'] = results.total_cost - results.name.map(naive)
    results['cost_reduction_vs_naive_pct'] = np.where(results.name.map(naive) > 0,
        100 * (results.name.map(naive) - results.total_cost) / results.name.map(naive), np.nan)
    results.to_csv(out / 'test_selected_results.csv', index=False)
    manifest = dict(configuration=vars(a), scenarios=scenarios,
        objective='Holding + lost-sales + fixed ordering cost; no service constraint',
        selection='Minimum validation cost per method/scenario; ties: lost units then buffer',
        no_retraining=True,
        period_initialisation='Independent resets, empty initial pipelines; startup retained; no terminal value',
        costs='Hypothetical unless independently estimated; no procurement cost or revenue',
        note='test_runs contains auxiliary method/buffer combinations for computation; only test_selected_results is the frozen-policy comparison',
        sha256={str(path.name): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                [Path(__file__), runner, Path(__file__).with_name('inventory_simulator.py'), Path(__file__).with_name('direct_ridge_forecast.py'), Path(a.model), Path(a.data)]})
    (out / 'cost_optimisation_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(results[['name','policy','safety_stock','fill_rate_pct','average_closing_stock','holding_cost',
                   'lost_sales_cost','ordering_cost','total_cost','cost_reduction_vs_naive_pct']].round(3).to_string(index=False))
    print('Saved to', out.resolve())


if __name__ == '__main__':
    main()
