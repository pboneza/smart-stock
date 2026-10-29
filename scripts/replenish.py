#!/usr/bin/env python3
"""Create one order-up-to recommendation from a completed batch forecast on S3.

Timing matches inventory_simulator.py: receive -> review/order -> demand.
Snapshot: opening stock BEFORE today's receipts and demand, plus ALL outstanding
confirmed orders. Forecast origin is yesterday's end-of-day. Protection=L+1.

This script does NOT submit orders, modify inventory, process today's demand,
or tune safety stock. Repeated recommendations are not additional orders.
A later approval/execution layer must enforce one committed decision per product
and review date, and include confirmed orders in subsequent inventory snapshots.

Requires boto3 and pandas. Run in the existing EC2 virtual environment.
See README.md for example inputs and the complete command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
from uuid import uuid4

import boto3
from botocore.exceptions import BotoCoreError, ClientError
import pandas as pd


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def s3_location(uri: str) -> tuple[str, str]:
    parts = urlsplit(uri)
    if parts.scheme != 's3' or not parts.netloc or parts.query or parts.fragment:
        raise ValueError('Use s3://bucket/prefix without a query or fragment.')
    return parts.netloc, parts.path.lstrip('/').rstrip('/')


def get_bytes(s3, bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    try:
        return response['Body'].read()
    finally:
        response['Body'].close()


def date(value, label):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError(f'{label} must be YYYY-MM-DD.')
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError(f'{label} is invalid.')
    return result


def nonnegative(value, label):
    if isinstance(value, bool):
        raise ValueError(f'{label} must be numeric, not boolean.')
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'{label} must be finite and nonnegative.')
    return result


def policy_from_csv(path, scenario, product_id, lead_time, pack_size):
    """Read only the explicitly selected direct_ridge safety-stock value.

    The optimisation CSV's holding_cost column is a total-cost result, not a
    per-unit policy parameter; it is deliberately not used for ordering.
    """
    if not 1 <= lead_time <= 6 or pack_size < 1:
        raise ValueError('Require lead-time 1..6 and positive integer pack-size.')
    raw = Path(path).read_bytes()
    import io
    frame = pd.read_csv(io.BytesIO(raw), dtype={'policy': str, 'name': str, 'StockCode': str})
    if not {'name', 'policy', 'safety_stock'}.issubset(frame.columns):
        raise ValueError('Selected-policy CSV needs name, policy, safety_stock.')
    if 'StockCode' in frame and not frame.StockCode.eq(product_id).all():
        raise ValueError('Policy CSV StockCode differs from the requested product.')
    chosen = frame.loc[frame['name'].eq(scenario) & frame.policy.eq('direct_ridge')]
    if len(chosen) != 1:
        raise ValueError('Require exactly one direct_ridge row for the chosen scenario.')
    return {'product_id': product_id, 'strategy': 'direct_ridge', 'scenario': scenario,
            'safety_stock': nonnegative(chosen.iloc[0].safety_stock, 'safety_stock'),
            'lead_time': lead_time, 'review_interval_days': 1, 'pack_size': pack_size,
            'selected_policies_sha256': digest(raw)}


def recommend(forecasts, manifest, snapshot, policy):
    """Mirror simulator's review equation, using an explicit before-receipts state."""
    product = policy['product_id']
    if snapshot.get('product_id') != product:
        raise ValueError('Inventory snapshot product_id must match the policy.')
    if snapshot.get('stage') != 'opening_before_receipts':
        raise ValueError('Inventory stage must be opening_before_receipts; do not double-count receipts.')
    if not isinstance(snapshot.get('is_simulated'), bool):
        raise ValueError('Inventory snapshot must declare is_simulated as true or false.')
    review = date(snapshot['review_date'], 'review_date')
    origin = review - pd.Timedelta(days=1)
    if manifest.get('product_id') != product or manifest.get('origin_date') != str(origin.date()):
        raise ValueError('Use the forecast run issued at the end of the day immediately before review_date.')
    required = {'product_id', 'origin_date', 'target_date', 'horizon', 'prediction', 'run_id', 'model_version'}
    if not required.issubset(forecasts.columns) or len(forecasts) != 7:
        raise ValueError('Require one complete seven-row batch_forecast.py output.')
    if not forecasts.product_id.astype(str).eq(product).all():
        raise ValueError('Forecast rows contain another product.')
    if not forecasts.origin_date.eq(str(origin.date())).all():
        raise ValueError('Forecast rows contain mismatched origins.')
    if not forecasts.run_id.eq(manifest['run_id']).all() or not forecasts.model_version.eq(manifest['model_version']).all():
        raise ValueError('Forecast run/model identifiers differ from manifest.')
    horizons = pd.to_numeric(forecasts.horizon, errors='raise')
    if sorted(horizons.tolist()) != list(range(1, 8)):
        raise ValueError('Require unique integer horizons 1..7.')
    ordered = forecasts.assign(horizon=horizons).sort_values('horizon')
    expected_dates = [str((origin + pd.Timedelta(days=h)).date()) for h in range(1, 8)]
    if ordered.target_date.tolist() != expected_dates:
        raise ValueError('Forecast target dates do not align with origin/horizon.')
    prediction = pd.to_numeric(ordered.prediction, errors='raise')
    if not all(math.isfinite(float(x)) and x >= 0 for x in prediction):
        raise ValueError('Forecast predictions must be finite and nonnegative.')
    opening = nonnegative(snapshot['on_hand'], 'on_hand')
    orders = snapshot['outstanding_orders']
    if not isinstance(orders, list):
        raise ValueError('outstanding_orders must be a list (empty only when no orders are pending).')
    ids, receipts, remaining = set(), 0.0, []
    for order in orders:
        order_id = order.get('order_id')
        if not isinstance(order_id, str) or not order_id.strip() or order_id in ids:
            raise ValueError('Outstanding orders require distinct nonempty order_id values.')
        ids.add(order_id)
        arrival = date(order['arrival_date'], 'arrival_date')
        quantity = nonnegative(order['quantity'], 'outstanding order quantity')
        if quantity <= 0:
            raise ValueError('Outstanding order quantity must be positive.')
        if arrival < review:
            raise ValueError('Overdue outstanding order: resolve its receipt/status before recommendation.')
        if arrival == review:
            receipts += quantity
        else:
            remaining.append({'order_id': order_id, 'arrival_date': str(arrival.date()), 'quantity': quantity})
    available = opening + receipts
    pipeline = sum(x['quantity'] for x in remaining)
    inventory_position = available + pipeline
    steps = policy['lead_time'] + 1
    expected = float(prediction.iloc[:steps].sum())
    target = expected + policy['safety_stock']
    order_quantity = policy['pack_size'] * math.ceil(max(0.0, target - inventory_position) / policy['pack_size'])
    decision_key = f'{product}:{review.date()}'
    recommendation = {
        'product_id': product, 'review_date': str(review.date()), 'decision_key': decision_key,
        'forecast_origin': str(origin.date()), 'forecast_run_id': manifest['run_id'],
        'model_version': manifest['model_version'], 'policy': policy,
        'inventory_is_simulated': snapshot['is_simulated'],
        'opening_stock': opening, 'receipts': receipts, 'available_stock': available,
        'pipeline_before_order': pipeline, 'inventory_position_before_order': inventory_position,
        'protection_days': steps, 'protection_end_date': str((review + pd.Timedelta(days=policy['lead_time'])).date()),
        'forecast_protection_demand': expected, 'order_up_to_target': target,
        'recommended_order_quantity': order_quantity,
        'proposed_order_date': str(review.date()),
        'proposed_arrival_date': str((review + pd.Timedelta(days=policy['lead_time'])).date()) if order_quantity else None,
        'order_submitted': False, 'inventory_updated': False,
        'status': 'recommendation_only',
        'notes': [
            'Receipts are applied before review; today demand is not processed here.',
            'All outstanding quantities count in inventory position, matching the simulator.',
            'A recommendation is not an order: do not add it to pipeline until confirmed.',
            'Enforce one committed decision per decision_key in the later execution layer.',
        ],
    }
    return recommendation, {'product_id': product, 'review_date': str(review.date()),
                            'stage': 'after_receipts_before_order_and_demand',
                            'on_hand': available, 'outstanding_orders': remaining,
                            'is_simulated': snapshot['is_simulated']}


def run(args, s3=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.product_id):
        raise ValueError("Product ID must contain letters, digits, underscores or hyphens.")
    if s3 is None:
        s3 = boto3.Session(region_name=args.region, profile_name=args.profile).client('s3')
    bucket, prefix = s3_location(args.forecast_run)
    if not prefix:
        raise ValueError('--forecast-run must be the exact completed run folder.')
    marker = json.loads(get_bytes(s3, bucket, prefix + '/_SUCCESS.json'))
    manifest_bytes = get_bytes(s3, bucket, prefix + '/run_manifest.json')
    if marker.get('status') != 'completed' or marker.get('manifest_sha256') != digest(manifest_bytes):
        raise ValueError('Forecast completion marker/manifest integrity check failed.')
    manifest = json.loads(manifest_bytes)
    if marker.get('run_id') != manifest.get('run_id'):
        raise ValueError('Completion marker and manifest run IDs differ.')
    forecast_bytes = get_bytes(s3, bucket, prefix + '/forecasts.csv')
    if manifest['file_sha256']['forecasts.csv'] != digest(forecast_bytes):
        raise ValueError('Forecast CSV checksum differs from the completed run manifest.')
    import io
    forecasts = pd.read_csv(io.BytesIO(forecast_bytes), dtype={'product_id': str})
    snapshot_bytes = Path(args.inventory).read_bytes()
    snapshot = json.loads(snapshot_bytes)
    policy = policy_from_csv(args.selected_policies, args.scenario, args.product_id, args.lead_time, args.pack_size)
    recommendation, review_state = recommend(forecasts, manifest, snapshot, policy)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid4().hex[:8]
    relative = f"product={args.product_id}/review={recommendation['review_date']}/run={run_id}"
    destination = Path(args.local_output) / relative
    destination.mkdir(parents=True, exist_ok=False)
    output_bucket, output_prefix = s3_location(args.output_s3)
    output_prefix = '/'.join(filter(None, [output_prefix, relative]))
    recommendation['recommendation_id'] = digest(json.dumps({
        'decision_key': recommendation['decision_key'], 'forecast_sha256': digest(forecast_bytes),
        'inventory_sha256': digest(snapshot_bytes), 'policy': policy,
    }, sort_keys=True).encode())[:24]
    evidence = {'run_id': run_id, 'forecast_run': args.forecast_run,
                'forecast_manifest_sha256': digest(manifest_bytes),
                'inventory_snapshot_sha256': digest(snapshot_bytes), 'policy': policy,
                'script_sha256': digest(Path(__file__).read_bytes()),
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'simulator_timing': 'receive -> order -> demand; arrival=review+lead_time',
                'inventory_state_mutated': False, 'order_submitted': False}
    files = {'recommendation.json': recommendation, 'inventory_review_state.json': review_state,
             'inventory_input.json': snapshot, 'run_manifest.json': evidence}
    for name, payload in files.items():
        path = destination / name
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        s3.upload_file(str(path), output_bucket, f'{output_prefix}/{name}')
    success = {'status': 'completed', 'run_id': run_id,
               'recommendation_id': recommendation['recommendation_id'],
               'manifest_sha256': digest((destination / 'run_manifest.json').read_bytes())}
    raw = (json.dumps(success, indent=2) + '\n').encode()
    s3.put_object(Bucket=output_bucket, Key=f'{output_prefix}/_SUCCESS.json', Body=raw, ContentType='application/json')
    (destination / '_SUCCESS.json').write_bytes(raw)
    return recommendation, f's3://{output_bucket}/{output_prefix}/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forecast-run', required=True, help='Exact S3 run folder printed by batch_forecast.py.')
    parser.add_argument('--inventory', required=True, help='Local before-receipts inventory snapshot JSON.')
    parser.add_argument('--selected-policies', required=True, help='Local frozen selected_policies.csv from validation.')
    parser.add_argument('--scenario', required=True, help='Explicit cost scenario name in selected_policies.csv.')
    parser.add_argument('--product-id', default='22041')
    parser.add_argument('--lead-time', type=int, default=3)
    parser.add_argument('--pack-size', type=int, default=1)
    parser.add_argument('--output-s3', required=True)
    parser.add_argument('--local-output', default='results/replenishment')
    parser.add_argument('--region', default='eu-north-1')
    parser.add_argument('--profile')
    args = parser.parse_args()
    try:
        recommendation, uri = run(args)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, BotoCoreError, ClientError) as error:
        print(json.dumps({'event': 'replenishment_failed', 'error': str(error)}), file=sys.stderr)
        sys.exit(1)
    print(json.dumps(recommendation, indent=2))
    print(json.dumps({'event': 'replenishment_completed', 'output_s3': uri}))


if __name__ == '__main__':
    main()
