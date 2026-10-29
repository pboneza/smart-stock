#!/usr/bin/env python3
"""Advance one historical inventory day, with an S3 checkpoint and retry ledger.

Run: python scripts/daily_replay.py --config config/replay.json --execution-id demo-001
Same ID = retry of the same event; new ID = request to advance one more day.
S3 conditional writes provide a single commit point for stock, orders and ledger.
No supplier orders are sent. Forecast/recommendation evidence can be duplicated
by failed attempts; only results referenced in the checkpoint are committed days.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import tempfile
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
import pandas as pd
import batch_forecast
import replenish


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def hashed(data):
    return hashlib.sha256(data).hexdigest()


def get(s3, uri):
    bucket, key = replenish.s3_location(uri)
    if not key:
        raise ValueError('Checkpoint and input URIs must identify exact objects.')
    response = s3.get_object(Bucket=bucket, Key=key)
    try:
        return response['Body'].read(), response.get('ETag')
    finally:
        response['Body'].close()


def put(s3, uri, payload, **conditions):
    bucket, key = replenish.s3_location(uri)
    return s3.put_object(Bucket=bucket, Key=key, Body=encoded(payload),
                         ContentType='application/json', **conditions)


def missing(error):
    return error.response.get('Error', {}).get('Code') in ['NoSuchKey', '404', 'NotFound']


def conflict(error):
    return error.response.get('Error', {}).get('Code') in ['PreconditionFailed', '412', 'ConditionalRequestConflict', '409']


def read_checkpoint(s3, uri):
    raw, etag = get(s3, uri)
    if not etag:
        raise ValueError('S3 checkpoint has no ETag; conditional commit cannot proceed.')
    return json.loads(raw), etag


def load_data(s3, location):
    return get(s3, location)[0] if location.startswith('s3://') else Path(location).read_bytes()


def validate_config(cfg):
    needed = ['product_id', 'model_s3', 'data', 'state_s3', 'output_s3', 'initial_inventory',
              'selected_policies', 'scenario', 'lead_time', 'pack_size', 'holding_cost_per_unit_day',
              'lost_unit_cost', 'order_cost']
    if any(key not in cfg for key in needed):
        raise ValueError('Missing configuration keys: ' + str([k for k in needed if k not in cfg]))
    if not re.fullmatch(r'[A-Za-z0-9_-]+', cfg['product_id']):
        raise ValueError('Invalid product_id.')
    for key in ['lead_time', 'pack_size']:
        if type(cfg[key]) is not int:
            raise ValueError(key + ' must be an integer.')
    for key in ['holding_cost_per_unit_day', 'lost_unit_cost', 'order_cost']:
        replenish.nonnegative(cfg[key], key)
    if cfg['state_s3'].rstrip('/') == cfg['model_s3'].rstrip('/'):
        raise ValueError('State must not overwrite the model.')


def run(cfg, execution_id, s3):
    validate_config(cfg)
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,120}', execution_id):
        raise ValueError('execution-id must be 1..120 letters/digits/underscore/dot/hyphen.')
    # Bind a replay to the configuration and exact inputs; never silently change it.
    configuration_hash = hashed(encoded(cfg))
    try:
        state, etag = read_checkpoint(s3, cfg['state_s3'])
    except ClientError as error:
        if not missing(error):
            raise
        initial = json.loads(Path(cfg['initial_inventory']).read_bytes())
        if initial.get('is_simulated') is not True:
            raise ValueError('Replay requires an explicitly simulated initial inventory snapshot.')
        state = {'schema_version': 1, 'config_sha256': configuration_hash,
                 'inventory': initial, 'committed_executions': {}, 'days_processed': 0,
                 'cumulative': {'fulfilled': 0., 'lost_sales': 0., 'total_cost': 0.}}
        try:
            put(s3, cfg['state_s3'], state, IfNoneMatch='*')
        except ClientError as race:
            if not conflict(race):
                raise
        state, etag = read_checkpoint(s3, cfg['state_s3'])
    if state['config_sha256'] != configuration_hash:
        raise ValueError('Config changed. Restore it, or use a NEW state/output prefix for a separate replay.')
    if execution_id in state['committed_executions']:
        return dict(state['committed_executions'][execution_id], event='replay_already_committed')
    inventory = state['inventory']
    if inventory.get('is_simulated') is not True:
        raise ValueError('Checkpoint inventory must be simulated.')
    review = replenish.date(inventory['review_date'], 'review_date')
    region = cfg.get('region', 'eu-north-1')
    policy = replenish.policy_from_csv(cfg['selected_policies'], cfg['scenario'], cfg['product_id'],
                                       cfg['lead_time'], cfg['pack_size'])
    # Read one model object and check that batch forecasting used those same bytes.
    model_bytes, _ = get(s3, cfg['model_s3'])
    data_bytes = load_data(s3, cfg['data'])
    inputs = {'model_sha256': hashed(model_bytes), 'data_sha256': hashed(data_bytes),
              'selected_policies_sha256': policy['selected_policies_sha256'],
              'initial_inventory_sha256': hashed(Path(cfg['initial_inventory']).read_bytes())}
    if 'inputs' in state and state['inputs'] != inputs:
        raise ValueError('Frozen replay inputs changed. Restore them or start a separate replay.')
    date_col, sales_col = cfg.get('date_col', 'forecast_date'), cfg.get('sales_col', 'units_sold')
    # Inspect date coverage, not today's quantity, before producing the decision.
    dates_frame = pd.read_csv(io.BytesIO(data_bytes), usecols=[date_col])
    dates = pd.to_datetime(dates_frame[date_col], errors='raise')
    if review > dates.max():
        return {'event': 'replay_dataset_exhausted', 'next_review_date': str(review.date()),
                'days_processed': state['days_processed'], 'state_s3': cfg['state_s3']}
    if int(dates.eq(review).sum()) != 1:
        raise ValueError('Require exactly one dataset row for review_date.')
    output = cfg['output_s3'].rstrip('/')
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid4().hex[:8]
    day_uri = f'{output}/days/product={cfg["product_id"]}/review={review.date()}/attempt={attempt}/day_result.json'
    start = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory(prefix='stock-replay-') as temp:
        temp = Path(temp)
        data_path = temp / 'daily_sales.csv'; data_path.write_bytes(data_bytes)
        inventory_path = temp / 'inventory.json'; inventory_path.write_bytes(encoded(inventory))
        forecast_args = Namespace(model_s3=cfg['model_s3'], data=str(data_path),
            origin=str((review - pd.Timedelta(days=1)).date()), product_id=cfg['product_id'],
            output_s3=output + '/forecasts', region=region, profile=None,
            local_output=cfg.get('local_output', 'results/daily_replay') + '/forecasts',
            date_col=date_col, sales_col=sales_col)
        _, forecast_folder, forecast_uri = batch_forecast.run_forecast(forecast_args, s3)
        forecast_manifest = json.loads((forecast_folder / 'run_manifest.json').read_bytes())
        if forecast_manifest['model_source']['sha256'] != inputs['model_sha256']:
            raise ValueError('Model changed during execution; no inventory commit made.')
        rec_args = Namespace(forecast_run=forecast_uri, inventory=str(inventory_path),
            selected_policies=cfg['selected_policies'], scenario=cfg['scenario'],
            product_id=cfg['product_id'], lead_time=cfg['lead_time'], pack_size=cfg['pack_size'],
            output_s3=output + '/recommendations', region=region, profile=None,
            local_output=cfg.get('local_output', 'results/daily_replay') + '/recommendations')
        rec, rec_uri = replenish.run(rec_args, s3)
        if rec['policy']['selected_policies_sha256'] != inputs['selected_policies_sha256']:
            raise ValueError('Policy changed during execution; no inventory commit made.')
        # Today's demand is revealed only AFTER the forecast and replenishment decision.
        frame = pd.read_csv(data_path)
        requested = replenish.nonnegative(frame.loc[pd.to_datetime(frame[date_col]).eq(review), sales_col].iloc[0], 'demand')
        pending = [copy.deepcopy(order) for order in inventory['outstanding_orders']
                   if replenish.date(order['arrival_date'], 'arrival_date') > review]
        quantity = rec['recommended_order_quantity']
        simulated_order = None
        if quantity:
            simulated_order = {'order_id': 'sim-' + rec['decision_key'],
                               'arrival_date': rec['proposed_arrival_date'], 'quantity': quantity}
            pending.append(simulated_order)
        available = rec['available_stock']
        fulfilled = min(available, requested)
        lost = requested - fulfilled
        closing = available - fulfilled
        holding = closing * cfg['holding_cost_per_unit_day']
        shortage = lost * cfg['lost_unit_cost']
        ordering = cfg['order_cost'] if quantity else 0.
        next_inventory = {'product_id': cfg['product_id'], 'review_date': str((review + pd.Timedelta(days=1)).date()),
                          'stage': 'opening_before_receipts', 'on_hand': closing,
                          'outstanding_orders': pending, 'is_simulated': True}
        result = {'event': 'replay_day_committed', 'execution_id': execution_id,
            'product_id': cfg['product_id'], 'review_date': str(review.date()),
            'execution_started_at_utc': start.isoformat(), 'forecast_run': forecast_uri,
            'recommendation_run': rec_uri, 'decision_key': rec['decision_key'],
            'opening_stock': inventory['on_hand'], 'receipts': rec['receipts'],
            'available_stock': available, 'pipeline_before_order': rec['pipeline_before_order'],
            'inventory_position_before_order': rec['inventory_position_before_order'],
            'forecast_protection_demand': rec['forecast_protection_demand'],
            'order_up_to_target': rec['order_up_to_target'], 'order_quantity': quantity,
            'simulated_order': simulated_order, 'demand': requested, 'fulfilled': fulfilled,
            'lost_sales': lost, 'closing_stock': closing,
            'pipeline_after_order': sum(x['quantity'] for x in pending),
            'holding_cost': holding, 'lost_sales_cost': shortage, 'ordering_cost': ordering,
            'total_cost': holding + shortage + ordering, 'next_inventory': next_inventory,
            'day_result_s3': day_uri, 'state_s3': cfg['state_s3'],
            'seconds_before_commit': (datetime.now(timezone.utc) - start).total_seconds(),
            'external_order_submitted': False, 'inputs': inputs}
        # This object is prepared evidence; it is authoritative ONLY if referenced by checkpoint.
        put(s3, day_uri, result, IfNoneMatch='*')
    new_state = copy.deepcopy(state)
    new_state.update(inventory=next_inventory, inputs=inputs, days_processed=state['days_processed'] + 1)
    for key in ['fulfilled', 'lost_sales', 'total_cost']:
        new_state['cumulative'][key] += result[key]
    new_state['committed_executions'][execution_id] = result
    new_state['last_result_s3'] = day_uri
    new_state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    try:
        put(s3, cfg['state_s3'], new_state, IfMatch=etag)
    except Exception:
        # Covers a lost success response as well as a concurrent conditional-write conflict.
        latest, _ = read_checkpoint(s3, cfg['state_s3'])
        if execution_id in latest['committed_executions']:
            return dict(latest['committed_executions'][execution_id], event='replay_already_committed')
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--execution-id', required=True, help='Stable event ID; reuse on retry, change to advance another day.')
    args = parser.parse_args()
    try:
        cfg = json.loads(Path(args.config).read_bytes())
        session = boto3.Session(region_name=cfg.get('region', 'eu-north-1'), profile_name=cfg.get('profile'))
        result = run(cfg, args.execution_id, session.client('s3'))
        print(json.dumps(result, indent=2))
    except Exception as error:
        print(json.dumps({'event': 'replay_failed', 'execution_id': args.execution_id,
                          'execution_time_utc': datetime.now(timezone.utc).isoformat(),
                          'error_type': type(error).__name__, 'error': str(error)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
