#!/usr/bin/env python3
"""Monitor the existing replay without modifying its configuration or commit logic.

Modes: probe (telemetry only), run (one replay event), watchdog (state freshness).
Keep beside daily_replay.py and its existing dependencies. Use monitoring.json
separately from the frozen replay.json. CloudWatch data are operational telemetry;
S3 checkpoints remain authoritative accounting records. Metrics can be delivered
at least once on ambiguous AWS responses; do not sum them as an order ledger.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from uuid import uuid4
import io

import boto3
from botocore.exceptions import ClientError
import pandas as pd
import daily_replay


def dims(cfg, mon, horizon=None):
    result = [{'Name': 'ProductId', 'Value': cfg['product_id']},
              {'Name': 'Workflow', 'Value': mon['workflow_id']}]
    if horizon is not None:
        result.append({'Name': 'Horizon', 'Value': str(horizon)})
    return result


class Reporter:
    def __init__(self, session, cfg, mon):
        self.cfg, self.mon = cfg, mon
        self.logs = session.client('logs')
        self.cw = session.client('cloudwatch')
        self.stream = datetime.now(timezone.utc).strftime('%Y/%m/%d/') + uuid4().hex
        self.created = False

    def emit(self, event, metrics=()):
        """Try log and metrics independently; propagate telemetry problems to caller."""
        now = datetime.now(timezone.utc)
        event = dict(event, observed_at_utc=now.isoformat(),
                     product_id=self.cfg['product_id'], workflow=self.mon['workflow_id'])
        print(json.dumps(event, default=str), flush=True)
        errors = []
        try:
            if not self.created:
                self.logs.create_log_stream(logGroupName=self.mon['log_group'], logStreamName=self.stream)
                self.created = True
            response = self.logs.put_log_events(logGroupName=self.mon['log_group'], logStreamName=self.stream,
                logEvents=[{'timestamp': int(now.timestamp() * 1000), 'message': json.dumps(event, default=str)}])
            if response.get('rejectedLogEventsInfo'):
                raise RuntimeError(str(response['rejectedLogEventsInfo']))
        except Exception as error:
            errors.append('logs: ' + str(error))
        try:
            if metrics:
                data = [{'MetricName': name, 'Value': float(value), 'Unit': unit,
                         'Timestamp': now, 'Dimensions': dims(self.cfg, self.mon, horizon)}
                        for name, value, unit, horizon in metrics]
                self.cw.put_metric_data(Namespace=self.mon['namespace'], MetricData=data)
        except Exception as error:
            errors.append('metrics: ' + str(error))
        if errors:
            raise RuntimeError('; '.join(errors))


def forecast_errors(result, state, s3):
    """Score only forecasts whose target is the just-revealed historical day.

    Each committed origin contributes at most one forecast for this target. No
    future actuals are read. Target-day demand comes from the committed replay.
    """
    target = pd.Timestamp(result['review_date'])
    points = []
    for committed in state['committed_executions'].values():
        origin = pd.Timestamp(committed['review_date']) - pd.Timedelta(days=1)
        horizon = (target - origin).days
        if not 1 <= horizon <= 7:
            continue
        prefix = committed['forecast_run'].rstrip('/')
        raw, _ = daily_replay.get(s3, prefix + '/forecasts.csv')
        manifest_raw, _ = daily_replay.get(s3, prefix + '/run_manifest.json')
        marker_raw, _ = daily_replay.get(s3, prefix + '/_SUCCESS.json')
        manifest, marker = json.loads(manifest_raw), json.loads(marker_raw)
        if marker.get('manifest_sha256') != daily_replay.hashed(manifest_raw) or manifest['file_sha256']['forecasts.csv'] != daily_replay.hashed(raw):
            raise ValueError('Forecast evidence integrity check failed during scoring.')
        frame = pd.read_csv(io.BytesIO(raw), dtype={'product_id': str})
        selected = frame.loc[frame.target_date.eq(str(target.date())) & frame.horizon.eq(horizon)]
        if len(selected) != 1 or selected.iloc[0].product_id != result['product_id']:
            raise ValueError('Forecast evidence target/product mismatch.')
        bias = float(selected.iloc[0].prediction) - float(result['demand'])
        points.extend([('ForecastAbsoluteError', abs(bias), 'Count', horizon),
                       ('ForecastBias', bias, 'Count', horizon)])
    return points


def execute(cfg, mon, mode, execution_id, session):
    reporter = Reporter(session, cfg, mon)
    s3 = session.client('s3')
    if mode == 'probe':
        reporter.emit({'event': 'monitoring_probe', 'inventory_changed': False},
                      [('MonitoringProbe', 1, 'Count', None)])
        return 0
    if mode == 'watchdog':
        state, _ = daily_replay.read_checkpoint(s3, cfg['state_s3'])
        updated = state.get('updated_at_utc')
        age = max(0., (datetime.now(timezone.utc) - datetime.fromisoformat(updated)).total_seconds()) if updated else 31 * 3600.
        reporter.emit({'event': 'replay_watchdog', 'seconds_since_last_commit': age,
                       'next_review_date': state['inventory']['review_date'],
                       'days_processed': state['days_processed']},
                      [('SecondsSinceLastCommit', age, 'Seconds', None),
                       ('WatchdogHeartbeat', 1, 'Count', None)])
        return 0
    if not execution_id:
        raise ValueError('--execution-id is required in run mode.')
    # Observability must not block an otherwise valid inventory transaction.
    telemetry_errors = []
    try:
        reporter.emit({'event': 'replay_started', 'execution_id': execution_id})
    except Exception as error:
        telemetry_errors.append(str(error))
    started = time.monotonic()
    try:
        result = daily_replay.run(cfg, execution_id, s3)
    except Exception as error:
        try:
            reporter.emit({'event': 'replay_failed', 'execution_id': execution_id,
                           'error_type': type(error).__name__, 'error': str(error)},
                          [('RunFailure', 1, 'Count', None),
                           ('RunDurationSeconds', time.monotonic() - started, 'Seconds', None)])
        except Exception as monitoring_error:
            print(json.dumps({'event': 'telemetry_failed', 'error': str(monitoring_error)}), file=sys.stderr)
        raise
    event = result['event']
    metrics = [('RunDurationSeconds', time.monotonic() - started, 'Seconds', None)]
    if event == 'replay_day_committed':
        metrics.append(('RunSuccess', 1, 'Count', None))
        metrics += [(name, result[key], unit, None) for name, key, unit in [
            ('OpeningStock', 'opening_stock', 'Count'), ('ClosingStock', 'closing_stock', 'Count'),
            ('DemandUnits', 'demand', 'Count'), ('FulfilledUnits', 'fulfilled', 'Count'),
            ('LostSalesUnits', 'lost_sales', 'Count'), ('OrderQuantity', 'order_quantity', 'Count'),
            ('PipelineUnits', 'pipeline_after_order', 'Count'), ('SimulatedTotalCost', 'total_cost', 'None')]]
        if result['demand'] > 0:
            metrics.append(('DailyFillRate', 100 * result['fulfilled'] / result['demand'], 'Percent', None))
    elif event == 'replay_already_committed':
        metrics.append(('RunRetryNoOp', 1, 'Count', None))
    else:
        metrics.append(('DatasetExhausted', 1, 'Count', None))
    try:
        reporter.emit({'event': event, 'execution_id': execution_id,
                       'review_date': result.get('review_date'), 'day_result_s3': result.get('day_result_s3'),
                       'order_quantity': result.get('order_quantity'), 'lost_sales': result.get('lost_sales'),
                       'total_cost': result.get('total_cost'), 'inventory_commit_unchanged_by_monitoring': True}, metrics)
    except Exception as error:
        telemetry_errors.append(str(error))
    if event == 'replay_day_committed':
        try:
            state, _ = daily_replay.read_checkpoint(s3, cfg['state_s3'])
            reporter.emit({'event': 'forecast_errors_scored', 'target_date': result['review_date']},
                          forecast_errors(result, state, s3))
        except Exception as error:
            telemetry_errors.append('forecast scoring: ' + str(error))
    if telemetry_errors:
        # The transaction may already be committed. Do not label it a training/inventory failure.
        try:
            reporter.emit({'event': 'telemetry_partial_failure', 'execution_id': execution_id,
                           'errors': telemetry_errors, 'replay_event': event},
                          [('TelemetryFailure', 1, 'Count', None)])
        except Exception:
            pass
        print(json.dumps({'event': 'telemetry_partial_failure', 'errors': telemetry_errors,
                          'replay_event': event, 'retry_with_same_execution_id': True}), file=sys.stderr)
        return 2
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/replay.json')
    parser.add_argument('--monitor-config', default='config/monitoring.json')
    parser.add_argument('--mode', choices=['probe', 'run', 'watchdog'], default='run')
    parser.add_argument('--execution-id')
    args = parser.parse_args()
    try:
        cfg = json.loads(Path(args.config).read_bytes())
        mon = json.loads(Path(args.monitor_config).read_bytes())
        session = boto3.Session(region_name=cfg.get('region', 'eu-north-1'), profile_name=cfg.get('profile'))
        code = execute(cfg, mon, args.mode, args.execution_id, session)
    except Exception as error:
        print(json.dumps({'event': 'monitored_command_failed', 'error': str(error)}), file=sys.stderr)
        code = 1
    sys.exit(code)


if __name__ == '__main__':
    main()
