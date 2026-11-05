#!/usr/bin/env python3
"""Generate a frozen seven-day direct Ridge forecast and publish a versioned run.

Keep the validated train.py beside this file. Run in the EC2 training environment.
Dependencies: boto3, numpy, pandas, scikit-learn, joblib.

Example (from ~/smart-stock; origin is END of the last fully observed sales day):
    python scripts/batch_forecast.py --model-s3 s3://smart-stock-s3-bucket/models/ridge_validation/direct_ridge.joblib --data data/processed/product_22041_daily_sales.csv --origin 2011-08-24 --product-id 22041 --output-s3 s3://smart-stock-s3-bucket/forecasts --region eu-north-1

--data accepts a local CSV or an exact S3 object URI. --model-s3 must point to the
trusted direct_ridge.joblib object, not a directory or model.tar.gz archive.
Default CSV columns: forecast_date, units_sold. For an older feature CSV specify
--sales-col target_sales. No training, evaluation, ordering or scheduling occurs.

A new run folder is created for EVERY invocation. Origin+target+horizon identify
forecast timing; run_id distinguishes reruns. Consumers must use ONLY folders
containing _SUCCESS.json, which is uploaded after all other files. There is no
'latest' pointer. The scheduler will later select the successful run for a decision.
The seven-day horizon does not mean seven-day execution intervals: run daily after
sales are finalised. For historical replay, supply a dataset date, never today's
calendar date unless today's completed sales really exist in your input.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
from urllib.parse import urlsplit
from uuid import uuid4

import boto3
from botocore.exceptions import BotoCoreError, ClientError
import joblib
import numpy as np
import pandas as pd
import sklearn

import train as training


def sha256(path: Path) -> str:
    """Hash in chunks so larger inputs do not need a second in-memory copy."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def split_s3(uri: str, require_key: bool = True) -> tuple[str, str]:
    parts = urlsplit(uri)
    key = parts.path.lstrip('/')
    if parts.scheme != 's3' or not parts.netloc or parts.query or parts.fragment:
        raise ValueError('Use an S3 URI such as s3://bucket/path, without query or fragment.')
    if require_key and (not key or key.endswith('/')):
        raise ValueError('Input S3 URI must identify an exact object, not a folder.')
    return parts.netloc, key.rstrip('/') if not require_key else key


def fetch_s3(s3, uri: str, destination: Path) -> dict:
    """Download one response and capture that same object's version metadata."""
    bucket, key = split_s3(uri)
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response['Body']
    try:
        with destination.open('wb') as handle:
            shutil.copyfileobj(body, handle)
    finally:
        body.close()
    return {'uri': uri, 'version_id': response.get('VersionId'),
            'etag': response.get('ETag'), 'sha256': sha256(destination)}


def read_history(path: Path, origin: pd.Timestamp, product_id: str,
                 date_col: str, sales_col: str) -> tuple[pd.Series, dict]:
    """Restrict to the origin BEFORE constructing sales-derived features.

    Daily input is already prepared: never infer missing dates as zero here.
    StockCode, if present, must match the explicitly requested product. If absent,
    the caller is responsible for providing the correct single-product CSV.
    """
    frame = pd.read_csv(path, dtype={'StockCode': 'string'})
    missing = {date_col, sales_col}.difference(frame.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    dates = pd.to_datetime(frame[date_col], errors='raise')
    if dates.isna().any() or dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError('Dates must be complete timezone-naive calendar dates at midnight.')
    if 'StockCode' in frame:
        codes = frame.StockCode.str.strip()
        if codes.isna().any() or not codes.eq(product_id).all():
            raise ValueError('Every StockCode must match --product-id.')
    observed = dates.le(origin)
    values = pd.to_numeric(frame.loc[observed, sales_col], errors='raise')
    history = pd.Series(values.to_numpy(dtype=float),
                        index=pd.DatetimeIndex(dates.loc[observed]), name='observed_sales').sort_index()
    if history.empty or history.index.has_duplicates:
        raise ValueError('Observed sales must be nonempty with no duplicate dates.')
    if not np.isfinite(history.to_numpy()).all() or history.lt(0).any():
        raise ValueError('Observed sales must be finite and nonnegative.')
    if origin not in history.index:
        raise ValueError('Origin must have an actual sales value. Choose a completed date in the dataset.')
    if not history.index.equals(pd.date_range(history.index.min(), origin, freq='D')):
        raise ValueError('Observed sales contain missing calendar days; repair the prepared input explicitly.')
    if len(history) < 28:
        raise ValueError('At least 28 consecutive observed days are required through the origin.')
    # Only 28 values are needed by this fixed feature contract.
    return history.tail(28), {
        'input_rows': len(frame), 'observed_rows': len(history),
        'rows_after_origin_ignored': int((~observed).sum()),
        'product_column_verified': 'StockCode' in frame,
    }


def predict(bundle: dict, history: pd.Series, origin: pd.Timestamp,
            product_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the frozen package and reuse train.py's exact feature functions."""
    if not isinstance(bundle, dict) or not {'models', 'metadata'}.issubset(bundle):
        raise ValueError('Model package must contain models and metadata.')
    models, metadata = bundle['models'], bundle['metadata']
    if set(models) != set(training.HORIZONS):
        raise ValueError('Expected integer horizon keys 1 through 7.')
    if metadata.get('strategy') != 'direct' or metadata.get('features') != training.FEATURES:
        raise ValueError('Model strategy/features differ from the validated train.py contract.')
    if metadata.get('sklearn_version') != sklearn.__version__:
        raise ValueError(f"Use scikit-learn {metadata.get('sklearn_version')}; installed {sklearn.__version__}.")
    if metadata.get('product_id') != product_id:
        raise ValueError('Saved product_id differs from --product-id.')
    cutoff = pd.Timestamp(metadata['train_end'])
    if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize():
        raise ValueError('Invalid training cutoff in model metadata.')
    if origin < cutoff:
        raise ValueError('Origin precedes the model training cutoff; this would leak future training information.')
    base = training.historical_features(history).tail(1)
    if base.isna().any().any():
        raise ValueError('Incomplete historical features.')
    rows, snapshots = [], []
    for horizon in training.HORIZONS:
        features = training.horizon_features(base, horizon)
        if list(models[horizon].feature_names_in_) != training.FEATURES:
            raise ValueError('Fitted model feature order differs from train.py.')
        raw = float(np.asarray(models[horizon].predict(features)).ravel()[0])
        if not np.isfinite(raw):
            raise ValueError('Model returned a non-finite prediction.')
        target = origin + pd.Timedelta(days=horizon)
        rows.append({'product_id': product_id, 'origin_date': str(origin.date()),
                     'target_date': str(target.date()), 'horizon': horizon,
                     'prediction_raw': raw, 'prediction': max(0.0, raw)})
        features = features.reset_index(drop=True)
        features.insert(0, 'horizon', horizon)
        snapshots.append(features)
    return pd.DataFrame(rows), pd.concat(snapshots, ignore_index=True)


def run_forecast(args, s3=None) -> tuple[pd.DataFrame, Path, str]:
    """Write local evidence, upload data, then publish a successful completion marker.

    Every retry creates a distinct run. Files from a partial upload have no marker
    and must not be consumed. Model selection and duplicate-order prevention are
    responsibilities of later orchestration/replenishment steps.
    """
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.product_id):
        raise ValueError('Product ID must contain letters, digits, underscores or hyphens.')
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', args.origin):
        raise ValueError('Origin must use YYYY-MM-DD.')
    origin = pd.Timestamp(args.origin)
    if pd.isna(origin):
        raise ValueError('Invalid origin date.')
    split_s3(args.model_s3)
    out_bucket, out_prefix = split_s3(args.output_s3, require_key=False)
    if s3 is None:
        session = boto3.Session(region_name=args.region, profile_name=args.profile)
        s3 = session.client('s3')
    started = datetime.now(timezone.utc)
    run_id = started.strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid4().hex[:8]
    relative = f'product={args.product_id}/origin={args.origin}/run={run_id}'
    run_prefix = '/'.join(filter(None, [out_prefix, relative]))
    destination = Path(args.local_output) / relative
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='stock-forecast-') as temp:
        temp = Path(temp)
        model_file = temp / 'direct_ridge.joblib'
        model_source = fetch_s3(s3, args.model_s3, model_file)
        if args.data.startswith('s3://'):
            data_file = temp / 'sales.csv'
            data_source = fetch_s3(s3, args.data, data_file)
        else:
            data_file = Path(args.data).expanduser()
            data_source = {'path': str(data_file.resolve()), 'sha256': sha256(data_file)}
        history, checks = read_history(data_file, origin, args.product_id,
                                       args.date_col, args.sales_col)
        # joblib is executable serialization. Only use artifacts from your trusted bucket.
        bundle = joblib.load(model_file)
        forecasts, features = predict(bundle, history, origin, args.product_id)
        version = 'ridge-' + model_source['sha256'][:16]
        forecasts['model_version'] = version
        forecasts['run_id'] = run_id
        forecasts.to_csv(destination / 'forecasts.csv', index=False)
        features.to_csv(destination / 'features.csv', index=False)
        history.rename_axis('date').to_csv(destination / 'observed_history.csv', date_format='%Y-%m-%d')
        manifest = {
            'run_id': run_id, 'started_at_utc': started.isoformat(),
            'product_id': args.product_id, 'origin_date': args.origin,
            'origin_timing': 'end_of_day_after_actual_sales_are_finalised',
            'horizons': list(training.HORIZONS), 'model_version': version,
            'model_source': model_source, 'data_source': data_source,
            'training_metadata': bundle['metadata'], 'input_checks': checks,
            'refitted': False, 'clip_negative_predictions': True,
            'seven_day_forecast_total': float(forecasts.prediction.sum()),
            'forecast_seconds': (datetime.now(timezone.utc) - started).total_seconds(),
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'scikit-learn': sklearn.__version__,
                        'joblib': joblib.__version__, 'boto3': boto3.__version__},
            'feature_code_sha256': sha256(Path(training.__file__)),
            'forecast_code_sha256': sha256(Path(__file__)),
            'output_s3': f's3://{out_bucket}/{run_prefix}/',
            'file_sha256': {name: sha256(destination / name) for name in
                            ['forecasts.csv', 'features.csv', 'observed_history.csv']},
            'notes': [
                'Seven forecasts are issued from one observed information set; no recursive predictions.',
                'Overlapping forecasts are separate estimates, not quantities to add across origins.',
                'Only successfully committed runs may be consumed; no automatic latest-model selection.',
                'The caller selects the product when the CSV has no StockCode column.',
                'No observed actuals for future target dates are required or used.',
            ],
        }
        (destination / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name in ['forecasts.csv', 'features.csv', 'observed_history.csv', 'run_manifest.json']:
        s3.upload_file(str(destination / name), out_bucket, f'{run_prefix}/{name}')
    marker = {'run_id': run_id, 'status': 'completed', 'forecast_rows': len(forecasts),
              'committed_at_utc': datetime.now(timezone.utc).isoformat(),
              'manifest_sha256': sha256(destination / 'run_manifest.json')}
    payload = (json.dumps(marker, indent=2) + '\n').encode('utf-8')
    s3.put_object(Bucket=out_bucket, Key=f'{run_prefix}/_SUCCESS.json',
                  Body=payload, ContentType='application/json')
    # Local marker also indicates the S3 publish finished, not just local prediction.
    (destination / '_SUCCESS.json').write_bytes(payload)
    return forecasts, destination, f's3://{out_bucket}/{run_prefix}/'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-s3', required=True, help='Exact trusted direct_ridge.joblib S3 object URI.')
    parser.add_argument('--data', required=True, help='Local daily CSV or exact S3 CSV object URI.')
    parser.add_argument('--origin', required=True, help='Last completed observed sales day, YYYY-MM-DD.')
    parser.add_argument('--product-id', required=True)
    parser.add_argument('--output-s3', required=True, help='S3 output prefix for versioned runs.')
    parser.add_argument('--region', default='eu-north-1')
    parser.add_argument('--profile', help='Optional local AWS profile; omit on EC2 to use its instance role.')
    parser.add_argument('--date-col', default='forecast_date')
    parser.add_argument('--sales-col', default='units_sold')
    parser.add_argument('--local-output', default='results/batch_forecasts')
    args = parser.parse_args()
    try:
        forecasts, folder, uri = run_forecast(args)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, BotoCoreError, ClientError) as error:
        logging.error(json.dumps({'event': 'batch_forecast_failed', 'error': str(error)}))
        sys.exit(1)
    print(forecasts.to_string(index=False))
    print(json.dumps({'event': 'batch_forecast_completed', 'forecast_rows': len(forecasts),
                      'local_output': str(folder.resolve()), 'output_s3': uri}))


if __name__ == '__main__':
    main()
