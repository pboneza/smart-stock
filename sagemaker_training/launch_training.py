#!/usr/bin/env python3
"""Upload local CSV and scripts; submit a single SageMaker training job via boto3.

Requires boto3 in the launching environment. The training image must already
contain python3, numpy, pandas, scikit-learn >=1.2, and joblib. Use a compatible
scikit-learn training image in your region, or your existing compatible ECR image.
No SageMaker Python SDK dependency, endpoint deployment, or automatic evaluation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time


def build_request(a, job_name):
    prefix = f"{a.prefix.strip('/')}/{job_name}"
    def channel(name):
        return {'ChannelName': name, 'DataSource': {'S3DataSource': {
            'S3DataType': 'S3Prefix', 'S3Uri': f's3://{a.bucket}/{prefix}/{name}/',
            'S3DataDistributionType': 'FullyReplicated'}}, 'InputMode': 'File'}
    return {
        'TrainingJobName': job_name, 'RoleArn': a.role_arn,
        'AlgorithmSpecification': {
            'TrainingImage': a.image_uri, 'TrainingInputMode': 'File',
            'ContainerEntrypoint': ['python3', '/opt/ml/input/data/code/sagemaker_entry.py'],
            'ContainerArguments': ['--product-id', a.product_id, '--train-end', a.train_end,
                                   '--alpha', str(a.alpha), '--sales-col', a.sales_col,
                                   '--date-col', a.date_col]},
        'InputDataConfig': [channel('train'), channel('code')],
        'OutputDataConfig': {'S3OutputPath': f's3://{a.bucket}/{prefix}/output'},
        'ResourceConfig': {'InstanceType': a.instance_type, 'InstanceCount': 1,
                           'VolumeSizeInGB': 10},
        'StoppingCondition': {'MaxRuntimeInSeconds': 900},
        'EnableNetworkIsolation': True,
        'Tags': [{'Key': 'Project', 'Value': 'smart-stock'}],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--region', required=True)
    parser.add_argument('--bucket', required=True, help='Existing bucket name in the training region.')
    parser.add_argument('--role-arn', required=True)
    parser.add_argument('--image-uri', required=True)
    parser.add_argument('--train-end', required=True)
    parser.add_argument('--product-id', default='22041')
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--sales-col', default='units_sold')
    parser.add_argument('--date-col', default='forecast_date')
    parser.add_argument('--instance-type', default='ml.m5.large')
    parser.add_argument('--prefix', default='smart-stock/training')
    parser.add_argument('--job-name')
    parser.add_argument('--profile', help='Optional local AWS profile; omit in SageMaker Studio.')
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print request without AWS calls or uploads.')
    a = parser.parse_args()
    import math
    if not a.data.is_file():
        parser.error('Input CSV does not exist.')
    if not math.isfinite(a.alpha) or a.alpha <= 0:
        parser.error('Alpha must be finite and positive.')
    scripts = Path(__file__).resolve().parent
    for name in ['train.py', 'sagemaker_entry.py']:
        if not (scripts / name).is_file():
            parser.error(f'Missing sibling file: {name}')
    job_name = a.job_name or 'smart-stock-ridge-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    if len(job_name) > 63 or not re.fullmatch(r'[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?', job_name):
        parser.error('Job name must be 1–63 alphanumeric/hyphen characters, starting and ending alphanumeric.')
    request = build_request(a, job_name)
    if a.dry_run:
        print(json.dumps(request, indent=2))
        return
    import boto3
    session = boto3.Session(region_name=a.region, profile_name=a.profile)
    identity = session.client('sts').get_caller_identity()
    print(f"Submitting as {identity['Arn']}", flush=True)
    s3 = session.client('s3'); sm = session.client('sagemaker')
    prefix = f"{a.prefix.strip('/')}/{job_name}"
    s3.upload_file(str(a.data), a.bucket, f'{prefix}/train/sales.csv')
    for name in ['train.py', 'sagemaker_entry.py']:
        s3.upload_file(str(scripts / name), a.bucket, f'{prefix}/code/{name}')
    result = sm.create_training_job(**request)
    print(json.dumps(result, default=str, indent=2), flush=True)
    print(f'https://{a.region}.console.aws.amazon.com/sagemaker/home?region={a.region}#/jobs/{job_name}', flush=True)
    print(f'CloudWatch log group: /aws/sagemaker/TrainingJobs; stream prefix: {job_name}', flush=True)
    if a.wait:
        previous = None
        while True:
            state = sm.describe_training_job(TrainingJobName=job_name)
            status = state['TrainingJobStatus']
            detail = (status, state.get('SecondaryStatus'))
            if detail != previous:
                print(f'{detail[0]}: {detail[1]}', flush=True); previous = detail
            if status in ['Completed', 'Failed', 'Stopped']:
                if status != 'Completed':
                    raise SystemExit(f"Training {status}: {state.get('FailureReason', 'See job logs')}")
                print('Model artifacts: ' + state['ModelArtifacts']['S3ModelArtifacts'], flush=True)
                break
            time.sleep(20)


if __name__ == '__main__':
    main()
