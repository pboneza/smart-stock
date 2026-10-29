#!/usr/bin/env python3
"""Create a dashboard/log group/alarms; generate a least-scope EC2 publishing policy.

Run from Windows with your AWS CLI login/profile credentials. This identity needs
CloudWatch provisioning, log-group creation/retention and STS access. EC2's runtime
publishing policy alone intentionally does not grant dashboard/alarm administration.
No SNS subscription is created. Optional --alarm-topic is an existing SNS topic ARN.
"""
import argparse
import json
from pathlib import Path
import re
import boto3
from botocore.exceptions import ClientError


def build(region, account, workflow, product, topic=None):
    group = '/smart-stock/replay/' + workflow
    namespace = 'SmartStock/Replay'
    dimensions = [{'Name': 'ProductId', 'Value': product}, {'Name': 'Workflow', 'Value': workflow}]
    def alarm(label, metric, threshold, operator, missing, periods=1, points=1, period=300, statistic='Sum'):
        return dict(AlarmName=f'smart-stock-{workflow}-{label}', Namespace=namespace, MetricName=metric,
                    Dimensions=dimensions, Threshold=threshold, ComparisonOperator=operator,
                    Period=period, EvaluationPeriods=periods, DatapointsToAlarm=points, Statistic=statistic,
                    TreatMissingData=missing, AlarmActions=[topic] if topic else [],
                    AlarmDescription='Historical replay operational alarm; timestamps are execution time, not dataset time.')
    alarms = [alarm('run-failed', 'RunFailure', 0, 'GreaterThanThreshold', 'notBreaching'),
              alarm('telemetry-failed', 'TelemetryFailure', 0, 'GreaterThanThreshold', 'notBreaching'),
              alarm('no-progress-30h', 'SecondsSinceLastCommit', 30*3600, 'GreaterThanThreshold',
                    'notBreaching', 1, 1, 3600, 'Maximum'),
              alarm('watchdog-missing', 'WatchdogHeartbeat', 1, 'LessThanThreshold',
                    'breaching', 3, 3, 3600, 'Maximum')]
    def metric(name, horizon=None):
        row = [namespace, name, 'ProductId', product, 'Workflow', workflow]
        if horizon is not None:row += ['Horizon', str(horizon)]
        return row
    def widget(title, names, x, y, stat='Maximum'):
        return {'type':'metric','x':x,'y':y,'width':12,'height':7,'properties':{
            'title':title,'region':region,'view':'timeSeries','period':60,'stat':stat,
            'metrics':[metric(name) for name in names]}}
    dashboard = {'start':'-PT24H','widgets':[
        widget('Execution outcomes (reporting time)', ['RunSuccess','RunFailure','RunRetryNoOp','TelemetryFailure','DatasetExhausted'],0,0,'Sum'),
        widget('Execution duration, seconds', ['RunDurationSeconds'],12,0),
        widget('Stock and replenishment', ['ClosingStock','OrderQuantity','PipelineUnits'],0,7),
        widget('Demand and fulfilment', ['DemandUnits','FulfilledUnits','LostSalesUnits'],12,7),
        widget('Simulated cost (hypothetical units)', ['SimulatedTotalCost'],0,14),
        widget('Freshness: seconds since committed day', ['SecondsSinceLastCommit'],12,14),
        {'type':'metric','x':0,'y':21,'width':12,'height':7,'properties':{
            'title':'Mean absolute forecast error by horizon (matured targets)','region':region,'period':300,
            'stat':'Average','metrics':[metric('ForecastAbsoluteError',h) for h in range(1,8)]}},
        {'type':'metric','x':12,'y':21,'width':12,'height':7,'properties':{
            'title':'Mean forecast bias by horizon (positive = overprediction)','region':region,'period':300,
            'stat':'Average','metrics':[metric('ForecastBias',h) for h in range(1,8)]}},
        {'type':'log','x':0,'y':28,'width':24,'height':7,'properties':{
            'title':'Execution logs and simulation dates','region':region,'view':'table',
            'query':f"SOURCE '{group}' | fields @timestamp, event, execution_id, review_date, error | sort @timestamp desc | limit 50"}}]}
    policy = {'Version':'2012-10-17','Statement':[
        {'Effect':'Allow','Action':['cloudwatch:PutMetricData'],'Resource':'*',
         'Condition':{'StringEquals':{'cloudwatch:namespace':namespace}}},
        {'Effect':'Allow','Action':['logs:CreateLogStream','logs:PutLogEvents'],
         'Resource':f'arn:aws:logs:{region}:{account}:log-group:{group}:log-stream:*'}]}
    config = {'namespace':namespace,'workflow_id':workflow,'log_group':group}
    return group, alarms, dashboard, policy, config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--region',default='eu-north-1')
    parser.add_argument('--workflow-id',default='demo-v1')
    parser.add_argument('--product-id',default='22041')
    parser.add_argument('--profile')
    parser.add_argument('--alarm-topic',help='Optional existing SNS topic ARN in this region.')
    parser.add_argument('--output-dir',type=Path,default=Path('cloudwatch_generated'))
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.workflow_id):parser.error('Invalid workflow-id.')
    if args.alarm_topic and not args.alarm_topic.startswith(f'arn:aws:sns:{args.region}:'):
        parser.error('Alarm topic must be an SNS ARN in the chosen region.')
    session=boto3.Session(region_name=args.region,profile_name=args.profile)
    account=session.client('sts').get_caller_identity()['Account']
    group,alarms,dashboard,policy,config=build(args.region,account,args.workflow_id,args.product_id,args.alarm_topic)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,payload in [('monitoring.json',config),('ec2_monitoring_policy.json',policy)]:
        (args.output_dir/name).write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    logs=session.client('logs');cw=session.client('cloudwatch')
    try:logs.create_log_group(logGroupName=group)
    except ClientError as error:
        if error.response['Error']['Code']!='ResourceAlreadyExistsException':raise
    logs.put_retention_policy(logGroupName=group,retentionInDays=14)
    response=cw.put_dashboard(DashboardName='SmartStock-'+args.workflow_id,DashboardBody=json.dumps(dashboard))
    if response.get('DashboardValidationMessages'):raise RuntimeError(str(response['DashboardValidationMessages']))
    for alarm in alarms:cw.put_metric_alarm(**alarm)
    print('Created dashboard: SmartStock-'+args.workflow_id)
    print('Log group: '+group)
    print('Generated EC2 publishing policy: '+str(args.output_dir/'ec2_monitoring_policy.json'))
    print('Generated separate monitoring config: '+str(args.output_dir/'monitoring.json'))
    print('Alarm notifications: '+(args.alarm_topic or 'NO SNS TOPIC CONFIGURED; alarms visible in console only'))
    print('The watchdog missing-data alarm can enter ALARM until hourly monitoring begins.')


if __name__=='__main__':main()
