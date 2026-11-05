# CloudWatch monitoring for Smart Stock historical replay

This package wraps the working daily replay without changing replay.json, the S3
checkpoint structure, the model or inventory transaction rules. It creates no
endpoint and configures no email subscription. Real AWS setup must be run in your
account; tests here used mocked AWS APIs.

## What you receive

- cloudwatch_replay.py: run/probe/watchdog modes with structured CloudWatch logs
  and custom metrics; reuse the existing daily_replay.py and its dependencies.
- setup_cloudwatch.py: create a log group (14-day retention), dashboard and alarms;
  generate an exact EC2 publishing policy and a separate monitoring.json.
- run_monitored_replay.sh: replacement cron wrapper for the daily replay.
- check_replay_health.sh: hourly checkpoint freshness check.

Metrics use actual UTC reporting timestamps. Historical dates are fields in logs.
No Product/date/run-ID dimensions are created per execution: metric dimensions are
ProductId and Workflow, plus Horizon for forecast error metrics. This limits metric
cardinality. Custom metrics, alarms, dashboards and log ingestion can incur charges;
Free Tier coverage and available credits depend on your account. No cost estimate
or guarantee is implied by this package.

## 1. Set up the AWS monitoring resources from Windows

Extract cloudwatch_monitoring into your local project folder.
Use an AWS identity allowed to create log groups, set retention, put dashboards,
put metric alarms and call STS GetCallerIdentity. The existing EC2 S3 role does
not have these administration permissions and should not receive them just to run
monitoring. If your local identity is restricted, an administrator can run setup.

Windows PowerShell, local project root:

```powershell
Set-Location C:\Users\patri\smart-stock
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade boto3
aws login --region eu-north-1
aws sts get-caller-identity

python .\cloudwatch_monitoring\setup_cloudwatch.py --region eu-north-1 --workflow-id demo-v1 --product-id 22041
```

Use the workflow ID matching this replay demonstration. If using a named profile,
pass --profile NAME to both AWS commands and setup_cloudwatch.py.

The setup creates:

- Log group: /smart-stock/replay/demo-v1
- Metric namespace: SmartStock/Replay
- Dashboard: SmartStock-demo-v1
- Four alarms (see section 7)
- Local files: cloudwatch_generated/monitoring.json and
  cloudwatch_generated/ec2_monitoring_policy.json

Setup is repeatable for the same workflow names. It updates the dashboard, retention
and alarm definitions; it does not delete log data. Rerunning without --alarm-topic
removes alarm notification actions previously configured by this script.

## 2. Give EC2 permission to publish

In IAM → Roles → SmartStock-EC2Role → Permissions → Add permissions → Create inline
policy → JSON, paste the contents of cloudwatch_generated/ec2_monitoring_policy.json.
Name the policy SmartStockCloudWatchPublishing and save it. Do not edit the trust
policy; keep the existing S3 access policy.

Optional CLI equivalent, using an identity permitted to modify this role:

```powershell
aws iam put-role-policy --role-name SmartStock-EC2Role --policy-name SmartStockCloudWatchPublishing --policy-document file://cloudwatch_generated/ec2_monitoring_policy.json
```

The generated policy allows PutMetricData only in SmartStock/Replay, and stream
creation/log publishing only within this workflow's precreated log group. It does
not grant dashboard or alarm administration. Wait briefly for IAM changes to propagate.

## 3. Transfer the monitoring files

Windows PowerShell:

```powershell
$keyPath = "$env:USERPROFILE\Downloads\smart-stock-key.pem"
$ec2Address = "YOUR_CURRENT_PUBLIC_IP"
$ec2Target = "ubuntu@$ec2Address"
scp -i $keyPath .\cloudwatch_monitoring\cloudwatch_replay.py .\cloudwatch_monitoring\run_monitored_replay.sh .\cloudwatch_monitoring\check_replay_health.sh "${ec2Target}:/home/ubuntu/smart-stock/scripts/"
scp -i $keyPath .\cloudwatch_generated\monitoring.json "${ec2Target}:/home/ubuntu/smart-stock/config/"
```

Keep the original daily_replay.py, batch_forecast.py, replenish.py and train.py in
scripts/. Do not modify config/replay.json: its hash is frozen in the checkpoint.

## 4. Test telemetry without advancing inventory

Ubuntu SSH:

```bash
cd ~/smart-stock
source .venv/bin/activate
python -m pip install --upgrade boto3

python scripts/cloudwatch_replay.py \
  --config config/replay.json \
  --monitor-config config/monitoring.json \
  --mode probe
```

Expected event: monitoring_probe. It publishes one log record and MonitoringProbe
metric; it does not call the replay or change inventory.

In the AWS Console, region eu-north-1:

- CloudWatch → Logs → Log groups → /smart-stock/replay/demo-v1 → newest stream.
- CloudWatch → Metrics → All metrics → SmartStock/Replay → ProductId, Workflow.

Allow time for new metrics and dashboard series to appear. A probe alone does not
populate inventory or forecast-error charts.

## 5. Run one monitored historical day

```bash
python scripts/cloudwatch_replay.py \
  --config config/replay.json \
  --monitor-config config/monitoring.json \
  --mode run \
  --execution-id monitored-001
```

A NEW execution ID advances the next dataset day. To verify retries, run exactly
the same command again: expect replay_already_committed and RunRetryNoOp, without
new inventory quantities or forecast-error samples.

Exit codes:

- 0: replay/probe/watchdog command and required reporting succeeded.
- 1: command or replay failed; inspect the emitted error and local log.
- 2: replay completed or was already committed, but some telemetry failed.
  Inventory is NOT rolled back. Inspect the S3 checkpoint and retry using the same
  execution ID, never a new one solely to recover telemetry.

The wrapper tries logs and metrics independently. If monitoring is unavailable,
it does not block the underlying inventory transaction. A retry intentionally
skips publishing prior inventory metrics to avoid double reporting. It does not
backfill missing original metrics; use the durable S3 result for reconstruction.
Network ambiguities can still duplicate CloudWatch samples: CloudWatch is an
operational view, not an exactly-once business ledger.

## 6. Run the hourly watchdog

```bash
python scripts/cloudwatch_replay.py \
  --config config/replay.json \
  --monitor-config config/monitoring.json \
  --mode watchdog
```

This reads the checkpoint and emits SecondsSinceLastCommit and WatchdogHeartbeat.
It never advances the dataset or changes stock. It uses checkpoint updated_at_utc,
so retry no-ops do not disguise a stalled replay as progress.

If the checkpoint has no committed days, freshness is treated as stale. If state
access fails or the script cannot start, the heartbeat is absent; a separate alarm
covers that case. Monitoring cannot publish a failure while the entire instance
or network is offline, which is why missing-heartbeat monitoring is also needed.

## 7. Dashboard and alarms

Open CloudWatch → Dashboards → SmartStock-demo-v1.

Dashboard panels:

- Successful, failed, duplicate and dataset-exhausted execution events.
- Execution duration.
- Closing stock, order quantities and outstanding units.
- Demand, fulfilled units and lost sales.
- Hypothetical simulated total cost.
- Seconds since the last committed historical day.
- Mean absolute error and signed bias, separately for horizons 1..7.
- Structured execution logs with historical review dates.

Forecast scoring happens only after a new historical day is committed. It scores
previous committed forecasts targeting that day against that day's revealed demand.
Horizon 1 appears first; longer horizons mature as replay advances. No future actual
is read for scoring or ordering. Previous unmonitored days are not automatically
backfilled, but their forecasts can contribute to the newly revealed target day.
If cost/scoring publication fails, original S3 evidence remains available.

The dashboard uses real reporting time. Several manually accelerated historical
days can fall into one chart period. Per-run maximums are used for inventory/cost
panels, not cumulative sums. Use Logs Insights or committed S3 records for exact
day-by-day analysis. Forecast-error panels show averages of samples received in
each execution-time bucket; retry no-ops add no new forecast samples.

Alarms (same product/workflow dimensions):

| Alarm suffix | Trigger |
|---|---|
| run-failed | RunFailure > 0 in a five-minute period |
| telemetry-failed | TelemetryFailure > 0 in a five-minute period |
| no-progress-30h | Hourly freshness measurement exceeds 30 hours since last committed day |
| watchdog-missing | Three missing/breaching hourly heartbeat periods |

Failure metrics treat missing data as not breaching. Watchdog missing data is
breaching. Alarm evaluation and notification are not instantaneous; allow the
configured periods and CloudWatch's evaluation range. Initial heartbeat alarms
can be in ALARM before the hourly monitor is first installed.

By default alarms are visible in CloudWatch but send NO email. To add notifications,
use an existing same-region SNS topic and rerun setup with --alarm-topic TOPIC_ARN.
Ensure the topic permits CloudWatch alarm delivery and its intended subscriptions
are confirmed. The package does not create a topic or subscribe anyone.

CloudWatch alarm notifications are generally sent on state transitions, not once
for every failing run while an alarm remains ALARM.

## 8. Replace the schedule and add watchdog cron

Ubuntu SSH:

```bash
cd ~/smart-stock
sed -i 's/\r$//' scripts/run_monitored_replay.sh scripts/check_replay_health.sh
chmod +x scripts/run_monitored_replay.sh scripts/check_replay_health.sh
mkdir -p logs
crontab -e
```

REMOVE the old daily replay cron entry, then add these two lines:

```cron
5 1 * * * /home/ubuntu/smart-stock/scripts/run_monitored_replay.sh >> /home/ubuntu/smart-stock/logs/daily_replay.log 2>&1
15 * * * * /home/ubuntu/smart-stock/scripts/check_replay_health.sh >> /home/ubuntu/smart-stock/logs/replay_health.log 2>&1
```

The daily wrapper retains scheduled-YYYY-MM-DD IDs and uses the same local replay
lock as before. If the unmonitored job already committed today's scheduled ID,
the monitored wrapper will report a retry no-op, not advance another day.

Times are in the server timezone (check timedatectl), normally UTC. The execution
ID uses the UTC calendar date. Keep that timezone relationship consistent.

Check:

```bash
crontab -l
tail -n 50 logs/daily_replay.log
tail -n 50 logs/replay_health.log
```

This is local EC2 cron, not an AWS service starting your instance. EC2 must remain
running for both scripts. If you stop it, watchdog-missing is expected to alarm.
Before a planned pause, disable alarm actions in CloudWatch if notifications are
configured. Resume actions after the instance/schedule is restored.

At dataset exhaustion, the replay emits DatasetExhausted without committing new
days. Disable the schedule/freshness alarms when the demonstration ends; otherwise
no-progress will correctly report that the replay stopped advancing.

## 9. Troubleshooting and clean-up

- AccessDenied for PutMetricData: confirm SmartStockCloudWatchPublishing is attached
  to the actual EC2 role and namespace matches the policy.
- AccessDenied/CreateLogStream or missing log group: run setup in eu-north-1 and
  verify monitoring.json uses the created log group.
- Empty inventory panels after probe/retry: run one NEW execution ID; probes and
  duplicate retries intentionally do not publish business quantities.
- Config hash error: restore replay.json; monitoring settings belong in the separate
  monitoring.json, not the frozen replay configuration.
- Missing horizon metrics: allow enough historical days to reveal those targets;
  inspect forecast_errors_scored or telemetry_partial_failure log events.
- TelemetryFailure missing during total CloudWatch outage: inspect local stderr;
  heartbeat absence is the independent fallback signal.
- Dependency error: scripts must share the same directory and existing EC2 venv.

To end the experiment: remove both cron entries; delete/disable the four alarms
and optional dashboard in CloudWatch; remove the workflow log group if you no longer
need logs; stop EC2 if no other workloads need it. Custom metrics have no direct
manual-delete operation; they age out under CloudWatch retention. Your S3 replay
state and model artifacts are not deleted by these monitoring steps.

## Validation

Python syntax, shell syntax and isolated tests passed with mocked AWS clients:
probe publication; real execution-time metric timestamps; no inventory metrics on
retries; telemetry failure isolation; replay failure metric; freshness watchdog;
dashboard/alarm/policy shapes; matured target errors by horizon. Actual AWS IAM,
resource provisioning, cron and metric appearance must be checked in your account.

References:
https://docs.aws.amazon.com/AmazonCloudWatchLogs/latest/APIReference/API_PutLogEvents.html
https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/alarms-and-missing-data.html
