# Smart Stock daily historical replay

One invocation advances one dataset day. Forecasts and recommendations use actual
sales only through yesterday; today's historical demand is applied afterwards.
Orders are simulated, not submitted to suppliers. The workflow is designed for
your single product 22041 and the existing EC2 Python environment.

## Included files

- daily_replay.py: orchestration, demand processing, S3 inventory state and retry ledger.
- batch_forecast.py, replenish.py, train.py: the previously tested dependencies,
  included so all imports are available. No model fitting is invoked by the replay.
- replay_example.json: example configuration, using the medium_shortage scenario.
- inventory_example.json: simulated initial stock of 28 units, no pipeline, review
  date 2011-08-25. It is a NEW replay startup state, not your observed live stock.
- run_daily_replay.sh: cron entrypoint with a local execution lock and stable daily ID.

The model remains the May-trained validation candidate. This demonstrates the
lifecycle; it does not claim to reproduce the previous final-model test results.
The example cost parameters are hypothetical: 0.1 holding per unit/day, 5 per lost
unit and 0 fixed ordering cost. Adjust before beginning if your intended scenario
is different; do not use the selected-policy CSV's total holding_cost result as
a unit cost. No policy recalibration is performed.

## 1. Transfer from Windows PowerShell

Extract the ZIP into C:\Users\patri\smart-stock so that daily_replay is a subfolder.
Use the current EC2 public IP, which can change after stop/start.

```powershell
Set-Location C:\Users\patri\smart-stock
$keyPath = "$env:USERPROFILE\Downloads\smart-stock-key.pem"
$ec2Address = "YOUR_CURRENT_PUBLIC_IP"
$ec2Target = "ubuntu@$ec2Address"
ssh -i $keyPath $ec2Target "mkdir -p /home/ubuntu/smart-stock/scripts /home/ubuntu/smart-stock/config /home/ubuntu/smart-stock/logs"
scp -i $keyPath .\daily_replay\daily_replay.py .\daily_replay\batch_forecast.py .\daily_replay\replenish.py .\daily_replay\train.py .\daily_replay\run_daily_replay.sh "${ec2Target}:/home/ubuntu/smart-stock/scripts/"
scp -i $keyPath .\daily_replay\replay_example.json "${ec2Target}:/home/ubuntu/smart-stock/config/replay.json"
```

You already have config/inventory_example.json and config/selected_policies.csv on
EC2. Inspect the inventory snapshot. For a NEW replay beginning on 2011-08-25,
use the included startup snapshot if the existing snapshot was altered. Preserve
any inventory record you need before replacing it. This replay stores subsequent
inventory states in S3; it does not modify the initial snapshot file.

## 2. Check configuration in Ubuntu SSH

```bash
cd ~/smart-stock
source .venv/bin/activate
python -m pip install --upgrade boto3
nano config/replay.json
```

Check these values:

- model_s3: EXACT direct_ridge.joblib object URI. Add the timestamp folder if your
  model upload used one. This is not a model directory or model.tar.gz archive.
- data: the prepared daily sales CSV, either the existing local file or an exact
  S3 CSV URI. Keep the source dataset frozen for this replay.
- initial_inventory: config/inventory_example.json, with is_simulated=true and
  stage=opening_before_receipts. Startup review_date must be after training cutoff.
- selected_policies: config/selected_policies.csv (your validation-selected file).
- scenario, lead_time, pack_size, costs: explicitly chosen experiment settings.
- state_s3 and output_s3: a fresh replay namespace, by default replay/demo-v1.

Update boto3 because the checkpoint needs current S3 PutObject IfMatch support.
No sklearn or model dependency upgrades are needed; keep the environment that
successfully trained and forecasted your model.

Example commands to inspect inputs:

```bash
cat config/inventory_example.json
head -n 4 config/selected_policies.csv
ls -lh data/processed/product_22041_daily_sales.csv
```

The model, dataset, initial inventory and selected policies are fingerprinted.
Changing them mid-replay is rejected. The configuration is also frozen. If you
want another experiment, create a new configuration and NEW state/output prefix,
for example replay/demo-v2. Do not delete the old state to simulate a retry.

## 3. Execute the first historical day

```bash
cd ~/smart-stock
source .venv/bin/activate
python -u scripts/daily_replay.py --config config/replay.json --execution-id demo-001
```

On first use, the script creates state_s3 with a conditional create. It then:

1. Loads the initial inventory state (for the example: morning 2011-08-25).
2. Forecasts seven days from end-of-day 2011-08-24 using the frozen model.
3. Runs replenishment after receipts, using four forecast days for L=3.
4. Records the proposed quantity as a simulated confirmed order arriving at t+3.
5. Reveals historical demand for 2011-08-25, fulfils available stock, and records losses.
6. Calculates holding, shortage and ordering costs for the day.
7. Writes prepared day evidence to S3, then atomically commits the next inventory
   state, simulated orders and execution ledger in one conditional checkpoint write.

The output event replay_day_committed means the next opening state is ready.

## 4. Verify retry protection, then advance another day

Run the SAME command with demo-001 again:

```bash
python -u scripts/daily_replay.py --config config/replay.json --execution-id demo-001
```

Expected event: replay_already_committed. The historical date and inventory do not
advance again. Even if the first request succeeded but its network response was
lost, the stored execution ledger identifies that event as already completed.

To advance one more day, use a NEW ID:

```bash
python -u scripts/daily_replay.py --config config/replay.json --execution-id demo-002
```

That invocation reads the S3 checkpoint and processes 2011-08-26 automatically.
You do not update the origin or inventory JSON manually between replay days.

A new execution ID means advance the next day, NOT retry. Keep the same ID when
recovering a failed or ambiguous invocation. Do not launch different IDs in parallel.
The S3 ETag condition rejects stale concurrent commits; the local scheduler lock
also prevents overlap on this EC2 instance. External workflows must use the same
state object and conditional protocol rather than overwrite it manually.

## 5. Inspect state and monitoring data

Each successful day records simulation date and UTC execution time separately,
plus stock, receipts, forecast, order quantity, demand, fulfilled/lost units and
costs. The S3 checkpoint also holds cumulative costs, fulfilment and lost sales.

```bash
python - <<'PY'
import json
from pathlib import Path
from urllib.parse import urlsplit
import boto3
cfg = json.loads(Path('config/replay.json').read_text())
u = urlsplit(cfg['state_s3'])
r = boto3.client('s3', region_name=cfg.get('region', 'eu-north-1')).get_object(
    Bucket=u.netloc, Key=u.path.lstrip('/'))
try:
    state = json.loads(r['Body'].read())
finally:
    r['Body'].close()
print('Days processed:', state['days_processed'])
print('Next simulated review date:', state['inventory']['review_date'])
print('Opening stock:', state['inventory']['on_hand'])
print('Outstanding orders:', state['inventory']['outstanding_orders'])
print('Cumulative:', state['cumulative'])
print('Last committed result:', state.get('last_result_s3'))
PY
```

The authoritative record of a completed historical day is the checkpoint's
committed_executions entry. Prepared day_result.json objects from failed or losing
concurrent attempts can exist in S3; do not aggregate every object by prefix.
Use only results referenced in committed_executions. Forecast/recommendation run
_SUCCESS markers confirm those individual artifacts were published, not that the
inventory day itself was committed.

An error prints event=replay_failed to stderr and exits nonzero. A failure before
commit leaves the previous inventory intact. Retry with the same execution ID.
The initial empty checkpoint may already exist even if the first day failed.

When the dataset is exhausted, replay_dataset_exhausted is returned without another
inventory update. No model is retrained, no date beyond the dataset is invented,
and no demand value from the next day is used in the ordering decision.

This package supplies basic execution logs and S3 metrics. It does NOT yet configure
CloudWatch dashboards, automatic alerts, model drift checks, or automated retraining.
Those are the next monitoring extensions.

## 6. Schedule one historical day per actual day

First complete the manual run and same-ID retry check. The schedule below runs
locally on EC2; the instance MUST remain running. If it is stopped, cron does not
run and this setup does not automatically catch up missed executions.

```bash
cd ~/smart-stock
sudo apt install -y cron util-linux
sudo systemctl enable --now cron
sed -i 's/\r$//' scripts/run_daily_replay.sh
chmod +x scripts/run_daily_replay.sh
mkdir -p logs
```

Optional wrapper smoke test (advances one day on the first invocation per UTC day):

```bash
scripts/run_daily_replay.sh >> logs/daily_replay.log 2>&1
```

The wrapper ID is scheduled-YYYY-MM-DD in UTC. Repeating it on the same UTC date
returns the prior committed result. It uses the existing remote virtual environment
by its full path, so interactive activation is unnecessary for cron.

Inspect the server timezone:

```bash
timedatectl
```

Open the Ubuntu user's crontab:

```bash
crontab -e
```

Add ONE line, using the server's timezone (normally UTC on this Ubuntu setup):

```cron
5 1 * * * /home/ubuntu/smart-stock/scripts/run_daily_replay.sh >> /home/ubuntu/smart-stock/logs/daily_replay.log 2>&1
```

This means 01:05 in the SERVER timezone, not necessarily Stockholm local time.
Do not add the entry repeatedly. Verify and inspect logs:

```bash
crontab -l
tail -n 80 ~/smart-stock/logs/daily_replay.log
```

The schedule is not installed by this package; saving the crontab activates it.
A continuously running EC2 instance consumes credits/can incur compute charges.
For a short demonstration you can execute several manual IDs instead, then stop
EC2. A later AWS-native scheduler can start/stop compute or use an event-based
runner. Do not leave EC2 running solely because cron was configured unless that
cost is intentional.

To disable scheduling: crontab -e, remove the replay line, save. Stop EC2 when no
jobs need to run; keep S3 outputs. Stopping preserves the EBS environment, but the
public IP generally changes when restarted.

## Validation and limits

Tested locally with real model fitting, forecasts and inventory arithmetic and a
mocked S3 transport: seven-day equality with inventory_simulator.py for orders,
receipts, stock, losses and costs; duplicate event no-op; failed commit rollback;
and recovery from a lost checkpoint-success response. Actual AWS access, scheduler
execution and the EC2 environment require your account-side smoke test.

This is a bounded, single-product historical demonstration. The execution ledger
is stored in one checkpoint and grows with replay length; a large production
system should use a dedicated transactional state/event store and a proper model
registry. Do not interpret simulated cost results as live retailer outcomes.

S3 conditional write reference:
https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html
