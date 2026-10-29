# Reproduction guide

## Scope and required inputs

The checked-in scripts support training from prepared daily sales, batch inference, replenishment, replay, and monitoring. Full raw-data-to-final-results reproduction still requires the project's final preparation, policy-selection, simulator, and evaluation artifacts; see [results](results.md).

Use a Python environment matching the scikit-learn version recorded in the trusted model metadata. The scripts use boto3, numpy, pandas, scikit-learn, and joblib. A new dependency lock file should be exported from the environment that actually produced the model; this documentation does not invent pinned versions.

Required local inputs:

- `data/processed/product_22041_daily_sales.csv`, with `forecast_date` and `units_sold`.
- `config/selected_policies.csv`, from validation selection. The replenishment reader requires `name`, `policy`, and `safety_stock`, and exactly one `direct_ridge` row for the configured scenario. Do not substitute fabricated values for a selected policy.
- A trusted model artifact in S3 and permissions to read it and write workflow outputs.
- AWS authentication through the configured environment or EC2 instance role.

Prepared daily sales must have unique, consecutive calendar dates and finite, nonnegative values. Batch forecasting requires at least 28 observed days through its origin. Explicitly document upstream missing-day handling.

## Train a validation candidate

From the project root in the validated environment:

```bash
python scripts/train.py --data data/processed/product_22041_daily_sales.csv --product-id 22041 --train-end 2011-05-09 --alpha 0.01 --output models/ridge_validation
```

This fits the example validation candidate. It does not tune alpha or reproduce a later final test model. Upload the resulting trusted `direct_ridge.joblib` to the S3 object used by your configuration.

## Publish a forecast

Replace `YOUR_BUCKET` with the configured bucket:

```bash
python scripts/batch_forecast.py \
  --model-s3 s3://YOUR_BUCKET/models/ridge_validation/direct_ridge.joblib \
  --data data/processed/product_22041_daily_sales.csv \
  --origin 2011-08-24 --product-id 22041 \
  --output-s3 s3://YOUR_BUCKET/forecasts --region eu-north-1
```

The origin is the last fully observed sales day. Use the complete run URI returned by the command when inspecting evidence. Consumers must ignore folders lacking `_SUCCESS.json`.

## Configure a separate demo replay

Copy `config/replay_example.json` to `config/replay.json`. Replace `YOUR_BUCKET`, verify the model and data paths, and use a fresh state/output namespace for a new experiment. `config/inventory_example.json` supplies simulated opening stock, not observed retailer stock.

Keep the selected scenario, safety stock, lead time, pack size, and unit costs consistent with policy calibration. Freeze the dataset, model, selected policies, initial inventory, and replay configuration once replay begins: the implementation fingerprints these inputs and rejects changes.

```bash
python scripts/daily_replay.py --config config/replay.json --execution-id portfolio-001
```

Repeat the identical command to verify `replay_already_committed`. Use `portfolio-002` only when you intend to advance another historical day. Do not run different IDs concurrently.

## Set up monitoring

Using an AWS identity permitted to create monitoring resources:

```bash
python scripts/setup_cloudwatch.py --region eu-north-1 --workflow-id demo-v1 --product-id 22041
```

Setup creates a log group, dashboard, four alarms, and `cloudwatch_generated/monitoring.json` plus an EC2 publishing policy. Attach that generated policy to the actual EC2 runtime role using your existing IAM process. Place the monitoring configuration at `config/monitoring.json`. Setup administration and runtime publishing use different permissions.

A resource-creation success does not establish that runtime publishing works. Verify both:

```bash
python scripts/cloudwatch_replay.py --config config/replay.json --monitor-config config/monitoring.json --mode probe
python scripts/cloudwatch_replay.py --config config/replay.json --monitor-config config/monitoring.json --mode run --execution-id portfolio-monitored-001
python scripts/cloudwatch_replay.py --config config/replay.json --monitor-config config/monitoring.json --mode watchdog
```

The probe and watchdog do not advance inventory. A new run execution ID does. Monitoring configuration is separate from the frozen replay configuration.

## Schedule on EC2

The wrappers currently assume `/home/ubuntu/smart-stock` and its `.venv`; inspect these paths before installation. Set executable permissions and create the `logs/` directory. Install one daily monitored entry and one hourly health entry, removing any earlier unmonitored daily entry:

```cron
5 1 * * * /home/ubuntu/smart-stock/scripts/run_monitored_replay.sh >> /home/ubuntu/smart-stock/logs/daily_replay.log 2>&1
15 * * * * /home/ubuntu/smart-stock/scripts/check_replay_health.sh >> /home/ubuntu/smart-stock/logs/replay_health.log 2>&1
```

Cron follows the server timezone; execution IDs use UTC dates. EC2 must remain running, and missed runs are not automatically caught up. Confirm `crontab -l`, inspect logs, and inspect the checkpoint after a scheduled run.

## End the demonstration

Remove the demo cron entries, disable or delete its alarms as appropriate, and stop EC2 when no other work requires it. Retain the evidence needed for the portfolio. Monitoring and compute resources may remain billable while configured or running; stopping the instance does not delete its storage or CloudWatch resources.
