# Architecture and decision timing

## Implemented path

EC2 hosts the validated Python environment. S3 stores the trusted model, forecast artifacts, replenishment evidence, and a persistent replay checkpoint. Cron triggers the monitored wrapper; CloudWatch receives operational telemetry. The demonstrated inference path does not require a SageMaker endpoint.

SageMaker training was explored separately. Its training launcher is not evidence that a training job completed. Identify the actual source of the demonstrated model from its artifact metadata and execution records before describing training in the published portfolio.

## One historical day

1. Read the S3 checkpoint and detect whether the execution ID was already committed.
2. Read opening simulated inventory and derive the next historical review date.
3. Load the frozen model and sales through the preceding day; generate seven direct forecasts.
4. Receive orders due today and calculate the replenishment quantity using the configured protection period.
5. Record the accepted simulated order with its future arrival date.
6. Reveal today's historical demand, fulfil what is available, and calculate lost units and costs.
7. Write day evidence, then conditionally commit the next opening state and execution ledger.
8. Report the outcome to CloudWatch and score forecasts whose target day has now been revealed.

Simulation date and real UTC execution time are recorded separately. Multiple historical days can be replayed during a short demonstration without suggesting that historical data are live transactions.

## Two levels of completion

| Evidence | Meaning |
|---|---|
| Forecast `_SUCCESS.json` | Forecast artifact upload completed |
| Recommendation `_SUCCESS.json` | Recommendation artifact upload completed |
| Checkpoint `committed_executions` entry | Historical inventory day committed |
| CloudWatch sample | Telemetry was received; it is not a transaction receipt |

Prepared day artifacts can exist after failed or concurrent attempts. Aggregate only results referenced in the checkpoint ledger.

## Recovery behaviour

A repeated committed execution ID does not advance inventory. A conditional S3 write rejects a stale checkpoint update, while the wrapper lock prevents overlapping local scheduled invocations. These controls protect the replay's state transition; they do not implement supplier-side order execution.

A monitoring failure after a successful inventory commit does not roll back inventory. Exit code 2 indicates a committed or previously committed replay with incomplete telemetry. Retry using the same execution ID. Retry no-ops intentionally avoid republishing inventory and forecast-error samples; missing telemetry is not automatically backfilled.

The checkpoint ledger grows with replay length. Larger deployments need a dedicated transactional event/state store and more complete orchestration.

## Monitoring interpretation

Metrics use real reporting timestamps. Historical dates are retained in structured logs. The dashboard includes execution health, duration, stock, orders, demand fulfilment, lost sales, simulated cost, freshness, and MAE/bias by horizon.

Longer-horizon error metrics appear only as their target days mature. The four configured alarm types cover run failure, telemetry failure, no progress for 30 hours, and missing watchdog heartbeats. Email notification requires a separately configured SNS topic and confirmed subscription.

For exact historical day totals, use committed S3 results. Several accelerated replay days can occupy one CloudWatch chart bucket, so dashboard values should not be treated as a complete evaluation table.
