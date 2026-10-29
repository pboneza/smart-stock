# Evaluation and release evidence

## Current evidence status

The documentation review inspected the public repository and the archived replay, forecasting, replenishment, training, and monitoring implementations. It did not access the AWS account, rerun training, or verify a final inventory evaluation file.

Consequently, this package reports no new model score, inventory savings percentage, or successful AWS deployment claim. Earlier experiment metrics must be tied to their exact model, period, forecasting protocol, and policy settings before publication.

## Forecast evaluation

Evaluate each method on the same target dates and information available at each origin. For the seven-model direct strategy, report results separately for horizons 1–7. Overlapping forecasts are distinct origin–target–horizon observations; specify any aggregation used.

Use chronological splits. Select model hyperparameters on validation data and keep the test period untouched until final evaluation. Record the model hash, training cutoff, split boundaries, feature contract, dependency versions, and any clipping of negative predictions.

Compare Ridge with at least seasonal naive under an identical forecast protocol. Report MAE, RMSE, WAPE, and signed bias, defining bias as prediction minus actual. WAPE is undefined when total actual sales are zero.

Suggested evidence file: `results/forecast_metrics.csv` with columns `model`, `model_version`, `train_end`, `evaluation_start`, `evaluation_end`, `horizon`, `n`, `mae`, `rmse`, `wape_pct`, and `bias`.

## Inventory evaluation

Calibrate each competing policy on validation data under a declared objective. Freeze both policies before testing. Use identical demand, initial stock, pipeline, lead time, pack size, review timing, and cost assumptions. Report any terminal treatment for remaining stock and outstanding orders.

Compare total holding, shortage, and ordering costs, alongside fulfilled units, lost units, unit fill rate, and average closing stock. Service and cost should be assessed together.

Cost reduction (%) = 100 × (baseline total cost − Ridge total cost) / baseline total cost. If baseline cost is zero, the percentage is undefined. Negative reduction means Ridge costs more.

A comparison with the earlier final model cannot be inferred from the validation-candidate replay. Run a matched experiment or explicitly label the replay as integration evidence.

Suggested evidence file: `results/inventory_metrics.csv` with columns `model`, `scenario`, `evaluation_start`, `evaluation_end`, `initial_stock`, `lead_time`, `pack_size`, `safety_stock`, `holding_cost`, `shortage_cost`, `ordering_cost`, `total_cost`, `demand_units`, `fulfilled_units`, `lost_units`, and `fill_rate`.

## Release checklist

- [ ] Add the final model metadata and exact environment versions.
- [ ] Add the prepared-data creation steps and record the daily zero-sales/missing-date convention.
- [ ] Add the validation-selected policy CSV and its selection procedure.
- [ ] Add final forecast and inventory tables with matching experimental settings.
- [ ] Add the simulator and evaluation scripts used to generate those tables.
- [ ] Resolve or remove the empty feature-engineering notebook.
- [ ] Capture a successful AWS forecast and its completion marker.
- [ ] Capture a committed replay day and a same-ID retry no-op.
- [ ] Capture CloudWatch logs, populated metrics, and watchdog operation.
- [ ] State whether alarm notifications were tested or only alarm resources configured.
- [ ] Add screenshots and a recording made from actual outputs.
- [ ] Inspect screenshots and manifests for credentials, account details, and local paths before publication.
- [ ] Choose a software licence and retain dataset attribution.

## Appropriate claims

Supported by the inspected implementation: seven-horizon direct Ridge inference, versioned artifacts, conditional replay commits, retry no-ops, simulated replenishment, and CloudWatch integration code.

Claims requiring execution evidence: a completed scheduled cloud run, visible monitoring telemetry, successful alerts, and the identity of the model actually executed.

Claims requiring matched evaluation: reduced inventory cost, improved service, superior forecast accuracy, or generalisation to additional products.
