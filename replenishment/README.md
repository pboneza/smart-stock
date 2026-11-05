# Daily replenishment recommendations

Place replenish.py in ~/smart-stock/scripts on EC2. Requires boto3 and pandas,
already installed in your training virtual environment. No simulator module needs
to be installed beside it: the recommendation equation is implemented and tested
against inventory_simulator.py.

## Inputs

1. The exact S3 run folder printed by batch_forecast.py. It must contain
   _SUCCESS.json, run_manifest.json, and forecasts.csv. Use the complete folder
   including product, origin, and run ID, not a parent folder.
2. Your frozen selected_policies.csv. The script selects the direct_ridge row
   matching --scenario. Available names in the previously reviewed file are
   lower_shortage, medium_shortage, higher_shortage. Choose the scenario you want
   to demonstrate; the script does not select or retune a scenario.
3. A dated inventory snapshot JSON. The example supplied is a hypothetical
   startup state (28 units, empty pipeline), matching your simulator's earlier
   initialisation. It is NOT an observed current inventory balance.

The snapshot must describe opening stock BEFORE receipts, orders, and demand for
review_date. For a forecast origin of 2011-08-24, review_date is 2011-08-25.
If your current forecast has another origin, change the snapshot review date to
the next day and provide stock/pipeline for that date.

Example pending order schema (include only confirmed, not proposed, orders):

```json
{
  "product_id": "22041",
  "review_date": "2011-08-25",
  "stage": "opening_before_receipts",
  "on_hand": 28,
  "outstanding_orders": [
    {"order_id": "EXAMPLE-001", "arrival_date": "2011-08-25", "quantity": 10},
    {"order_id": "EXAMPLE-002", "arrival_date": "2011-08-27", "quantity": 15}
  ],
  "is_simulated": true
}
```

An order arriving on review_date is received once before calculating inventory
position. Later arrivals remain in the pipeline. Overdue entries are rejected:
resolve actual receipt status instead of dropping old orders silently.

## Transfer from Windows PowerShell

Assuming the downloaded package is extracted into your local project folder as
replenishment, and selected_policies.csv is in results/cost_optimisation:

```powershell
Set-Location C:\Users\patri\smart-stock
$keyPath = "$env:USERPROFILE\Downloads\smart-stock-key.pem"
$ec2Address = "YOUR_CURRENT_PUBLIC_IP"
$ec2Target = "ubuntu@$ec2Address"
ssh -i $keyPath $ec2Target "mkdir -p /home/ubuntu/smart-stock/config"
scp -i $keyPath .\replenishment\replenish.py "${ec2Target}:/home/ubuntu/smart-stock/scripts/"
scp -i $keyPath .\replenishment\inventory_example.json "${ec2Target}:/home/ubuntu/smart-stock/config/"
scp -i $keyPath .\results\cost_optimisation\selected_policies.csv "${ec2Target}:/home/ubuntu/smart-stock/config/"
```

Adjust the local selected_policies.csv path to its real location. You can also
obtain the file from the saved policy results. Do not substitute another CSV.

## Execute in Ubuntu SSH

```bash
cd ~/smart-stock
source .venv/bin/activate
python scripts/replenish.py \
  --forecast-run 's3://smart-stock-s3-bucket/forecasts/product=22041/origin=2011-08-24/run=REPLACE_WITH_ACTUAL_RUN_ID/' \
  --inventory config/inventory_example.json \
  --selected-policies config/selected_policies.csv \
  --scenario medium_shortage \
  --product-id 22041 \
  --lead-time 3 \
  --pack-size 1 \
  --output-s3 s3://smart-stock-s3-bucket/replenishment
```

The scenario in this command is an explicit example, not a newly selected optimum.
Lead time and pack size should match the configuration used to select the policy;
the selected-policy CSV alone does not store these settings. Forecasts from the
current May-trained validation model demonstrate integration; they do not establish
that the prior inventory-cost result is reproduced with this particular model.

## Decision rule

At review t, origin is end of t-1. Protection covers t through t+L (L+1 days).
For L=3, use forecast horizons 1..4, not all seven forecasts.

inventory position = opening stock + today's receipts + remaining confirmed pipeline

order-up-to target = sum of the first L+1 forecasts + selected safety stock

recommended quantity = pack_size * ceil(max(0, target - inventory position) / pack_size)

An accepted order placed on t arrives at the START of t+L. All outstanding
quantities count in inventory position, even if outside the protection window,
matching the existing simulator rather than introducing a new policy.

## Outputs and next day

Each run writes recommendation.json, inventory_input.json,
inventory_review_state.json, run_manifest.json, and _SUCCESS.json locally and to
S3. inventory_review_state is a derived after-receipts, BEFORE-ORDER state for audit;
it is not automatically saved back into the source inventory system.

No order is submitted and no next-day inventory is fabricated. A recommendation
is NOT an outstanding order. On the next day, provide updated opening stock after
actual/simulated demand and include only accepted outstanding orders. Never reuse
the startup example unchanged to represent a continuous simulation.

Reruns generate separate evidence folders. decision_key identifies product/date;
recommendation_id identifies the inputs. The script does NOT provide transactional
order execution or prevent double submission to a supplier. A later execution
layer must enforce one committed decision per decision_key and record confirmed
orders. Use only folders containing _SUCCESS.json.

Next automation step: supply daily inventory snapshots, chain successful forecasting
to this recommendation step, then add scheduling, failure alerts and monitoring.
Until that state update exists, this is a manual recommendation demonstration.
