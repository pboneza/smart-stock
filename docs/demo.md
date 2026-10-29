# Four-minute portfolio demonstration

## Aim

Show the complete path from observed sales to a committed simulated inventory decision, including a safe retry and operational monitoring. Use real artifacts from the dedicated demo replay. The demonstration is a historical simulation, with no supplier orders or live demand feed.

## Recording sequence

| Time | Screen | Narration |
|---|---|---|
| 0:00–0:30 | README business question and architecture | “Smart Stock examines whether retail sales forecasts can support replenishment decisions that balance lost sales and excess inventory. I evaluate forecast error and inventory costs separately.” |
| 0:30–1:00 | Model metadata and prepared daily sales | “The workflow begins with one product. Seven direct Ridge models forecast the next seven days from information available at the forecast origin. This replay uses a frozen validation candidate.” |
| 1:00–1:40 | One monitored replay execution and structured output | “One invocation advances one historical day. It forecasts from yesterday's information, receives due stock, calculates an order, and then reveals today's historical demand.” |
| 1:40–2:10 | S3 forecast manifest, completion marker, and checkpoint | “Each forecast has versioned evidence. The authoritative inventory transition is a conditional checkpoint commit containing the execution ledger and next opening state.” |
| 2:10–2:35 | Repeat the same execution ID | “Repeating this ID returns an already-committed result. Inventory and simulated orders do not advance again.” |
| 2:35–3:15 | CloudWatch dashboard and structured logs | “Monitoring separates workflow health from business outcomes. Forecast errors are scored by horizon only after the corresponding target sales become available.” |
| 3:15–4:00 | Matched evaluation table, if verified; otherwise limitations and next steps | “Inventory findings depend on assumed costs, lead time, and the use of sales as a demand proxy. The next extension is a matched multi-product evaluation.” |

If no verified final comparison is available, state that the recording demonstrates integration and reliability. Do not replace a missing evaluation table with example savings.

## Before recording

1. Use a separate, fully configured demo namespace and the validated environment.
2. Confirm the model metadata, policy configuration, and dataset dates.
3. Run monitoring probe and watchdog checks.
4. Prepare a unique execution ID. Record it so the retry uses exactly the same ID.
5. Check that the chosen day is within the dataset and after the model training cutoff.
6. Open the dashboard, log group, and relevant S3 prefixes in advance.
7. Allow CloudWatch time to display metrics; use previously populated panels if ingestion is delayed, identifying their execution date.
8. Present only the required application output and evidence. Keep credentials and personal account details outside the recording.

Use the commands in [reproduction](reproduction.md). A new execution ID changes simulation state, so do not repeatedly issue new IDs just to improve a recording take. Reuse the same ID to demonstrate recovery, or create a separately configured experiment.

## Evidence to include with the video

- A forecast CSV with seven horizon rows and its manifest/completion marker.
- A committed `day_result.json` referenced by the checkpoint.
- A same-ID retry result.
- A dashboard screenshot and a structured log containing the historical date.
- Final evaluation tables only when generated under the documented matched protocol.

Suggested filenames are `docs/assets/architecture.png`, `docs/assets/cloudwatch-dashboard.png`, and `docs/assets/retry-no-op.png` after actual captures exist. This package intentionally contains no fabricated screenshots or demonstration outputs.

## Questions the demonstration should answer

**Why batch inference?** The decision is reviewed daily, so a scheduled forecast is sufficient and an always-running endpoint is unnecessary for this implementation.

**Why forecast daily with a seven-day horizon?** Each new observation updates the forecast. Forecast origin, target date, and horizon keep overlapping estimates distinct.

**Why can a better forecast yield a worse inventory result?** Policy settings, cost asymmetry, and lead time affect decisions. Forecast and policy performance require separate evaluation.

**How are repeated orders avoided?** The replay stores the execution ID and inventory transition in one conditional checkpoint. This protects simulated replay decisions; real supplier integration would need its own execution controls.

**What happens if monitoring fails?** A committed inventory day remains committed. The durable checkpoint is checked before retrying with the same execution ID.
