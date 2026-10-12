# Smart Stock

An end-to-end AWS machine learning and decision analysis project that uses historical retail transactions to forecast daily sales as a proxy for demand and support inventory replenishment decisions.

## Objective

Evaluate whether forecasts and their uncertainty can support inventory policies that reduce total holding, shortage, and ordering costs compared with baseline approaches.

Forecast accuracy and inventory performance will be evaluated separately: a more accurate forecast does not necessarily produce a lower-cost inventory decision.

## Project scope

Development follows a single-product-first approach:

1. Develop and validate the complete data preparation, forecasting, inventory simulation, and AWS workflow for one product.
2. Apply the validated workflow to a small portfolio of products, initially making replenishment decisions independently for each product.
3. Evaluate performance for individual products and aggregate inventory costs across the portfolio.

Shared purchasing budgets, storage constraints, and joint inventory optimisation are later extensions.

### Initial inventory setting

- One stock location.
- Daily sales aggregation and daily inventory review.
- Next-day forecasting initially, extending to demand over the replenishment lead time.
- A fixed, explicitly assumed replenishment lead time.
- Inventory decisions that account for available stock and outstanding orders.
- Unfulfilled demand treated as lost sales initially.
- Explicit holding, shortage, and ordering cost assumptions, evaluated through sensitivity analysis.

The exact product, lead time, cost values, and inventory policy will be selected and documented during development.

## Data

The project uses the UCI Online Retail dataset.

Initial exploration examines sales frequency, history length, missing values, cancellations, returns, and unusual transactions. Product selection follows this exploration, beginning with a product that has sufficient history and reasonably regular sales.

The preparation workflow will create a daily product-level series and document how days without recorded sales are handled.

### Interpretation and limitations

Observed sales may differ from underlying demand when products are unavailable. Sales will therefore be used as a proxy for demand, with this limitation stated explicitly.

Historical stock levels, stockouts, replenishment records, and inventory costs will not be assumed to be available. Missing operational information will be represented through documented simulation assumptions. Inventory results will describe performance under those assumptions rather than verified savings in an actual retail operation.

## Development plan

| Phase | Main activities | Expected outcome |
|---|---|---|
| 1. Problem definition and data exploration | Define the decision setting, inspect transactions, and select one initial product | Documented scope and data preparation rules |
| 2. Daily dataset and baseline forecasts | Aggregate sales, create features, and establish simple forecasting benchmarks | Reproducible daily dataset and baseline results |
| 3. Forecasting and uncertainty | Develop models and assess demand uncertainty using chronological validation | Forecasts and uncertainty estimates for inventory decisions |
| 4. Inventory simulation | Simulate replenishment and compare inventory policies under explicit assumptions | Cost and service evaluation against baseline policies |
| 5. AWS lifecycle | Implement data preparation, training, model storage, batch inference, and monitoring on AWS | Reproducible end-to-end cloud workflow |
| 6. Product portfolio evaluation | Apply the workflow to additional products and aggregate results | Product-level findings and combined inventory performance |

AWS implementation will be introduced incrementally as the underlying workflow is validated. Service choices and deployment details will be documented as they are implemented.

## Evaluation
The project will be regularly evaluated across the two defined objectives: Forecasting and Inventory optimization decisions.

### Forecasting

- Use chronological training, validation, and test periods.
- Prevent future information from entering features or model selection.
- Compare models with simple forecasting baselines.
- Assess forecast error and the quality of uncertainty estimates.

### Inventory decisions

- Compare forecast-based policies with baseline inventory policies using the same products, evaluation periods, and simulation assumptions.
- Report holding, shortage, ordering, and total costs.
- Report service outcomes, including the proportion of demand fulfilled.
- Examine sensitivity to cost assumptions and replenishment lead times.

For multiple products, report both individual results and combined costs. Portfolio fulfilment will be calculated from total fulfilled units divided by total simulated demanded units, rather than an unweighted average of product percentages.
