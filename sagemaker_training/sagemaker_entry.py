"""SageMaker custom entrypoint; model logic remains in the validated train.py."""
import argparse
import json
from pathlib import Path
import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--product-id', required=True)
    parser.add_argument('--train-end', required=True)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--date-col', default='forecast_date')
    parser.add_argument('--sales-col', default='units_sold')
    a = parser.parse_args()
    # Explicit platform paths: custom entrypoints do not depend on toolkit SM_* setup.
    report = train.train(Path('/opt/ml/input/data/train/sales.csv'),
                         Path('/opt/ml/model'), a.product_id, a.train_end,
                         a.alpha, a.date_col, a.sales_col)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
