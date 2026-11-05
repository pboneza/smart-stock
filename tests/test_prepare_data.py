"""Regression coverage for transaction exclusions and daily aggregation."""
import unittest

import pandas as pd

from scripts.prepare_data import clean_transactions, aggregate_daily_sales, NON_PRODUCT_CODES


class CleaningTests(unittest.TestCase):
    def test_rules_and_sequential_counts(self):
        base = dict(Invoice="100", StockCode="22041", InvoiceDate="2011-01-01 12:00",
                    Quantity=2, Price=1, Description=None, **{"Customer ID": None})
        rows = [base.copy(), base.copy()]
        changes = [
            dict(Invoice=None), dict(StockCode=" "), dict(InvoiceDate="bad"),
            dict(Quantity="bad"), dict(Price=float("inf")),
            dict(Invoice=" c123 "), dict(Quantity=-1), dict(Quantity=0),
            dict(Price=0), dict(Price=-1),
            dict(Invoice="581483", StockCode="23843"),
            dict(Invoice="541431", StockCode="23166"),
        ]
        changes += [dict(StockCode=f" {code.lower()} ") for code in NON_PRODUCT_CODES]
        changes += [dict(StockCode="gift_0001_10"), dict(StockCode="test001")]
        changes += [dict(Invoice=" 101 ", StockCode=" 22041 ", Quantity=100000,
                         InvoiceDate="2011-01-03 12:00")]
        rows += [dict(base, **change) for change in changes]
        clean, report = clean_transactions(pd.DataFrame(rows))
        self.assertEqual(clean.Quantity.tolist(), [2, 100000])
        self.assertTrue(clean.StockCode.eq("22041").all())
        self.assertTrue(clean["Customer ID"].isna().all())
        self.assertTrue(clean.Description.isna().all())
        counts = report["row_counts"]
        self.assertEqual(counts["input_rows"], counts["retained_rows"] +
                         sum(v for k, v in counts.items() if k.endswith("_removed")))
        self.assertEqual(counts["invalid_identifier_rows_removed"], 2)
        self.assertEqual(counts["invalid_date_rows_removed"], 1)
        self.assertEqual(counts["invalid_numeric_rows_removed"], 2)
        daily = aggregate_daily_sales(clean, "22041")
        self.assertEqual(daily.units_sold.tolist(), [2, 0, 100000])

    def test_duplicate_detection_precedes_normalization(self):
        rows = pd.DataFrame(dict(Invoice=["1", " 1 ", "1"], StockCode=["22041"] * 3,
                                 InvoiceDate=["2011-01-01"] * 3,
                                 Quantity=["2", "2", "2.0"], Price=[1] * 3))
        clean, report = clean_transactions(rows)
        self.assertEqual(len(clean), 3)
        self.assertEqual(report["row_counts"]["exact_duplicates_removed"], 0)

    def test_schema_and_empty_result_fail(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            clean_transactions(pd.DataFrame({"Invoice": ["1"]}))
        with self.assertRaisesRegex(ValueError, "No transactions remain"):
            clean_transactions(pd.DataFrame(dict(Invoice=["C1"], StockCode=["22041"],
                                                InvoiceDate=["2011-01-01"], Quantity=[-1], Price=[1])))


if __name__ == "__main__":
    unittest.main()
