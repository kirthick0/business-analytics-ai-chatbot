"""Integration checks for the sample dataset and core analyst components."""

from __future__ import annotations

import unittest

from modules.chatbot import answer_question
from modules.data_analysis import create_sqlite_connection, numeric_correlations, sqlite_dataset_preview
from modules.data_cleaning import clean_data, quality_report
from modules.data_loader import load_sample_dataset
from modules.machine_learning import segment_customers, train_classification, train_regression
from modules.report_generator import generate_pdf_report
from modules.visualization import make_chart
from modules.business_insights import generate_business_insights


class AnalystModulesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load_sample_dataset()

    def test_sample_dashboard_facts(self) -> None:
        quality = quality_report(self.frame)
        self.assertEqual((quality["rows"], quality["columns"]), (10, 10))
        self.assertEqual(quality["missing_cells"], 0)
        self.assertEqual(int(self.frame["Revenue"].sum()), 548000)

    def test_chat_answers_calculated_values(self) -> None:
        total = answer_question("What is the total revenue?", self.frame)
        self.assertIn("548,000", total["answer"])
        product = answer_question("Which product generates the highest sales?", self.frame)
        self.assertIn("Laptop", product["answer"])
        self.assertEqual(product["table"].iloc[0]["Product"], "Laptop")
        region = answer_question("Which region has the highest profit?", self.frame)
        self.assertEqual(region["table"].iloc[0]["Region"], "South")
        customers = answer_question("Show the top 10 customers", self.frame)
        self.assertEqual(customers["table"].iloc[0]["Customer"], "Customer A")
        self.assertIn("Product with the highest summed Revenue: Laptop", " ".join(generate_business_insights(self.frame)))

    def test_monthly_trend_has_computed_growth(self) -> None:
        result = answer_question("What is the monthly sales growth?", self.frame)
        self.assertEqual(result["table"]["Period"].tolist(), ["2026-01", "2026-02", "2026-03"])
        self.assertEqual(int(result["table"]["Revenue"].iloc[0]), 214000)
        self.assertAlmostEqual(float(result["table"]["Growth %"].iloc[1]), -29.44, places=2)

    def test_unsupported_metric_is_explained(self) -> None:
        response = answer_question("What is the total advertising spend?", self.frame)
        self.assertIn("could not identify", response["answer"])
        self.assertIsNone(response["table"])

    def test_sqlite_analysis_and_visualization(self) -> None:
        connection = create_sqlite_connection(self.frame)
        try:
            self.assertEqual(len(sqlite_dataset_preview(connection)), 10)
        finally:
            connection.close()
        self.assertFalse(numeric_correlations(self.frame).empty)
        chart = make_chart(self.frame, "Histogram", "Revenue")
        self.assertIsNotNone(chart)
        self.assertTrue(generate_business_insights(self.frame))

    def test_cleaning_preserves_source_and_pdf_builds(self) -> None:
        frame = self.frame.copy()
        frame.loc[0, "Profit"] = None
        original_missing = int(frame.isna().sum().sum())
        cleaned = clean_data(frame, missing_strategy="Fill numeric with median; text with mode")
        self.assertEqual(int(frame.isna().sum().sum()), original_missing)
        self.assertEqual(int(cleaned.isna().sum().sum()), 0)
        pdf = generate_pdf_report("Test report", generate_business_insights(self.frame), self.frame)
        self.assertTrue(pdf.startswith(b"%PDF"))

    def test_machine_learning_pipelines(self) -> None:
        import pandas as pd

        frame = pd.DataFrame({
            "spend": list(range(1, 31)),
            "visits": [value * 2 for value in range(1, 31)],
            "channel": ["store", "web"] * 15,
            "Revenue": [value * 125 for value in range(1, 31)],
            "Group": ["A", "B"] * 15,
        })
        regression = train_regression(frame, "Revenue", "Linear Regression")
        classification = train_classification(frame, "Group")
        segmented = segment_customers(frame, ["spend", "visits"], clusters=3)
        self.assertIn("RMSE", regression["metrics"])
        self.assertIn("Accuracy", classification["metrics"])
        self.assertEqual(segmented["Customer Segment"].nunique(), 3)


if __name__ == "__main__":
    unittest.main()
