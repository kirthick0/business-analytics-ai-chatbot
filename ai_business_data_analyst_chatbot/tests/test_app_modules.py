import io
import unittest

import pandas as pd

from modules.chatbot import answer_question
from modules.data_analysis import (
    create_sqlite_connection,
    descriptive_statistics,
    numeric_correlations,
    outlier_summary,
    sqlite_dataset_preview,
)
from modules.data_cleaning import clean_data, quality_report
from modules.data_loader import load_dataset, load_sample_dataset
from modules.machine_learning import (
    segment_customers,
    train_classification,
    train_regression,
)
from modules.report_generator import generate_pdf_report
from modules.visualization import correlation_heatmap, make_chart


class DataAndAnalysisTests(unittest.TestCase):
    def test_sample_loader_and_quality(self):
        frame = load_sample_dataset()
        self.assertEqual(len(frame), 10)
        report = quality_report(frame)
        self.assertEqual(report["missing_cells"], 0)
        self.assertIn("Revenue", frame.columns)

    def test_empty_csv_is_rejected(self):
        source = io.BytesIO(b"")
        source.name = "empty.csv"
        with self.assertRaises(ValueError):
            load_dataset(source)

    def test_duplicate_columns_are_made_unique(self):
        source = io.BytesIO(b"A,A,A\n1,2,3\n")
        source.name = "duplicates.csv"
        frame = load_dataset(source)
        self.assertEqual(len(frame.columns), 3)
        self.assertEqual(len(set(frame.columns)), 3)

    def test_cleaning_preserves_original(self):
        frame = pd.DataFrame({"A": [1.0, None, 3.0], "B": ["x", None, "z"]})
        cleaned = clean_data(frame, remove_duplicates=True, missing_strategy="Fill numeric with median; text with mode")
        self.assertTrue(frame.isna().any().any())
        self.assertFalse(cleaned.isna().any().any())

    def test_descriptive_and_sqlite_helpers(self):
        frame = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1], "C": ["x", "y", "z"]})
        self.assertFalse(descriptive_statistics(frame).empty)
        self.assertFalse(numeric_correlations(frame).empty)
        self.assertFalse(outlier_summary(frame).empty)
        connection = create_sqlite_connection(frame)
        try:
            self.assertEqual(len(sqlite_dataset_preview(connection)), 3)
        finally:
            connection.close()


class AnalystTests(unittest.TestCase):
    def setUp(self):
        self.frame = load_sample_dataset()

    def test_revenue_total(self):
        result = answer_question("What is the total revenue?", self.frame)
        self.assertIn("548,000", result["answer"])
        self.assertEqual(float(result["table"].iloc[0, 0]), 548000)

    def test_product_ranking(self):
        result = answer_question("Which product generates the highest sales?", self.frame)
        self.assertIn("Laptop", result["answer"])
        self.assertEqual(result["table"].iloc[0]["Product"], "Laptop")

    def test_monthly_growth(self):
        result = answer_question("Monthly sales growth", self.frame)
        self.assertEqual(result["kind"], "trend")
        self.assertEqual(result["table"].iloc[-1]["Period"], "2026-03")

    def test_customer_count_and_unique_count(self):
        result = answer_question("How many unique customers are in the dataset?", self.frame)
        self.assertIn("8", result["answer"])
        result = answer_question("How many records are in the dataset?", self.frame)
        self.assertIn("10", result["answer"])


class ModelingAndReportingTests(unittest.TestCase):
    def setUp(self):
        self.frame = load_sample_dataset()

    def test_regression_guards_and_training(self):
        with self.assertRaises(ValueError):
            train_regression(self.frame, "Customer")
        with self.assertRaises(ValueError):
            train_regression(self.frame, "Revenue", "Invalid model")
        outcome = train_regression(self.frame, "Revenue", "Linear Regression")
        self.assertIn("R²", outcome["metrics"])

    def test_classification_training(self):
        outcome = train_classification(self.frame, "Category")
        self.assertIn("Accuracy", outcome["metrics"])

    def test_segmentation_rejects_text_and_handles_numeric(self):
        with self.assertRaises(ValueError):
            segment_customers(self.frame, ["Revenue", "Product"], 2)
        segmented = segment_customers(self.frame, ["Revenue", "Profit"], 2)
        self.assertIn("Customer Segment", segmented.columns)

    def test_chart_and_pdf_generation(self):
        chart = make_chart(self.frame, "Bar chart", "Product", "Revenue")
        self.assertIsNotNone(chart)
        heatmap = correlation_heatmap(self.frame.select_dtypes(include="number").corr())
        self.assertIsNotNone(heatmap)
        pdf = generate_pdf_report("Test Report", ["Finding"], self.frame)
        self.assertTrue(pdf.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
