"""AI Business Data Analyst Chatbot — Streamlit application entry point."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.business_insights import generate_business_insights
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
from modules.machine_learning import train_classification, train_regression, segment_customers
from modules.report_generator import generate_pdf_report
from modules.visualization import correlation_heatmap, make_chart


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="AI Business Data Analyst", page_icon="📊", layout="wide")


def apply_styles(theme: str) -> None:
    css_path = ROOT / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    if theme == "Dark":
        st.markdown("<style>.stApp,.stApp label,.stApp [data-testid='stMarkdownContainer']{background:#101820;color:#edf3f7}.hero{background:linear-gradient(115deg,#183d3a,#202e50)}[data-testid='stSidebar']{background:#111c25;color:#edf3f7}[data-testid='stAlert']{background:#1d3448;color:#edf3f7}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>.stApp,.stApp label,.stApp [data-testid='stMarkdownContainer']{background:#f7f9fc;color:#182735}.hero{background:linear-gradient(115deg,#e8fbf5,#eef1ff)}[data-testid='stSidebar']{background:#f1f5f9;color:#182735}[data-testid='stAlert']{background:#eaf3ff;color:#173b57}</style>", unsafe_allow_html=True)


def initialise_state() -> None:
    defaults = {"data": None, "filename": None, "db": None, "chat_history": [], "chat_result": None,
                "cleaned_data": None, "analysis_result": None, "theme": "Light", "upload_id": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_dataset(frame: pd.DataFrame, filename: str) -> None:
    old_db = st.session_state.get("db")
    if old_db is not None:
        try:
            old_db.close()
        except Exception:
            pass
    st.session_state.update(data=frame, filename=filename, db=create_sqlite_connection(frame),
                            chat_history=[], chat_result=None, cleaned_data=None, analysis_result=None)


def metric_value(frame: pd.DataFrame, candidates: tuple[str, ...]) -> float | None:
    for column in frame.select_dtypes(include="number").columns:
        if any(word in str(column).lower() for word in candidates):
            return float(frame[column].sum())
    return None


def render_header() -> None:
    st.markdown("""<div class="hero"><div class="hero-kicker">Business intelligence workspace</div>
    <div class="hero-title">AI Business Data Analyst</div>
    <p class="hero-copy">Turn your business data into clear, evidence-based decisions — no API key required.</p></div>""", unsafe_allow_html=True)


def dashboard(frame: pd.DataFrame) -> None:
    report = quality_report(frame)
    revenue = metric_value(frame, ("revenue", "sales", "turnover"))
    profit = metric_value(frame, ("profit", "earnings"))
    metrics = st.columns(5)
    metrics[0].metric("Total records", f"{len(frame):,}")
    metrics[1].metric("Total columns", f"{frame.shape[1]:,}")
    metrics[2].metric("Missing cells", f"{report['missing_cells']:,}")
    metrics[3].metric("Duplicate rows", f"{report['duplicate_rows']:,}")
    metrics[4].metric("Total revenue / sales", f"{revenue:,.2f}" if revenue is not None else "Not detected")
    if profit is not None:
        st.caption(f"Observed total profit: {profit:,.2f}  ·  Dataset memory: {report['memory_mb']:.2f} MB")
    else:
        st.caption(f"Dataset memory: {report['memory_mb']:.2f} MB  ·  Upload a dataset with a Profit column to see profit totals.")
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Dataset preview")
        st.dataframe(frame.head(8), width="stretch", hide_index=True)
    with right:
        st.subheader("Column profile")
        st.dataframe(report["column_details"], width="stretch", hide_index=True, height=310)
    findings = generate_business_insights(frame)
    st.subheader("Business snapshot")
    for finding in findings[:5]:
        st.markdown(f"- {finding}")


def data_explorer(frame: pd.DataFrame) -> None:
    st.subheader("Explore and clean your data")
    st.caption("The uploaded dataset remains unchanged unless you apply the selected cleaning steps.")
    preview_tab, clean_tab, sqlite_tab = st.tabs(["Data preview", "Data cleaning", "SQLite snapshot"])
    with preview_tab:
        st.dataframe(frame, width="stretch", height=420)
        st.download_button("Download original dataset as CSV", frame.to_csv(index=False).encode("utf-8"),
                           "uploaded_dataset.csv", "text/csv")
        st.write("Data types")
        st.dataframe(frame.dtypes.astype(str).rename("Data type").to_frame(), width="stretch")
    with clean_tab:
        st.write("Current data quality")
        st.dataframe(quality_report(frame)["column_details"], width="stretch", hide_index=True)
        remove_duplicates = st.checkbox("Remove exact duplicate rows", value=False)
        strategy = st.selectbox("Missing-value handling", ["Keep missing values", "Drop rows with missing values",
                                                             "Fill numeric with median; text with mode"])
        if st.button("Apply cleaning to an analysis copy", type="primary"):
            st.session_state.cleaned_data = clean_data(frame, remove_duplicates=remove_duplicates,
                                                       missing_strategy=strategy)
        cleaned = st.session_state.cleaned_data
        if cleaned is not None:
            st.success(f"Analysis copy ready: {len(cleaned):,} rows ({len(frame) - len(cleaned):,} rows removed). Original data is preserved.")
            st.dataframe(cleaned.head(20), width="stretch")
            st.download_button("Download cleaned copy", cleaned.to_csv(index=False).encode("utf-8"),
                               "cleaned_dataset.csv", "text/csv")
    with sqlite_tab:
        st.caption("For this session, the current dataset is also held in an in-memory SQLite table named uploaded_data.")
        st.dataframe(sqlite_dataset_preview(st.session_state.db), width="stretch", hide_index=True)


def analyst(frame: pd.DataFrame) -> None:
    st.subheader("Ask your data analyst")
    st.markdown("Try: **What is the total revenue?** · **Which product has the highest sales?** · **Top 10 customers by revenue** · **Monthly sales growth**")
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("table") is not None:
                st.dataframe(item["table"], width="stretch", hide_index=True)
                if item.get("chart") is not None:
                    st.plotly_chart(item["chart"], width="stretch")
    question = st.chat_input("Ask a business question about the uploaded data")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        result = answer_question(question, frame)
        chart = None
        table = result.get("table")
        if table is not None and not table.empty and result.get("kind") in {"bar", "trend"}:
            chart = make_chart(table, "Line chart" if result["kind"] == "trend" else "Bar chart",
                               result.get("x", table.columns[0]), result.get("y", table.columns[-1]))
        st.session_state.chat_history.append({"role": "assistant", "content": result["answer"], "table": table, "chart": chart})
        st.session_state.chat_result = result
        st.rerun()
    if st.session_state.chat_result and st.session_state.chat_result.get("table") is not None:
        table = st.session_state.chat_result["table"]
        st.download_button("Download latest answer as CSV", table.to_csv(index=False).encode("utf-8"),
                           "chat_analysis.csv", "text/csv")


def automated_analysis(frame: pd.DataFrame) -> None:
    st.subheader("Automated data analysis")
    overview, correlations, outliers = st.tabs(["Descriptive statistics", "Correlation", "Outliers"])
    with overview:
        st.dataframe(descriptive_statistics(frame), width="stretch", hide_index=True)
    with correlations:
        corr = numeric_correlations(frame)
        if corr.empty:
            st.info("At least two numeric columns are required for correlation analysis.")
        else:
            st.plotly_chart(correlation_heatmap(corr), width="stretch")
            st.dataframe(corr, width="stretch")
            st.download_button("Download correlation matrix", corr.to_csv().encode("utf-8"), "correlations.csv", "text/csv")
    with outliers:
        summary = outlier_summary(frame)
        if summary.empty:
            st.info("No numeric columns were detected for IQR outlier analysis.")
        else:
            st.caption("IQR rule: values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR are flagged for review, not automatically removed.")
            st.dataframe(summary, width="stretch", hide_index=True)


def visualizations(frame: pd.DataFrame) -> None:
    st.subheader("Interactive visualizations")
    categorical = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    numeric = frame.select_dtypes(include="number").columns.tolist()
    if not numeric:
        st.info("Add or upload numeric columns to build business charts.")
        return
    chart_type = st.selectbox("Chart type", ["Bar chart", "Line chart", "Pie chart", "Scatter plot", "Histogram"])
    if chart_type == "Histogram":
        x = st.selectbox("Numeric column", numeric)
        figure = make_chart(frame, chart_type, x)
    else:
        x_options = categorical + numeric
        x = st.selectbox("X axis / category", x_options)
        y = st.selectbox("Measure", numeric, index=0)
        if chart_type == "Pie chart" and x == y:
            st.info("Choose a category for the pie chart labels.")
            return
        if chart_type in {"Bar chart", "Pie chart"} and x in numeric:
            st.info("Choose a categorical column for bar or pie chart categories.")
            return
        figure = make_chart(frame, chart_type, x, y)
    if figure is not None:
        st.plotly_chart(figure, width="stretch")
        st.download_button("Download chart as interactive HTML", figure.to_html(include_plotlyjs="cdn").encode("utf-8"),
                           "business_chart.html", "text/html")


def machine_learning(frame: pd.DataFrame) -> None:
    st.subheader("Optional machine-learning lab")
    st.caption("Models are trained only when the selected columns and row counts are suitable. Use results for exploration, not as guaranteed forecasts.")
    regression, classification, segmentation = st.tabs(["Regression / sales prediction", "Classification", "Customer segmentation"])
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    with regression:
        if len(numeric) < 2:
            st.info("Regression needs a numeric target and at least one usable feature column.")
        else:
            target = st.selectbox("Numeric prediction target", numeric, key="reg_target")
            model_name = st.selectbox("Regression model", ["Random Forest", "Linear Regression"], key="reg_model")
            if st.button("Train regression model"):
                try:
                    outcome = train_regression(frame, target, model_name)
                    st.session_state.analysis_result = outcome["predictions"]
                    cols = st.columns(len(outcome["metrics"]))
                    for column, (name, value) in zip(cols, outcome["metrics"].items()):
                        column.metric(name, f"{value:,.4f}")
                    st.dataframe(outcome["predictions"], width="stretch")
                except (ValueError, TypeError) as exc:
                    st.error(str(exc))
    with classification:
        possible_targets = [c for c in categorical if 2 <= frame[c].nunique(dropna=True) <= 30]
        if not possible_targets or not numeric:
            st.info("Classification needs a categorical target (2–30 classes) and at least one feature column.")
        else:
            target = st.selectbox("Category to predict", possible_targets, key="class_target")
            if st.button("Train classifier"):
                try:
                    outcome = train_classification(frame, target)
                    st.session_state.analysis_result = outcome["predictions"]
                    st.metric("Holdout accuracy", f"{outcome['metrics']['Accuracy']:.1%}")
                    st.dataframe(outcome["predictions"], width="stretch")
                except (ValueError, TypeError) as exc:
                    st.error(str(exc))
    with segmentation:
        if len(numeric) < 2 or len(frame) < 4:
            st.info("K-Means segmentation needs at least two numeric columns and four rows.")
        else:
            features = st.multiselect("Customer behavior features", numeric, default=numeric[:min(3, len(numeric))])
            maximum_clusters = min(10, len(frame) - 1)
            count = st.slider("Number of segments", 2, maximum_clusters, min(3, maximum_clusters))
            if st.button("Create customer segments"):
                try:
                    segmented = segment_customers(frame, features, count)
                    st.session_state.analysis_result = segmented
                    st.dataframe(segmented, width="stretch")
                    st.bar_chart(segmented["Customer Segment"].value_counts().sort_index())
                    st.download_button("Download segmented data", segmented.to_csv(index=False).encode("utf-8"),
                                       "customer_segments.csv", "text/csv")
                except (ValueError, TypeError) as exc:
                    st.error(str(exc))


def insights_and_reports(frame: pd.DataFrame) -> None:
    st.subheader("Business insights and reports")
    findings = generate_business_insights(frame)
    for finding in findings:
        st.markdown(f"- {finding}")
    result = st.session_state.analysis_result
    if result is not None:
        st.markdown("#### Latest model result")
        st.dataframe(result.head(30), width="stretch")
        st.download_button("Download latest analysis result", result.to_csv(index=False).encode("utf-8"),
                           "analysis_results.csv", "text/csv")
    report_bytes = generate_pdf_report("AI Business Data Analyst Report", findings, frame, result)
    st.download_button("📄 Download PDF business report", report_bytes, "business_analysis_report.pdf", "application/pdf", type="primary")


def main() -> None:
    initialise_state()
    with st.sidebar:
        st.markdown("## 📊 Business Analyst")
        st.caption("Upload data · ask questions · discover insights")
        st.session_state.theme = st.selectbox("Appearance", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1)
        uploaded = st.file_uploader("Upload business data", type=["csv", "xlsx", "xls"], help="CSV and Excel files are processed locally in your session.")
        upload_id = None
        if uploaded is not None:
            upload_id = hashlib.sha256(uploaded.getvalue()).hexdigest()
        if uploaded is not None and upload_id != st.session_state.upload_id:
            try:
                set_dataset(load_dataset(uploaded), uploaded.name)
                st.session_state.upload_id = upload_id
                st.success(f"Loaded {uploaded.name}")
            except Exception as exc:
                st.error(f"Upload failed: {exc}")
        if st.button("Load included sample sales data", width="stretch"):
            try:
                set_dataset(load_sample_dataset(), "sample_sales.csv")
                st.session_state.upload_id = None
                st.success("Sample data loaded.")
            except Exception as exc:
                st.error(f"Could not load sample: {exc}")
        if st.session_state.filename:
            st.caption(f"Current dataset: {st.session_state.filename}")
        page = st.radio("Workspace", ["Dashboard", "Data explorer", "AI analyst", "Automated analysis",
                                        "Visualizations", "Machine learning", "Business insights & reports"])
        st.divider()
        st.caption("Rule-based analytics run without an API key. Data and SQLite table are session-local.")
    apply_styles(st.session_state.theme)
    render_header()
    frame = st.session_state.data
    if frame is None:
        st.info("Upload a CSV/Excel file in the sidebar, or load the included sample dataset to begin.")
        st.markdown("#### Example questions")
        st.markdown("- What is the total revenue?\n- Which product generates the highest sales?\n- Which region has the highest profit?\n- Show the top 10 customers.\n- What is the monthly sales growth?")
        st.stop()
    try:
        if page == "Dashboard":
            dashboard(frame)
        elif page == "Data explorer":
            data_explorer(frame)
        elif page == "AI analyst":
            analyst(frame)
        elif page == "Automated analysis":
            automated_analysis(frame)
        elif page == "Visualizations":
            visualizations(frame)
        elif page == "Machine learning":
            machine_learning(frame)
        else:
            insights_and_reports(frame)
    except Exception as exc:
        st.error(f"This section could not complete the request: {exc}")


if __name__ == "__main__":
    main()