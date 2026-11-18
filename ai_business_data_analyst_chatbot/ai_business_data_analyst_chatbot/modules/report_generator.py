"""CSV/PDF report generation with a data-derived chart where possible."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _chart_image(frame: pd.DataFrame) -> BytesIO | None:
    numeric = frame.select_dtypes(include="number")
    categorical = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if numeric.empty or not categorical:
        return None
    category, metric = categorical[0], numeric.columns[0]
    grouped = frame.groupby(category, dropna=False)[metric].sum().nlargest(10).sort_values()
    image = BytesIO()
    figure, axis = plt.subplots(figsize=(8, 4))
    grouped.plot(kind="barh", ax=axis, color="#20a982")
    axis.set_title(f"Top {len(grouped)} {category} by {metric}")
    axis.set_xlabel(str(metric))
    figure.tight_layout()
    figure.savefig(image, format="png", dpi=140)
    plt.close(figure)
    image.seek(0)
    return image


def generate_pdf_report(title: str, findings: list[str], frame: pd.DataFrame,
                        result: pd.DataFrame | None = None) -> bytes:
    """Build a portable PDF with findings, a chart, and an optional result preview."""
    output = BytesIO()
    document = SimpleDocTemplate(output, pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm,
                                 topMargin=16 * mm, bottomMargin=16 * mm)
    styles = getSampleStyleSheet()
    styles["BodyText"].alignment = TA_LEFT
    story = [Paragraph(escape(title), styles["Title"]),
             Paragraph(f"Generated {datetime.now():%d %b %Y, %H:%M}", styles["Normal"]), Spacer(1, 10)]
    story.append(Paragraph("Executive findings", styles["Heading2"]))
    story.extend(Paragraph(f"• {escape(str(item))}", styles["BodyText"]) for item in findings)
    story.append(Spacer(1, 8))
    chart = _chart_image(frame)
    if chart is not None:
        story.extend([Paragraph("Dataset chart", styles["Heading2"]), Image(chart, width=165 * mm, height=82 * mm), Spacer(1, 8)])
    if result is not None and not result.empty:
        story.append(Paragraph("Analysis result preview", styles["Heading2"]))
        preview = result.head(20).iloc[:, :8].fillna("").astype(str)
        matrix = [[str(column)[:35] for column in preview.columns]] + [[str(value)[:50] for value in row] for row in preview.to_numpy()]
        table = Table(matrix, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173b57")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
            ("FONTSIZE", (0, 0), (-1, -1), 7), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(table)
    document.build(story)
    return output.getvalue()