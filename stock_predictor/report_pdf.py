from __future__ import annotations

import re
from collections.abc import Sequence
from io import BytesIO

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .engine import ModelReport
from .features import add_technical_features


FOREST = colors.HexColor("#103D35")
FOREST_LIGHT = colors.HexColor("#1C594A")
GOLD = colors.HexColor("#D8B66A")
INK = colors.HexColor("#1C2A26")
MUTED = colors.HexColor("#65756E")
GRID = colors.HexColor("#E4E9E4")
PALE_GOLD = colors.HexColor("#F8F3E7")


def _short_number(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 10_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.2f}"


class LineChart(Flowable):
    def __init__(
        self,
        title: str,
        values: Sequence[tuple[str, float]],
        color: colors.Color = FOREST_LIGHT,
        width: float = 245,
        height: float = 142,
    ) -> None:
        super().__init__()
        self.title = title
        self.values = values
        self.color = color
        self.width = width
        self.height = height

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        self.width = min(self.width, available_width)
        return self.width, self.height

    def draw(self) -> None:
        canvas = self.canv
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(INK)
        canvas.drawString(0, self.height - 11, self.title)
        if len(self.values) < 2:
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(MUTED)
            canvas.drawString(0, self.height / 2, "Not enough data to chart.")
            return

        left, right, bottom, top = 38, self.width - 8, 24, self.height - 28
        numbers = [float(value) for _, value in self.values]
        low, high = min(numbers), max(numbers)
        spread = high - low
        if spread == 0:
            spread = max(abs(high) * 0.02, 1.0)
        low -= spread * 0.08
        high += spread * 0.08

        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.setFont("Helvetica", 6)
        canvas.setFillColor(MUTED)
        for index in range(4):
            ratio = index / 3
            y = bottom + ratio * (top - bottom)
            value = low + ratio * (high - low)
            canvas.line(left, y, right, y)
            canvas.drawRightString(left - 4, y - 2, _short_number(value))

        points = []
        denominator = len(numbers) - 1
        for index, value in enumerate(numbers):
            x = left + index / denominator * (right - left)
            y = bottom + (value - low) / (high - low) * (top - bottom)
            points.append((x, y))

        path = canvas.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        canvas.setStrokeColor(self.color)
        canvas.setLineWidth(1.7)
        canvas.drawPath(path)
        canvas.setFillColor(self.color)
        canvas.circle(*points[-1], 2.5, stroke=0, fill=1)
        canvas.setFont("Helvetica", 6)
        canvas.setFillColor(MUTED)
        canvas.drawString(left, 9, self.values[0][0])
        canvas.drawRightString(right, 9, self.values[-1][0])


class BarChart(Flowable):
    def __init__(
        self,
        title: str,
        labels: Sequence[str],
        values: Sequence[float],
        *,
        percent: bool = False,
        width: float = 245,
        height: float = 142,
    ) -> None:
        super().__init__()
        if len(labels) != len(values):
            raise ValueError("Chart labels and values must have the same length.")
        self.title = title
        self.labels = labels
        self.values = values
        self.percent = percent
        self.width = width
        self.height = height

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        self.width = min(self.width, available_width)
        return self.width, self.height

    def draw(self) -> None:
        canvas = self.canv
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(INK)
        canvas.drawString(0, self.height - 11, self.title)
        if not self.values:
            return

        left, right, bottom, top = 28, self.width - 5, 37, self.height - 27
        max_abs = max((abs(value) for value in self.values), default=0.0)
        if max_abs == 0:
            max_abs = 1.0
        baseline = bottom + (top - bottom) * (max_abs / (2 * max_abs)) if min(self.values) < 0 else bottom
        canvas.setStrokeColor(GRID)
        canvas.line(left, baseline, right, baseline)
        slot = (right - left) / len(self.values)
        bar_width = min(25, slot * 0.58)
        for index, (label, value) in enumerate(zip(self.labels, self.values, strict=True)):
            center = left + slot * (index + 0.5)
            bar_height = abs(value) / max_abs * (top - bottom) * (
                0.48 if min(self.values) < 0 else 0.9
            )
            y = baseline if value >= 0 else baseline - bar_height
            canvas.setFillColor(FOREST_LIGHT if value >= 0 else GOLD)
            canvas.roundRect(center - bar_width / 2, y, bar_width, bar_height, 2, stroke=0, fill=1)
            text = f"{value:+.2f}%" if self.percent else f"{value:.4f}"
            canvas.setFont("Helvetica", 6)
            canvas.setFillColor(INK)
            canvas.drawCentredString(center, y + bar_height + 3, text)
            canvas.saveState()
            canvas.translate(center, bottom - 6)
            canvas.rotate(35)
            canvas.setFillColor(MUTED)
            canvas.drawString(0, 0, label[:14])
            canvas.restoreState()


def create_pdf_report(report: ModelReport, history: pd.DataFrame) -> bytes:
    if history.empty:
        raise ValueError("Cannot create a PDF report without historical prices.")

    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title=f"Ticker Predictor — {report.ticker}",
        author="Vruksh Enterprises",
        subject=f"Forecast report as of {report.as_of}",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BrandTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=FOREST,
            alignment=TA_LEFT,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BrandSubtitle",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=MUTED,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontSize=13,
            leading=17,
            textColor=FOREST,
            spaceBefore=8,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallBody",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
            textColor=INK,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=colors.white,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["Normal"],
            fontSize=7,
            leading=9,
            textColor=INK,
        )
    )

    story: list[Flowable] = [
        Paragraph("Ticker Predictor", styles["BrandTitle"]),
        Paragraph(
            f"Market research report for <b>{report.ticker}</b> · "
            f"Data through {report.as_of}<br/>Rahul Pagaria · Vruksh Enterprises",
            styles["BrandSubtitle"],
        ),
    ]

    key_metrics = [
        ["Latest close", "Labeled sessions", "BUY / SELL threshold"],
        [
            f"{report.current_close:,.2f}",
            f"{report.training_samples:,}",
            f"±{report.threshold_pct:.2f}%",
        ],
    ]
    metric_table = Table(key_metrics, colWidths=[57 * mm, 57 * mm, 62 * mm])
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), FOREST),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 1), (-1, 1), PALE_GOLD),
                ("TEXTCOLOR", (0, 1), (-1, 1), INK),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("BOX", (0, 0), (-1, -1), 0.5, GRID),
            ]
        )
    )
    story.extend(
        [
            metric_table,
            Spacer(1, 7 * mm),
            Paragraph("Next-session estimates", styles["SectionTitle"]),
        ]
    )

    forecast_data = [
        ["Technique", "Predicted close", "Predicted return", "Signal"],
    ]
    for forecast in report.forecasts:
        forecast_data.append(
            [
                forecast.model.title(),
                f"{forecast.predicted_next_close:,.2f}",
                f"{forecast.predicted_return_pct:+.3f}%",
                forecast.signal,
            ]
        )
    forecast_table = Table(
        forecast_data,
        colWidths=[38 * mm, 47 * mm, 47 * mm, 44 * mm],
        repeatRows=1,
    )
    forecast_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), FOREST_LIGHT),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TEXTCOLOR", (0, 1), (-1, -1), INK),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_GOLD]),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
            ]
        )
    )
    story.append(forecast_table)

    story.extend(
        [
            Spacer(1, 4 * mm),
            Paragraph("Walk-forward validation", styles["SectionTitle"]),
            Paragraph(
                "Scores average chronological held-out folds. MAE and RMSE are in "
                "decimal return units; lower is better. Direction accuracy applies "
                "to fitted models only. The zero-return baseline makes no directional call.",
                styles["SmallBody"],
            ),
            Spacer(1, 2 * mm),
        ]
    )
    baseline = report.evaluation["zero_return_baseline"]["mae"]
    model_names = [forecast.model for forecast in report.forecasts]
    metrics_data = [
        ["Technique", "MAE", "RMSE", "Directional accuracy", "MAE vs baseline"]
    ]
    for model_name in model_names:
        metrics = report.evaluation[model_name]
        if metrics["mae"] < baseline:
            baseline_comparison = "Better"
        elif metrics["mae"] > baseline:
            baseline_comparison = "Worse"
        else:
            baseline_comparison = "Equal"
        metrics_data.append(
            [
                model_name.title(),
                f"{metrics['mae']:.5f}",
                f"{metrics['rmse']:.5f}",
                f"{metrics['directional_accuracy']:.1%}",
                baseline_comparison,
            ]
        )
    baseline_metrics = report.evaluation["zero_return_baseline"]
    metrics_data.append(
        [
            "Zero-return baseline",
            f"{baseline_metrics['mae']:.5f}",
            f"{baseline_metrics['rmse']:.5f}",
            "N/A",
            "Reference",
        ]
    )
    metrics_table = Table(
        metrics_data,
        colWidths=[43 * mm, 28 * mm, 28 * mm, 50 * mm, 27 * mm],
        repeatRows=1,
    )
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), FOREST_LIGHT),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("TEXTCOLOR", (0, 1), (-1, -1), INK),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_GOLD]),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
            ]
        )
    )
    story.append(metrics_table)

    story.append(Paragraph("Model inferences", styles["SectionTitle"]))
    for forecast in report.forecasts:
        metrics = report.evaluation[forecast.model]
        story.append(
            Paragraph(
                f"<b>{forecast.model.title()} · {forecast.signal}</b> — "
                f"Forecast return {forecast.predicted_return_pct:+.3f}%; "
                f"MAE {metrics['mae']:.5f}; RMSE {metrics['rmse']:.5f}; "
                f"direction accuracy {metrics['directional_accuracy']:.1%}. "
                f"{forecast.mae_vs_baseline} {forecast.directional_accuracy_inference}",
                styles["SmallBody"],
            )
        )
        story.append(Spacer(1, 2 * mm))

    story.extend(
        [
            PageBreak(),
            Paragraph("Charts & indicators", styles["SectionTitle"]),
            Paragraph(
                "Historical charts use the downloaded, adjusted daily market data.",
                styles["SmallBody"],
            ),
            Spacer(1, 4 * mm),
        ]
    )

    sampling_step = max(1, len(history) // 90)

    def sampled_series(series: pd.Series) -> list[tuple[str, float]]:
        sampled = series.iloc[::sampling_step]
        if len(sampled) == 0 or sampled.index[-1] != series.index[-1]:
            sampled = pd.concat([sampled, series.iloc[[-1]]])
        return [
            (pd.Timestamp(index).strftime("%Y-%m-%d"), float(value))
            for index, value in sampled.items()
            if pd.notna(value)
        ]

    close_chart = LineChart(
        "Adjusted closing price",
        sampled_series(history["Close"]),
        color=FOREST_LIGHT,
        width=260,
    )
    volume_chart = LineChart(
        "Daily volume",
        sampled_series(history["Volume"]),
        color=GOLD,
        width=260,
    )
    story.append(Table([[close_chart, volume_chart]], colWidths=[88 * mm, 88 * mm]))

    features = add_technical_features(history)
    indicator_chart = LineChart(
        "RSI (14)",
        sampled_series(features["rsi_14"].dropna()),
        color=GOLD,
        width=260,
    )
    forecast_chart = BarChart(
        "Forecast return by technique",
        [forecast.model.title() for forecast in report.forecasts],
        [forecast.predicted_return_pct for forecast in report.forecasts],
        percent=True,
        width=260,
    )
    story.append(Spacer(1, 5 * mm))
    story.append(Table([[indicator_chart, forecast_chart]], colWidths=[88 * mm, 88 * mm]))

    mae_chart = BarChart(
        "Walk-forward MAE (lower is better)",
        [name.title() for name in [*model_names, "zero_return_baseline"]],
        [float(report.evaluation[name]["mae"]) for name in [*model_names, "zero_return_baseline"]],
        width=260,
    )
    accuracy_chart = BarChart(
        "Directional accuracy",
        [name.title() for name in model_names],
        [
            float(report.evaluation[name]["directional_accuracy"]) * 100
            for name in model_names
        ],
        percent=True,
        width=260,
    )
    story.append(Spacer(1, 5 * mm))
    story.append(Table([[mae_chart, accuracy_chart]], colWidths=[88 * mm, 88 * mm]))
    story.extend(
        [
            Spacer(1, 7 * mm),
            Paragraph(
                "<b>Important:</b> Forecasts and signals are statistical estimates "
                "for research and education, not financial advice or a guarantee of "
                "future performance. The next trading date is not inferred because "
                "market holidays vary.",
                styles["SmallBody"],
            ),
        ]
    )

    def draw_page(canvas: object, doc: object) -> None:
        canvas.saveState()
        page_width, _ = A4
        canvas.setStrokeColor(GRID)
        canvas.line(17 * mm, 13 * mm, page_width - 17 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(17 * mm, 8 * mm, "Ticker Predictor · Vruksh Enterprises")
        canvas.drawRightString(page_width - 17 * mm, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return output.getvalue()


def pdf_filename(ticker: str, as_of: str) -> str:
    safe_ticker = re.sub(r"[^A-Za-z0-9._-]+", "-", ticker.strip()).strip("-")
    if not safe_ticker:
        safe_ticker = "ticker"
    return f"ticker-predictor-{safe_ticker}-{as_of}.pdf"
