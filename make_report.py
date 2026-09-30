"""Create the assessment PDF from validated model results and scorer chart."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent


def main() -> None:
    metrics_path = ROOT / "reports" / "validation_metrics.csv"
    chart_path = ROOT / "scorer_results" / "candidate_december.png"
    output_path = ROOT / "reports" / "freight_rate_assessment.pdf"
    if not metrics_path.is_file() or not chart_path.is_file():
        raise SystemExit("Run train_model.py and score.py before creating the report.")

    data = pd.read_csv(ROOT / "data" / "train_test.csv")
    validation = pd.read_csv(ROOT / "data" / "validation.csv")
    december = pd.read_csv(ROOT / "data" / "december_chart_inputs.csv")
    results = pd.read_csv(metrics_path)
    candidates = results[results["model"].str.startswith("CatBoost")]
    chosen = candidates.groupby("model")["selection_score"].mean().idxmin()
    chosen_rows = results[results["model"] == chosen].set_index("fold")
    baseline = results[results["model"] == "equipment median rate-per-mile"].set_index("fold")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=colors.HexColor("#064A56"), alignment=TA_LEFT, spaceAfter=8))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#064A56"), spaceBefore=12, spaceAfter=5))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    doc = SimpleDocTemplate(str(output_path), pagesize=letter, rightMargin=0.68*inch, leftMargin=0.68*inch, topMargin=0.62*inch, bottomMargin=0.62*inch, title="Spotter Freight Rate Prediction Assessment")
    story = [
        Paragraph("Freight Rate Prediction", styles["Title2"]),
        Paragraph("Machine Learning Engineer Assessment | Spotter AI", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Approach", styles["Section"]),
        Paragraph("The development set contains 48,000 labeled loads from January through October 2025. Validation rows cover November and December 2025, so random splitting would mix future and past conditions. We used two expanding-window holdouts: January-August to predict September, then January-September to predict October. October is the primary proxy for the unseen months.", styles["BodyText"]),
        Paragraph("We compared an equipment-level median rate-per-mile baseline with CatBoost using MAE and RMSE training losses. Selection is the lowest mean of relative MAE and RMSE versus baseline across the two holdouts. The selected loss is refit using all labeled rows before predicting the supplied validation loads and fixed December scenario.", styles["BodyText"]),
        Paragraph("Model inputs include origin/destination and equipment categories, mileage, coordinates, straight-line distance and route geometry, weight, market and quote signals, and calendar features. load_id is excluded. CatBoost handles missing numeric values natively; negative weights are converted to missing because they are physically invalid. Extreme positive target rates are retained so model selection measures performance against the supplied labels.", styles["BodyText"]),
        Paragraph("Validation results", styles["Section"]),
    ]
    table_data = [["Holdout", "Rows", "Model", "MAE ($)", "RMSE ($)", "Baseline MAE", "Baseline RMSE"]]
    for fold in ("September", "October"):
        row = chosen_rows.loc[fold]
        base = baseline.loc[fold]
        table_data.append([fold, f"{int(row.validation_rows):,}", chosen.replace("CatBoost ", "CatBoost / "), f"{row.mae:,.1f}", f"{row.rmse:,.1f}", f"{base.mae:,.1f}", f"{base.rmse:,.1f}"])
    table = Table(table_data, colWidths=[0.76*inch, 0.54*inch, 1.46*inch, 0.75*inch, 0.82*inch, 0.88*inch, 0.92*inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#064A56")), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#C7D5D7")), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EFF5F5")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.extend([table, Spacer(1, 8), Paragraph("Metrics are in-sample holdout estimates, not the hidden final Spotter evaluation score. Holdouts preserve time order and evaluate models trained only on earlier rows.", styles["Small"]), PageBreak()])

    missing_weight = int(data.weight.isna().sum())
    negative_weight = int((data.weight < 0).sum())
    missing_market = int(data.market_index.isna().sum())
    unseen = (set(validation.pickup) | set(validation.delivery)) - (set(data.pickup) | set(data.delivery))
    target_q99 = float(data.posted_rate.quantile(.99))
    story.extend([
        Paragraph("Exploration and data quality", styles["Title2"]),
        Paragraph(f"The source has {data.pickup.nunique()} pickup locations, {data.delivery.nunique()} delivery locations, three equipment types, and {data[['pickup','delivery']].drop_duplicates().shape[0]:,} observed route pairs. Distance is the strongest simple numeric correlate of posted rate (Pearson r = {data[['distance','posted_rate']].corr().iloc[0,1]:.3f}). The target median is ${data.posted_rate.median():,.2f}; the 99th percentile is ${target_q99:,.2f} and the maximum is ${data.posted_rate.max():,.2f}.", styles["BodyText"]),
        Paragraph(f"Quality checks found {missing_weight:,} missing weights, {negative_weight:,} negative weights, and {missing_market:,} missing market-index values among the {len(data):,} development rows. Missing weight and market values were not used to drop rows. Negative weights were treated as missing; the tree model's native missing-value handling was used for numeric gaps. The high-rate tail was kept intact rather than clipped. {len(unseen)} locations appear in validation but not in development; coordinates and route geometry provide continuous location information for such cases, while CatBoost supports unseen categorical values.", styles["BodyText"]),
        Paragraph("December fixed-input prediction", styles["Section"]),
        Paragraph(f"The supplied chart holds Lexington to Fort Wayne, 360 miles, Dry Van, and 32,000 lb constant while varying the date over all 31 days of December 2025. Predicted values range from ${december.predicted_rate.min():,.2f} to ${december.predicted_rate.max():,.2f}. The chart below is produced by the provided scorer from the completed December input file.", styles["BodyText"]),
        Spacer(1, 8),
    ])
    chart = Image(str(chart_path), width=6.95*inch, height=3.1*inch, kind="proportional")
    story.append(chart)
    story.extend([Spacer(1, 10), Paragraph("Submission checks", styles["Section"]), Paragraph(f"The completed submission contains {len(pd.read_csv(ROOT / 'validation_predictions.csv')):,} validation predictions and the December input contains {len(december)} daily predictions. Both output files were checked by the supplied scorer. Final labels and the official evaluation metric are not available locally.", styles["BodyText"])])
    doc.build(story)
    print(f"Created {output_path}")


if __name__ == "__main__":
    main()
