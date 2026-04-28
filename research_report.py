"""
Research Comparison Report

Runs simulations comparing three AI strategies:
    - Random baseline (33% expected)
    - Heuristic AI (FairPlayAI / ChallengeAI)
    - ML Prediction AI (trained model)

Against all simulated player strategies, then generates
a formatted Excel report with comparison tables.

Usage:
    python research_report.py

Output:
    ~/Desktop/CapStone/research_comparison_report.xlsx
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from simulation_mode import run_simulation, PLAYER_STRATEGIES

ML_MODEL_PATH = os.path.join(
    os.path.expanduser("~"), "Desktop", "CapStone", "rps_ml_model.pkl"
)
OUTPUT_PATH = Path.home() / "Desktop" / "CapStone" / "research_comparison_report.xlsx"

RUNS_PER_COMBO = 10
ROUNDS_PER_RUN = 100


# ==================================================================
# Formatting helpers
# ==================================================================

HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=11)
SUBHEADER_FILL = PatternFill("solid", fgColor="D6E4F0")
SUBHEADER_FONT = Font(bold=True, size=11)
TITLE_FONT = Font(bold=True, size=14)
SUBTITLE_FONT = Font(bold=True, size=12, color="1F4E78")
BODY_FONT = Font(size=11)
GOOD_FILL = PatternFill("solid", fgColor="C6EFCE")
BAD_FILL = PatternFill("solid", fgColor="FFC7CE")
NEUTRAL_FILL = PatternFill("solid", fgColor="FFEB9C")

THIN_BORDER = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin"),
)


def _format_header_row(ws, row_num):
    for cell in ws[row_num]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = THIN_BORDER


def _format_data_cell(ws, row, col, value, fmt=None):
    cell = ws.cell(row=row, column=col, value=value)
    cell.font = BODY_FONT
    cell.alignment = Alignment(horizontal="center")
    cell.border = THIN_BORDER
    if fmt == "pct":
        cell.number_format = "0.0%"
    elif fmt == "dec1":
        cell.number_format = "0.0"
    return cell


def _auto_width(ws, min_width=10, max_width=22):
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        max_len = min_width
        for cell in col:
            if cell.value:
                max_len = max(max_len, len(str(cell.value)) + 2)
        ws.column_dimensions[letter].width = min(max_len, max_width)


# ==================================================================
# Run the three AI comparisons
# ==================================================================

def run_all_comparisons():
    """
    Run simulations for random, heuristic (fair_play + challenge), and ML.
    Returns a structured dict of results.
    """
    ai_configs = [
        ("Random", ["random"]),
        ("Heuristic (FairPlay)", ["fair_play"]),
        ("Heuristic (Challenge)", ["challenge"]),
    ]

    # Check if ML model exists.
    ml_available = os.path.exists(ML_MODEL_PATH)
    if ml_available:
        ai_configs.append(("ML Prediction", ["ml"]))
        print(f"ML model found at {ML_MODEL_PATH}")
    else:
        print(f"ML model NOT found at {ML_MODEL_PATH} — skipping ML comparison.")

    all_results = {}

    for ai_label, ai_list in ai_configs:
        print(f"\nRunning: {ai_label}...")
        results = run_simulation(
            player_strategies=PLAYER_STRATEGIES,
            ai_opponents=ai_list,
            runs_per_combo=RUNS_PER_COMBO,
            rounds_per_run=ROUNDS_PER_RUN,
            save_excel=False,
        )
        all_results[ai_label] = results

    return all_results, ml_available


# ==================================================================
# Build Excel report
# ==================================================================

def build_report(all_results, ml_available, output_path):
    wb = Workbook()

    _build_overview_sheet(wb, all_results)
    _build_comparison_table(wb, all_results)
    _build_per_strategy_sheet(wb, all_results)
    _build_key_findings_sheet(wb, all_results, ml_available)

    if ml_available:
        _build_ml_details_sheet(wb)

    # Save.
    try:
        wb.save(output_path)
        print(f"\nReport saved to: {output_path}")
    except PermissionError:
        print(f"\nCould not save — close {output_path} in Excel and retry.")
    finally:
        wb.close()


def _build_overview_sheet(wb, all_results):
    ws = wb.active
    ws.title = "Overview"

    ws.merge_cells("A1:F1")
    ws["A1"] = "RPS AI Research Comparison Report"
    ws["A1"].font = TITLE_FONT

    ws["A3"] = "Generated:"
    ws["B3"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ws["A4"] = "Runs per combination:"
    ws["B4"] = RUNS_PER_COMBO
    ws["A5"] = "Rounds per run:"
    ws["B5"] = ROUNDS_PER_RUN
    ws["A6"] = "Player strategies tested:"
    ws["B6"] = len(PLAYER_STRATEGIES)

    row = 8
    ws.cell(row=row, column=1, value="AI Type").font = SUBHEADER_FONT
    ws.cell(row=row, column=2, value="Avg Robot Win Rate").font = SUBHEADER_FONT
    ws.cell(row=row, column=3, value="Avg Player Win Rate").font = SUBHEADER_FONT
    ws.cell(row=row, column=4, value="Avg Draw Rate").font = SUBHEADER_FONT
    ws.cell(row=row, column=5, value="Total Rounds").font = SUBHEADER_FONT
    _format_header_row(ws, row)

    row += 1
    for ai_label, results in all_results.items():
        combos = results["combo_results"]
        if not combos:
            continue

        avg_rwr = sum(c["robot_win_rate"] for c in combos) / len(combos)
        avg_pwr = sum(c["player_win_rate"] for c in combos) / len(combos)
        avg_dr = sum(c["draw_rate"] for c in combos) / len(combos)
        total_r = results["total_rounds"]

        _format_data_cell(ws, row, 1, ai_label)
        _format_data_cell(ws, row, 2, avg_rwr, "pct")
        _format_data_cell(ws, row, 3, avg_pwr, "pct")
        _format_data_cell(ws, row, 4, avg_dr, "pct")
        _format_data_cell(ws, row, 5, total_r)

        # Color code robot win rate.
        cell = ws.cell(row=row, column=2)
        if avg_rwr > 0.38:
            cell.fill = GOOD_FILL
        elif avg_rwr < 0.30:
            cell.fill = BAD_FILL

        row += 1

    _auto_width(ws)


def _build_comparison_table(wb, all_results):
    ws = wb.create_sheet("AI Comparison")

    ai_labels = list(all_results.keys())

    # Header row.
    ws.cell(row=1, column=1, value="Player Strategy")
    for col_idx, ai_label in enumerate(ai_labels):
        ws.cell(row=1, column=2 + col_idx * 2, value=f"{ai_label} Robot WR")
        ws.cell(row=1, column=3 + col_idx * 2, value=f"{ai_label} Streak")
    _format_header_row(ws, 1)

    # Build lookup: (strategy, ai) -> combo data.
    lookup = {}
    for ai_label, results in all_results.items():
        for combo in results["combo_results"]:
            lookup[(combo["strategy"], ai_label)] = combo

    row = 2
    for strategy in PLAYER_STRATEGIES:
        ws.cell(row=row, column=1, value=strategy).font = BODY_FONT
        ws.cell(row=row, column=1).border = THIN_BORDER

        for col_idx, ai_label in enumerate(ai_labels):
            combo = lookup.get((strategy, ai_label))
            if combo:
                _format_data_cell(ws, row, 2 + col_idx * 2, combo["robot_win_rate"], "pct")
                _format_data_cell(ws, row, 3 + col_idx * 2, combo["avg_streak"], "dec1")
            else:
                _format_data_cell(ws, row, 2 + col_idx * 2, "N/A")
                _format_data_cell(ws, row, 3 + col_idx * 2, "N/A")

        row += 1

    _auto_width(ws)


def _build_per_strategy_sheet(wb, all_results):
    ws = wb.create_sheet("Per Strategy Detail")

    ws.append([
        "Player Strategy",
        "AI Type",
        "Player Win Rate",
        "Robot Win Rate",
        "Draw Rate",
        "Avg Max Streak",
        "Runs",
    ])
    _format_header_row(ws, 1)

    row = 2
    for ai_label, results in all_results.items():
        for combo in results["combo_results"]:
            _format_data_cell(ws, row, 1, combo["strategy"])
            _format_data_cell(ws, row, 2, ai_label)
            _format_data_cell(ws, row, 3, combo["player_win_rate"], "pct")
            _format_data_cell(ws, row, 4, combo["robot_win_rate"], "pct")
            _format_data_cell(ws, row, 5, combo["draw_rate"], "pct")
            _format_data_cell(ws, row, 6, combo["avg_streak"], "dec1")
            _format_data_cell(ws, row, 7, combo["runs"])

            # Color code.
            cell = ws.cell(row=row, column=4)
            if combo["robot_win_rate"] > 0.40:
                cell.fill = GOOD_FILL
            elif combo["robot_win_rate"] < 0.28:
                cell.fill = BAD_FILL

            row += 1

    _auto_width(ws)


def _build_key_findings_sheet(wb, all_results, ml_available):
    ws = wb.create_sheet("Key Findings")

    ws.merge_cells("A1:D1")
    ws["A1"] = "Key Research Findings"
    ws["A1"].font = TITLE_FONT

    findings = []

    # Compare AI performance.
    ai_avg = {}
    for ai_label, results in all_results.items():
        combos = results["combo_results"]
        if combos:
            ai_avg[ai_label] = sum(c["robot_win_rate"] for c in combos) / len(combos)

    if ai_avg:
        best_ai = max(ai_avg, key=ai_avg.get)
        worst_ai = min(ai_avg, key=ai_avg.get)

        findings.append(f"Strongest AI overall: {best_ai} ({ai_avg[best_ai]:.1%} robot win rate)")
        findings.append(f"Weakest AI overall: {worst_ai} ({ai_avg[worst_ai]:.1%} robot win rate)")

        random_wr = ai_avg.get("Random", 0.333)
        for label, wr in ai_avg.items():
            if label != "Random":
                lift = wr - random_wr
                findings.append(f"  {label} lift over random: {lift:+.1%}")

    findings.append("")

    # ML vs heuristic comparison.
    if ml_available and "ML Prediction" in ai_avg:
        ml_wr = ai_avg["ML Prediction"]
        for label in ["Heuristic (FairPlay)", "Heuristic (Challenge)"]:
            if label in ai_avg:
                h_wr = ai_avg[label]
                diff = ml_wr - h_wr
                direction = "outperforms" if diff > 0 else "underperforms"
                findings.append(f"ML {direction} {label} by {abs(diff):.1%}")

    findings.append("")

    # Per-strategy insights.
    findings.append("Strategy vulnerability analysis:")
    for strategy in PLAYER_STRATEGIES:
        rates = {}
        for ai_label, results in all_results.items():
            for combo in results["combo_results"]:
                if combo["strategy"] == strategy:
                    rates[ai_label] = combo["robot_win_rate"]

        if rates:
            best_against = max(rates, key=rates.get)
            findings.append(f"  {strategy}: most exploited by {best_against} ({rates[best_against]:.1%})")

    row = 3
    for finding in findings:
        ws.cell(row=row, column=1, value=finding).font = BODY_FONT
        row += 1

    ws.column_dimensions["A"].width = 70


def _build_ml_details_sheet(wb):
    ws = wb.create_sheet("ML Model Details")

    ws.merge_cells("A1:C1")
    ws["A1"] = "ML Model Information"
    ws["A1"].font = TITLE_FONT

    try:
        from ml_model import RPSModel
        model = RPSModel.load(ML_MODEL_PATH)

        ws["A3"] = "Model type:"
        ws["B3"] = type(model.model).__name__ if model.model else "N/A"
        ws["A4"] = "Trained:"
        ws["B4"] = "Yes" if model.is_trained else "No"
        ws["A5"] = "Lookback:"
        ws["B5"] = model.lookback

        # Feature importance.
        importance = model.get_feature_importance()
        if importance:
            ws["A7"] = "Feature Importance Ranking"
            ws["A7"].font = SUBTITLE_FONT

            ws["A8"] = "Rank"
            ws["B8"] = "Feature"
            ws["C8"] = "Importance"
            _format_header_row(ws, 8)

            for i, (name, score) in enumerate(importance):
                row = 9 + i
                _format_data_cell(ws, row, 1, i + 1)
                _format_data_cell(ws, row, 2, name)
                _format_data_cell(ws, row, 3, round(score, 4))

    except Exception as exc:
        ws["A3"] = f"Could not load model details: {exc}"

    _auto_width(ws)


# ==================================================================
# Main
# ==================================================================

def main():
    print("=" * 60)
    print("RPS Research Comparison Report Generator")
    print("=" * 60)
    print()

    start = time.time()
    all_results, ml_available = run_all_comparisons()
    elapsed = time.time() - start

    print(f"\nAll simulations complete in {elapsed:.1f}s")

    # Print summary to terminal.
    print("\n--- AI Performance Summary ---")
    for ai_label, results in all_results.items():
        combos = results["combo_results"]
        if combos:
            avg_rwr = sum(c["robot_win_rate"] for c in combos) / len(combos)
            print(f"  {ai_label:30s}  Robot WR: {avg_rwr:.1%}")

    print()
    build_report(all_results, ml_available, OUTPUT_PATH)


if __name__ == "__main__":
    main()
