"""
Results logger — auto-append experiment results to RESULTS_LOG.md.

Usage:
    from utils.result_logger import log_experiment

    log_experiment(
        name="PAS加权消融",
        script="benchmark_pas_weighted.py",
        description="异常分数加权得分的10类对比",
        columns=["Class", "Orig", "PAS", "OrigW", "PASW", "D_uw", "D_w"],
        rows=[["bagel", 0.8683, 0.8683, 0.9530, 0.9530, 0.0, 0.0], ...],
        mean_row=["MEAN", 0.7350, 0.7350, 0.8231, 0.8231, 0.0, 0.0],
        conclusion="异常加权有效，+0.088 平均提升",
        extra_sections={"加权 vs 不加权": {
            "columns": ["Class", "原版", "加权后", "提升"],
            "rows": [["bagel", 0.8683, 0.9530, 0.085], ...],
            "mean_row": ["MEAN", 0.7350, 0.8231, 0.088],
        }},
    )
"""
import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "RESULTS_LOG.md")


def _fmt_row(columns, values, widths=None):
    """Format a markdown table row."""
    if widths is None:
        widths = [max(len(str(c)), 10) for c in columns]
    cells = []
    for v, w in zip(values, widths):
        s = str(v)
        if isinstance(v, float):
            s = f"{v:.4f}"
        cells.append(s.rjust(w))
    return "| " + " | ".join(cells) + " |"


def log_experiment(name, script, description, columns, rows,
                   mean_row=None, conclusion=None, extra_sections=None):
    """
    Append a formatted experiment result to RESULTS_LOG.md.

    Args:
        name:        Experiment title
        script:      Script file name
        description: One-line description of the method
        columns:     List of column headers
        rows:        List of lists (per-class results)
        mean_row:    Optional [label, val1, val2, ...] for mean
        conclusion:  One-line conclusion
        extra_sections: Dict of {title: {columns, rows, mean_row}} for sub-tables
    """
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    # Header
    lines.append(f"\n---\n\n## {name}\n")
    lines.append(f"**日期**: {date}\n")
    lines.append(f"**脚本**: `{script}`\n")
    if description:
        lines.append(f"**方法**: {description}\n")
    lines.append("")

    # Main table
    widths = [max(len(str(c)), 10) for c in columns]
    # Expand first col for class names
    widths[0] = max(widths[0], 14)

    lines.append("### 结果\n")
    lines.append(_fmt_row(columns, columns, widths))
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    lines.append(sep)
    for row in rows:
        lines.append(_fmt_row(columns, row, widths))
    if mean_row:
        lines.append(sep)
        lines.append(_fmt_row(columns, mean_row, widths))
    lines.append("")

    # Extra sub-tables
    if extra_sections:
        for sec_title, sec_data in extra_sections.items():
            lines.append(f"### {sec_title}\n")
            scols = sec_data["columns"]
            swidths = [max(len(str(c)), 10) for c in scols]
            swidths[0] = max(swidths[0], 14)
            lines.append(_fmt_row(scols, scols, swidths))
            ssep = "|" + "|".join("-" * (w + 2) for w in swidths) + "|"
            lines.append(ssep)
            for row in sec_data["rows"]:
                lines.append(_fmt_row(scols, row, swidths))
            if sec_data.get("mean_row"):
                lines.append(ssep)
                lines.append(_fmt_row(scols, sec_data["mean_row"], swidths))
            lines.append("")

    # Conclusion
    if conclusion:
        lines.append(f"### 结论\n{conclusion}\n")

    # Append to log file
    with open(LOG_PATH, "a") as f:
        f.write("\n".join(lines))

    print(f"[results] 已保存到 {LOG_PATH}")
