"""Conservative shape checks for parser tables, not an accuracy or visual validator."""
import re


def table_quality(text):
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in text.splitlines() if line.strip().startswith("|")]
    rows = [row for row in rows if not all(re.fullmatch(r"[:\- ]+", cell) for cell in row)]
    flags = []
    if len(rows) < 2:
        flags.append("table_rows_missing")
    else:
        widths = {len(row) for row in rows}
        if len(widths) > 1:
            flags.append("table_column_counts_inconsistent")
        if len(rows[0]) < 2:
            flags.append("single_column_table_needs_visual_review")
        if len(rows) == 2 and max(len(cell) for cell in rows[1]) > 180:
            flags.append("possible_collapsed_table_rows")
    return {"status": "needs_visual_review" if flags else "shape_check_only",
        "quality_flags": flags, "body_rows": max(0, len(rows) - 1),
        "limitation": "Well-shaped rows do not prove correct cells, symbols, units, headings or footnotes"}
