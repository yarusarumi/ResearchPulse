"""Conservative numeric presence gate, complementary to model and human verification."""
import re
from decimal import Decimal


def numeric_literals(text):
    text = text.replace("{,}", "").replace("−", "-").replace("–", "-")
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    words = {word: str(i) for i, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve".split())}
    # Recognize explicit small number words; do not read 'five hundred' as the quantity five.
    text = re.sub(r"\b(" + "|".join(words) + r")\b(?![ -]+(?:hundred|thousand|million|billion)\b)",
                  lambda match: words[match[0].lower()], text, flags=re.I)
    return {Decimal(value) for value in re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)}


def check_numbers(claim, evidence):
    if claim.kind != "paper_fact":
        return {"status": "not_applicable", "missing": []}
    numbers = numeric_literals(claim.text)
    source_numbers = numeric_literals("\n".join(e.text for e in evidence))
    missing = sorted(numbers - source_numbers)
    return {"status": "missing_numeric_literals" if missing else
            ("literal_numbers_present" if numbers else "no_numeric_literals"),
            "missing": [str(value) for value in missing],
            "limitation": "Presence does not verify units, rows, conditions, calculations or statistical significance"}
