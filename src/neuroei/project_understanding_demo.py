"""Offline executable fixture demo. Never creates a model client or accesses sources."""
import argparse
import json
from pathlib import Path

from .project_understanding import (
    JournalBatch,
    UnderstandingExtraction,
    identity,
    understand,
)
from .store import Store


class FixtureExtractor:
    audit_ref = "offline-fixed-fixture-no-model-call"

    def __init__(self, extractions):
        self.extractions = iter(extractions)

    def extract(self, payload):
        return UnderstandingExtraction.model_validate(next(self.extractions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON fixture containing steps")
    parser.add_argument("--out", type=Path, required=True, help="New isolated output directory")
    args = parser.parse_args()
    if (args.out / "neuroei.sqlite3").exists():
        parser.error("Use a fresh output directory to preserve previous demo state")
    data = json.loads(args.input.read_text(encoding="utf-8"))
    store = Store(args.out)
    results = []
    try:
        for step in data["steps"]:
            results.append(understand(store, FixtureExtractor(step["extractions"]),
                JournalBatch.model_validate(step["batch"]), user_id=data["user_id"],
                project_id=data["project_id"], as_of=step["as_of"], bootstrap=step["bootstrap"]))
        pointer = store.get("understanding_head", identity([data["user_id"], data["project_id"]]))
        output = {"validation": "offline_fixed_fixture_not_scientific_acceptance", "results": results,
                  "state": store.get("understanding_state", pointer["key"])}
        (args.out / "demo-output.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(args.out / "demo-output.json"),
                          "statuses": [r["status"] for r in results]}, ensure_ascii=False))
    finally:
        store.close()


if __name__ == "__main__":
    main()
