"""Compare the LLM judge to the human labels in evals/judge_labels.jsonl.

Workflow: run `make eval` (judge on), read each listed case's next steps in
results/latest.json, put your own verdict in judge_labels.jsonl as
"human_label": true (steps are sensible and safe) or false, then run this.
It refuses to report until every label is filled in, so the README number
cannot be produced from unlabelled data.
"""

import json
import sys
from pathlib import Path

LABELS = Path(__file__).parent / "judge_labels.jsonl"


def agreement(labels: list[dict], rows: list[dict]) -> dict:
    by_id = {r["id"]: r for r in rows}
    missing = [x["id"] for x in labels if x["human_label"] is None]
    if missing:
        raise SystemExit(f"unlabelled cases (fill human_label): {missing}")
    absent = [x["id"] for x in labels if "next_steps_safe" not in by_id.get(x["id"], {}).get("grades", {})]
    if absent:
        raise SystemExit(f"no judge verdict in results for: {absent} (run the full eval with the judge on)")
    pairs = [(x["id"], x["human_label"], by_id[x["id"]]["grades"]["next_steps_safe"]) for x in labels]
    agree = sum(h == j for _, h, j in pairs)
    return {
        "n": len(pairs),
        "agree": agree,
        "rate": agree / len(pairs),
        "disagreements": [{"id": i, "human": h, "judge": j} for i, h, j in pairs if h != j],
    }


def main() -> None:
    labels = [json.loads(x) for x in LABELS.read_text().splitlines() if x.strip()]
    rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/latest.json"))["rows"]
    print(json.dumps(agreement(labels, rows), indent=2))


if __name__ == "__main__":
    main()
