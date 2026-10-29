"""Run the golden set and write results/latest.json + results/latest.md.

With LANGSMITH_API_KEY set, the same target and scorer run through a LangSmith
experiment on a versioned dataset, so runs are comparable in the UI. Without it
the loop runs locally. Either way the local files are written.
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

from evals.graders import grade
from evals.judge import judge_next_steps
from evals.report import summarize, to_markdown
from field_triage.config import default_llm, triage_model
from field_triage.graph import run_triage
from field_triage.kb import Retriever

GOLDEN = Path(__file__).parent / "golden.jsonl"
RESULTS = Path("results")


def load_cases(subset: str) -> list[dict]:
    cases = [json.loads(x) for x in GOLDEN.read_text().splitlines() if x.strip()]
    return [c for c in cases if c["smoke"]] if subset == "smoke" else cases


def run_case(case: dict, experiment: str, llm, retriever) -> dict:
    trace_id = str(uuid.uuid4())
    state = run_triage(
        case["report"],
        llm=llm,
        retriever=retriever,
        trace_id=trace_id,
        config={
            "extra_tags": (f"experiment:{experiment}", f"case:{case['id']}", f"category:{case['category']}"),
            "extra_metadata": {"case_id": case["id"]},
        },
    )
    return {
        "result": state["final"].model_dump(),
        "available_tool_ids": [r.id for r in state.get("tool_results", [])],
        "retriever": state.get("retriever"),
    }


def score_case(case: dict, out: dict, use_judge: bool) -> tuple[dict[str, bool], str | None]:
    grades = grade(case, out)
    reason = None
    if use_judge:
        verdict = judge_next_steps(case, out)
        grades["next_steps_safe"], reason = verdict.passed, verdict.reason
    return grades, reason


def run_local(cases, experiment, use_judge, llm, retriever):
    rows = []
    for c in cases:
        out = run_case(c, experiment, llm, retriever)
        rows.append(make_row(c, out, *score_case(c, out, use_judge)))
        print(f"{c['id']}: {out['result']['severity']} (expected {c['expected_severity']})")
    return rows


def run_langsmith(cases, experiment, use_judge, llm, retriever, dataset_version):
    from langsmith import Client, evaluate

    client = Client()
    name = f"field-triage-golden-{dataset_version}"
    # Datasets are append-only by version: edit golden.jsonl => bump DATASET_VERSION,
    # so old experiments keep pointing at the data they were run on.
    if not client.has_dataset(dataset_name=name):
        ds = client.create_dataset(name, description="Synthetic maintenance triage golden set")
        client.create_examples(
            dataset_id=ds.id,
            inputs=[{"id": c["id"], "report": c["report"]} for c in load_cases("full")],
            outputs=[{k: v for k, v in c.items() if k != "report"} for c in load_cases("full")],
        )
    wanted = {c["id"] for c in cases}
    examples = [e for e in client.list_examples(dataset_name=name) if e.inputs["id"] in wanted]
    by_id = {c["id"]: c for c in cases}
    collected: dict[str, dict] = {}

    def target(inputs: dict) -> dict:
        out = run_case(by_id[inputs["id"]], experiment, llm, retriever)
        collected[inputs["id"]] = out
        return out

    def scorer(run, example) -> dict:
        case = by_id[example.inputs["id"]]
        grades, reason = score_case(case, run.outputs, use_judge)
        collected[case["id"]]["judge_reason"] = reason
        return {"results": [{"key": k, "score": float(v)} for k, v in grades.items()]}

    evaluate(
        target,
        data=examples,
        evaluators=[scorer],
        experiment_prefix=experiment,
        metadata={"model": triage_model(), "dataset_version": dataset_version},
        max_concurrency=4,
    )
    rows = []
    for c in cases:
        out = collected[c["id"]]
        rows.append(make_row(c, out, *score_case(c, out, use_judge)))
    return rows


def make_row(case: dict, out: dict, grades: dict, judge_reason: str | None) -> dict:
    r = out["result"]
    return {
        "id": case["id"],
        "category": case["category"],
        "expected_severity": case["expected_severity"],
        "severity": r["severity"],
        "severity_floor": r["severity_floor"],
        "floor_enforced": r["floor_enforced"],
        "needs_human_review": r["needs_human_review"],
        "next_steps": r["recommended_next_steps"],
        "cited": [t["id"] for t in r["tool_results_used"]],
        "errors": [e["message"] for e in r["errors"]],
        "trace_id": r["trace_id"],
        "grades": grades,
        "judge_reason": judge_reason,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", choices=["smoke", "full"], default="full")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--experiment", default=f"triage-{time.strftime('%Y%m%d-%H%M%S')}")
    args = p.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set; copy .env.example to .env and fill it in")

    cases, llm, retriever = load_cases(args.subset), default_llm(), Retriever()
    version = os.getenv("DATASET_VERSION", "v1")
    use_judge = not args.no_judge
    if os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        rows = run_langsmith(cases, args.experiment, use_judge, llm, retriever, version)
    else:
        print("LangSmith off (need LANGSMITH_API_KEY and LANGSMITH_TRACING=true): running locally")
        rows = run_local(cases, args.experiment, use_judge, llm, retriever)

    meta = {"model": triage_model(), "dataset_version": version, "subset": args.subset,
            "experiment": args.experiment, "judge": use_judge}
    summary = summarize(rows)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latest.json").write_text(json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=2))
    (RESULTS / "latest.md").write_text(to_markdown(meta, summary) + "\n")
    print((RESULTS / "latest.md").read_text())


if __name__ == "__main__":
    main()
