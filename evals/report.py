"""Turn per-case rows into a summary dict and a markdown table."""

from collections import defaultdict

GRADERS = (
    "severity_match",
    "floor_respected",
    "floor_unchanged",
    "schema_valid",
    "citations_exist",
    "must_include_tools",
    "no_forbidden_phrases",
    "next_steps_safe",  # LLM judge; absent when run with --no-judge
)


def _rate(rows: list[dict], key: str) -> float | None:
    vals = [r["grades"][key] for r in rows if key in r["grades"]]
    return sum(vals) / len(vals) if vals else None


def summarize(rows: list[dict]) -> dict:
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    return {
        "n": len(rows),
        "overall": {g: _rate(rows, g) for g in GRADERS},
        "by_category": {c: {g: _rate(rs, g) for g in GRADERS} for c, rs in by_cat.items()},
        "human_review_rate": sum(r["needs_human_review"] for r in rows) / len(rows),
        "floor_enforced_rate": sum(r["floor_enforced"] for r in rows) / len(rows),
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def to_markdown(meta: dict, summary: dict) -> str:
    cats = ["overall", *sorted(summary["by_category"])]
    lines = [
        f"Model `{meta['model']}` · dataset `{meta['dataset_version']}` · subset `{meta['subset']}` · {summary['n']} cases",
        "",
        "| grader | " + " | ".join(cats) + " |",
        "|---|" + "---|" * len(cats),
    ]
    for g in GRADERS:
        vals = [summary["overall"][g], *(summary["by_category"][c][g] for c in cats[1:])]
        lines.append(f"| {g} | " + " | ".join(_pct(v) for v in vals) + " |")
    lines += [
        "",
        f"Human-review rate: {_pct(summary['human_review_rate'])} · "
        f"floor clamped by validate: {_pct(summary['floor_enforced_rate'])}",
    ]
    return "\n".join(lines)
