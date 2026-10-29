"""The golden file is hand-labelled. These tests catch label drift: if someone
edits a limit in equipment.py or a case in golden.jsonl, they must still agree."""

import json
from collections import Counter
from pathlib import Path

import pytest

from field_triage.schema import ParsedReport, Readings, severity_rank
from field_triage.tools import run_tools

CASES = [json.loads(x) for x in (Path(__file__).parents[1] / "evals" / "golden.jsonl").read_text().splitlines()]


def truth_parse(case) -> ParsedReport:
    t = case["truth"]
    return ParsedReport(
        equipment_id=t["equipment_id"],
        readings=Readings(bearing_temp_c=t["bearing_temp_c"], vibration_mm_s=t["vibration_mm_s"]),
        months_since_service=t["months_since_service"],
    )


def test_split_sizes():
    assert Counter(c["category"] for c in CASES) == {"normal": 15, "edge": 8, "adversarial": 7}
    assert sum(c["smoke"] for c in CASES) == 10


def test_smoke_subset_covers_every_category():
    assert {c["category"] for c in CASES if c["smoke"]} == {"normal", "edge", "adversarial"}


def test_ids_unique():
    assert len({c["id"] for c in CASES}) == len(CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_hand_labelled_floor_matches_tools(case):
    results = {r.id: r for r in run_tools(truth_parse(case))}
    assert results["severity_floor"].floor == case["expected_floor"]
    # every tool the case says must be cited has to exist for the true parse
    assert set(case["must_include_tools"]) <= set(results)
    assert severity_rank(case["expected_severity"]) >= severity_rank(case["expected_floor"])
