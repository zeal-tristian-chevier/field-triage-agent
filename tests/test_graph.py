from field_triage.graph import run_triage

from .conftest import BrokenRetriever, FakeLLM, draft, hot_pump

TID = "11111111-1111-4111-8111-111111111111"
REPORT = "Pump P-104 vibrating, bearing temp 92C, last serviced 14 months ago"


def run(llm, retriever):
    return run_triage(REPORT, llm=llm, retriever=retriever, trace_id=TID)


def test_happy_path_returns_cited_tool_results(bm25_retriever):
    state = run(FakeLLM(hot_pump(), [draft("high")]), bm25_retriever)
    final = state["final"]
    assert final.severity == "high" and not final.floor_enforced and not final.needs_human_review
    assert [r.id for r in final.tool_results_used] == ["threshold:bearing_temp_c", "severity_floor"]
    assert final.trace_id == TID


def test_llm_below_floor_is_retried_then_accepted(bm25_retriever):
    llm = FakeLLM(hot_pump(), [draft("low"), draft("high")])
    final = run(llm, bm25_retriever)["final"]
    assert llm.calls[type(draft())] == 2
    assert final.severity == "high" and not final.floor_enforced


def test_llm_below_floor_twice_is_clamped_up(bm25_retriever):
    # The core rule: even a stubborn LLM cannot return less than the tool floor
    final = run(FakeLLM(hot_pump(), [draft("low"), draft("medium")]), bm25_retriever)["final"]
    assert final.severity == "high" and final.floor_enforced
    assert not final.needs_human_review


def test_llm_above_floor_is_left_alone(bm25_retriever):
    final = run(FakeLLM(hot_pump(), [draft("critical")]), bm25_retriever)["final"]
    assert final.severity == "critical" and not final.floor_enforced


def test_unknown_citation_retried_then_human_review(bm25_retriever):
    bad = draft("high", cited=("threshold:made_up",))
    llm = FakeLLM(hot_pump(), [bad, bad])
    final = run(llm, bm25_retriever)["final"]
    assert final.needs_human_review and "made_up" in final.review_reason
    # fail-safe severity is the floor, not "low"
    assert final.severity == "high" and final.tool_results_used == []


def test_reason_failure_then_recovery(bm25_retriever):
    llm = FakeLLM(hot_pump(), [ValueError("bad schema"), draft("high")])
    state = run(llm, bm25_retriever)
    assert not state["final"].needs_human_review
    # the first failure stays on the record even though the retry succeeded
    assert [e.node for e in state["final"].errors] == ["reason"]


def test_reason_fails_twice_is_human_review(bm25_retriever):
    llm = FakeLLM(hot_pump(), [ValueError("x"), ValueError("y")])
    final = run(llm, bm25_retriever)["final"]
    assert final.needs_human_review and final.severity == "high"


def test_parse_failure_skips_to_human_review_without_retry(bm25_retriever):
    llm = FakeLLM(None)
    llm.queues[type(hot_pump())] = [RuntimeError("api down")]
    state = run(llm, bm25_retriever)
    final = state["final"]
    assert final.needs_human_review and "parse_report" in final.review_reason
    assert llm.calls[type(draft())] == 0  # reason never ran
    # no floor is known, so fail-safe is "high", never "low"
    assert final.severity == "high" and final.severity_floor is None


def test_retrieval_failure_is_recorded_but_not_fatal():
    state = run(FakeLLM(hot_pump(), [draft("high")]), BrokenRetriever())
    final = state["final"]
    assert not final.needs_human_review
    assert [e.node for e in final.errors] == ["lookup"]


def test_bm25_fallback_finds_relevant_snippet(bm25_retriever):
    backend, hits = bm25_retriever.search("pump bearing temperature lubrication")
    assert backend == "bm25" and hits[0]["id"] == "KB-01"
