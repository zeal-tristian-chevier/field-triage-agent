import io
import json
import logging
import uuid

from field_triage.graph import run_triage
from field_triage.observability import JsonFormatter, log, run_config

from .conftest import FakeLLM, draft, hot_pump

TID = "22222222-2222-4222-8222-222222222222"


def test_run_config_uses_trace_id_as_langsmith_run_id(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL", "m-x")
    monkeypatch.setenv("DATASET_VERSION", "v9")
    cfg = run_config(TID, extra_tags=("experiment:e1",))
    assert cfg["run_id"] == uuid.UUID(TID) and cfg["run_name"] == "field-triage"
    assert {"model:m-x", "dataset:v9", "experiment:e1"} <= set(cfg["tags"])
    assert cfg["metadata"]["trace_id"] == TID


def test_every_log_line_carries_the_trace_id(bm25_retriever):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    log.addHandler(handler)
    try:
        run_triage("r", llm=FakeLLM(hot_pump(), [draft("high")]), retriever=bm25_retriever, trace_id=TID)
    finally:
        log.removeHandler(handler)
    lines = [json.loads(x) for x in stream.getvalue().splitlines()]
    events = [x["event"] for x in lines]
    assert events[0] == "run_start" and events[-1] == "run_done"
    assert events.count("node_done") == 5  # parse, lookup, calculate, reason, validate
    assert {x["trace_id"] for x in lines} == {TID}
