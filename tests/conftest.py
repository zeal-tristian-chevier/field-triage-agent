import pytest

from field_triage.kb import Retriever
from field_triage.schema import Draft, ParsedReport, Readings


class FakeLLM:
    """Stands in for ChatAnthropic. Returns queued outputs per schema; an
    Exception in the queue is raised, so failure paths are testable."""

    def __init__(self, parsed=None, drafts=()):
        self.queues = {ParsedReport: [parsed] if parsed is not None else [], Draft: list(drafts)}
        self.calls = {ParsedReport: 0, Draft: 0}

    def with_structured_output(self, schema):
        outer = self

        class _Bound:
            def invoke(self, _messages):
                outer.calls[schema] += 1
                item = outer.queues[schema].pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

        return _Bound()


class BrokenRetriever:
    def search(self, query, k=3):
        raise RuntimeError("index down")


@pytest.fixture
def bm25_retriever(monkeypatch):
    r = Retriever()
    # force the fallback path so tests never download an embedding model
    monkeypatch.setattr(r, "_chroma", lambda: (_ for _ in ()).throw(RuntimeError("no chroma")))
    return r


def hot_pump():
    """P-104, 92C: tools give floor 'high' (temp exceeded) with service overdue 'medium'."""
    return ParsedReport(
        equipment_id="P-104",
        readings=Readings(bearing_temp_c=92),
        months_since_service=14,
        symptoms=["vibrating"],
    )


def draft(severity="high", cited=("threshold:bearing_temp_c", "severity_floor")):
    return Draft(
        severity=severity,
        likely_cause="bearing lubrication failure",
        recommended_next_steps=["Check oil level"],
        parts_to_check=["bearing"],
        cited_tool_ids=list(cited),
    )
