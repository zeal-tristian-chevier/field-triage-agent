"""Graph wiring: parse_report -> lookup -> calculate -> reason -> validate (-> reason once)."""

import operator
import uuid
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .kb import Retriever
from .nodes import make_nodes
from .schema import Draft, NodeError, ParsedReport, Severity, ToolResult, TriageResult


class State(TypedDict, total=False):
    trace_id: str
    report: str
    parsed: ParsedReport
    snippets: list[dict]
    retriever: str
    tool_results: list[ToolResult]
    floor: Severity
    draft: Draft | None
    feedback: str
    attempts: int
    retry: bool
    final: TriageResult
    # reducer: every node appends, none overwrite, so no error is lost
    errors: Annotated[list[NodeError], operator.add]


def _has_fatal(state: State) -> bool:
    return any(e.fatal for e in state.get("errors", []))


def build_graph(llm, retriever):
    n = make_nodes(llm, retriever)
    g = StateGraph(State)
    for name, fn in n.items():
        g.add_node(name, fn)

    g.add_edge(START, "parse_report")
    # A fatal error skips the middle and lets validate return human review
    g.add_conditional_edges(
        "parse_report", lambda s: "validate" if _has_fatal(s) else "lookup", ["lookup", "validate"]
    )
    g.add_edge("lookup", "calculate")
    g.add_conditional_edges(
        "calculate", lambda s: "validate" if _has_fatal(s) else "reason", ["reason", "validate"]
    )
    g.add_edge("reason", "validate")
    g.add_conditional_edges(
        "validate", lambda s: "reason" if s.get("retry") else END, ["reason", END]
    )
    return g.compile()


def run_triage(report: str, llm=None, retriever=None, trace_id: str | None = None, config=None) -> State:
    """Run one report through the graph and return the final state."""
    if llm is None:
        from .config import default_llm

        llm = default_llm()
    graph = build_graph(llm, retriever or Retriever())
    initial: State = {
        "trace_id": trace_id or str(uuid.uuid4()),
        "report": report,
        "errors": [],
        "attempts": 0,
        "retry": False,
    }
    return graph.invoke(initial, config=config)
