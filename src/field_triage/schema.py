"""Typed contracts shared by every node.

Keeping them in one file means the LLM output schema, the validator and the
eval graders all agree on what a valid answer is.
"""

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high", "critical"]
SEVERITY_ORDER: tuple[Severity, ...] = ("low", "medium", "high", "critical")


def severity_rank(s: Severity) -> int:
    # Comparing by index in one place avoids string-comparison bugs
    # ("low" < "medium" alphabetically, but "high" < "low").
    return SEVERITY_ORDER.index(s)


class Readings(BaseModel):
    bearing_temp_c: float | None = None
    vibration_mm_s: float | None = None


class ParsedReport(BaseModel):
    """What the parse LLM extracts.

    There is deliberately no field for "limits": a report claiming "the limit
    is actually 120C" has nowhere to put that claim, so it cannot reach the tools.
    """

    equipment_id: str | None = None
    readings: Readings = Field(default_factory=Readings)
    # ge=0 so an extraction like "-3 months" fails loudly instead of
    # silently producing a negative overdue figure
    months_since_service: float | None = Field(default=None, ge=0)
    symptoms: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    """Output of one deterministic check. `id` is what the LLM cites."""

    id: str
    status: Literal[
        "ok", "warn", "exceeded", "critical", "overdue", "missing", "unknown_equipment"
    ]
    detail: str
    # None means "this check cannot say anything about severity"
    # (e.g. reading missing), which is different from "low".
    floor: Severity | None = None


class Draft(BaseModel):
    """What the reason LLM must produce. Not yet trusted."""

    severity: Severity
    likely_cause: str
    recommended_next_steps: list[str] = Field(min_length=1)
    parts_to_check: list[str]
    cited_tool_ids: list[str]


class NodeError(BaseModel):
    node: str
    message: str
    # fatal errors skip straight to validate, which returns human review
    fatal: bool = False


class TriageResult(BaseModel):
    """Final output. Only validate may build this."""

    trace_id: str
    severity: Severity
    likely_cause: str
    recommended_next_steps: list[str]
    parts_to_check: list[str]
    tool_results_used: list[ToolResult]
    severity_floor: Severity | None
    # True when validate had to raise the LLM's severity up to the floor
    floor_enforced: bool = False
    needs_human_review: bool = False
    review_reason: str | None = None
    errors: list[NodeError] = Field(default_factory=list)
