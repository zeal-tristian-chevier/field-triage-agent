"""Deterministic checks. No LLM, no I/O.

The LLM never compares a number to a limit; it only reads these results and
cites them by id. validate() later enforces the floor computed here.
"""

from .equipment import SPECS, Limits, equipment_kind
from .schema import SEVERITY_ORDER, ParsedReport, Severity, ToolResult, severity_rank


def check_threshold(name: str, value: float | None, limits: Limits) -> ToolResult:
    tid = f"threshold:{name}"
    if value is None:
        # Missing is not "fine": floor stays None so it cannot lower or raise anything
        return ToolResult(id=tid, status="missing", detail=f"{name} not reported")
    # Strict ">" everywhere: a reading exactly at a limit is still within it
    if value > limits.critical:
        status, floor = "critical", "critical"
    elif value > limits.max:
        status, floor = "exceeded", "high"
    elif value > limits.warn:
        status, floor = "warn", "medium"
    else:
        status, floor = "ok", "low"
    detail = (
        f"{name}={value:g} (warn>{limits.warn:g}, max>{limits.max:g}, "
        f"critical>{limits.critical:g}) -> {status}"
    )
    return ToolResult(id=tid, status=status, detail=detail, floor=floor)


def check_service(months_since: float | None, interval: float) -> ToolResult:
    tid = "service_interval"
    if months_since is None:
        return ToolResult(id=tid, status="missing", detail="last service date not reported")
    overdue = months_since - interval
    if overdue > 0:
        detail = f"{months_since:g} months since service, interval {interval:g}: overdue by {overdue:g}"
        return ToolResult(id=tid, status="overdue", detail=detail, floor="medium")
    detail = f"{months_since:g} months since service, interval {interval:g}: within interval"
    return ToolResult(id=tid, status="ok", detail=detail, floor="low")


def severity_floor(results: list[ToolResult]) -> Severity:
    """Highest floor among the checks; 'low' if no check had an opinion."""
    floors = [r.floor for r in results if r.floor is not None]
    return max(floors, key=severity_rank, default=SEVERITY_ORDER[0])


def run_tools(parsed: ParsedReport) -> list[ToolResult]:
    kind = equipment_kind(parsed.equipment_id)
    if kind is None:
        # Without a known type there are no limits to check against. The floor
        # stays "low", which here means "no evidence", not "safe".
        results = [
            ToolResult(
                id="equipment_lookup",
                status="unknown_equipment",
                detail=f"equipment id {parsed.equipment_id!r} not recognised; no limits applied",
            )
        ]
    else:
        spec = SPECS[kind]
        results = [
            ToolResult(id="equipment_lookup", status="ok", detail=f"{parsed.equipment_id} is a {kind}"),
            check_threshold("bearing_temp_c", parsed.readings.bearing_temp_c, spec.bearing_temp_c),
            check_threshold("vibration_mm_s", parsed.readings.vibration_mm_s, spec.vibration_mm_s),
            check_service(parsed.months_since_service, spec.service_interval_months),
        ]
    floor = severity_floor(results)
    results.append(
        ToolResult(id="severity_floor", status="ok", detail=f"rule floor is {floor}", floor=floor)
    )
    return results
