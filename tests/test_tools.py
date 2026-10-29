import pytest
from pydantic import ValidationError

from field_triage.equipment import SPECS, Limits, equipment_kind
from field_triage.schema import ParsedReport, Readings
from field_triage.tools import check_service, check_threshold, run_tools, severity_floor

PUMP_TEMP = SPECS["pump"].bearing_temp_c  # warn 80, max 90, critical 105


@pytest.mark.parametrize(
    "value, status, floor",
    [
        (80, "ok", "low"),  # exactly at warn is still ok
        (80.1, "warn", "medium"),
        (90, "warn", "medium"),  # exactly at max is not "exceeded"
        (90.1, "exceeded", "high"),
        (105, "exceeded", "high"),  # exactly at critical is not "critical"
        (105.1, "critical", "critical"),
    ],
)
def test_threshold_boundaries(value, status, floor):
    r = check_threshold("bearing_temp_c", value, PUMP_TEMP)
    assert (r.status, r.floor) == (status, floor)


def test_missing_reading_gives_no_floor():
    # Must be None, not "low": a missing reading must not look like a passing one
    r = check_threshold("bearing_temp_c", None, PUMP_TEMP)
    assert r.status == "missing" and r.floor is None


def test_zero_reading_is_a_reading_not_missing():
    # Guards against `if not value` style truthiness bugs
    assert check_threshold("vibration_mm_s", 0.0, Limits(4.5, 7.1, 11.0)).status == "ok"


def test_service_exactly_at_interval_is_not_overdue():
    assert check_service(12, 12).status == "ok"


def test_service_overdue_reports_amount():
    r = check_service(14, 12)
    assert r.status == "overdue" and r.floor == "medium" and "overdue by 2" in r.detail


def test_service_missing():
    assert check_service(None, 12).floor is None


def test_limits_differ_by_equipment_type():
    # 80C is fine on a pump but past warn on a fan: limits must not be shared
    pump = run_tools(ParsedReport(equipment_id="P-1", readings=Readings(bearing_temp_c=80)))
    fan = run_tools(ParsedReport(equipment_id="F-1", readings=Readings(bearing_temp_c=80)))
    by_id = lambda rs: {r.id: r for r in rs}  # noqa: E731
    assert by_id(pump)["threshold:bearing_temp_c"].status == "ok"
    assert by_id(fan)["threshold:bearing_temp_c"].status == "warn"


@pytest.mark.parametrize(
    "tag, kind",
    [("P-104", "pump"), ("p104", "pump"), (" M-7 ", "motor"), ("P-104A", "pump"),
     ("CV-12", None), ("Pump", None), ("", None), (None, None)],
)
def test_equipment_kind(tag, kind):
    assert equipment_kind(tag) == kind


def test_floor_is_max_and_ignores_missing():
    parsed = ParsedReport(
        equipment_id="P-104",
        readings=Readings(bearing_temp_c=92, vibration_mm_s=None),
        months_since_service=14,
    )
    results = run_tools(parsed)
    # temp exceeded -> high beats service overdue -> medium; missing vibration adds nothing
    assert severity_floor(results) == "high"
    assert results[-1].id == "severity_floor" and results[-1].floor == "high"


def test_floor_defaults_low_when_nothing_known():
    assert severity_floor([]) == "low"
    results = run_tools(ParsedReport(equipment_id="P-104"))
    assert results[-1].floor == "low"


def test_unknown_equipment_applies_no_limits():
    # Readings are high, but we have no limits for an unrecognised tag
    results = run_tools(ParsedReport(equipment_id="X-9", readings=Readings(bearing_temp_c=500)))
    assert [r.id for r in results] == ["equipment_lookup", "severity_floor"]
    assert results[0].status == "unknown_equipment"


def test_negative_service_months_rejected_at_parse_boundary():
    with pytest.raises(ValidationError):
        ParsedReport(months_since_service=-3)
