"""Synthetic equipment limits. Real limits would come from the OEM / CMMS.

Limits live in code, not in the prompt or the report, so nothing the LLM or the
technician writes can change them.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    warn: float
    max: float
    # Explicit number instead of "max * 1.15": float rounding makes a
    # percentage margin flip at the boundary (90 * 1.15 == 103.49999...).
    critical: float


@dataclass(frozen=True)
class EquipmentSpec:
    bearing_temp_c: Limits
    vibration_mm_s: Limits
    service_interval_months: float


SPECS: dict[str, EquipmentSpec] = {
    "pump": EquipmentSpec(Limits(80, 90, 105), Limits(4.5, 7.1, 11.0), 12),
    "motor": EquipmentSpec(Limits(85, 95, 110), Limits(4.5, 7.1, 11.0), 12),
    "compressor": EquipmentSpec(Limits(90, 105, 120), Limits(5.6, 9.0, 14.0), 6),
    "fan": EquipmentSpec(Limits(75, 85, 100), Limits(4.5, 7.1, 11.0), 12),
    "gearbox": EquipmentSpec(Limits(85, 100, 115), Limits(3.5, 5.6, 9.0), 18),
}

# Site tag convention: letter prefix = equipment type, e.g. P-104 is a pump.
PREFIXES = {"P": "pump", "M": "motor", "C": "compressor", "F": "fan", "G": "gearbox"}

_TAG = re.compile(r"([A-Z]+)-?\d+[A-Z]?")


def equipment_kind(equipment_id: str | None) -> str | None:
    """Map a tag like 'P-104' to 'pump'. None if the tag is not recognised."""
    if not equipment_id:
        return None
    m = _TAG.fullmatch(equipment_id.strip().upper())
    return PREFIXES.get(m.group(1)) if m else None
