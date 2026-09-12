"""Validator stage — structural + statistical QC on canonical records.

Pydantic already enforces hard schema rules at construction time; this
stage applies *cross-record* rules (bounds, spike filters, dedupe) and
returns a QC report so operators can audit every drop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from aqf_delhi.config import QualityConfig
from aqf_delhi.schemas.cpcb import CpcbObservation
from aqf_delhi.schemas.firms import FireDetection
from aqf_delhi.schemas.gfs import GfsFieldAsset


@dataclass
class QCReport:
    kept: int = 0
    dropped: int = 0
    reasons: dict = field(default_factory=dict)

    def record_drop(self, reason: str) -> None:
        self.dropped += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def validate_cpcb(
    records: Iterable[CpcbObservation], q: QualityConfig
) -> tuple[list[CpcbObservation], QCReport]:
    report = QCReport()
    kept: list[CpcbObservation] = []
    for rec in records:
        if rec.value < q.cpcb.explicit_lower_bound:
            report.record_drop("below_lower_bound")
            continue
        if rec.value > q.cpcb.max_value_ug_m3:
            report.record_drop("above_physical_ceiling")
            continue
        if rec.quality_flag.value == "MISSING":
            report.record_drop("marked_missing")
            continue
        kept.append(rec)
    report.kept = len(kept)
    return kept, report


def mad_spike_filter(
    batches: Iterable[tuple[str, str, list[float]]],
    cutoff_sigma: float,
) -> tuple[int, int]:
    """MAD-based spike suppression per (station, metric) batch.

    Returns (flagged, kept). Rows deemed statistical outliers by this
    simple test are counted; the actual flagging is left to the caller so
    the canonical table can carry ``SUSPECT`` instead of dropping data.
    """
    import statistics

    flagged = 0
    kept = 0
    for _station, _metric, values in batches:
        if len(values) < 3:
            kept += len(values)
            continue
        median = statistics.median(values)
        mad = statistics.median([abs(v - median) for v in values]) or 1.0
        for v in values:
            if abs(v - median) / (1.4826 * mad) > cutoff_sigma:
                flagged += 1
            else:
                kept += 1
    return flagged, kept


def validate_firms(
    records: Iterable[FireDetection], q: QualityConfig
) -> tuple[list[FireDetection], QCReport]:
    report = QCReport()
    seen: set[str] = set()
    kept: list[FireDetection] = []
    lo, hi = q.firms.frp_range_mw
    for rec in records:
        if rec.fire_id in seen:
            report.record_drop("duplicate_fire_id")
            continue
        seen.add(rec.fire_id)
        if not (lo <= rec.frp_mw <= hi):
            report.record_drop("frp_out_of_range")
            continue
        if rec.confidence_percent < q.firms.confidence_min:
            report.record_drop("low_confidence")
            continue
        kept.append(rec)
    report.kept = len(kept)
    return kept, report


def validate_gfs(
    records: Iterable[GfsFieldAsset], q: QualityConfig
) -> tuple[list[GfsFieldAsset], QCReport]:
    report = QCReport()
    seen: set[str] = set()
    kept: list[GfsFieldAsset] = []
    for rec in records:
        if rec.h_key in seen:  # duplicate (run, lead, variable, level)
            report.record_drop("duplicate_asset")
            continue
        seen.add(rec.h_key)
        if rec.missing_fraction > q.gfs.max_missing_fraction:
            report.record_drop("excessive_missing_data")
            continue
        kept.append(rec)
    report.kept = len(kept)
    return kept, report


VALIDATORS = {
    "cpcb": validate_cpcb,
    "firms": validate_firms,
    "gfs": validate_gfs,
}