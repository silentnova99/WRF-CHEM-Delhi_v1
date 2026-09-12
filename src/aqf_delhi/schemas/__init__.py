"""Canonical record schemas for Phase-1 data ingestion."""

from aqf_delhi.schemas.common import CanonicalRecord, QualityFlag
from aqf_delhi.schemas.cpcb import CpcbObservation
from aqf_delhi.schemas.firms import FireDetection
from aqf_delhi.schemas.gfs import GfsFieldAsset

__all__ = [
    "CanonicalRecord",
    "QualityFlag",
    "CpcbObservation",
    "FireDetection",
    "GfsFieldAsset",
]