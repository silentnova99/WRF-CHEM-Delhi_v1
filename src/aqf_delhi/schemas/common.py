"""Shared canonical record primitives."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict


class QualityFlag(str, Enum):
    OK = "OK"
    SUSPECT = "SUSPECT"
    MISSING = "MISSING"


class CanonicalRecord(BaseModel):
    """Base class: every ingested record carries provenance & QC state."""

    model_config = ConfigDict(extra="forbid")

    source: str
    observed_at_utc: datetime
    ingest_ts: datetime
    quality_flag: QualityFlag = QualityFlag.OK