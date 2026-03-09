from __future__ import annotations

"""Performance tracking stubs."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PerformanceSnapshot:
    """Basic performance snapshot."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    window: str
    notes: Optional[str] = None
