"""
trace_unit.py

Contains the TraceUnit class, which encapsulates a single
trace of the agentic system's prediction process for IC50 values.
Each TraceUnit includes identifying information, the agent's
predictions, explanations, and optionally the true IC50 values
for evaluation purposes.

Classes:
- TraceUnit: A Pydantic model representing a single trace of the
  agentic system's prediction for a drug-cell line pair.
"""

from __future__ import annotations
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field

class TraceUnit(BaseModel):

    # basic identifying information
    drug: str
    cell_line: str
    experimental_description: Optional[str] = None
    output_unit: str = "nM"

    # agentic trace (all optional now)
    ic50_pred: Optional[float] = Field(default=None, gt=0)
    confidence: Optional[int] = None
    explanation: Optional[str] = None
    trajecory: Optional[Dict[str, Any]] = None

    # truth and evaluation (all optional)
    ic50_true: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None
