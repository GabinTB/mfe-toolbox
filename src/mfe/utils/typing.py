"""Shared type aliases for the mfe package."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

# Core numeric arrays
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

# Sampling types (match MATLAB mfe-toolbox convention)
SamplingType = Literal["CalendarTime", "CalendarUniform", "BusinessTime", "BusinessUniform", "Fixed"]
TimeType = Literal["Wall", "BusinessTime", "Seconds", "Unit"]
KernelType = Literal["Parzen", "Bartlett", "Tukey-Hanning", "BNHLS", "Cubic", "Epanechnikov"]

__all__ = [
    "FloatArray",
    "IntArray",
    "BoolArray",
    "SamplingType",
    "TimeType",
    "KernelType",
]
