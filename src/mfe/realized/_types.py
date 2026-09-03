"""
Types and result classes for the realized module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from mfe.utils.typing import FloatArray


class SamplingType(str, Enum):
    """Price sampling scheme."""
    CALENDAR_TIME = "CalendarTime"
    CALENDAR_UNIFORM = "CalendarUniform"
    BUSINESS_TIME = "BusinessTime"
    BUSINESS_UNIFORM = "BusinessUniform"
    FIXED = "Fixed"


class TimeType(str, Enum):
    """Time representation of the tick timestamps."""
    WALL = "Wall"          # Python datetime or POSIX float
    SECONDS = "Seconds"    # seconds since start of session
    UNIT = "Unit"          # fractional day [0, 1]
    BUSINESS = "Business"  # tick index


class KernelType(str, Enum):
    """Kernel weight function for realized kernel estimator."""
    PARZEN = "Parzen"
    BARTLETT = "Bartlett"
    TUKEY_HANNING = "Tukey-Hanning"
    CUBIC = "Cubic"
    EPANECHNIKOV = "Epanechnikov"
    FLAT_TOP = "FlatTop"  # Barndorff-Nielsen et al. 2008 recommended


@dataclass
class RealizedResult:
    """Generic container for realized estimator output."""
    value: float
    subsampled_value: float | None = None
    debiased_value: float | None = None
    n_returns: int = 0
    sampling_type: SamplingType = SamplingType.CALENDAR_TIME
    sampling_interval: float | int = 300  # seconds or ticks
    diagnostics: dict = field(default_factory=dict)


@dataclass
class RealizedKernelResult:
    """Output from realized_kernel()."""
    rk: float                          # realized kernel estimate
    rk_adjusted: float                 # end-point corrected (jittered) version
    bandwidth: int                     # number of lags H used
    noise_variance: float              # estimated microstructure noise variance
    iq_lower_bound: float              # lower bound for integrated quarticity
    kernel_type: KernelType = KernelType.PARZEN
    n_returns: int = 0
    diagnostics: dict = field(default_factory=dict)


@dataclass
class RealizedCovarianceResult:
    """Output from realized_covariance() or realized_hayashi_yoshida()."""
    cov: FloatArray                    # (K, K) realized covariance matrix
    method: str = "synchronous"        # "synchronous" | "hayashi-yoshida"
    n_assets: int = 0
    n_returns: int = 0                 # per asset (or minimum across assets)
