"""
mfe.tests_stat — Statistical tests for financial time series.

Serial correlation
    ljung_box         Ljung-Box Q statistic (not robust to heteroskedasticity)
    lm_test           HAC-robust LM serial correlation test (MFE lmtest1)

Conditional heteroskedasticity
    arch_lm           Engle (1982) ARCH-LM test

Forecast evaluation
    mincer_zarnowitz  MZ regression-based forecast evaluation
    diebold_mariano   DM test for equal predictive accuracy (MSE/MAE/QLIKE)
"""

from mfe.tests_stat.serial import ljung_box, lm_test, LjungBoxResult, LMTestResult
from mfe.tests_stat.arch_lm import arch_lm, ARCHLMResult
from mfe.tests_stat.forecast_eval import mincer_zarnowitz, diebold_mariano, MZResult, DMResult

__all__ = [
    "ljung_box", "lm_test", "LjungBoxResult", "LMTestResult",
    "arch_lm", "ARCHLMResult",
    "mincer_zarnowitz", "diebold_mariano", "MZResult", "DMResult",
]
