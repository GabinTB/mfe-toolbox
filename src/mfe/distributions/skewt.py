"""
Hansen's Skewed Student-t distribution (Hansen 1994).

Hansen, B.E. (1994): "Autoregressive Conditional Density Estimation",
International Economic Review, 35(3), 705-730.

Parameters:
    nu    : degrees of freedom (nu > 2)
    lam   : skewness parameter (-1 < lam < 1)

PDF:
    f(x; nu, lam) = bc * (1 + 1/(nu-2) * ((bx+a)/(1+/-lam))^2)^{-(nu+1)/2}

with sign depending on whether x < -a/b or x >= -a/b.

The MATLAB mfe-toolbox computes gradients numerically. We provide analytic
score functions for faster GARCH estimation.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma
from scipy.stats import t as student_t

from mfe.utils.typing import FloatArray


def _hansen_constants(nu: float, lam: float) -> tuple[float, float, float]:
    """
    Compute the normalizing constants a, b, c for Hansen's Skew-t.

    a = 4 * lam * c * (nu-2)/(nu-1)
    b = sqrt(1 + 3*lam^2 - a^2)
    c = Gamma((nu+1)/2) / (sqrt(pi*(nu-2)) * Gamma(nu/2))
    """
    c = float(gamma((nu + 1) / 2) / (np.sqrt(np.pi * (nu - 2)) * gamma(nu / 2)))
    a = 4 * lam * c * (nu - 2) / (nu - 1)
    b = float(np.sqrt(1 + 3 * lam ** 2 - a ** 2))
    return a, b, c


def skewt_logpdf(
    x: FloatArray,
    nu: float,
    lam: float,
) -> FloatArray:
    """
    Log-PDF of Hansen's Skew-t distribution.

    Parameters
    ----------
    x   : (T,) standardized residuals
    nu  : degrees of freedom (>2)
    lam : skewness (-1, 1)
    """
    x = np.asarray(x, dtype=np.float64)
    a, b, c = _hansen_constants(nu, lam)

    mask = x < -a / b
    z = np.where(mask,
                 (b * x + a) / (1 - lam),
                 (b * x + a) / (1 + lam))

    log_kernel = -(nu + 1) / 2 * np.log(1 + z ** 2 / (nu - 2))
    log_pdf = np.log(b) + np.log(c) + log_kernel

    return log_pdf


def skewt_score(
    x: FloatArray,
    nu: float,
    lam: float,
) -> tuple[FloatArray, FloatArray]:
    """
    Analytic score of log-skewt PDF with respect to (nu, lam).

    Returns (d_log_f / d_nu, d_log_f / d_lam), each (T,).

    These are used in the outer MLE loop over distribution parameters,
    avoiding the finite-difference approach from the MATLAB source.
    """
    from scipy.special import digamma

    x = np.asarray(x, dtype=np.float64)
    a, b, c = _hansen_constants(nu, lam)

    mask = x < -a / b
    sign = np.where(mask, 1 - lam, 1 + lam)
    z = (b * x + a) / sign

    u = z ** 2 / (nu - 2)
    denom = 1 + u

    # d/d_nu: from log kernel
    d_kernel_dnu = (
        -0.5 * np.log(denom)
        + (nu + 1) / 2 * z ** 2 / ((nu - 2) ** 2 * denom)
    )
    d_c_dnu = 0.5 * (digamma((nu + 1) / 2) - digamma(nu / 2) - 1 / (nu - 2))
    score_nu = d_c_dnu + d_kernel_dnu

    # d/d_lam: via chain rule through z(lam) and a(lam), b(lam)
    # This is the expensive part — see Hansen (1994) Appendix
    da_dlam = 4 * c * (nu - 2) / (nu - 1)  # simplified: ignoring dc/dlam for now
    db_dlam = (6 * lam - 2 * a * da_dlam) / (2 * b)

    dz_dlam_pos = (db_dlam * x + da_dlam) * (1 + lam) - (b * x + a)
    dz_dlam_pos /= (1 + lam) ** 2
    dz_dlam_neg = (db_dlam * x + da_dlam) * (1 - lam) + (b * x + a)
    dz_dlam_neg /= (1 - lam) ** 2

    dz_dlam = np.where(mask, dz_dlam_neg, dz_dlam_pos)
    d_kernel_dlam = -(nu + 1) * z * dz_dlam / ((nu - 2) * denom)
    score_lam = db_dlam / b + d_kernel_dlam

    return score_nu, score_lam


def skewt_ppf(
    p: FloatArray,
    nu: float,
    lam: float,
) -> FloatArray:
    """
    Quantile function (inverse CDF) of Hansen's Skew-t.
    Used for VaR/ES computation.
    """
    p = np.asarray(p, dtype=np.float64)
    a, b, c = _hansen_constants(nu, lam)

    p1 = (1 - lam) / 2  # mass in the left tail

    result = np.where(
        p < p1,
        (student_t.ppf(p / (1 - lam), df=nu) * (1 - lam) - a) / b,
        (student_t.ppf((p - (1 - lam) / 2) / (1 + lam) + 0.5, df=nu) * (1 + lam) - a) / b,
    )
    return result
