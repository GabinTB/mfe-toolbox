"""
mfe.multivariate — Multivariate volatility models.

All of these are missing from the `arch` package (as of 2026).

Models
------
DCC        Dynamic Conditional Correlation (Engle 2002); variants: dcc, cdcc, deco
CCC        Constant Conditional Correlation (Bollerslev 1990)
BEKK       BEKK model (Engle & Kroner 1995); variants: scalar, diagonal
OGARCH     Orthogonal GARCH (Alexander 2001)
GOGARCH    Generalized Orthogonal GARCH (van der Weide 2002); rotations: ica, moments
RCC        Rotated Conditional Correlation (Noureldin, Shephard & Sheppard 2014)
"""

from mfe.multivariate.dcc import DCC
from mfe.multivariate.ccc import CCC
from mfe.multivariate.bekk import BEKK, BEKKVariant
from mfe.multivariate.gogarch import GOGARCH, OGARCH, GOGARCHResult
from mfe.multivariate.rcc import RCC, RCCResult

__all__ = ["DCC", "CCC", "BEKK", "BEKKVariant", "GOGARCH", "OGARCH", "GOGARCHResult", "RCC", "RCCResult"]
