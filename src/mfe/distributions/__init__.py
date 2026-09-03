"""
mfe.distributions — Fat-tailed and multivariate distributions.

skewt_logpdf    Hansen (1994) Skew-t log-PDF with analytic score
skewt_ppf       Skew-t quantile function (VaR/ES)
ged_logpdf      Generalized Error Distribution log-PDF
ged_ppf         GED quantile function
mvnorm_loglik   Multivariate normal log-likelihood (time-varying Sigma_t)
mahalanobis     Mahalanobis distances under a covariance sequence
standardize_mvn Extract standardized multivariate residuals
"""

from mfe.distributions.skewt import skewt_logpdf, skewt_score, skewt_ppf
from mfe.distributions.ged import ged_logpdf, ged_ppf, ged_score
from mfe.distributions.mvnorm import mvnorm_loglik, mvnorm_loglik_t, mahalanobis, standardize_mvn

__all__ = [
    "skewt_logpdf", "skewt_score", "skewt_ppf",
    "ged_logpdf", "ged_ppf", "ged_score",
    "mvnorm_loglik", "mvnorm_loglik_t", "mahalanobis", "standardize_mvn",
]
