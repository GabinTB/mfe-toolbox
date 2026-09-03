"""
Build Cython extensions for mfe.

Usage:
    python setup_cython.py build_ext --inplace

Or via uv:
    uv run python setup_cython.py build_ext --inplace

Extensions compiled
-------------------
mfe.realized._core
    _autocovariance_sum     realized kernel inner loop
    _bpv_sum                BPV/skip-k inner loop
    _medvar_triplets        MedRV median-of-three triplets
    _hy_sweep               Hayashi-Yoshida O((N1+N2)log(N1+N2)) sweep
    _refresh_time_indices   refresh-time sync for K series

mfe.multivariate._core
    _dcc_q_recursion        DCC Q-process recursion
    _dcc_corr_loglik        DCC step-2 correlation log-likelihood
    _bekk_scalar_recursion  scalar BEKK recursion + Gaussian QML
    _bekk_diagonal_recursion diagonal BEKK recursion + Gaussian QML
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="mfe.realized._core",
        sources=["src/mfe/realized/_core.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3", "-march=native", "-ffast-math", "-funroll-loops"],
    ),
    Extension(
        name="mfe.multivariate._core",
        sources=["src/mfe/multivariate/_core.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3", "-march=native", "-ffast-math", "-funroll-loops"],
    ),
]

setup(
    name="mfe",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
        },
        annotate=True,  # generates .html annotation for profiling
    ),
    package_dir={"": "src"},
)
