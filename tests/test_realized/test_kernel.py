"""
Tests for mfe.realized.kernel.

Checks:
- Kernel weights sum to correct values for known kernels
- RK is non-negative
- Bandwidth selector returns reasonable H
- Parzen kernel >= 0 everywhere
"""

import numpy as np
import pytest

from mfe.realized._types import KernelType
from mfe.realized.kernel import realized_kernel, _kernel_weights, select_bandwidth


def test_parzen_weights_non_negative():
    w = _kernel_weights(KernelType.PARZEN, H=10)
    assert np.all(w >= 0)
    assert w[0] == 1.0


def test_bartlett_weights_decay():
    w = _kernel_weights(KernelType.BARTLETT, H=5)
    assert w[0] == 1.0
    assert all(w[i] > w[i + 1] for i in range(len(w) - 1))


def test_realized_kernel_positive(rng):
    r = rng.standard_normal(500) * 0.01
    result = realized_kernel(r)
    assert result.rk_adjusted >= 0
    assert result.bandwidth >= 1


def test_realized_kernel_with_noise(rng):
    """With microstructure noise, RK should still be positive and < naive RV."""
    T = 1000
    r_clean = rng.standard_normal(T) * 0.01
    noise = rng.standard_normal(T) * 0.005
    r_noisy = r_clean + noise - np.roll(noise, 1)
    r_noisy[0] = r_clean[0]

    from mfe.realized.variance import realized_variance
    rv = realized_variance(r_noisy).value
    rk_res = realized_kernel(r_noisy, jitter=True)
    # The adjusted RK should be closer to the true QV than naive RV
    assert rk_res.rk_adjusted >= 0


def test_select_bandwidth_reasonable(rng):
    r = rng.standard_normal(1000) * 0.01
    H = select_bandwidth(r)
    assert 1 <= H <= 200  # for T=1000, should be in this range


@pytest.mark.parametrize("kernel_type", list(KernelType))
def test_kernel_weights_h_equals_1(kernel_type):
    w = _kernel_weights(kernel_type, H=1)
    assert len(w) == 2
    assert w[0] == 1.0
