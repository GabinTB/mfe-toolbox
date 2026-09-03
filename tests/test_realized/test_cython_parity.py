"""
Numerical parity tests: Cython extensions must produce results
identical (to float64 precision) to the numpy reference implementations.

These tests are skipped when Cython is not compiled.
"""

import numpy as np
import pytest

try:
    from mfe.realized._core import (
        _autocovariance_sum as _acov_cy,
        _bpv_sum as _bpv_cy,
        _medvar_triplets as _med_cy,
        _hy_sweep as _hy_cy,
    )
    from mfe.multivariate._core import (
        _dcc_q_recursion as _dcc_cy,
        _bekk_scalar_recursion as _bekk_scalar_cy,
    )
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

pytestmark = pytest.mark.skipif(not HAS_CYTHON, reason="Cython not compiled")


@pytest.fixture
def rng():
    return np.random.default_rng(99)


class TestAutocovariance:
    def test_parity_small(self, rng):
        r = np.ascontiguousarray(rng.standard_normal(500) * 0.01)
        H = 10
        cy = np.asarray(_acov_cy(r, H))
        # numpy reference
        np_ref = np.array([float(r[h:] @ r[:len(r)-h]) if h > 0 else float(r @ r)
                           for h in range(H+1)])
        np.testing.assert_allclose(cy, np_ref, rtol=1e-12, atol=1e-15)

    def test_parity_large(self, rng):
        r = np.ascontiguousarray(rng.standard_normal(50_000) * 0.001)
        H = 30
        cy = np.asarray(_acov_cy(r, H))
        np_ref = np.array([float(r[h:] @ r[:len(r)-h]) if h > 0 else float(r @ r)
                           for h in range(H+1)])
        np.testing.assert_allclose(cy, np_ref, rtol=1e-10, atol=0)


class TestBPV:
    def test_parity_skip0(self, rng):
        absr = np.ascontiguousarray(np.abs(rng.standard_normal(1000) * 0.01))
        cy = float(_bpv_cy(absr, 0))
        np_ref = float(np.sum(absr[1:] * absr[:-1]))
        assert abs(cy - np_ref) / max(abs(np_ref), 1e-30) < 1e-12

    def test_parity_skip2(self, rng):
        absr = np.ascontiguousarray(np.abs(rng.standard_normal(1000) * 0.01))
        skip = 2
        cy = float(_bpv_cy(absr, skip))
        np_ref = float(np.sum(absr[skip+1:] * absr[:len(absr)-skip-1]))
        assert abs(cy - np_ref) / max(abs(np_ref), 1e-30) < 1e-12


class TestMedRV:
    def test_parity(self, rng):
        absr = np.ascontiguousarray(np.abs(rng.standard_normal(2000) * 0.01))
        cy = float(_med_cy(absr))
        # reference: np.partition
        triplets = np.stack([absr[:-2], absr[1:-1], absr[2:]], axis=1)
        np_ref = float(np.sum(np.partition(triplets, 1, axis=1)[:, 1] ** 2))
        assert abs(cy - np_ref) / max(abs(np_ref), 1e-30) < 1e-12


class TestHayShi:
    def _make_sync(self, rng, N1, N2):
        p1 = np.exp(np.cumsum(rng.standard_normal(N1) * 0.001))
        t1 = np.sort(rng.uniform(0, 23400, N1))
        p2 = np.exp(np.cumsum(rng.standard_normal(N2) * 0.001))
        t2 = np.sort(rng.uniform(0, 23400, N2))
        r1 = np.ascontiguousarray(np.diff(np.log(p1)))
        a1 = np.ascontiguousarray(t1[:-1]); b1 = np.ascontiguousarray(t1[1:])
        r2 = np.ascontiguousarray(np.diff(np.log(p2)))
        a2 = np.ascontiguousarray(t2[:-1]); b2 = np.ascontiguousarray(t2[1:])
        return r1, a1, b1, r2, a2, b2

    def _hy_numpy(self, r1, a1, b1, r2, a2, b2):
        """Reference: fully vectorized overlap matrix."""
        overlap = (a1[:, None] < b2[None, :]) & (a2[None, :] < b1[:, None])
        return float(r1 @ (overlap @ r2))

    def test_parity_small(self, rng):
        r1, a1, b1, r2, a2, b2 = self._make_sync(rng, 200, 180)
        cy = float(_hy_cy(r1, a1, b1, r2, a2, b2))
        np_ref = self._hy_numpy(r1, a1, b1, r2, a2, b2)
        assert abs(cy - np_ref) < 1e-10 * max(abs(np_ref), 1.0)

    def test_parity_medium(self, rng):
        r1, a1, b1, r2, a2, b2 = self._make_sync(rng, 2000, 1800)
        cy = float(_hy_cy(r1, a1, b1, r2, a2, b2))
        np_ref = self._hy_numpy(r1, a1, b1, r2, a2, b2)
        assert abs(cy - np_ref) < 1e-8 * max(abs(np_ref), 1.0)

    def test_symmetric(self, rng):
        """HY(A, B) == HY(B, A)."""
        r1, a1, b1, r2, a2, b2 = self._make_sync(rng, 500, 400)
        fwd = float(_hy_cy(r1, a1, b1, r2, a2, b2))
        rev = float(_hy_cy(r2, a2, b2, r1, a1, b1))
        assert abs(fwd - rev) < 1e-10 * max(abs(fwd), 1.0)


class TestDCCRecursion:
    def test_parity_k2(self, rng):
        T, K = 500, 2
        z = np.ascontiguousarray(rng.standard_normal((T, K)) * 0.01)
        qb = np.ascontiguousarray(np.eye(K) * 0.01)
        a, b = 0.05, 0.90

        from mfe.multivariate.dcc import _dcc_recursion_numpy
        Q_np = _dcc_recursion_numpy(z, qb, a, b)
        Q_cy = np.asarray(_dcc_cy(z, qb, a, b))

        np.testing.assert_allclose(Q_cy, Q_np, rtol=1e-12, atol=1e-15)

    def test_parity_k5(self, rng):
        T, K = 300, 5
        z = np.ascontiguousarray(rng.standard_normal((T, K)) * 0.01)
        qb = np.ascontiguousarray(np.eye(K) * 0.01)
        a, b = 0.03, 0.92

        from mfe.multivariate.dcc import _dcc_recursion_numpy
        Q_np = _dcc_recursion_numpy(z, qb, a, b)
        Q_cy = np.asarray(_dcc_cy(z, qb, a, b))

        np.testing.assert_allclose(Q_cy, Q_np, rtol=1e-11, atol=1e-14)


class TestBEKKRecursion:
    def test_scalar_parity_k2(self, rng):
        T, K = 500, 2
        eps = np.ascontiguousarray(rng.standard_normal((T, K)) * 0.01)
        C = np.eye(K) * 0.001
        CC = np.ascontiguousarray(C.T @ C)
        H0 = np.ascontiguousarray(np.eye(K) * 0.0001)
        a, b = 0.10, 0.85

        from mfe.multivariate.bekk import _bekk_scalar_recursion_numpy
        H_np, ll_np = _bekk_scalar_recursion_numpy(eps, C, a, b, H0)
        H_cy, ll_cy = _bekk_scalar_cy(eps, CC, a**2, b**2, H0)

        np.testing.assert_allclose(np.asarray(H_cy), H_np, rtol=1e-12, atol=1e-15)
        assert abs(ll_cy - ll_np) / max(abs(ll_np), 1.0) < 1e-10
