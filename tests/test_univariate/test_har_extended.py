"""Tests for extended HAR — matrix intervals, MODIFIED, HAR-J, forecast."""
import numpy as np
import pytest
from mfe.univariate import har_rv, har_rv_j, har_forecast


@pytest.fixture
def rv_series(rng):
    return np.abs(rng.standard_normal(500)) * 0.01


class TestHARIntervals:
    def test_vector_notation(self, rv_series):
        res = har_rv(rv_series, p=[1, 5, 22])
        assert len(res.params) == 4  # const + 3 regressors
        assert res.spec == "standard"

    def test_matrix_notation_overlapping(self, rv_series):
        res = har_rv(rv_series, p=[[1,1],[1,5],[1,22]])
        assert len(res.params) == 4
        np.testing.assert_allclose(
            res.r_squared,
            har_rv(rv_series, p=[1,5,22]).r_squared,
            atol=1e-10,
        )

    def test_matrix_notation_nonoverlapping(self, rv_series):
        res = har_rv(rv_series, p=[[1,1],[2,5],[6,22]])
        assert len(res.params) == 4
        assert res.param_names[2] == "RV_avg2to5"
        assert res.param_names[3] == "RV_avg6to22"

    def test_modified_spec(self, rv_series):
        res = har_rv(rv_series, p=[1,5,22], spec="modified")
        assert res.spec == "modified"
        # Non-overlapping intervals
        assert res.intervals[1] == (2, 5)
        assert res.intervals[2] == (6, 22)

    def test_modified_same_r2_as_standard(self, rv_series):
        """Modified and standard are equivalent reparameterisations."""
        r_std = har_rv(rv_series, p=[1,5,22], spec="standard")
        r_mod = har_rv(rv_series, p=[1,5,22], spec="modified")
        np.testing.assert_allclose(r_std.r_squared, r_mod.r_squared, atol=1e-10)

    def test_two_component_har(self, rv_series):
        res = har_rv(rv_series, p=[1, 5])
        assert len(res.params) == 3

    def test_horizon_5(self, rv_series):
        res = har_rv(rv_series, p=[1,5,22], horizon=5)
        assert np.isfinite(res.r_squared)


class TestHARJ:
    def test_param_count(self, rv_series):
        jump = np.maximum(rv_series - rv_series * 0.8, 0.0)
        res = har_rv_j(rv_series, jump)
        assert len(res.params) == 5   # const + 3 HAR + jump

    def test_jump_name_in_params(self, rv_series):
        jump = np.maximum(rv_series - rv_series * 0.8, 0.0)
        res = har_rv_j(rv_series, jump)
        assert "Jump" in res.param_names[-1]

    def test_se_nonnegative(self, rv_series):
        jump = np.maximum(rv_series - rv_series * 0.8, 0.0)
        res = har_rv_j(rv_series, jump)
        assert np.all(res.std_errors >= 0)


class TestHARForecast:
    def test_forecast_shape(self, rv_series):
        res = har_rv(rv_series)
        fc = har_forecast(res, rv_series[-30:], horizon=5)
        assert fc.shape == (5,)

    def test_forecast_positive(self, rv_series):
        res = har_rv(rv_series)
        fc = har_forecast(res, rv_series[-30:], horizon=10)
        assert np.all(fc > 0)

    def test_forecast_horizon_1(self, rv_series):
        res = har_rv(rv_series)
        fc = har_forecast(res, rv_series[-30:], horizon=1)
        assert len(fc) == 1
