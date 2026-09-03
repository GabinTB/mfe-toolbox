# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
# cython: nonecheck=False, initializedcheck=False
"""
Cython extensions for mfe.realized hot paths.

Functions
---------
_autocovariance_sum     RK kernel inner loop
_bpv_sum                BPV skip-k inner loop
_medvar_triplets        MedRV median-of-three per triplet
_hy_sweep               Hayashi-Yoshida O((N1+N2)log(N1+N2)) estimator
_refresh_time_indices   Refresh-time index computation
"""

import numpy as np
cimport numpy as np
from numpy cimport ndarray, float64_t, int64_t
from libc.math cimport fabs, sqrt, log


# ────────────────────────────────────────────────────────────────────────────
# 1. Autocovariance (realized kernel inner loop)
# ────────────────────────────────────────────────────────────────────────────

def _autocovariance_sum(double[::1] returns, int H):
    """gamma[h] = sum_{t=h}^{T-1} r[t]*r[t-h], h=0..H"""
    cdef int T = returns.shape[0], h, t
    cdef double acc

    gamma_arr = np.zeros(H + 1, dtype=np.float64)
    cdef double[::1] g = gamma_arr
    cdef double[::1] r = returns

    acc = 0.0
    for t in range(T):
        acc += r[t] * r[t]
    g[0] = acc

    for h in range(1, H + 1):
        acc = 0.0
        for t in range(h, T):
            acc += r[t] * r[t - h]
        g[h] = acc

    return gamma_arr


# ────────────────────────────────────────────────────────────────────────────
# 2. BPV sum
# ────────────────────────────────────────────────────────────────────────────

def _bpv_sum(double[::1] abs_returns, int skip):
    """sum_{t=skip+1}^{T-1} |r[t]| * |r[t-skip-1]|"""
    cdef int T = abs_returns.shape[0], t, gap = skip + 1
    cdef double acc = 0.0
    cdef double[::1] a = abs_returns
    for t in range(gap, T):
        acc += a[t] * a[t - gap]
    return acc


# ────────────────────────────────────────────────────────────────────────────
# 3. MedRV triplet sum
# ────────────────────────────────────────────────────────────────────────────

cdef inline double _median3(double a, double b, double c) nogil:
    """Branchless median of three values."""
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        b = a
    return b

def _medvar_triplets(double[::1] abs_returns):
    """sum_{t=1}^{T-2} median(|r[t-1]|, |r[t]|, |r[t+1]|)^2"""
    cdef int T = abs_returns.shape[0], t
    cdef double med, acc = 0.0
    cdef double[::1] a = abs_returns
    for t in range(1, T - 1):
        med = _median3(a[t - 1], a[t], a[t + 1])
        acc += med * med
    return acc


# ────────────────────────────────────────────────────────────────────────────
# 4. Hayashi-Yoshida sweep-line  O((N1+N2)*k) where k = avg open r1 count
# ────────────────────────────────────────────────────────────────────────────

def _hy_sweep(
    double[::1] r1, double[::1] a1, double[::1] b1,
    double[::1] r2, double[::1] a2, double[::1] b2,
):
    """
    Hayashi-Yoshida estimator via event sweep with swap-remove open-list.

    Correctness: when r2[t] opens, we add r2[t] to ALL currently-open r1
    accumulators. This correctly handles r2 intervals that both open and
    close while r1[s] is active.

    Complexity: O((N1+N2)*log(N1+N2)) for sort + O((N1+N2)*k_open) for sweep,
    where k_open = max simultaneous open r1 intervals.
    For typical HFT data (intervals << trading day) k_open ~ O(1).

    The open_list uses swap-remove for O(1) deletion.
    """
    cdef int N1 = r1.shape[0]
    cdef int N2 = r2.shape[0]
    cdef int n_events = 2 * (N1 + N2)
    cdef int i, k = 0, ev, idx, etype, n_open = 0, j

    ev_time_arr = np.empty(n_events, dtype=np.float64)
    ev_type_arr = np.empty(n_events, dtype=np.int32)
    ev_idx_arr  = np.empty(n_events, dtype=np.int32)
    cdef double[::1] ev_time = ev_time_arr
    cdef int[::1]    ev_type = ev_type_arr
    cdef int[::1]    ev_idx  = ev_idx_arr

    for i in range(N1):
        ev_time[k] = a1[i]; ev_type[k] = 1; ev_idx[k] = i; k += 1
        ev_time[k] = b1[i]; ev_type[k] = 2; ev_idx[k] = i; k += 1
    for i in range(N2):
        ev_time[k] = a2[i]; ev_type[k] = 0; ev_idx[k] = i; k += 1
        ev_time[k] = b2[i]; ev_type[k] = 3; ev_idx[k] = i; k += 1

    # Sort: type tie-break ensures r2-open(0) < r1-open(1) < r1-close(2) < r2-close(3)
    # lexsort: primary key = ev_time, secondary = ev_type (0 < 1 < 2 < 3)
    # np.lexsort keys are applied right-to-left, so pass (type, time)
    order = np.lexsort((ev_type_arr, ev_time_arr)).astype(np.int64)
    cdef int64_t[::1] ord_v = order

    # Per-r1 accumulator: sum of r2[t] values overlapping r1[s]
    r1_accum_arr = np.zeros(N1, dtype=np.float64)
    cdef double[::1] r1_accum = r1_accum_arr

    # Compact open-list for O(1) swap-remove
    open_list_arr = np.empty(N1, dtype=np.int32)
    cdef int[::1] open_list = open_list_arr
    open_pos_arr = np.zeros(N1, dtype=np.int32)
    cdef int[::1] open_pos = open_pos_arr
    is_open_arr = np.zeros(N1, dtype=np.int32)
    cdef int[::1] is_open = is_open_arr

    cdef double total = 0.0, rv2, active_r2_sum = 0.0
    cdef int last_pos, last_idx

    for ev in range(n_events):
        idx   = ev_idx[ord_v[ev]]
        etype = ev_type[ord_v[ev]]

        if etype == 0:
            # r2[idx] opens: (1) track active sum; (2) add to all open r1
            rv2 = r2[idx]
            active_r2_sum += rv2
            for j in range(n_open):
                r1_accum[open_list[j]] += rv2

        elif etype == 1:
            # r1[idx] opens: seed accumulator with r2 already active, then add to open list
            r1_accum[idx] = active_r2_sum
            open_pos[idx] = n_open
            open_list[n_open] = idx
            is_open[idx] = 1
            n_open += 1

        elif etype == 2:
            # r1[idx] closes: finalize, swap-remove from open list
            if is_open[idx]:
                total += r1[idx] * r1_accum[idx]
                last_pos = n_open - 1
                last_idx = open_list[last_pos]
                open_list[open_pos[idx]] = last_idx
                open_pos[last_idx] = open_pos[idx]
                n_open -= 1
                is_open[idx] = 0

        else:
            # r2[idx] closes: update active sum only (r1 accumulators already got rv2 on open)
            active_r2_sum -= r2[idx]

    return total


# ────────────────────────────────────────────────────────────────────────────
# 5. Refresh-time index computation
# ────────────────────────────────────────────────────────────────────────────

def _refresh_time_indices(list times_list):
    """
    Compute refresh-time indices for K asynchronous series.
    Returns (sync_indices (M,K), sync_times (M,)).
    """
    cdef int K = len(times_list)
    times = [np.asarray(t, dtype=np.float64) for t in times_list]
    lengths = [len(t) for t in times]

    cdef double t_start = max(float(t[0]) for t in times)

    cur_idx = np.array([
        max(0, int(np.searchsorted(times[k], t_start, side='right')) - 1)
        for k in range(K)
    ], dtype=np.int64)

    sync_times_list = []
    sync_idx_list   = []

    for _ in range(min(lengths) * 2):
        t_refresh = max(float(times[k][cur_idx[k]]) for k in range(K))
        sync_times_list.append(t_refresh)
        sync_idx_list.append(cur_idx.copy())

        new_idx = np.array([
            int(np.searchsorted(times[k], t_refresh, side='left'))
            for k in range(K)
        ], dtype=np.int64)

        if any(new_idx[k] >= lengths[k] for k in range(K)):
            break
        cur_idx = new_idx

    return (
        np.array(sync_idx_list, dtype=np.int64),
        np.array(sync_times_list, dtype=np.float64),
    )
