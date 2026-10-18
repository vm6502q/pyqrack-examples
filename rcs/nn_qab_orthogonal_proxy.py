# Nearest-neighbor RCS on QrackAceBackend, validated AT SCALE by a K-patch
# (K = 2, 3, 4, ...) proxy whose cuts are ORTHOGONAL to ACE's own patch seams.
#
# By Dan Strano and (Anthropic) Claude.
#
# Same circuit family, gate set and ACE settings as nn_qab.py. The difference
# is how the samples are scored:
#
#   * ACE cuts its layout into patches ONLY along the long dimension
#     (long_range_columns). long_range_rows never cuts it into patches
#     (asserted below), so every ACE seam is a full "wall" spanning all rows.
#
#   * The validation proxy cuts along the OTHER dimension, orthogonal to the
#     long_range_columns seams: the R rows are split into K contiguous bands
#     (as equal as possible), with the torus wrap between the last and first
#     row also cut. Every coupler that crosses a band boundary is removed from
#     the circuit; nothing else changes (same random draw, same order).
#
#   * The proxy circuit then factorizes into K independent bands, and its
#     ideal distribution is exactly p(x) = prod_k p_k(x_k). Each band is
#     simulated exactly on its own, so the ideal is computable for widths
#     where 2^width is not. K must not exceed R (the short grid dimension).
#
#   * Because the cuts are orthogonal, EVERY proxy band still runs straight
#     through every ACE seam. ACE gets no help from the cuts: its boundary
#     reconciliation is exercised exactly as in the full circuit, and it
#     cannot line its patches up with the proxy's.
#
# The score is the same normalized XEB as nn_qab.py's calc_stats,
#     XEB = (N * sum_b f_b p_b - 1) / (N * sum_b p_b^2 - 1),   N = 2^width,
# evaluated in factorized form:  N p = prod_k (N_k p_k),  N sum p^2 = prod_k (N_k sum p_k^2).
#
# With --full (small widths only), the script ALSO runs ACE on the unpatched
# circuit (the same draw, cut couplers included) and scores it against the
# full 2^width ideal, so you can show the proxy tracks the full-circuit XEB
# wherever both are computable.
#
# usage: python3 nn_qab_orthogonal_proxy.py [width] [depth] [long_range_columns=4] [long_range_rows=4] [proxy_patches=2] [--full]

import ctypes
import math
import random
import sys
import time

from collections import Counter
from importlib.metadata import version

import numpy as np
from pyqrack import QrackSimulator, QrackAceBackend
from pyqrack.qrack_system import Qrack


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (row_len, col_len)


def u(sim, q, th, ph, lm):
    sim.u(q, th, ph, lm)


def cx(sim, q1, q2):
    sim.mcx([q1], q2)


def cy(sim, q1, q2):
    sim.mcy([q1], q2)


def cz(sim, q1, q2):
    sim.mcz([q1], q2)


def acx(sim, q1, q2):
    sim.macx([q1], q2)


def acy(sim, q1, q2):
    sim.macy([q1], q2)


def acz(sim, q1, q2):
    sim.macz([q1], q2)


def swap(sim, q1, q2):
    sim.swap(q1, q2)


def iswap(sim, q1, q2):
    sim.iswap(q1, q2)


def iiswap(sim, q1, q2):
    sim.adjiswap(q1, q2)


def pswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)


def mswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.iswap(q1, q2)


def nswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.adjiswap(q1, q2)


def run_circuit(sim, circ):
    for g in circ:
        g[0](sim, *g[1:])


# ---------------------------------------------------------------------------
# Circuit: identical generator to nn_qab.py
# ---------------------------------------------------------------------------

def random_circuit(width, depth):
    lcv_range = range(width)
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    qc = []
    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
            # Keep it Haar-random towards the poles:
            th = math.asin(th / math.pi)
            qc.append((u, i, th, ph, lm))

        # Nearest-neighbor couplers:
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
                temp_col = temp_col + (1 if (gate & 1) else 0)

                if temp_row < 0:
                    temp_row = temp_row + row_len
                if temp_col < 0:
                    temp_col = temp_col + col_len
                if temp_row >= row_len:
                    temp_row = temp_row - row_len
                if temp_col >= col_len:
                    temp_col = temp_col - col_len

                b1 = col * row_len + row
                b2 = temp_col * row_len + temp_row

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                qc.append((g, b1, b2))

    return qc


# ---------------------------------------------------------------------------
# Orthogonal cut
# ---------------------------------------------------------------------------
#
# Qubit index is b = col * row_len + row, with "row" the LONG coordinate
# (ACE's column index, which long_range_columns cuts) and "col" the SHORT one
# (ACE's row index, which long_range_rows does not cut). Banding the SHORT
# coordinate gives each band a contiguous index range
# [offset_k, offset_k + n_k), so a sample x splits as
#     x_k = (x >> offset_k) & (2^n_k - 1).

def orthogonal_bands(width, k):
    """Split the short grid dimension into k contiguous bands of rows.
    Returns [(offset_k, n_k)] in qubit-index units."""
    row_len, col_len = factor_width(width)
    if k < 2:
        raise RuntimeError("proxy_patches must be at least 2.")
    if k > col_len:
        raise RuntimeError(
            f"width {width} factors to a {col_len} x {row_len} grid; "
            f"at most {col_len} orthogonal proxy patches are possible."
        )
    bounds = [(i * col_len) // k for i in range(k + 1)]
    return [(bounds[i] * row_len, (bounds[i + 1] - bounds[i]) * row_len) for i in range(k)]


def make_proxy(qc, bands):
    """Drop every coupler that crosses a band boundary (including the torus
    wrap); return (proxy circuit, [gates for each band, locally indexed], #removed)."""
    def band_of(q):
        for i, (off, n) in enumerate(bands):
            if off <= q < off + n:
                return i
        raise ValueError(q)

    proxy = []
    per_band = [[] for _ in bands]
    removed = 0
    for g in qc:
        if g[0] is u:
            i = band_of(g[1])
            proxy.append(g)
            per_band[i].append((g[0], g[1] - bands[i][0]) + tuple(g[2:]))
            continue
        i, j = band_of(g[1]), band_of(g[2])
        if i != j:
            removed += 1
            continue
        proxy.append(g)
        per_band[i].append((g[0], g[1] - bands[i][0], g[2] - bands[i][0]))
    return proxy, per_band, removed


# ---------------------------------------------------------------------------
# Exact ideal of one half, straight into a numpy buffer
# ---------------------------------------------------------------------------

def ideal_probs(n, gates):
    sim = QrackSimulator(n)
    run_circuit(sim, gates)
    dt, ct = (np.float32, ctypes.c_float) if Qrack.fppow < 6 else (np.float64, ctypes.c_double)
    probs = np.empty(1 << n, dtype=dt)
    Qrack.qrack_lib.OutProbs(sim.sid, probs.ctypes.data_as(ctypes.POINTER(ct)))
    sim._throw_if_error()
    del sim
    return probs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats(ideal_probs, counts, shots):
    """nn_qab.py's estimator, on a full 2^n ideal (used only with --full)."""
    ideal = np.asarray(ideal_probs, dtype=np.float64)
    threshold = np.median(ideal)
    u_u = ideal.mean()
    ideal_centered = ideal - u_u
    denom = np.dot(ideal_centered, ideal_centered)
    keys = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    patch = np.fromiter(counts.values(), dtype=np.float64, count=len(counts)) / shots
    numer = np.dot(ideal_centered[keys], patch) - u_u * ideal_centered.sum()
    hog_prob = patch[ideal[keys] > threshold].sum()

    return numer / denom, hog_prob


def calc_stats_factorized(band_probs, bands, counts, shots, median_draws=1 << 16):
    """The same estimator as calc_stats, for p(x) = prod_k p_k(x_k).

    XEB is exact. HOG uses the median of the product distribution, which is
    estimated from `median_draws` uniformly random bitstrings (the median of
    2^width values is not computed exactly), so HOG is approximate.
    """
    keys = np.fromiter(counts.keys(), dtype=np.uint64, count=len(counts))
    freq = np.fromiter(counts.values(), dtype=np.float64, count=len(counts)) / shots

    sum_sq = 1.0
    scaled = np.ones(len(keys))
    rng = np.random.default_rng(12345)
    draws = np.ones(median_draws)
    for p, (off, n) in zip(band_probs, bands):
        n_pow = float(p.size)
        sum_sq *= n_pow * np.dot(p.astype(np.float64), p)
        x_k = ((keys >> np.uint64(off)) & np.uint64((1 << n) - 1)).astype(np.int64)
        scaled *= n_pow * p[x_k].astype(np.float64)
        draws *= n_pow * p[rng.integers(0, p.size, median_draws)].astype(np.float64)

    xeb = (np.dot(freq, scaled) - 1.0) / (sum_sq - 1.0)
    hog_approx = freq[scaled > np.median(draws)].sum()

    return xeb, hog_approx


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def ace_counts(width, lrc, lrr, circ, shots):
    sim = QrackAceBackend(width, long_range_columns=lrc, long_range_rows=lrr)
    # Fairness-critical settings: assert and log, so a run with any of them
    # silently changed cannot pass as this benchmark.
    if not sim.is_error_detection:
        raise RuntimeError("is_error_detection must be True for this benchmark.")
    if not all(sim._is_row_long_range):
        raise RuntimeError(
            "long_range_rows cuts this layout into row patches; the orthogonal proxy assumes it never does. "
            "Raise long_range_rows."
        )
    config = {
        "pyqrack": version("pyqrack"),
        "is_error_detection": sim.is_error_detection,
        "is_torus": sim.is_torus,
        "use_crossbars": sim.use_crossbars,
        "ace_column_seams": [c for c, ok in enumerate(sim._is_col_long_range) if not ok],
        "ace_patch_qubits": list(sim._sim_counts) if hasattr(sim, "_sim_counts") else None,
    }
    run_circuit(sim, circ)
    counts = dict(Counter(sim.measure_shots(list(range(width)), shots)))
    del sim
    return counts, config


def bench(width, depth, lrc=4, lrr=4, k=2, full=False):
    shots = 1 << min(13, width + 2)
    row_len, col_len = factor_width(width)
    bands = orthogonal_bands(width, k)

    t0 = time.perf_counter()
    qc = random_circuit(width, depth)
    proxy, per_band, removed = make_proxy(qc, bands)
    t1 = time.perf_counter()
    print(f"circuit_seconds: {t1 - t0:.4f}  (grid {col_len} x {row_len}; {k} proxy patches of "
          f"{' + '.join(str(n) for _, n in bands)} qubits; {removed} of {len(qc) - width * depth} couplers cut)", flush=True)

    counts, config = ace_counts(width, lrc, lrr, proxy, shots)
    t2 = time.perf_counter()
    print(f"ace_proxy_seconds: {t2 - t1:.4f}", flush=True)
    for k, v in config.items():
        print(f"  {k}: {v}")

    band_probs = [ideal_probs(n, g) for (_, n), g in zip(bands, per_band)]
    t3 = time.perf_counter()
    print(f"ideal_patches_seconds: {t3 - t2:.4f}", flush=True)

    xeb_proxy, hog_proxy = calc_stats_factorized(band_probs, bands, counts, shots)
    del band_probs

    result = {
        "width": width,
        "depth": depth,
        "long_range_columns": lrc,
        "long_range_rows": lrr,
        "proxy_patches": k,
        "proxy_patch_qubits": [n for _, n in bands],
        "couplers_cut": removed,
        "xeb_ace_proxy": float(xeb_proxy),
        "hog_ace_proxy_approx": float(hog_proxy),
    }

    if full:
        counts_full, _ = ace_counts(width, lrc, lrr, qc, shots)
        t4 = time.perf_counter()
        print(f"ace_full_seconds: {t4 - t3:.4f}", flush=True)
        sim_ideal = QrackSimulator(width)
        run_circuit(sim_ideal, qc)
        probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
        del sim_ideal
        xeb_full, hog_full = calc_stats(probs, counts_full, shots)
        print(f"ideal_full_seconds: {time.perf_counter() - t4:.4f}", flush=True)
        result["xeb_ace_full"] = float(xeb_full)
        result["hog_ace_full"] = float(hog_full)

    return result


def main():
    args = [a for a in sys.argv[1:] if a != "--full"]
    full = "--full" in sys.argv[1:]
    if len(args) < 2:
        raise RuntimeError(
            "Usage: python3 nn_qab_orthogonal_proxy.py [width] [depth] "
            "[long_range_columns=4] [long_range_rows=4] [proxy_patches=2] [--full]"
        )
    width = int(args[0])
    depth = int(args[1])
    lrc = int(args[2]) if len(args) > 2 else 4
    lrr = int(args[3]) if len(args) > 3 else 4
    k = int(args[4]) if len(args) > 4 else 2
    result = bench(width, depth, lrc, lrr, k, full)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
