# Fully-connected RCS with built-in greedy ACE consensus.
#
# Four QrackSimulator instances run the same random circuit with different
# coupler orderings, developing approximately orthogonal separability structures.
# Their coherent superposition (consensus state) identifies heavy outputs.
#
# The consensus state replaces MPS amplitude queries: we sieve the top width**2
# heavy-output candidates by consensus probability, estimate each candidate's
# probability from the (phase-canonicalized, renormalized) consensus state,
# then mix 50/50 with uniform — exactly the QV protocol from the MPS script.
# XEB and HOG are computed against a full ideal simulator for ground truth.
#
# Coupler orderings (two orthogonal cuts):
#   inst 0: natural   [0,1,2,...,k-1]          — sequential
#   inst 1: stride    [1,3,5,...,0,2,4,...]     — every-other starting from second
#
# By Dan Strano and (Anthropic) Claude.

import math
import random
import sys
import time
from itertools import combinations

import numpy as np
from pyqrack import QrackSimulator


# ---------------------------------------------------------------------------
# Coupler ordering
# ---------------------------------------------------------------------------

def _order_pairs(pairs, inst):
    """
    inst 0: natural order          [0,1,2,...,k-1]
    inst 1: stride (every-other,   [1,3,5,...,0,2,4,...])
            odds first then evens — orthogonal ACE greedy path
    """
    k = len(pairs)
    if k == 0:
        return pairs
    if inst == 0:
        return pairs
    # inst 1: odds-first stride
    return [pairs[i] for i in range(1, k, 2)] +            [pairs[i] for i in range(0, k, 2)]



# ---------------------------------------------------------------------------
# Walsh-Hadamard transform (fast, O(n * 2^n))
# ---------------------------------------------------------------------------

def hadamard_transform(v):
    """Unnormalized Walsh-Hadamard transform. Inverse = hadamard_transform(v) / len(v)."""
    n = len(v)
    h = v.copy()
    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            lo = h[i:i+step].copy()
            hi = h[i+step:i+2*step].copy()
            h[i:i+step]        = lo + hi
            h[i+step:i+2*step] = lo - hi
        step *= 2
    return h


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats_np(ideal_ket, split_ket):
    n_pow  = len(ideal_ket)
    u_u    = 1.0 / n_pow
    ideal  = np.asarray(ideal_ket, dtype=np.complex128)
    split  = np.asarray(split_ket, dtype=np.complex128)
    l2     = complex(np.dot(split, ideal.conj()))
    l2     = (l2 * l2.conjugate()).real
    p      = (ideal * ideal.conj()).real
    q      = (split * split.conj()).real
    p_c    = p - u_u
    q_c    = q - u_u
    denom  = float(np.dot(p_c, p_c))
    xeb    = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    hog    = float(q[p > float(np.median(p))].sum())
    return xeb, l2, hog


def calc_stats_sparse(ideal_probs, exp_probs_sparse, n_pow):
    """
    XEB and HOG from a full ideal probability vector and a sparse
    experimental distribution (50/50 mixed with uniform, QV protocol).
    """
    u_u     = 1.0 / n_pow
    model   = 0.5

    exp_dense = np.zeros(n_pow, dtype=np.float64)
    for k, v in exp_probs_sparse.items():
        exp_dense[k] = v
    exp_mixed = (1.0 - model) * exp_dense + model * u_u

    p_c   = ideal_probs - u_u
    q_c   = exp_mixed   - u_u
    denom = float(np.dot(p_c, p_c))
    xeb   = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    hog   = float(exp_mixed[ideal_probs > float(np.median(ideal_probs))].sum())
    return xeb, hog


def calc_stats(ideal_probs, split_probs):
    n_pow  = len(ideal_probs)
    u_u    = 1.0 / n_pow
    p      = np.asarray(ideal_probs, dtype=np.float64)
    q      = np.asarray(split_probs, dtype=np.float64)
    p_c    = p - u_u
    q_c    = q - u_u
    denom  = float(np.dot(p_c, p_c))
    xeb    = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    hog    = float(q[p > float(np.median(p))].sum())
    return xeb, hog


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, sdrp=0.0):
    lcv_range    = range(width)
    all_bits     = list(lcv_range)
    n_inst       = 2
    n_candidates = width ** 2

    # Full ideal simulator (no ACE limit) — ground truth
    sim_ideal = QrackSimulator(width)

    # Four consensus instances with ACE limit
    sims = [QrackSimulator(width) for _ in range(n_inst)]
    for s in sims:
        if sdrp > 0.0:
            s.set_sdrp(sdrp)
        s.set_ace_max_qb((width + 3) >> 2)

    rng_state = random.getstate()
    t_start   = time.perf_counter()

    # -----------------------------------------------------------------------
    # Ideal — full simulation, no ACE limit
    # -----------------------------------------------------------------------
    random.setstate(rng_state)
    for _ in range(depth):
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            sim_ideal.u(i, th, ph, lm)
        unused = all_bits.copy()
        random.shuffle(unused)
        while len(unused) > 1:
            c = unused.pop(); t = unused.pop()
            sim_ideal.mcx([c], t)

    ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
    del sim_ideal

    # -----------------------------------------------------------------------
    # Two consensus instances
    # -----------------------------------------------------------------------
    kets = []
    for inst in range(n_inst):
        random.setstate(rng_state)
        sim = sims[inst]
        for _ in range(depth):
            for i in lcv_range:
                th = random.uniform(0, 2 * math.pi)
                ph = random.uniform(0, 2 * math.pi)
                lm = random.uniform(0, 2 * math.pi)
                sim.u(i, th, ph, lm)
            pairs = []
            unused = all_bits.copy()
            random.shuffle(unused)
            while len(unused) > 1:
                pairs.append((unused.pop(), unused.pop()))
            for c, t in _order_pairs(pairs, inst):
                sim.mcx([c], t)
        kets.append(np.asarray(sim.out_ket(), dtype=np.complex128))

    t_elapsed = time.perf_counter() - t_start

    # -----------------------------------------------------------------------
    # Phase canonicalization: rotate each ket so that the amplitude at the
    # common gauge index (first above-uniform index in the ensemble mean field)
    # is real and positive.  Same index for all kets => common gauge.
    # -----------------------------------------------------------------------
    n_pow   = 1 << width
    u_u     = 1.0 / n_pow
    mean_p  = sum((k * k.conj()).real for k in kets) / n_inst
    gauge_idx = int(np.argmax(mean_p > u_u))

    phase_fixed = []
    for k in kets:
        ref   = k[gauge_idx]
        phase = ref / abs(ref) if abs(ref) > 1e-30 else 1.0
        phase_fixed.append(k / phase)

    # -----------------------------------------------------------------------
    # Symmetrized outer product density matrix.
    # rho = (|psi_A><psi_B| + |psi_B><psi_A|) / 2
    # diagonal: rho[i,i] = Re(psi_A[i] * psi_B[i].conj())
    # This is a heuristic for the ideal state that incorporates cross-instance
    # coherence without requiring global phase canonicalization beyond what
    # the common gauge already provides.
    # Use instances 0 and 1 (the two orthogonal cuts).
    # -----------------------------------------------------------------------
    p_dm = (phase_fixed[0] * phase_fixed[1].conj() + phase_fixed[1] * phase_fixed[0].conj()).real
    # Shift to non-negative (diagonal of a valid density matrix is non-negative,
    # but numerical errors from approximate orthogonality can give small negatives)
    p_dm = np.maximum(p_dm, 0.0)
    dm_sum = p_dm.sum()
    if dm_sum > 0:
        p_dm /= dm_sum

    xeb_dm, hog_dm = calc_stats(ideal_probs, p_dm)

    # -----------------------------------------------------------------------
    # Hadamard-basis piecewise combination.
    # Transform each patch ket to Hadamard basis, combine piecewise:
    #   - only one patch has support (|phi| > threshold): full weight
    #   - both patches have support: average
    # Transform back to computational basis for XEB/HOG.
    # -----------------------------------------------------------------------
    n_pow   = 1 << width
    phi_0   = hadamard_transform(phase_fixed[0]) / np.sqrt(n_pow)
    phi_1   = hadamard_transform(phase_fixed[1]) / np.sqrt(n_pow)
    thresh  = 1.0 / np.sqrt(n_pow)   # above uniform in Hadamard basis
    supp_0  = np.abs(phi_0) > thresh
    supp_1  = np.abs(phi_1) > thresh
    phi_had = np.where(supp_0 & supp_1, (phi_0 + phi_1) / 2.0,
              np.where(supp_0, phi_0,
              np.where(supp_1, phi_1, 0.0)))
    psi_had = hadamard_transform(phi_had) / np.sqrt(n_pow)
    p_had   = np.abs(psi_had) ** 2
    had_sum = p_had.sum()
    if had_sum > 0:
        p_had /= had_sum
    xeb_had, hog_had = calc_stats(ideal_probs, p_had)

    xeb_vs_cons = []
    for k in kets:
        xeb_i, _, _ = calc_stats_np(k, p_dm)
        xeb_vs_cons.append(xeb_i)

    result = {
        "width":              width,
        "depth":              depth,
        "seconds":            t_elapsed,
        "xeb_dm_vs_ideal":    xeb_dm,
        "hog_dm_vs_ideal":    hog_dm,
        "xeb_had_vs_ideal":   xeb_had,
        "hog_had_vs_ideal":   hog_had,
    }

    for i, x in enumerate(xeb_vs_cons):
        result[f"xeb_{i}_vs_cons"] = x

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_consensus.py [width] [depth] [sdrp=0]"
        )
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    result = bench_qrack(width, depth, sdrp)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
