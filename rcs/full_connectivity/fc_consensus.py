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
# Coupler orderings (2-bit Gray code on pair index):
#   inst 0: natural              [0,1,2,...,k-1]
#   inst 1: stride               [1,3,5,...,0,2,4,...]
#   inst 2: half-rotation        [k/2,...,k-1,0,...,k/2-1]
#   inst 3: half-rotation+stride [k/2+1,k/2+3,...,k/2,k/2+2,...]
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
    k = len(pairs)
    if k == 0:
        return pairs
    offset  = (k >> 1) if (inst & 2) else 0
    rotated = pairs[offset:] + pairs[:offset]
    if inst & 1:
        rotated = [rotated[i] for i in range(1, k, 2)] + \
                  [rotated[i] for i in range(0, k, 2)]
    return rotated


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


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, sdrp=0.0):
    lcv_range    = range(width)
    all_bits     = list(lcv_range)
    n_inst       = 4
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
    # Four consensus instances
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

    # Coherent superposition, renormalized
    mix = sum(phase_fixed) / n_inst
    mix_norm = float(np.sqrt((mix * mix.conj()).real.sum()))
    if mix_norm > 0:
        mix /= mix_norm

    # -----------------------------------------------------------------------
    # Sieve: top n_candidates heavy outputs by consensus probability
    # -----------------------------------------------------------------------
    cons_probs = (mix * mix.conj()).real   # shape (n_pow,)
    top_idx    = np.argpartition(cons_probs, -n_candidates)[-n_candidates:]
    top_idx    = top_idx[np.argsort(cons_probs[top_idx])[::-1]]
    bottom_idx = np.argpartition(cons_probs, n_candidates)[n_candidates:]
    bottom_idx = bottom_idx[np.argsort(cons_probs[bottom_idx])[::1]]

    # Sparse ideal: consensus-identified candidates, probability from consensus,
    # renormalized, then 50/50 mixed with uniform (QV protocol)
    exp_probs_sparse = {int(i): float(cons_probs[i]) for i in top_idx}
    s = sum(exp_probs_sparse.values())
    exp_probs_sparse = {int(k): (v / s) for k, v in exp_probs_sparse.items()}

    _exp_probs_sparse = {int(i): float(u_u - cons_probs[i]) for i in bottom_idx}
    s = sum(_exp_probs_sparse.values())
    _exp_probs_sparse = {int(k): (u_u - (v / s)) for k, v in _exp_probs_sparse.items()}

    exp_probs_sparse = {int(k): exp_probs_sparse.get(k, 0) + _exp_probs_sparse.get(k, 0) for k in set(exp_probs_sparse) | set(_exp_probs_sparse)}
    s = sum(exp_probs_sparse.values())
    exp_probs_sparse = {int(k): (v / s) for k, v in exp_probs_sparse.items()}

    xeb_sparse, hog_sparse = calc_stats_sparse(ideal_probs, exp_probs_sparse, n_pow)
    del exp_probs_sparse
    del _exp_probs_sparse

    # -----------------------------------------------------------------------
    # Cross-instance and consensus-vs-ideal diagnostics
    # -----------------------------------------------------------------------
    cross_xeb = {}
    for i, j in combinations(range(n_inst), 2):
        xeb_ij, _, _ = calc_stats_np(kets[i], kets[j])
        cross_xeb[f"xeb_{i}_vs_{j}"] = xeb_ij

    xeb_cons_ideal, l2_cons_ideal, hog_cons_ideal = calc_stats_np(
        ideal_probs, cons_probs)

    xeb_vs_cons = []
    for k in kets:
        xeb_i, _, _ = calc_stats_np(k, mix)
        xeb_vs_cons.append(xeb_i)

    result = {
        "width":              width,
        "depth":              depth,
        "seconds":            t_elapsed,
        # Headline: sparse XEB against ideal using consensus-sieved candidates
        "xeb_sparse":         xeb_sparse,
        "hog_sparse":         hog_sparse,
        # Consensus state vs ideal (full probability comparison)
        "xeb_cons_vs_ideal":  xeb_cons_ideal,
        "l2_cons_vs_ideal":   l2_cons_ideal,
        "hog_cons_vs_ideal":  hog_cons_ideal,
        # Instance consensus quality
        "xeb_consensus_mean": float(np.mean(xeb_vs_cons)),
    }
    for i, x in enumerate(xeb_vs_cons):
        result[f"xeb_{i}_vs_cons"] = x
    result.update(cross_xeb)

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
