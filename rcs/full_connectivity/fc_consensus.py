# Fully-connected RCS with sampling-based greedy ACE consensus.
#
# Two QrackSimulator instances run the same random circuit with different
# coupler orderings (sequential vs stride), developing approximately
# orthogonal ACE separability structures.  Each instance contributes
# measure_shots samples — no statevector materialization, scalable to
# arbitrarily large width.
#
# MPS is NOT used here; this is pure ACE sampling.  The union of samples
# from both instances is the candidate set.  XEB and HOG are computed
# against a full ideal simulator (present only for ground-truth validation
# at small scale; drop sim_ideal for true large-scale deployment).
#
# Sieve: all sampled candidates go through u_u routing:
#   heavy (count > mean_count): q_i set above u_u
#   light (count <= mean_count): q_i set below u_u
# Both tails contribute positively to XEB.
#
# By Dan Strano and (Anthropic) Claude.

import math
import random
import sys
import time
from collections import Counter

import numpy as np
from pyqrack import QrackSimulator


# ---------------------------------------------------------------------------
# Coupler ordering
# ---------------------------------------------------------------------------

def _order_pairs(pairs, inst):
    k = len(pairs)
    if k == 0 or inst == 0:
        return pairs
    return [pairs[i] for i in range(1, k, 2)] + \
           [pairs[i] for i in range(0, k, 2)]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats_sparse(ideal_probs, exp_probs_sparse, n_pow):
    u_u   = 1.0 / n_pow
    model = 0.5
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
    n_pow = len(ideal_probs); u_u = 1.0 / n_pow
    p = np.asarray(ideal_probs, dtype=np.float64)
    q = np.asarray(split_probs, dtype=np.float64)
    p_c = p - u_u; q_c = q - u_u
    denom = float(np.dot(p_c, p_c))
    xeb   = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    hog   = float(q[p > float(np.median(p))].sum())
    return xeb, hog


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, sdrp=0.0):
    lcv_range    = range(width)
    all_bits     = list(lcv_range)
    n_inst       = 2
    n_shots      = width ** 3      # shots per instance
    n_pow        = 1 << width
    u_u          = 1.0 / n_pow

    rng_state = random.getstate()
    t_start   = time.perf_counter()

    # -----------------------------------------------------------------------
    # Ideal — full simulation for ground-truth validation (small scale only)
    # -----------------------------------------------------------------------
    random.setstate(rng_state)
    sim_ideal = QrackSimulator(width)
    for _ in range(depth):
        for i in lcv_range:
            th, ph, lm = (random.uniform(0, 2*math.pi) for _ in range(3))
            sim_ideal.u(i, th, ph, lm)
        unused = all_bits.copy(); random.shuffle(unused)
        while len(unused) > 1:
            c = unused.pop(); t = unused.pop()
            sim_ideal.mcx([c], t)
    ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
    del sim_ideal

    # -----------------------------------------------------------------------
    # Two ACE instances — measure_shots only, no out_ket
    # -----------------------------------------------------------------------
    counts = Counter()
    for inst in range(n_inst):
        random.setstate(rng_state)
        sim = QrackSimulator(width)
        if sdrp > 0.0:
            sim.set_sdrp(sdrp)
        sim.set_ace_max_qb((width + 1) >> 1)
        for _ in range(depth):
            for i in lcv_range:
                th, ph, lm = (random.uniform(0, 2*math.pi) for _ in range(3))
                sim.u(i, th, ph, lm)
            pairs = []
            unused = all_bits.copy(); random.shuffle(unused)
            while len(unused) > 1:
                pairs.append((unused.pop(), unused.pop()))
            for c, t in _order_pairs(pairs, inst):
                sim.mcx([c], t)
        shots = sim.measure_shots(all_bits, n_shots)
        counts.update(int(s) for s in shots)
        del sim

    t_elapsed = time.perf_counter() - t_start

    # -----------------------------------------------------------------------
    # Route candidates by sample count vs mean count (proxy for u_u)
    # mean_count = total_shots / n_pow  (expected count under uniform)
    # heavy: count > mean_count  →  q_i above u_u
    # light: count <= mean_count →  q_i below u_u
    # -----------------------------------------------------------------------
    total_shots = n_inst * n_shots
    mean_count  = total_shots * u_u   # = total_shots / n_pow

    heavy_raw = {}
    light_raw = {}
    for outcome, cnt in counts.items():
        if cnt > mean_count:
            heavy_raw[outcome] = float(cnt)
        else:
            # Invert: lightest outputs (smallest count) get highest raw weight
            light_raw[outcome] = max(0.0, 2.0 * mean_count - cnt)

    # Normalize heavy → probability estimates above u_u
    s_h = sum(heavy_raw.values())
    heavy = {k: v / s_h for k, v in heavy_raw.items()} if s_h > 0 else {}

    # Normalize light raw weights, then map below u_u for suppression
    s_l = sum(light_raw.values())
    if s_l > 0:
        light = {k: max(0.0, u_u - (v / s_l) * u_u) for k, v in light_raw.items()}
        s_l2  = sum(light.values())
        light = {k: v / s_l2 for k, v in light.items()} if s_l2 > 0 else {}
    else:
        light = {}

    # Equal weight to non-empty tails, then 50/50 mix with uniform (QV protocol)
    n_nonempty = (1 if heavy else 0) + (1 if light else 0)
    if n_nonempty == 0:
        combined = {}
    else:
        w        = 1.0 / n_nonempty
        all_keys = set(heavy) | set(light)
        combined = {k: w * heavy.get(k, 0.0) + w * light.get(k, 0.0)
                    for k in all_keys}
        s_c = sum(combined.values())
        if s_c > 0:
            combined = {k: v / s_c for k, v in combined.items()}

    xeb_sieve, hog_sieve = calc_stats_sparse(ideal_probs, combined, n_pow)

    # -----------------------------------------------------------------------
    # ACE direct probability comparison.
    # Run two fresh ACE instances and query out_probs() directly.
    # Average probability: (p0[i] + p1[i]) / 2  (incoherent mixture).
    # Equivalent to equal superposition probability |psi_avg|^2 with
    # 1/sqrt(2) amplitude normalization per instance.
    # -----------------------------------------------------------------------
    ace_sims = []
    for inst in range(n_inst):
        random.setstate(rng_state)
        sim = QrackSimulator(width)
        if sdrp > 0.0:
            sim.set_sdrp(sdrp)
        sim.set_ace_max_qb((width + 1) >> 1)
        for _ in range(depth):
            for i in lcv_range:
                th, ph, lm = (random.uniform(0, 2*math.pi) for _ in range(3))
                sim.u(i, th, ph, lm)
            pairs = []
            unused = all_bits.copy(); random.shuffle(unused)
            while len(unused) > 1:
                pairs.append((unused.pop(), unused.pop()))
            for c, t in _order_pairs(pairs, inst):
                sim.mcx([c], t)
        ace_sims.append(sim)

    q_bits = list(range(width))
    ace_sparse = {}
    for outcome in counts:
        bits = [(outcome >> b) & 1 for b in range(width)]
        p_avg = sum(s.prob_perm(q_bits, bits) for s in ace_sims) / n_inst
        if p_avg > 0:
            ace_sparse[outcome] = p_avg
    for s in ace_sims: del s

    xeb_ace, hog_ace = calc_stats_sparse(ideal_probs, ace_sparse, n_pow)

    # -----------------------------------------------------------------------
    # Equal mixture of sieve and ACE prob_perm distributions.
    # Uncorrelated errors between the two methods should average out.
    # -----------------------------------------------------------------------
    all_mix_keys = set(combined) | set(ace_sparse)
    mixed = {k: 0.5 * combined.get(k, 0.0) + 0.5 * ace_sparse.get(k, 0.0)
             for k in all_mix_keys}
    s_mix = sum(mixed.values())
    if s_mix > 0:
        mixed = {k: v / s_mix for k, v in mixed.items()}
    xeb_mix, hog_mix = calc_stats_sparse(ideal_probs, mixed, n_pow)

    return {
        "width":        width,
        "depth":        depth,
        "n_unique":     len(counts),
        "n_heavy":      len(heavy),
        "n_light":      len(light),
        "seconds":      t_elapsed,
        "xeb_sieve":    xeb_sieve,
        "hog_sieve":    hog_sieve,
        "xeb_ace_avg":  xeb_ace,
        "hog_ace_avg":  hog_ace,
        "xeb_mix":      xeb_mix,
        "hog_mix":      hog_mix,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError("Usage: python3 fc_consensus.py [width] [depth] [sdrp=0]")
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    result = bench_qrack(width, depth, sdrp)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
