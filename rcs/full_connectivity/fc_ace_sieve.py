# Fully-connected RCS: ACE consensus sieve for heavy outputs.
#
# Candidate outcomes are drawn by uniform random selection. Each candidate's
# probability is estimated solely as the average prob_perm across n_inst
# ACE instances. Candidates are routed into heavy/light buckets relative
# to the uniform distribution, and XEB / HOG are computed against a full
# ideal Qrack simulation.
#
# By Dan Strano and (Anthropic) Claude.

import math
import random
import sys
import time

import numpy as np
from qiskit import QuantumCircuit
from pyqrack import QrackSimulator


# ---------------------------------------------------------------------------
# Statistics (sparse heavy/light representation)
# ---------------------------------------------------------------------------

def route_heavy_light(prob_dict, u_u):
    """
    Split a {outcome: p} dict into (heavy, light) dicts centered on u_u.
    heavy: outcomes where p > u_u, values normalised to sum 1.
    light: outcomes where p < u_u, values (stored positive) normalised to sum 1.
    """
    heavy_raw = {}
    light_raw = {}
    for outcome, p in prob_dict.items():
        c = p - u_u
        if c > 0:
            heavy_raw[outcome] = c
        elif c < 0:
            light_raw[outcome] = -c          # store as positive

    s_h = sum(heavy_raw.values())
    s_l = sum(light_raw.values())
    heavy = {k: v / s_h for k, v in heavy_raw.items()} if s_h > 0 else {}
    light = {k: v / s_l for k, v in light_raw.items()} if s_l > 0 else {}
    return heavy, light


def calc_stats_sparse(ideal_probs, exp_probs_sparse, n_pow):
    """
    Reconstruct a dense approximate distribution from the heavy/light split
    and compute XEB and HOG against ideal_probs.

    Dense reconstruction:
        exp_mixed[k] = 0.5 * heavy[k]  +  0.5 * u_u * (1 - light[k])
    Heavy outcomes sit above u_u; light outcomes are suppressed below u_u.
    Everything not sampled sits exactly at u_u in the light half, which is
    conservative and avoids phantom signal from unvisited outcomes.
    """
    u_u = 1.0 / n_pow
    heavy, light = exp_probs_sparse

    h_dense = np.zeros(n_pow, dtype=np.float64)
    for k, v in heavy.items():
        h_dense[k] = v

    l_dense = np.zeros(n_pow, dtype=np.float64)
    for k, v in light.items():
        l_dense[k] = v

    exp_mixed = 0.5 * h_dense + 0.5 * u_u * (1.0 - l_dense)

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
    n_pow        = 1 << width
    u_u          = 1.0 / n_pow

    # Number of candidate outcomes to probe via the sieve.
    # width**2 keeps cost polynomial; sqrt(n_pow) is the natural "heavy
    # output" count for a Porter-Thomas distributed circuit.
    n_candidates = max(width ** 2, int(math.sqrt(n_pow) + 0.5))

    # -----------------------------------------------------------------------
    # Build n_inst independent random circuits (same single-qubit angles,
    # different coupler orderings — identical to fc_ace_consensus.py)
    # -----------------------------------------------------------------------
    t_circ = time.perf_counter()
    qc = [QuantumCircuit(width) for _ in range(n_inst)]

    for _ in range(depth):
        for i in lcv_range:
            th, ph, lm = (random.uniform(0, 2 * math.pi) for _ in range(3))
            for c in qc:
                c.u(th, ph, lm, i)
        shuffled = all_bits[:]
        random.shuffle(shuffled)
        cl = []
        while len(shuffled) > 1:
            cl.append(((shuffled.pop(), shuffled.pop()),
                       [random.uniform(0, 2 * math.pi) for _ in range(4)]))
        for c in qc:
            random.shuffle(cl)
            for g in cl:
                b, p = g
                c.cu(p[0], p[1], p[2], p[3], b[0], b[1])

    t_build = time.perf_counter()
    print(f"circuit_build_seconds: {t_build - t_circ}")

    # -----------------------------------------------------------------------
    # Ideal ground truth (full state vector via Qrack)
    # -----------------------------------------------------------------------
    sim_ideal = QrackSimulator(width)
    sim_ideal.run_qiskit_circuit(qc[0], shots=0)
    ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
    del sim_ideal

    t_ideal = time.perf_counter()
    print(f"ideal_seconds: {t_ideal - t_build}")

    # -----------------------------------------------------------------------
    # ACE consensus instances
    # -----------------------------------------------------------------------
    ace_sims = []
    for inst in range(n_inst):
        sim = QrackSimulator(width)
        if sdrp > 0.0:
            sim.set_sdrp(sdrp)
        sim.set_ace_max_qb((width + 1) >> 1)
        sim.run_qiskit_circuit(qc[inst], shots=0)
        ace_sims.append(sim)

    t_ace = time.perf_counter()
    print(f"ace_seconds: {t_ace - t_ideal}")

    # -----------------------------------------------------------------------
    # Uniform random sieve.
    #
    # Draw n_candidates outcomes uniformly at random from the full 2^n
    # Hilbert space with no prefix bias.  Each run explores a different
    # random subset, giving unbiased variance data across repeated calls.
    # -----------------------------------------------------------------------
    candidates = random.sample(range(n_pow), min(n_candidates, n_pow))

    q_bits = list(range(width))
    ace_probs = {}
    for idx in candidates:
        bits  = [(idx >> b) & 1 for b in range(width)]
        p_ace = sum(s.prob_perm(q_bits, bits) for s in ace_sims) / n_inst
        if p_ace > 0:
            ace_probs[idx] = p_ace

    for s in ace_sims:
        del s

    t_sieve = time.perf_counter()
    print(f"sieve_seconds: {t_sieve - t_ace}")

    # -----------------------------------------------------------------------
    # Route into heavy / light and compute statistics
    # -----------------------------------------------------------------------
    sparse = route_heavy_light(ace_probs, u_u)
    xeb, hog = calc_stats_sparse(ideal_probs, sparse, n_pow)

    return {
        "width":        width,
        "depth":        depth,
        "sdrp":         sdrp,
        "n_candidates": len(candidates),
        "xeb_ace":      xeb,
        "hog_ace":      hog,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_ace_sieve.py [width] [depth] [sdrp=0]")
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0  #((1 - 1 / math.sqrt(2)) / 2)
    result = bench_qrack(width, depth, sdrp)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
