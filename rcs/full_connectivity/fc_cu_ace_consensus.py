# Fully-connected RCS: Automatic circuit elision
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
# Statistics
# ---------------------------------------------------------------------------

def calc_stats(ideal_probs, exp_probs, n_pow):
    u_u   = 1.0 / n_pow
    model = 0.5
    exp_mixed = (1.0 - model) * exp_probs + model * u_u
    p_c   = ideal_probs - u_u
    q_c   = exp_probs   - u_u
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
    n_inst       = 2
    n_pow        = 1 << width
    u_u          = 1.0 / n_pow

    # -----------------------------------------------------------------------
    # Build circuit once in Qiskit
    # -----------------------------------------------------------------------
    t_circ = time.perf_counter()
    qc      = [QuantumCircuit(width) for _ in range(n_inst)]

    for _ in range(depth):
        for i in lcv_range:
            th, ph, lm = (random.uniform(0, 2*math.pi) for _ in range(3))
            for c in qc:
                c.u(th, ph, lm, i)
        shuffled = all_bits[:]
        random.shuffle(shuffled)
        cl = []
        while len(shuffled) > 1:
            # cl.append(((shuffled.pop(), shuffled.pop()), [random.uniform(0, 2*math.pi) for _ in range(4)]))
            cl.append((random.uniform(0, 2*math.pi), shuffled.pop(), shuffled.pop()))
        for c in qc:
            random.shuffle(cl)
            for g in cl:
                # b, p = g
                # c.cu(p[0], p[1], p[2], p[3], b[0], b[1])
                c.cu(g[0], 0, 0, 0, g[1], g[2])

    # -----------------------------------------------------------------------
    # Ideal ground truth
    # -----------------------------------------------------------------------
    sim_ideal = QrackSimulator(width)
    sim_ideal.run_qiskit_circuit(qc[0], shots=0)
    ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
    del sim_ideal

    t_ideal = time.perf_counter()

    print(f"qrack_circuit_seconds: {t_ideal - t_circ}")

    # -----------------------------------------------------------------------
    # Method: ACE prob_perm over full Hilbert space.
    # Since the ideal simulation is already materialized for ground truth,
    # we can afford to walk all 2^n permutations with prob_perm — giving
    # the complete ACE probability distribution, not just sampled candidates.
    # Two ACE instances (sequential + stride); average their prob_perm values.
    # -----------------------------------------------------------------------
    ace_sims = []
    for inst in range(n_inst):
        sim = QrackSimulator(width)
        if sdrp > 0.0:
            sim.set_sdrp(sdrp)
        sim.set_ace_max_qb((width + 1) >> 1)
        sim.run_qiskit_circuit(qc[inst], shots=0)
        ace_sims.append(sim)

    q_bits = list(range(width))
    ace_probs = np.empty(n_pow, dtype=np.float64)
    for outcome in range(n_pow):
        bits  = [(outcome >> b) & 1 for b in range(width)]
        ace_probs[outcome] = sum(s.prob_perm(q_bits, bits) for s in ace_sims) / n_inst
    for s in ace_sims: del s

    xeb_ace, hog_ace = calc_stats(ideal_probs, ace_probs, n_pow)

    t_elapsed = time.perf_counter() - t_ideal

    return {
        "width":         width,
        "depth":         depth,
        "xeb_ace":       xeb_ace,
        "hog_ace":       hog_ace,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_ace.py [width] [depth] [sdrp=0]")
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    result = bench_qrack(width, depth, sdrp)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
