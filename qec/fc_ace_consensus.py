# Fully-connected RCS: ACE majority-vote consensus validation.
#
# Three QrackAceBackend instances run the same circuit with three
# different coupler orderings (sequential, stride-1, stride-2).
# After each circuit layer, single-qubit marginals (prob()) are
# queried from all three instances. For each qubit, a majority vote
# over the three marginals determines the consensus outcome, which is
# then used to build a full probability estimate via product of
# per-qubit marginals (separability assumption within ACE patches).
#
# XEB and HOG are printed after each layer so progress is visible.
# QrackSimulator provides ideal ground truth.
#
# By Dan Strano and (Anthropic) Claude.

import math
import os
import random
import sys
import time
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit
from pyqrack import QrackSimulator, QrackAceBackend


# ---------------------------------------------------------------------------
# Coupler ordering
# ---------------------------------------------------------------------------

def _make_qubit_maps(width, n_inst):
    # Generate n_inst bijective qubit permutations by cyclic rotation.
    # Instance k maps RCS logical qubit q -> ACE logical qubit (q + k*width//n_inst) % width.
    # This shifts which qubits land on patch boundaries in each instance,
    # distributing boundary effects across the ensemble without reordering
    # gates or applying explicit corrections.
    maps     = []   # maps[inst][rcs_q] -> ace_q
    inv_maps = []   # inv_maps[inst][ace_q] -> rcs_q
    for inst in range(n_inst):
        shift = inst * width // n_inst
        fwd = [(q + shift) % width for q in range(width)]
        inv = [0] * width
        for rcs_q, ace_q in enumerate(fwd):
            inv[ace_q] = rcs_q
        maps.append(fwd)
        inv_maps.append(inv)
    return maps, inv_maps

def calc_stats(ideal_probs, counts, n_pow, shots):
    u_u       = 1.0 / n_pow
    model     = 0.5
    exp_dense = np.zeros(n_pow, dtype=np.float64)
    for k, v in counts.items():
        exp_dense[k] = v / shots
    exp_mixed = (1.0 - model) * exp_dense + model * u_u
    p_c   = ideal_probs - u_u
    q_c   = exp_dense   - u_u
    denom = float(np.dot(p_c, p_c))
    xeb   = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    hog   = float(exp_mixed[ideal_probs > float(np.median(ideal_probs))].sum())
    return xeb, hog


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, sdrp=0.0):
    lcv_range = range(width)
    all_bits  = list(lcv_range)
    n_inst    = 3
    n_pow     = 1 << width
    n_shots   = 1 << (width + 2)

    # -----------------------------------------------------------------------
    # Initialise three ACE instances and one ideal QrackSimulator.
    # All four start from |0> and receive gates layer by layer.
    # -----------------------------------------------------------------------
    aces = [QrackAceBackend(width, long_range_columns=2) for _ in range(n_inst)]

    # maps[inst][rcs_q] = ace_q,  inv_maps[inst][ace_q] = rcs_q
    maps, inv_maps = _make_qubit_maps(width, n_inst)
    if "QRACK_QUNIT_SEPARABILITY_THRESHOLD" not in os.environ:
        for ace in aces:
            for sim in ace.sim:
                sim.set_sdrp(0.03)

    sim_ideal = QrackSimulator(width)

    for d in range(depth):
        t_layer = time.perf_counter()

        # Build single-qubit gate layer (same for all instances)
        layer_u = []
        for i in lcv_range:
            th, ph, lm = (random.uniform(0, 2*math.pi) for _ in range(3))
            layer_u.append((i, th, ph, lm))

        # Build coupler pair list (same shuffle for all instances)
        pairs = []
        shuffled = all_bits[:]
        random.shuffle(shuffled)
        while len(shuffled) > 1:
            c, t = shuffled.pop(), shuffled.pop()
            pairs.append((c, t))

        # Apply gates via per-instance qubit permutation so patch
        # boundaries fall on different RCS qubits in each instance.
        for i, th, ph, lm in layer_u:
            for inst, ace in enumerate(aces):
                ace.u(maps[inst][i], th, ph, lm)
            sim_ideal.u(i, th, ph, lm)

        for inst, ace in enumerate(aces):
            for c, t in pairs:
                ace.cx(maps[inst][c], maps[inst][t])
        for c, t in pairs:
            sim_ideal.mcx([c], t)

        # -------------------------------------------------------------------
        # Query ideal probabilities (non-demolition on statevector)
        # -------------------------------------------------------------------
        ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)

        # -------------------------------------------------------------------
        # Consensus probability estimate via per-qubit majority vote.
        #
        # For each qubit q, collect p_q = prob(q) from all three instances.
        # Majority vote: average the three marginals, threshold at 0.5.
        # Build full 2^n probability vector as product of per-qubit marginals
        # (separability assumption — each qubit independently reflects its
        # consensus marginal).
        #
        # This resolves parity flips: a boundary bit-flip makes one instance
        # report p~0.9 while the others report p~0.1; the majority (2 vs 1)
        # correctly votes for the ~0.1 outcome.
        # -------------------------------------------------------------------
        # marginals = np.zeros((n_inst, width), dtype=np.float64)
        # for inst, ace in enumerate(aces):
        #     for q in lcv_range:
        #         marginals[inst, q] = ace.prob(q)

        # Majority vote: average then threshold
        # avg_marginals = marginals.mean(axis=0)   # shape (width,)

        # Apply X corrections in a dedicated pass — before any further
        # prob() calls or sampling — to avoid _correct() side-effects
        # that can corrupt physical qubit indices during measure_shots.
        # epsilon = aces[0]._epsilon
        # for inst, ace in enumerate(aces):
        #     for q in lcv_range:
        #         if avg_marginals[q] > (0.5 + epsilon) and marginals[inst, q] < (0.5 - epsilon):
        #             ace.x(q)
        #         elif avg_marginals[q] < (0.5 - epsilon) and marginals[inst, q] > (0.5 + epsilon):
        #             ace.x(q)

        # Invert the qubit permutation on readout so all shots are in
        # canonical RCS qubit ordering before pooling.
        pooled_counts = Counter()
        total_shots   = 0
        for inst, ace in enumerate(aces):
            raw_shots = ace.measure_shots(all_bits, n_shots)
            for raw in raw_shots:
                canonical = 0
                for ace_q in range(width):
                    if (raw >> ace_q) & 1:
                        canonical |= 1 << inv_maps[inst][ace_q]
                pooled_counts[canonical] += 1
            total_shots += n_shots

        xeb_ace, hog_ace = calc_stats(ideal_probs, pooled_counts, n_pow, total_shots)

        print({"qubits": width, "depth": d + 1,
               "xeb_ace": float(xeb_ace), "hog_ace": float(hog_ace),
               "layer_seconds": time.perf_counter() - t_layer})

    for ace in aces:
        del ace
    del sim_ideal


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError("Usage: python3 fc_ace.py [width] [depth] [sdrp=0]")
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    bench_qrack(width, depth, sdrp)
    return 0

if __name__ == "__main__":
    sys.exit(main())
