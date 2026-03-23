# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import json
import math
import operator
import random
import statistics
import sys

from collections import Counter

from scipy.stats import binom

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit, qpy

import quimb.tensor as tn
from qiskit_quimb import quimb_circuit


# Function by Google search AI
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


# Modified to use MPS by (Anthropic) Claude
def bench_qrack(width, depth, sdrp, is_sparse):
    lcv_range = range(width)
    all_bits = list(lcv_range)
    retained = min(width ** 2, 1 << (width - 1))
    checked = min(1 << 20, 1 << (width + 2))

    # chi controls approximation quality vs. speed
    # chi = width is cheap; chi = width**2 is closer to exact
    # for QV circuits at modest width, chi = 2*width is a reasonable start
    chi = min(width ** 3, 1 << width)

    # CircuitMPS maintains state as MPS with bounded bond dimension
    # Gate application is O(chi^2 * width) per gate instead of exact
    quimb_rcs = tn.CircuitMPS(width, max_bond=chi)

    rcs = QuantumCircuit(width)
    # excluded = [-1] * depth
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            rcs.u(th, ph, lm, i)
            quimb_rcs.apply_gate('U3', th, ph, lm, i, tags=f"LAYER_{d}")

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            rcs.cx(c, t)
            quimb_rcs.apply_gate('CX', c, t, tags=f"LAYER_{d}")

    with open("fc_mps.qpy", "wb") as file:
        qpy.dump(rcs, file)

    # Run Qrack for heavy output sieve
    if is_sparse:
        experiment = QrackSimulator(width, isTensorNetwork=False, isOpenCL=False, isSparse=True)
    else:
        experiment = QrackSimulator(width)
    if sdrp > 0:
        experiment.set_sdrp(sdrp)
    experiment.run_qiskit_circuit(rcs)
    highest_prob = experiment.highest_prob_perm()
    experiment_counts = dict(Counter(experiment.measure_shots(all_bits, checked)))
    experiment_counts[highest_prob] = checked
    experiment_counts = sorted(experiment_counts.items(), key=operator.itemgetter(1), reverse=True)
    experiment = None

    # Approximate amplitude estimation via MPS
    # amplitude() on CircuitMPS is O(chi^2 * width) per call
    # vs. O(exp(treewidth)) for exact contraction
    n_pow = 1 << width
    u_u = 1 / n_pow
    ideal_amps = {}
    sum_probs = 0

    for count_tuple in experiment_counts:
        if len(ideal_amps) >= retained and count_tuple[1] < 2:
            break
        key = count_tuple[0]
        bitstring = int_to_bitstring(key, width)
        # Approximate amplitude from MPS — cheap inner product
        amp = complex(quimb_rcs.amplitude(bitstring))
        prob = abs(amp) ** 2
        if prob <= u_u:
            continue
        ideal_amps[key] = amp
        sum_probs += prob

    with open('fc_mps.out', 'w') as f:
         json.dump(str(ideal_amps), f)

    return {
        "qubits": width,
        "depth": depth,
        "sum_sieved_probs": sum_probs,
    }


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc.py [width] [depth] [sdrp] [is_sparse]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp = 0 # (1.0 - 1.0 / math.sqrt(2)) / 2.0
    is_sparse = False
    if len(sys.argv) > 3:
        sdrp = float(sys.argv[3])
    if len(sys.argv) > 4:
        is_sparse = sys.argv[4] not in ["False", "0"]

    # Run the benchmarks
    print(bench_qrack(width, depth, sdrp, is_sparse))

    return 0


if __name__ == "__main__":
    sys.exit(main())
