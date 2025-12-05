# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import operator
import random
import statistics
import sys

from collections import Counter

from scipy.stats import binom

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit

import quimb.tensor as tn
from qiskit_quimb import quimb_circuit


# Function by Google search AI
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


def bench_qrack(width, depth, sdrp, is_sparse):
    lcv_range = range(width)
    all_bits = list(lcv_range)
    retained = width * width
    shots = retained * width

    rcs = QuantumCircuit(width)
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            for b in range(3):
                rcs.h(i)
                rcs.rz(random.uniform(0, 2 * math.pi), i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            rcs.cx(c, t)

    if is_sparse:
        experiment = QrackSimulator(width, isTensorNetwork=False, isOpenCL=False, isSparse=True)
    else:
        experiment = QrackSimulator(width, isTensorNetwork=False)
    if sdrp > 0:
        experiment.set_sdrp(sdrp)
    experiment.run_qiskit_circuit(rcs)
    experiment_counts = dict(Counter(experiment.measure_shots(all_bits, shots)))
    experiment_counts = sorted(experiment_counts.items(), key=operator.itemgetter(1))

    quimb_rcs = quimb_circuit(rcs)
    n_pow = 1 << width
    u_u =  1 / n_pow
    idx = 0
    ideal_amps = {}
    sum_probs = 0
    for count_tuple in experiment_counts:
        key = count_tuple[0]
        amp = complex(quimb_rcs.amplitude(int_to_bitstring(key, width), backend="jax"))
        prob = float((abs(amp) ** 2).real)
        if prob <= u_u:
            continue
        val = count_tuple[1]
        ideal_amps[key] = amp
        sum_probs += prob
        if len(ideal_amps) >= retained:
            break

    return {
        "qubits": width,
        "depth": depth,
        "sum_sieved_probs": sum_probs,
        "sieved_amps": ideal_amps
    }


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc.py [width] [depth] [sdrp] [is_sparse]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp = 0
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
