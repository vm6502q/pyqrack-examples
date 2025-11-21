# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import operator
import random
import statistics
import sys

from collections import Counter

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector

import quimb.tensor as tn
from qiskit_quimb import quimb_circuit


# Function by Google search AI
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


def bench_qrack(width, depth, sdrp):
    lcv_range = range(width)
    all_bits = list(lcv_range)
    shots = 1 << (width + 2)

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

    experiment = QrackSimulator(width, isTensorNetwork=False, isSparse=True, isOpenCL=False)
    if sdrp > 0:
        experiment.set_sdrp(sdrp)
    experiment.run_qiskit_circuit(rcs)
    experiment_counts = dict(Counter(experiment.measure_shots(all_bits, shots)))

    sorted_counts = sorted(experiment_counts.items(), key=operator.itemgetter(1))

    quimb_rcs = quimb_circuit(rcs)
    u_u =  1 / (1 << width)
    idx = 0
    ideal_probs = {}
    sum_probs = 0
    while (len(ideal_probs) < width):
        count_tuple = sorted_counts[idx]
        idx += 1
        key = count_tuple[0]
        prob = abs(quimb_rcs.amplitude(int_to_bitstring(key, width))) ** 2
        if prob > u_u:
            val = count_tuple[1]
            ideal_probs[key] = val
            sum_probs += val

    numer = 0
    denom = 0
    for key in ideal_probs.keys():
        ideal = ideal_probs[key]
        adj = ideal / sum_probs
        ideal_probs[key] = ideal
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (adj - u_u)

    adj_xeb = numer / denom

    rcs.save_statevector()
    control = AerSimulator(method="statevector")
    job = control.run(rcs)
    control_probs = Statevector(job.result().get_statevector()).probabilities()

    return calc_stats(control_probs, ideal_probs, adj_xeb, depth)


def calc_stats(ideal_probs, exp_probs, adj_xeb, depth):
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    u_u =  1 / n_pow
    model = max(1.0, 1 / adj_xeb)
    numer = 0
    denom = 0
    for i in range(n_pow):
        count = model * (exp_probs[i] if i in exp_probs else 0) + (1 - model) * u_u
        ideal = ideal_probs[i]

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (count - u_u)

    xeb = numer / denom

    return {
        "qubits": n,
        "depth": depth,
        "xeb": float(xeb)
    }


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_qiskit_validation.py [width] [depth] [trials]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp = 0
    if len(sys.argv) > 3:
        sdrp = float(sys.argv[3])

    # Run the benchmarks
    print(bench_qrack(width, depth, sdrp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
