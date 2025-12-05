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
    retained = width * width * 2
    shots = retained * width

    quimb_rcs = tn.Circuit(width)
    rcs = QuantumCircuit(width)
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

    if is_sparse:
        experiment = QrackSimulator(width, isTensorNetwork=False, isOpenCL=False, isSparse=True)
    else:
        experiment = QrackSimulator(width, isTensorNetwork=False)
    if sdrp > 0:
        experiment.set_sdrp(sdrp)
    experiment.run_qiskit_circuit(rcs)
    experiment_counts = dict(Counter(experiment.measure_shots(all_bits, shots)))
    experiment_counts = sorted(experiment_counts.items(), key=operator.itemgetter(1))
    experiment = None

    for l in range(depth):
        for q in range(width):
            quimb_rcs.psi.contract([f'I{q}', f'LAYER_{l}'], which='all')

    for p in range(2):
        s = 1 << p
        for l in range(0, depth - s + 1, s):
            l_end = l + s - 1
            for q in range(width):
                quimb_rcs.psi.contract_between(['CX', f'I{q}', f'LAYER_{l}'], ['CX', f'I{q}', f'LAYER_{l_end}'])

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
