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

from qiskit import QuantumCircuit, qpy

import quimb.tensor as tn
from qiskit_quimb import quimb_circuit


# Function by Google search AI
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


def cx(sim, q1, q2):
    sim.cx(q1, q2)


def cy(sim, q1, q2):
    sim.cy(q1, q2)


def cz(sim, q1, q2):
    sim.cz(q1, q2)


def acx(sim, q1, q2):
    sim.x(q1)
    sim.cx(q1, q2)
    sim.x(q1)


def acy(sim, q1, q2):
    sim.x(q1)
    sim.cy(q1, q2)
    sim.x(q1)


def acz(sim, q1, q2):
    sim.x(q1)
    sim.cz(q1, q2)
    sim.x(q1)


def bench_qrack(width, depth, sdrp, is_sparse):
    lcv_range = range(width)
    all_bits = list(lcv_range)
    retained = width * width
    checked = min(1 << width, retained * width)
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = cx, cy, cz, acx, acy, acz
    row_len, col_len = factor_width(width)

    rcs = QuantumCircuit(width)
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            for b in range(3):
                rcs.h(i)
                rcs.rz(random.uniform(0, 2 * math.pi), i)

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
                temp_col = temp_col + (1 if (gate & 1) else 0)

                if temp_row < 0:
                    temp_row = temp_row + row_len
                if temp_col < 0:
                    temp_col = temp_col + col_len
                if temp_row >= row_len:
                    temp_row = temp_row - row_len
                if temp_col >= col_len:
                    temp_col = temp_col - col_len

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width):
                    continue

                if d & 1:
                    t = b1
                    b1 = b2
                    b2 = t

                g = random.choice(two_bit_gates)
                g(rcs, b1, b2)

    with open("rcs_nn_tn.qpy", "wb") as file:
        qpy.dump(rcs, file)

    if is_sparse:
        experiment = QrackSimulator(width, isTensorNetwork=False, isOpenCL=False, isSparse=True)
    else:
        experiment = QrackSimulator(width, isTensorNetwork=False)
    if sdrp > 0:
        experiment.set_sdrp(sdrp)
    experiment.run_qiskit_circuit(rcs)
    experiment_perms = experiment.highest_n_prob_perm(checked)
    experiment = None

    quimb_rcs = quimb_circuit(rcs)

    cx_count = width >> 1
    for l in range(depth):
        layer_offset = (6 * width + cx_count) * l
        for q in range(width):
            h_idx = layer_offset + 6 * q
            quimb_rcs.psi.contract_between([f'GATE_{h_idx}'], [f'GATE_{h_idx + 5}'])

    n_pow = 1 << width
    u_u =  1 / n_pow
    idx = 0
    ideal_probs = {}
    sum_probs = 0
    for key in experiment_perms:
        prob = float((abs(complex(quimb_rcs.amplitude(int_to_bitstring(key, width), backend="jax"))) ** 2).real)
        if prob <= u_u:
            continue
        ideal_probs[key] = prob
        sum_probs += prob
        if len(ideal_probs) >= retained:
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
            "Usage: python3 rcs_nn_tn.py [width] [depth] [sdrp] [is_sparse]"
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
