# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import sys

from collections import Counter

from pyqrack import QrackStabilizer

from qiskit import QuantumCircuit


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


def swap(sim, q1, q2):
    sim.swap(q1, q2)


def iswap(sim, q1, q2):
    sim.swap(q1, q2)
    sim.cz(q1, q2)
    sim.s(q1)
    sim.s(q2)


def iiswap(sim, q1, q2):
    sim.s(q2)
    sim.s(q1)
    sim.cz(q1, q2)
    sim.swap(q1, q2)


def pswap(sim, q1, q2):
    sim.cz(q1, q2)
    sim.swap(q1, q2)


def mswap(sim, q1, q2):
    sim.swap(q1, q2)
    sim.cz(q1, q2)


def nswap(sim, q1, q2):
    sim.cz(q1, q2)
    sim.swap(q1, q2)
    sim.cz(q1, q2)


def bench_qrack(n_qubits, magic):
    # This is a "nearest-neighbor" coupler random circuit.
    shots = 100
    lcv_range = range(n_qubits)
    all_bits = list(lcv_range)

    print(f"{n_qubits} qubits, square circuit, {magic} units of 'magic', then mirrored for double")

    rz_count = magic
    rz_opportunities = n_qubits * n_qubits * 3
    rz_positions = []
    while len(rz_positions) < rz_count:
        rz_position = random.randint(0, rz_opportunities - 1)
        if rz_position in rz_positions:
            continue
        rz_positions.append(rz_position)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz
    row_len, col_len = factor_width(n_qubits)

    qc = QuantumCircuit(n_qubits)
    gate_count = 0
    for d in range(n_qubits >> 1):
        # Single-qubit gates
        for i in lcv_range:
            # Single-qubit gates
            for _ in range(3):
                qc.h(i)
                s_count = random.randint(0, 7)
                if s_count & 1:
                    qc.z(i)
                if s_count & 2:
                    qc.s(i)
                if gate_count in rz_positions:
                    # angle = random.uniform(0, math.pi / 2)
                    # qc.rz(angle, i)
                    if s_count & 4:
                        qc.t(i)
                    else:
                        qc.tdg(i)
                gate_count = gate_count + 1

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

                if (b1 >= n_qubits) or (b2 >= n_qubits):
                    continue

                g = random.choice(two_bit_gates)
                g(qc, b1, b2)

    qc = qc.compose(qc.inverse())

    # Round to nearest Clifford circuit
    zero_count = 0
    for i in range(shots):
        experiment = QrackStabilizer(n_qubits)
        experiment.run_qiskit_circuit(qc, shots=0)
        if experiment.m_all() == 0:
            zero_count += 1

    print(f"Fidelity: {zero_count} correct out of {shots} shots")

def main():
    n_qubits = 36
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    magic = 18
    if len(sys.argv) > 2:
        magic = int(sys.argv[2])

    # Run the benchmarks
    bench_qrack(n_qubits, magic)

    return 0


if __name__ == "__main__":
    sys.exit(main())
