# Orbifolded random circuit sampling
# How good are Google's own "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)
# (This is actually a different "elision" concept, but allow that it works.)

import math
import random
import sys
import time

from pyqrack import QrackAceBackend
from qiskit.providers.qrack import AceQasmSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile


def count_set_bits(n):
    return bin(n).count("1")


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")
    row_len = width // col_len

    return row_len, col_len


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
    sim.iswap(q1, q2)


def iiswap(sim, q1, q2):
    sim.iswap(q1, q2)
    sim.iswap(q1, q2)
    sim.iswap(q1, q2)


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


def bench_qrack(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    start = time.perf_counter()
    experiment = QrackAceBackend(width)

    lcv_range = range(width)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = (
        experiment.cx,
        experiment.cy,
        experiment.cz,
        experiment.acx,
        experiment.acy,
        experiment.acz,
    )

    row_len, col_len = factor_width(width)


def bench_qrack(depth):
    # This is a "nearest-neighbor" coupler random circuit.
    width = 64
    shots = 1000
    lcv_range = range(width)
    all_bits = list(lcv_range)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    results = {
        "qubits": 64,
        "depth": depth
    }

    circ = QuantumCircuit(width)
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            circ.u(th, ph, lm, i)

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

                g = random.choice(two_bit_gates)
                g(circ, b1, b2)

    noise_dummy=AceQasmSimulator(n_qubits=64, long_range_rows=3, long_range_columns=3)
    experiment = QrackAceBackend(64, long_range_rows=3, long_range_columns=3)

    start = time.perf_counter()
    circ = transpile(circ, optimization_level=3, backend=noise_dummy)
    transpile_seconds = time.perf_counter() - start

    start = time.perf_counter()
    experiment.run_qiskit_circuit(circ)
    midpoint = experiment.measure_shots(all_bits, shots)
    forward_seconds = time.perf_counter() - start

    circ = circ.inverse()
    start = time.perf_counter()
    experiment.run_qiskit_circuit(circ)
    terminal = experiment.measure_shots(all_bits, shots)
    backward_seconds = time.perf_counter() - start

    # Experiment results
    hamming_weight = sum(count_set_bits(r) for r in midpoint) / shots
    # Validation
    hamming_distance = sum(count_set_bits(r) for r in terminal) / shots

    results["transpile_seconds"] = transpile_seconds
    results["forward_seconds"] = forward_seconds
    results["backward_seconds"] = backward_seconds
    results["fidelity"] = terminal.count(0) / shots
    results["midpoint_weight"] = hamming_weight
    results["terminal_distance"] = hamming_distance
    results["hamming_fidelity"] = (
        hamming_weight - hamming_distance
    ) / hamming_weight

    return results


def main():
    if len(sys.argv) < 2:
        raise RuntimeError(
            "Usage: python3 rcs_nn_ace_time.py [depth]"
        )

    width = 64
    depth = int(sys.argv[1])

    # Run the benchmarks
    print(bench_qrack(depth))

    return 0


if __name__ == "__main__":
    sys.exit(main())
