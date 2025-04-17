# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import sys
import time

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while (((width // col_len) * col_len) != width):
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


def count_set_bits(n):
        return bin(n).count('1')


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


def bench_qrack(width, depth, trials, is_obfuscated):
    # This is a "nearest-neighbor" coupler random circuit.
    shots = 100
    n_perm = 1 << width
    lcv_range = range(width)
    all_bits = list(lcv_range)

    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    results = {
        'qubits': width,
        'depth': depth,
        'trials': trials,
        'forward_seconds_avg': 0,
        'backward_seconds_avg': 0,
        'fidelity_avg': 0,
        'midpoint_weight_avg': 0,
        'terminal_distance_avg': 0,
        'hamming_fidelity_avg': 0
    }

    for trial in range(trials):
        circ = QuantumCircuit(width)
        for d in range(depth):
            # Single-qubit gates
            for i in lcv_range:
                # Single-qubit gates
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
                    temp_row = temp_row + (1 if (gate & 2) else -1);
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

        start = time.perf_counter()

        experiment = QrackSimulator(width)

        # To ensure no dependence on initial |0> state,
        # initialize to a random permutation.
        rand_perm = random.randint(0, n_perm - 1)
        for bit_index in range(width):
            if (rand_perm >> bit_index) & 1:
                experiment.x(bit_index)

        # Run the experiment.
        experiment.run_qiskit_circuit(circ)
        # Collect the experimental observable results.
        midpoint = experiment.measure_shots(all_bits, shots)

        forward_seconds = time.perf_counter() - start

        # Optionally "obfuscate" the circuit adjoint.
        adj_circ = (transpile(circ, basis_gates=["u", "cx"]) if is_obfuscated else circ).inverse()

        # Uncompute the experiment
        experiment.run_qiskit_circuit(adj_circ)
        # Uncompute state preparation
        for bit_index in range(width):
            if (rand_perm >> bit_index) & 1:
                experiment.x(bit_index)

        # Check whether the experiment and state preparation
        # ("...and measurement", SPAM) is observably uncomputed.
        terminal = experiment.measure_shots(all_bits, shots)

        backward_seconds = time.perf_counter() - (forward_seconds + start)

        # Experiment results
        hamming_weight = sum(count_set_bits(r) for r in midpoint) / shots
        # Validation
        hamming_distance = sum(count_set_bits(r) for r in terminal) / shots

        results['forward_seconds_avg'] += forward_seconds
        results['backward_seconds_avg'] += backward_seconds
        results['fidelity_avg'] += terminal.count(0) / shots
        results['midpoint_weight_avg'] += hamming_weight
        results['terminal_distance_avg'] += hamming_distance
        results['hamming_fidelity_avg'] += (hamming_weight - hamming_distance) / hamming_weight

    results['forward_seconds_avg'] /= trials
    results['backward_seconds_avg'] /= trials
    results['fidelity_avg'] /= trials
    results['midpoint_weight_avg'] /= trials
    results['terminal_distance_avg'] /= trials
    results['hamming_fidelity_avg'] /= trials

    return results

def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 mirror_nn_depth_series.py [width] [depth] [trials] [is_obfuscated]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    trials = 1
    is_obfuscated = False
    if len(sys.argv) > 3:
        trials = int(sys.argv[3])
    if len(sys.argv) > 4:
        trials = (sys.argv[4] not in ['False', '0'])

    # Run the benchmarks
    print(bench_qrack(width, depth, trials, is_obfuscated))

    return 0


if __name__ == '__main__':
    sys.exit(main())
