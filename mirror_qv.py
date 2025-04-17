# Demonstrate mirror circuit simplification

import math
import random
import sys
import time

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit


def count_set_bits(n):
        return bin(n).count('1')


def bench_qrack(width, depth, trials):
    # This is a "nearest-neighbor" coupler random circuit.
    shots = 100
    n_perm = 1 << width
    lcv_range = range(width)
    all_bits = list(lcv_range)

    results = { 'qubits': width, 'depth': depth, 'trials': trials, 'seconds_avg': 0, 'midpoint_weight_avg': 0, 'terminal_distance_avg': 0 }

    for trial in range(trials):
        circ = QuantumCircuit(width)
        for d in range(depth):
            # Single-qubit gates
            for i in lcv_range:
                for _ in range(3):
                    circ.h(i)
                    circ.rz(random.uniform(0, 2 * math.pi), i)

            # 2-qubit couplers
            unused_bits = all_bits.copy()
            random.shuffle(unused_bits)
            while len(unused_bits) > 1:
                c = unused_bits.pop()
                t = unused_bits.pop()
                circ.cx(c, t)

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

        # Uncompute the experiment
        experiment.run_qiskit_circuit(circ.inverse())
        # Uncompute state preparation
        for bit_index in range(width):
            if (rand_perm >> bit_index) & 1:
                experiment.x(bit_index)

        # Check whether the experiment and state preparation
        # ("...and measurement", SPAM) is observably uncomputed.
        terminal = experiment.measure_shots(all_bits, shots)

        seconds = time.perf_counter() - start

        # Experiment results
        hamming_weight = sum(count_set_bits(r) for r in midpoint) / shots
        # Validation
        hamming_distance = sum(count_set_bits(r) for r in terminal) / shots

        results['seconds_avg'] += seconds
        results['midpoint_weight_avg'] += hamming_weight
        results['terminal_distance_avg'] += hamming_distance

    results['seconds_avg'] /= trials
    results['midpoint_weight_avg'] /= trials
    results['terminal_distance_avg'] /= trials

    return results


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 mirror_qv.py [width] [depth] [trials]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    trials = 1
    if len(sys.argv) > 3:
        trials = int(sys.argv[3])

    print(bench_qrack(width, depth, trials))

    return 0


if __name__ == '__main__':
    sys.exit(main())
