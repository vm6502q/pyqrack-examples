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
    lcv_range = range(width)
    all_bits = list(lcv_range)

    results = { 'qubits': width, 'depth': depth, 'midpoint_weight': 0, 'terminal_distance': 0 }

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
        sim = QrackSimulator(width)
        sim.run_qiskit_circuit(circ)
        midpoint = sim.measure_shots(all_bits, shots)
        sim.run_qiskit_circuit(circ.inverse())
        terminal = sim.measure_shots(all_bits, shots)
        seconds = time.perf_counter() - start

        hamming_weight = sum(count_set_bits(r) for r in midpoint) / shots
        hamming_distance = sum(count_set_bits(r) for r in terminal) / shots

        results['midpoint_weight'] += hamming_weight
        results['terminal_distance'] += hamming_distance

    results['midpoint_weight'] /= trials
    results['terminal_distance'] /= trials

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
