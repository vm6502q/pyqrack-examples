# Example of ("weak simulation condition") stochastic Pauli noise

import math
import random
import sys
import time

from pyqrack import Pauli, QrackSimulator


def inject_pauli_bit_flip_noise(simulator, basis, qubit, probability):
    if (not probability >= 1.) and ((probability <= 0.) or (random.uniform(0., 1.) >= probability)):
        # We avoid the bit flip error
        return

    # We apply a bit flip error
    if basis == Pauli.PauliX:
        simulator.x(qubit)
    elif basis == Pauli.PauliY:
        simulator.y(qubit)
    elif basis == Pauli.PauliZ:
        simulator.z(qubit)


def bench_qrack(n):
    # This is basically a "quantum volume" (random) circuit.
    start = time.perf_counter()

    sim = QrackSimulator(n)

    lcv_range = range(n)
    all_bits = list(lcv_range)

    noise_level = 0.01
    fidelity = 1.0
    for _ in lcv_range:
        # Single-qubit gates
        for i in lcv_range:
            sim.u(i, random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi))
            inject_pauli_bit_flip_noise(sim, Pauli.PauliX, i, noise_level)
            inject_pauli_bit_flip_noise(sim, Pauli.PauliZ, i, noise_level)
            fidelity = fidelity * (1.0 - noise_level)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            sim.mcx([unused_bits.pop()], unused_bits.pop())

    fidelity = fidelity * sim.get_unitary_fidelity()
    # Terminal measurement
    sim.m_all()

    return (time.perf_counter() - start, fidelity)


def main():
    bench_qrack(1)

    max_qb = 24
    samples = 1
    if len(sys.argv) > 1:
        max_qb = int(sys.argv[1])
    if len(sys.argv) > 2:
        samples = int(sys.argv[2])

    for n in range(1, max_qb + 1):
        width_results = []
        
        # Run the benchmarks
        for _ in range(samples):
            width_results.append(bench_qrack(n))

        time_result = sum(r[0] for r in width_results) / samples
        fidelity_result = sum(r[1] for r in width_results) / samples
        print(n, ": ", time_result, " seconds, ", fidelity_result, " out of 1.0 fidelity")

    return 0


if __name__ == '__main__':
    sys.exit(main())
