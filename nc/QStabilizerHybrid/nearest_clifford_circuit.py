import math
import os
import random
import sys
import time

from pyqrack import QrackSimulator, Pauli


def rand_1qb(sim, q):
    sim.r(Pauli.PauliZ, random.uniform(0, 4 * math.pi), q)


def random_circuit(width, max_magic, circ):
    t_count = 0
    single_bit_gates = circ.h, circ.x, circ.y
    single_bit_gates_with_phase = circ.h, circ.x, circ.y, circ.z, circ.s, circ.adjs
    two_bit_gates = (
        circ.mcx,
        circ.mcy,
        circ.mcz,
        circ.macx,
        circ.macy,
        circ.macz,
        circ.swap,
        circ.iswap,
        circ.adjiswap,
    )

    for n in range(3 * width):
        # Single bit gates
        for j in range(width):
            if (width * width * random.random()) < max_magic:
                random.choice(single_bit_gates)(j)
                rand_1qb(circ, j)
                t_count += 1
            else:
                random.choice(single_bit_gates_with_phase)(j)

        # Multi bit gates
        bit_set = [i for i in range(width)]
        while len(bit_set) > 1:
            b1 = random.choice(bit_set)
            bit_set.remove(b1)
            b2 = random.choice(bit_set)
            bit_set.remove(b2)
            g = random.choice(two_bit_gates)
            if g == circ.swap or g == circ.iswap or g == circ.adjiswap:
                g(b1, b2)
            else:
                g([b1], b2)

    return circ


def bench_qrack(n):
    # This is a demonstration of near-Clifford capabilities, with Clifford+RZ gate set.
    start = time.perf_counter()

    sim = QrackSimulator(
        n,
        isStabilizerHybrid=True,
        isTensorNetwork=False,
        isSchmidtDecomposeMulti=False,
        isSchmidtDecompose=False,
        isOpenCL=False,
    )

    # Run a near-Clifford circuit
    random_circuit(n, n + 1, sim)

    sim.out_to_file("_qrack_tableau.qnc")
    qrack_tableau = ""
    with open("_qrack_tableau.qnc", "r") as file:
        qrack_tableau = file.read()

    # fidelity = sim.get_unitary_fidelity()

    return (time.perf_counter() - start, qrack_tableau)


def main():
    print("Set environment variable QRACK_MAX_CPU_QB=-1 to use this script properly.")

    bench_qrack(1)

    max_qb = 24
    samples = 1
    if len(sys.argv) > 1:
        max_qb = int(sys.argv[1])
    if len(sys.argv) > 2:
        samples = int(sys.argv[2])

    os.environ["QRACK_MAX_CPU_QB"] = "-1"

    for n in range(1, max_qb + 1):
        width_results = []

        # Run the benchmarks
        for _ in range(samples):
            width_results.append(bench_qrack(n))
            print(width_results[-1][1])

        time_result = sum(r[0] for r in width_results) / samples
        print(n, ": ", time_result, " seconds, ")

    return 0


if __name__ == "__main__":
    sys.exit(main())
