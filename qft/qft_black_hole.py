# Benchmark the preparation of a state with maximal spatial information density (AKA "entropy," at that point)

import math
import random
import sys
import time

from pyqrack import QrackSimulator


def bench_qrack(n):
    # This is a discrete Fourier transform, after initializing all qubits randomly but separably.
    start = time.perf_counter()

    qsim = QrackSimulator(n, isTensorNetwork=False)

    lcv_range = range(n)

    theta_crit = math.acos(1.0 / math.sqrt(3))

    # Idealized maximum information density state...
    for i in lcv_range:
        qsim.u(
            i,
            theta_crit,
            theta_crit,
            theta_crit,
        )
        if 2 * random.random() < 1:
            qsim.x(i)
        if 2 * random.random() < 1:
            qsim.y(i)
        if 2 * random.random() < 1:
            qsim.z(i)

    # ...But this might be rather what happens in practice:
    # for i in lcv_range:
    #     qsim.u(
    #         i,
    #         random.uniform(0, 2 * math.pi),
    #         random.uniform(0, 2 * math.pi),
    #         random.uniform(0, 2 * math.pi),
    #     )

    qsim.qft(list(lcv_range))

    end = n - 1
    for i in range(n // 2):
        qsim.swap(i, end - i)

    fidelity = qsim.get_unitary_fidelity()
    # Terminal measurement
    qsim.m_all()

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
        for i in range(samples):
            width_results.append(bench_qrack(n))

        time_result = sum(r[0] for r in width_results) / samples
        fidelity_result = sum(r[1] for r in width_results) / samples
        print(
            n, ": ", time_result, " seconds, ", fidelity_result, " out of 1.0 fidelity"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
