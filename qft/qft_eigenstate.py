# Recreation of the eigenstate-input QFT from Qrack in https://arxiv.org/abs/2304.14969

import math
import random
import sys
import time

from pyqrack import QrackSimulator


def bench_qrack(n):
    # This is a discrete Fourier transform, after initializing all qubits randomly but separably.
    start = time.perf_counter()

    qsim = QrackSimulator(n)
    perm = random.randint(0, 1 << n)
    for b in range(n):
        if (perm >> b) & 1:
            qsim.x(b)
    qsim.qft(list(range(n)))
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
