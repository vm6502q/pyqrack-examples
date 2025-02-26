import math
import random
import sys
import time

from pyqrack import QrackSimulator


def bench_qrack(n):
    # This is a discrete Fourier transform, after initializing all qubits in a GHZ state.
    start = time.perf_counter()

    qsim = QrackSimulator(n, isTensorNetwork=False, isBinaryDecisionTree=True)

    lcv_range = range(n)

    # Single-qubit gates
    # for i in lcv_range:
    #     qsim.u(i, random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi))

    # GHZ state
    qsim.h(0)
    for i in range(1, n):
        qsim.mcx([i - 1], i)

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
    single_width = False
    if len(sys.argv) > 1:
        max_qb = int(sys.argv[1])
    if len(sys.argv) > 2:
        samples = int(sys.argv[2])
    if len(sys.argv) > 3:
        single_width = (sys.argv[2] not in ['False', '0'])

    min_qb = max_qb if single_width else 1

    for n in range(min_qb, max_qb + 1):
        width_results = []

        # Run the benchmarks
        for i in range(samples):
            width_results.append(bench_qrack(n))

        time_result = sum(r[0] for r in width_results) / samples
        fidelity_result = sum(r[1] for r in width_results) / samples
        print(n, ": ", time_result, " seconds, ", fidelity_result, " out of 1.0 fidelity")

    return 0


if __name__ == '__main__':
    sys.exit(main())
