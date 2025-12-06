# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys

from pyqrack import QrackSimulator


def bench_qrack(width, depth, sdrp):
    lcv_range = range(width)
    all_bits = list(lcv_range)

    results = []

    control = QrackSimulator(width, isTensorNetwork=False)
    experiment = QrackSimulator(width, isTensorNetwork=False, isOpenCL=False, isSparse=True)
    if sdrp > 0:
        experiment.set_sdrp(sdrp)
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            control.u(i, th, ph, lm)
            experiment.u(i, th, ph, lm)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            control.mcx([c], t)
            experiment.mcx([c], t)

        experiment_probs = experiment.out_probs()
        control_probs = control.out_probs()

        stats = calc_stats(control_probs, experiment_probs, d + 1)

        print(stats)


def calc_stats(ideal_probs, experiment_probs, depth):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    hog_prob = 0
    for i in range(n_pow):
        exp = experiment_probs[i]
        ideal = ideal_probs[i]

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (exp - u_u)

        # QV / HOG
        if ideal > threshold:
            hog_prob += exp

    xeb = numer / denom

    return {
        "qubits": n,
        "depth": depth,
        "xeb": float(xeb),
        "hog_prob": float(hog_prob)
    }


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_qiskit_validation.py [width] [depth] [trials]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    trials = 1
    sdrp = 0
    if len(sys.argv) > 4:
        sdrp = float(sys.argv[4])

    # Run the benchmarks
    bench_qrack(width, depth, sdrp)

    return 0


if __name__ == "__main__":
    sys.exit(main())
