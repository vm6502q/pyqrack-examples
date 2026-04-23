import math
import random
import statistics
import sys
import time

from collections import Counter

from pyqrack import QrackSimulator


def calc_stats(ideal_probs, exp_probs):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    mean_guess = 1 / n_pow
    model = 1 / 2
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    hog_prob = 0
    sqr_diff = 0
    for i in range(n_pow):
        exp = (1 - model) * (exp_probs[i] if i in exp_probs else 0) + model * mean_guess
        ideal = ideal_probs[i]

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (exp - u_u)

        # L2 norm
        sqr_diff += (ideal - exp) ** 2

        # QV / HOG
        if ideal > threshold:
            hog_prob += exp

    xeb = numer / denom
    rss = math.sqrt(sqr_diff)

    return {
        "qubits": n,
        "xeb": float(xeb),
        "hog_prob": float(hog_prob),
        "l2_diff": float(rss),
    }


def bench_qrack(n):
    control = QrackSimulator(n, isBinaryDecisionTree=False)
    experiment = QrackSimulator(n, isBinaryDecisionTree=True)

    # GHZ state
    control.h(0)
    experiment.h(0)
    for i in range(1, n):
        control.mcx([i - 1], i)
        experiment.mcx([i - 1], i)

    all_bits = list(range(n))
    control.qft(all_bits)
    experiment.qft(all_bits)

    end = n - 1
    for i in range(n // 2):
        control.swap(i, end - i)
        experiment.swap(i, end - i)

    control = control.out_probs()
    experiment = experiment.out_probs()

    return calc_stats(control, experiment)


def main():
    bench_qrack(1)

    n = 18
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    print(bench_qrack(n))

    return 0


if __name__ == "__main__":
    sys.exit(main())
