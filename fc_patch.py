# How good are Google's own "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)
# (This is actually a different "elision" concept, but allow that it works.)

import math
import os
import random
import statistics
import sys
import time

import numpy as np

import pandas as pd

from pyqrack import QrackSimulator

from scipy.stats import binom


def bench_qrack(width, depth):
    patch_size = (width + 1) >> 1
    # This is a fully-connected random circuit.
    control = QrackSimulator(width)
    experiment = QrackSimulator(width)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    for d in range(depth):
        start = time.perf_counter()
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            experiment.u(i, th, ph, lm)
            control.u(i, th, ph, lm)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            control.h(t)
            control.mcz([c], t)
            control.h(t)
            experiment.h(t)
            if (c < patch_size and t >= patch_size) or (t < patch_size and c >= patch_size):
                # This is our version of ("semi-classical") gate "elision":
                experiment.u(t, 0, 0, math.pi * experiment.prob(c))
            else:
                experiment.mcz([c], t)
            experiment.h(t)

    ideal_probs = control.out_probs()
    patch_probs = experiment.out_probs()

    return (ideal_probs, patch_probs, time.perf_counter() - start)


def calc_stats(ideal_probs, patch_probs, interval, depth):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    ideal_df = pd.DataFrame(ideal_probs)
    patch_df = pd.DataFrame(patch_probs)
    threshold = pd.median(ideal_probs)
    u_u = pd.mean(ideal_probs)

    # XEB / EPLG
    ideal_centered = ideal_df - u_u
    denom = (ideal_centered * ideal_centered).sum()
    numer = (ideal_centered * (patch - u_u)).sum()

    # QV / HOG
    hog_prob = (patch >= threshold).sum()

    xeb = numer / denom

    return {
        'qubits': n,
        'depth': depth,
        'seconds': interval,
        'xeb': xeb,
        'hog_prob': hog_prob,
        'qv_pass': hog_prob >= 2 / 3,
        'eplg':  (1 - (xeb ** (1 / depth))) if xeb < 1 else 0
    }


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 sycamore_2019_patch.py [width] [depth]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Run the benchmarks
    result = bench_qrack(width, depth)
    # Calc. and print the results
    print(calc_stats(result[0], result[1], result[2], depth))

    return 0


if __name__ == '__main__':
    sys.exit(main())
