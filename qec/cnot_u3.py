# Example of entanglement-breaking channel

import math
import random
import statistics
import sys

from collections import Counter

from pyqrack import QrackSimulator, QrackAceBackend


def calc_stats(ideal_probs, counts, shots):
    n_pow = len(ideal_probs)
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    hog_prob = 0
    for b in range(n_pow):
        ideal = ideal_probs[b]
        patch = (counts.get(b, 0) / shots)

        ideal_centered = ideal - u_u
        denom += ideal_centered * ideal_centered
        numer += ideal_centered * (patch - u_u)

        if ideal > threshold:
            hog_prob += patch

    xeb = numer / denom
    return {"xeb": xeb, "hog_prob": hog_prob}


def main():
    th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
    # Keep it Haar-random towards the poles:
    th = math.asin(th / math.pi)

    control = QrackSimulator(3)

    control.h(2)

    control.mcx([2], 1)
    control.u(1, th, ph, lm)
    control.mcx([1], 0)

    ideal_probs = control.out_probs()

    experiment = QrackAceBackend(10, long_range_columns=2, is_torus=False)

    # Experiment has a cleaved-QEC code ACE boundary.
    experiment.h(2)

    # With error-detection
    experiment.cx(0, 5)
    experiment.cx(2, 5)
    experiment.cx(1, 5)
    experiment.cx(2, 1)
    experiment.u(1, th, ph, lm)
    experiment.u(5, th, ph, lm)
    experiment.cx(1, 0)
    experiment.cx(0, 5)
    experiment.force_m(5, False)

    shots = 1024
    counts = dict(Counter(experiment.measure_shots([0, 1, 2], shots)))
    results = calc_stats(ideal_probs, counts, shots)

    print(results)


if __name__ == "__main__":
    sys.exit(main())
