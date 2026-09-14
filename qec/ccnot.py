# Example of entanglement-breaking channel

import math
import random
import statistics
import sys

from pyqrack import QrackAceBackend

def ccnot(experiment, c1, c2, t):
    experiment.h(t)
    experiment.cx(c2, t)
    experiment.adjt(t)
    experiment.cx(c1, t)
    experiment.t(t)
    experiment.cx(c2, t)
    experiment.adjt(t)
    experiment.cx(c1, t)
    experiment.t(t)
    experiment.h(t)
    experiment.t(c2)
    experiment.cx(c1, c2)
    experiment.t(c1)
    experiment.adjt(c2)
    experiment.cx(c1, c2)


def output(experiment, c1, c2, t):
    shots = 1024
    counts = experiment.measure_shots([c1, c2, t], shots)

    uncorrelated = 0
    three = 0
    four = 0
    seven = 0
    for count in counts:
        if count == 3:
            three += 1
            uncorrelated += 1
        elif count == 4:
            four += 1
            uncorrelated += 1
        elif count == 7:
            seven += 1
    correlated = shots - uncorrelated

    print("Correlation: " + str(correlated / shots))
    if uncorrelated:
        print("[1, 1, 0] frequency (of uncorrelated): " + str(three / uncorrelated))
        print("[0, 0, 1] frequency (of uncorrelated): " + str(four / uncorrelated))
    print("[1, 1, 1] frequency (of correlated): " + (str(seven / correlated) if correlated else "N/A"))
    print("[0, 0, 0] frequency (of correlated): " + (str(1 - (seven / correlated)) if correlated else "N/A"))


def main():
    c1, c2, t = 2, 1 if random.random() < 0.5 else 7, 0

    experiment = QrackAceBackend(10, long_range_columns=2)

    # Experiment has a cleaved-QEC code ACE boundary.
    experiment.h(c1)
    experiment.h(c2)

    ccnot(experiment, c1, c2, t)

    print("Uncorrected:")
    output(experiment, c1, c2, t)
    print()

    experiment = QrackAceBackend(10, long_range_columns=2)

    # Experiment has a cleaved-QEC code ACE boundary.
    experiment.h(c1)
    experiment.h(c2)

    # Error-detection
    experiment.cx(t, 5)
    experiment.cx(c1, 5)
    experiment.cx(c2, 5)
    experiment.cx(t, 6)
    experiment.cx(c2, 6)
    experiment.cx(c1, 6)

    ccnot(experiment, c1, c2, t)

    # Syndrome
    experiment.cx(t, 5)
    experiment.cx(c2, 5)
    experiment.cx(t, 6)
    experiment.cx(c1, 6)

    # Post-selection
    experiment.force_m(5, False)
    experiment.force_m(6, False)

    print("Corrected:")
    output(experiment, c1, c2, t)


if __name__ == "__main__":
    sys.exit(main())
