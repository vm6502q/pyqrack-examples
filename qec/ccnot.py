# Example of entanglement-breaking channel

import math
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


def output(experiment):
    shots = 1024
    counts = experiment.measure_shots([0, 1, 2], shots)

    one = 0
    uncorrelated = 0
    for count in counts:
        if (count == 1) or (count == 6):
            uncorrelated += 1
        elif count == 7:
            one += 1
    correlated = shots - uncorrelated

    print("Correlation: " + str(correlated / shots))
    print("[1, 1, 1] frequency: " + (str(one / correlated) if correlated else "N/A"))


def main():
    experiment = QrackAceBackend(15, long_range_columns=2, is_torus=False)

    # Experiment has a cleaved-QEC code ACE boundary.
    experiment.h(2)
    experiment.h(1)

    ccnot(experiment, 2, 1, 0)

    print("Uncorrected:")
    output(experiment)
    print()

    experiment = QrackAceBackend(15, long_range_columns=2)

    # Experiment has a cleaved-QEC code ACE boundary.
    experiment.h(2)
    experiment.h(1)

    # Error-detection
    experiment.cx(2, 5)
    experiment.cx(1, 5)
    experiment.cx(1, 6)
    experiment.cx(0, 10)

    ccnot(experiment, 2, 1, 0)

    # Syndrome
    experiment.cx(0, 10)
    experiment.acx(5, 10)
    experiment.acx(6, 10)

    # Uncompute
    experiment.cx(1, 6)
    experiment.cx(1, 5)
    experiment.cx(2, 5)

    # Post-selection
    experiment.force_m(10, False)
    experiment.force_m(6, False)
    experiment.force_m(5, False)

    print("Corrected:")
    output(experiment)


if __name__ == "__main__":
    sys.exit(main())
