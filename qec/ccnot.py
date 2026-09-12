# Example of entanglement-breaking channel

import math
import statistics
import sys

from pyqrack import QrackAceBackend

def ccnot(experiment):
    experiment.h(2)
    experiment.cx(1, 2)
    experiment.adjt(2)
    experiment.cx(0, 2)
    experiment.t(2)
    experiment.cx(1, 2)
    experiment.adjt(2)
    experiment.cx(0, 2)
    experiment.t(2)
    experiment.h(2)
    experiment.t(1)
    experiment.cx(0, 1)
    experiment.t(0)
    experiment.adjt(1)
    experiment.cx(0, 1)


def output(experiment):
    shots = 1024
    counts = experiment.measure_shots([0, 1, 2], shots)

    one = 0
    uncorrelated = 0
    for count in counts:
        if (count == 3) or (count == 4):
            uncorrelated += 1
        elif count == 7:
            one += 1
    correlated = shots - uncorrelated

    print("Correlation: " + str(correlated / shots))
    print("[1, 1, 1] count: " + (str(one / correlated) if correlated else "N/A"))


def main():
    experiment = QrackAceBackend(20, long_range_columns=2, is_torus=False)

    # Experiment has a cleaved-QEC code ACE boundary.
    experiment.h(0)
    experiment.h(1)

    ccnot(experiment)

    print("Uncorrected:")
    output(experiment)
    print()

    experiment = QrackAceBackend(20, long_range_columns=2)

    # Experiment has a cleaved-QEC code ACE boundary.
    experiment.h(0)
    experiment.h(1)

    # Error-detection
    experiment.cx(0, 7)
    experiment.cx(1, 7)
    experiment.cx(1, 11)
    experiment.cx(2, 16)

    ccnot(experiment)

    # Syndrome
    experiment.cx(2, 16)
    experiment.acx(6, 11)
    experiment.acx(7, 16)

    # Uncompute
    experiment.acx(6, 11)
    experiment.cx(1, 11)
    experiment.cx(1, 7)
    experiment.cx(0, 7)

    # Post-selection
    experiment.force_m(16, False)
    experiment.force_m(7, False)
    experiment.force_m(11, False)

    print("Corrected:")
    output(experiment)


if __name__ == "__main__":
    sys.exit(main())
