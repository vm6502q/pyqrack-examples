# Example of entanglement-breaking channel

import math
import statistics
import sys

from pyqrack import QrackAceBackend


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
    experiment = QrackAceBackend(10, long_range_columns=2)

    # Experiment has a cleaved-QEC code ACE boundary.
    experiment.h(2)

    experiment.cx(2, 1)
    experiment.cx(1, 0)

    print("Uncorrected:")
    output(experiment)
    print()

    experiment = QrackAceBackend(15, long_range_columns=2)

    # Experiment has a cleaved-QEC code ACE boundary.
    experiment.h(0)

    # With error-detection
    experiment.cx(0, 6)
    experiment.cx(2, 6)
    experiment.cx(2, 1)
    experiment.cx(1, 0)
    experiment.cx(0, 6)
    experiment.force_m(6, False)

    print("Corrected:")
    output(experiment)


if __name__ == "__main__":
    sys.exit(main())
