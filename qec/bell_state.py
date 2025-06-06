# Example of entanglement-breaking channel

import math
import statistics
import sys

from pyqrack import QrackAceBackend


def main():
    experiment = QrackAceBackend(2, long_range_columns=0)

    # Experiment has a cleaved-QEC code ACE boundary.
    experiment.h(0)
    experiment.cx(0, 1)

    # L2 fidelity should be 50%:
    shots = 1024
    counts = experiment.measure_shots([0, 1], shots)

    zero = 0
    correlated = 0
    for count in counts:
        if count == 0:
            zero += 1
            correlated += 1
        elif count == 3:
            correlated += 1

    print("Correlation: " + str(correlated / shots))
    print("0/1 balance: " + str(zero / correlated))


if __name__ == "__main__":
    sys.exit(main())
