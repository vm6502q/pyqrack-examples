# Example of entanglement-breaking channel

import math
import statistics
import sys

from pyqrack import QrackAceBackend


def main():
    experiment = QrackAceBackend(16, long_range_columns=1)

    # Experiment has a cleaved-QEC code ACE boundary.
    experiment.h(4)
    experiment.cx(4, 5)

    # Any correlation above 0.5 is entanglement non-locality.
    shots = 1024
    counts = experiment.measure_shots([4, 5], shots)

    zero = 0
    correlated = 0
    for count in counts:
        if count == 0:
            zero += 1
            correlated += 1
        elif count == 3:
            correlated += 1

    print("Correlation: " + str(correlated / shots))
    print("0/1 balance: " + (str(zero / correlated) if correlated else "N/A"))


if __name__ == "__main__":
    sys.exit(main())
