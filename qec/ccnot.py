# Example of entanglement-breaking channel

import math
import statistics
import sys

from pyqrack import QrackAceBackend


def main():
    experiment = QrackAceBackend(3, long_range_columns=2)

    # Experiment has a cleaved-QEC code ACE boundary.
    experiment.h(0)
    experiment.h(1)

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

    # Any correlation above 0.5 is entanglement non-locality.
    shots = 1024
    counts = experiment.measure_shots([0, 1, 2], shots)

    one = 0
    correlated = 0
    for count in counts:
        if count == 0:
            correlated += 1
        elif count == 1:
            correlated += 1
        elif count == 2:
            correlated += 1
        elif count == 7:
            correlated += 1
            one += 1

    print("Correlation: " + str(correlated / shots))
    print("1-1-1 count: " + (str(one / correlated) if correlated else "N/A"))


if __name__ == "__main__":
    sys.exit(main())
