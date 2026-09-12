# Example of entanglement-breaking channel

import math
import statistics
import sys

from pyqrack import QrackAceBackend


def main():
    experiment = QrackAceBackend(15, long_range_columns=2)

    # Experiment has a cleaved-QEC code ACE boundary.
    experiment.h(0)
    experiment.h(1)

    # Error-detection
    experiment.cx(0, 6)
    experiment.cx(1, 6)
    experiment.cx(1, 7)
    experiment.cx(2, 11)

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

    # Post-selection
    experiment.cx(2, 11)
    experiment.acx(6, 7)
    experiment.acx(7, 11)
    experiment.acx(6, 7)
    experiment.cx(1, 6)
    experiment.cx(0, 6)
    experiment.force_m(11, False)

    # Any correlation above 0.5 is entanglement non-locality.
    shots = 1024
    counts = experiment.measure_shots([0, 1, 2], shots)

    one = 0
    uncorrelated = 0
    for count in counts:
        count &= 7
        if count == 3:
            uncorrelated += 1
        elif count == 7:
            one += 1
    correlated = shots - uncorrelated

    print("Correlation: " + str(correlated / shots))
    print("[1, 1, 1] count: " + (str(one / correlated) if correlated else "N/A"))


if __name__ == "__main__":
    sys.exit(main())
