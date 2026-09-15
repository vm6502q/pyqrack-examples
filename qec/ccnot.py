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

    correlated = 0
    uncorrelated = 0
    zero = 0
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0
    six = 0
    seven = 0
    for count in counts:
        if count == 0:
            zero += 1
            correlated += 1
        elif count == 1:
            one += 1
            correlated += 1
        elif count == 2:
            two += 1
            correlated += 1
        elif count == 3:
            three += 1
            uncorrelated += 1
        elif count == 4:
            four += 1
            uncorrelated += 1
        elif count == 5:
            five += 1
            uncorrelated += 1
        elif count == 6:
            six += 1
            uncorrelated += 1
        elif count == 7:
            seven += 1
            correlated += 1

    print("Correlation: " + str(correlated / shots))
    if uncorrelated:
        print("[1, 1, 0] frequency (of uncorrelated): " + str(three / uncorrelated))
        print("[0, 0, 1] frequency (of uncorrelated): " + str(four / uncorrelated))
        print("[1, 0, 1] frequency (of uncorrelated): " + str(five / uncorrelated))
        print("[0, 1, 1] frequency (of uncorrelated): " + str(six / uncorrelated))
    if correlated:
        print("[1, 0, 0] frequency (of correlated): " + (str(one / correlated) if correlated else "N/A"))
        print("[0, 1, 0] frequency (of correlated): " + (str(two / correlated) if correlated else "N/A"))
        print("[1, 1, 1] frequency (of correlated): " + (str(seven / correlated) if correlated else "N/A"))
        print("[0, 0, 0] frequency (of correlated): " + (str(zero / correlated) if correlated else "N/A"))


def main():
    c1 = 2
    c2 = 7 if random.random() < 0.5 else 1
    t = 0

    print(f"Qubit indices:[[Boundary], {'[Boundary]' if c2 == 7 else '[Bulk]'}, [Bulk]]")
    print("(Boundary target qubit is handled same for corrected as uncorrected.)")
    print()

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

    experiment.ccx(c1, c2, t)

    print("Corrected:")
    output(experiment, c1, c2, t)


if __name__ == "__main__":
    sys.exit(main())
