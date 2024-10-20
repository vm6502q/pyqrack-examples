# Example of entanglement-breaking channel

import math
import statistics
import sys

from pyqrack import QrackSimulator


def calc_xeb(ideal_probs, split_probs):
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    for i in range(n_pow):
        split = split_probs[i]
        ideal = ideal_probs[i]

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (split - u_u)

    return numer / denom


def main():
    control = QrackSimulator(2)
    experiment = QrackSimulator(2, isTensorNetwork=False, isStabilizerHybrid=False)

    # Prepare a Bell pair.
    control.h(1)
    control.mcx([1], 2)

    experiment.h(1)
    experiment.mcx([1], 2)

    # Apply an entanglement-breaking channel
    experiment.separate([1])

    # Cross entropy should be 50%
    ideal_probs = control.out_probs()
    split_probs = experiment.out_probs()

    for i in range(len(ideal_probs)):
        print("|<" + str(i) + "|ideal>|^2: " + str(ideal_probs[i]))

    print()

    for i in range(len(ideal_probs)):
        print("|<" + str(i) + "|split>|^2: " + str(split_probs[i]))

    print()

    print("XEB: " + str(calc_xeb(ideal_probs, split_probs)))


if __name__ == '__main__':
    sys.exit(main())
