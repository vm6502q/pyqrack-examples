# Example of entanglement-breaking channel

import math
import statistics
import sys

from pyqrack import QrackSimulator


def calc_xeb(ideal_probs, split_probs):
    n_pow = len(ideal_probs)
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


def calc_fidelity(ideal_ket, split_ket):
    s = sum([x * y.conjugate() for x, y in zip(ideal_ket, split_ket)])
    return (s * s.conjugate()).real

def main():
    control = QrackSimulator(2)
    experiment = QrackSimulator(2, isTensorNetwork=False, isStabilizerHybrid=False)

    # Prepare a Bell pair.
    control.h(0)
    control.mcx([0], 1)

    experiment.h(0)
    experiment.mcx([0], 1)

    # Apply an entanglement-breaking channel
    experiment.separate([0])

    # L2 fidelity should be 50%:
    ideal_ket = control.out_ket()
    split_ket = experiment.out_ket()

    for i in range(len(ideal_ket)):
        print("<" + str(i) + "|ideal>: " + str(ideal_ket[i]))

    print()

    for i in range(len(ideal_ket)):
        print("<" + str(i) + "|split>: " + str(split_ket[i]))

    print()

    print("Fidelity: " + str(calc_fidelity(ideal_ket, split_ket)))

    print()

    # Cross entropy should be 0
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
