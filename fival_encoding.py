# "Fival encoding"
# https://arxiv.org/abs/hep-th/9409150 adopts a definition of a "maximally-entangled" state
# that relies on two qudits with the same (but otherwise arbitrary) dimensional cardinality.
# If padded with 0-amplitude entries so that the new dimensionality of the modified state
# has some cardinality that factors as only first powers of unique prime numbers, then the
# Schmidt rank becomes 1, and the state appears separable in the "Fival encoding."
#
# (The largest qudit in such a case might or might not be larger than the dimensionality of
# the logical qubit space)
#
# Also see https://physics.stackexchange.com/questions/441949/if-the-dimension-of-a-space-is-prime-are-quantum-states-in-it-guaranteed-to-be
# for context in under-appreciated basic considerations to product state factorizability.

import sys
import numpy as np

from pyqrack import QrackSimulator


def main():
    qsim = QrackSimulator(2)
    qsim.h(0)
    qsim.h(1)
    qsim.mcz([0], 1)
    qsim.h(1)

    ket = qsim.out_ket() + [0, 0]
    print(ket)

    r = np.reshape(ket, (2, 3))
    U, S, Vh = np.linalg.svd(r)
    ket1 = U[:, [0]]
    ket2 = Vh[:, [0]]

    print("2 by 3:")
    print(np.kron(ket1, ket2))
    print(S)


if __name__ == "__main__":
    sys.exit(main())
