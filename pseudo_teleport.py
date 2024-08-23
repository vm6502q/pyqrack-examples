# "pseudo_teleport.py"
#
# Imagine that Qrack's RNG carries 1:1 bits of von Neumann entropy.
# (It's possible `cmake -DENABLE_RDRAND=ON ..` will achieve this.)
# Then, we can carry out (decoherent) "quantum teleporation,"
# assuming qubit #2 is a physically-measured qubit (as opposed to
# a virtual qubit).

import math
import os
import random
import sys

from pyqrack import QrackSimulator


def state_prep(sim, qubit):
    sim.u(qubit, random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi))


def bench_qrack():
    rng = QrackSimulator(1)
    sim = QrackSimulator(3)

    sim.h(0)
    sim.mcx([0], 2)
        
    state_prep(sim, 1)
        
    sim.mcx([1], 0)
    sim.h(1)

    if sim.m(0):
        sim.x(2)
    if sim.m(1):
        sim.z(2)

    if sim.m(2):
        print("Measured |1>.")
    else:
        print("Measured |0>.")

def main():
    # Run the example
    bench_qrack()

    return 0


if __name__ == '__main__':
    sys.exit(main())
