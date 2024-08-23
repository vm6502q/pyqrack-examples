# "pseudo_bell_pairs.py"
#
# Imagine that Qrack's RNG carries 1:1 bits of von Neumann entropy.
# (It's possible `cmake -DENABLE_RDRAND=ON ..` will achieve this.)
# Then, if we act a Z gate conditionally on a virtual Qrack qubit,
# it can be as if "creating a Bell pair" between a Qrack qubit and
# a physical qubit (or Everettian "world" pair) from thermal noise.

import math
import os
import random
import sys

from pyqrack import QrackSimulator


def bench_qrack(width):

    rng = QrackSimulator(1)
    sim = QrackSimulator(width)

    # Loop over the sequence of conditional decision points
    for i in range(width):
        # Independent causal action for decision point
        sim.h(i)
        
        rng.h(0)
        if rng.m(0):
            sim.z(i)
            rng.x(0)

    return sim.m_all()


def main():
    width = 256
    if len(sys.argv) > 1:
        width = int(sys.argv[1])

    # Run the example
    result = bench_qrack(width)

    print("Width=" + str(width) + ": " + str(result) + " result.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
