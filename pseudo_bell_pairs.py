# "pseudo_bell_pairs.py"
#
# Imagine that Qrack's RNG carries 1:1 bits of von Neumann entropy.
# (`cmake -DENABLE_RDRAND=ON ..` get us partway to this condition,
# or load your own von Neumann entropy with `-DENABLE_RNDFILE=ON`.)
# Then, if we act a X gate conditionally on a virtual Qrack qubit,
# it can be as if "creating a Bell pair" between a Qrack qubit and
# a physical qubit (or Everettian "world" pair) from thermal noise.
#
# (Obviously, the state is already "collapsed"! The point is to
# question what "entanglement" could really mean, though.)

import math
import os
import random
import sys

from pyqrack import QrackSimulator


def bench_qrack(width):

    rng = QrackSimulator(width)
    sim = QrackSimulator(width)

    for i in range(width):
        # H(0)
        rng.h(i)

        # CNOT(0, 1)
        # (Remember that measurement and control commute!)
        if rng.m(i):
            sim.x(i)

        # Circuit diagram "creates a Bell pair"

    return (sim.m_all(), rng.m_all())


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
