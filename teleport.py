# Example of the quantum teleportation algorithm

import math
import random
import sys

from pyqrack import QrackSimulator


def print_bit(sim, q):
    print("Z Prob. =", sim.prob(q))
    sim.h(q)
    print("X Prob. =", sim.prob(q))
    sim.h(q)


def state_prep(sim):
    # "Alice" has a qubit to teleport.

    # (To test consistency, comment out this U() gate.)
    the = 2 * math.pi * random.uniform(0, 1)
    phi = 2 * math.pi * random.uniform(0, 1)
    lam = 2 * math.pi * random.uniform(0, 1)
    sim.u(0, the, phi, lam)

    # (Try with and without just an X() gate, instead.)
    # sim.x(0)

    print("Alice is sending:")
    print_bit(sim, 0)


def main():
    sim = QrackSimulator(3)

    # "Eve" prepares a Bell pair.
    sim.h(1)
    sim.mcx([1], 2)

    # Alice prepares her "message."
    state_prep(sim)
    # Alice entangles her message with her half of the Bell pair.
    sim.mcx([0], 1)
    sim.h(0)
    # Alice measures both of her bits
    q0 = sim.m(0)
    q1 = sim.m(1)

    # "Bob" receives classical message and prepares his half of the Bell pair to complete teleportation.
    if q0:
        sim.z(2)
    if q1:
        sim.x(2)
    print("Bob received:")
    print_bit(sim, 2)


if __name__ == '__main__':
    sys.exit(main())
