# Example of a (Hadamard) quantum random walk

import math
import random
import sys
import time

from pyqrack import QrackSimulator


def main():
    lattice_qb_count = 8
    lattice_qubits = list(range(lattice_qb_count))
    coin_qubit = lattice_qb_count
    steps = 1 << (lattice_qb_count - 1)
    limit = steps - 1

    sim = QrackSimulator(lattice_qb_count + 1)
    sim.x(lattice_qb_count - 1)

    for _ in range(steps):
        sim.h(coin_qubit)
        sim.mcadd(1, [coin_qubit], lattice_qubits)
        sim.x(coin_qubit)
        sim.mcsub(1, [coin_qubit], lattice_qubits)
        sim.x(coin_qubit)

    exp_pos = sim.permutation_expectation(lattice_qubits) - limit
    obs_pos = (sim.m_all() & ~(1 << coin_qubit)) - limit
    print("Expected position:", exp_pos)
    print("Observed position:", obs_pos)


if __name__ == '__main__':
    sys.exit(main())
