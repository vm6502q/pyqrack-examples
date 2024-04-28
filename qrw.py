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
    sign_power = 1 << (lattice_qb_count - 1)

    sim = QrackSimulator(lattice_qb_count + 1)

    for _ in range(sign_power):
        sim.h(coin_qubit)
        sim.mcadd(1, [coin_qubit], lattice_qubits)
        sim.x(coin_qubit)
        sim.mcsub(1, [coin_qubit], lattice_qubits)
        sim.x(coin_qubit)

    exp_pos = sim.permutation_expectation(lattice_qubits)
    if exp_pos >= sign_power:
        exp_pos = exp_pos - 2 * sign_power
    obs_pos = (sim.m_all() & ~(1 << coin_qubit))
    if obs_pos >= sign_power:
        obs_pos = obs_pos - 2 * sign_power
    print("Expected position:", exp_pos)
    print("Observed position:", obs_pos)


if __name__ == '__main__':
    sys.exit(main())
