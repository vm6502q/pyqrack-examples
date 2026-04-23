# (Clifford) Bernstein-Vazirani

import random
import sys

from pyqrack import QrackStabilizer


# All these example oracles happen to be stabilizer, so we can easily do more than hundreds of qubits.
o_qubits = 100
oracle_qubits = [*range(o_qubits)]
num_qubits = len(oracle_qubits) + 1
hidden_bits = random.randint(0, (1 << o_qubits) - 1)


def oracle(sim):
    for i in oracle_qubits:
        if ((hidden_bits >> i) & 1) > 0:
            sim.mcx([i], num_qubits - 1)


def main():
    # Prepare the initial register state:
    sim = QrackStabilizer(num_qubits)
    sim.x(num_qubits - 1)
    for i in range(num_qubits):
        sim.h(i)

    # Make exactly one query to the oracle:
    oracle(sim)

    # Finish the unitary portion of the algorithm, with the result from the oracle:
    for i in oracle_qubits:
        sim.h(i)

    # This measurement result is always the "hidden_bits" parameter of the oracle.
    result = 0
    for i in oracle_qubits:
        result |= sim.m(i) << i

    print("Output string: ", result)
    print("True answer: ", hidden_bits)


if __name__ == "__main__":
    sys.exit(main())
