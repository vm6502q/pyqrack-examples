# (Clifford) Deutsch-Josza

import random
import sys

from pyqrack import QrackSimulator


# All these example oracles happen to be stabilizer, so we can easily do more than hundreds of qubits.
oracle_qubits = [*range(100)]
num_qubits = len(oracle_qubits) + 1


def zero_oracle(sim):
    pass


def one_oracle(sim):
    sim.x(num_qubits - 1)

# Random subset to XOR (at least one)
xor_qubits = [q for q in oracle_qubits if random.random() < 0.5]
if not xor_qubits:
    xor_qubits = [oracle_qubits[0]]

# Random flips for variety
flip_qubits = [q for q in oracle_qubits if random.random() < 0.5]

def balanced_oracle(sim):
    oracle_qubits = list(range(num_qubits - 1))
    output_qubit = num_qubits - 1
    
    for q in flip_qubits:
        sim.x(q)
    for q in xor_qubits:
        sim.mcx([q], output_qubit)
    for q in flip_qubits:
        sim.x(q)


def main():
    # Prepare the initial register state:
    sim = QrackSimulator(num_qubits, isOpenCL=False)
    sim.x(num_qubits - 1)
    for i in range(num_qubits):
        sim.h(i)

    print("Zero oracle:")

    # Make exactly one query to the oracle:
    zero_oracle(sim)

    # Finish the unitary portion of the algorithm, with the result from the oracle:
    for i in oracle_qubits:
        sim.h(i)

    # Always, a constant oracle measurement result will be "0";
    # always, a balanced oracle measurement result will be anything besides "0":
    result = 0
    for i in oracle_qubits:
        result |= sim.m(i) << i
    if result == 0:
        print("Oracle is constant!")
    else:
        print("Oracle is balanced!")

    print()

    # Prepare the initial register state:
    sim = QrackSimulator(num_qubits, isOpenCL=False)
    sim.x(num_qubits - 1)
    for i in range(num_qubits):
        sim.h(i)

    print("One oracle:")

    # Make exactly one query to the oracle:
    one_oracle(sim)

    # Finish the unitary portion of the algorithm, with the result from the oracle:
    for i in oracle_qubits:
        sim.h(i)

    # Always, a constant oracle measurement result will be "0";
    # always, a balanced oracle measurement result will be anything besides "0":
    result = 0
    for i in oracle_qubits:
        result |= sim.m(i) << i
    if result == 0:
        print("Oracle is constant!")
    else:
        print("Oracle is balanced!")

    print()

    # Prepare the initial register state:
    sim = QrackSimulator(num_qubits, isOpenCL=False)
    sim.x(num_qubits - 1)
    for i in range(num_qubits):
        sim.h(i)

    print("Balanced oracle:")

    # Make exactly one query to the oracle:
    balanced_oracle(sim)

    # Finish the unitary portion of the algorithm, with the result from the oracle:
    for i in oracle_qubits:
        sim.h(i)

    # Always, a constant oracle measurement result will be "0";
    # always, a balanced oracle measurement result will be anything besides "0":
    result = 0
    for i in oracle_qubits:
        result |= sim.m(i) << i
    if result == 0:
        print("Oracle is constant!")
    else:
        print("Oracle is balanced!")


if __name__ == "__main__":
    sys.exit(main())
