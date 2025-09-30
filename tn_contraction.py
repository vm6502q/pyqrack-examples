import math
import random
import sys

import qiskit.qasm2

from qiskit_quimb import quimb_circuit
import quimb.tensor as qtn

from pyqrackising import convert_quimb_tree_to_tsp, tsp_symmetric


def generate_qv_circuit(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    lcv_range = range(width)
    all_bits = list(lcv_range)

    circ = qtn.Circuit(width)
    contraction_set = []
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            tag = f"U3_d{d}_q{i}"
            circ.apply_gate('U3', th, ph, lm, i, tags=tag)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            tag = f"CNOT_d{d}_c{c}_t{t}"
            circ.apply_gate('CNOT', (c, t), tags=tag)
            contraction_set.append((f"U3_d{d}_q{c}", tag))
            contraction_set.append((f"U3_d{d}_q{t}", tag))

    circ = circ.psi
    for tags in contraction_set:
        circ.contract_between(tags[0], tags[1])

    return circ


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_qiskit_validation.py [width] [depth]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Generate the circuit
    quimb_tn = generate_qv_circuit(width, depth)
    # Convert to TSP
    tsp, nodes = convert_quimb_tree_to_tsp(quimb_tn)
    # Solve TSP
    tsp_sol, raw_cost = tsp_symmetric(tsp, monte_carlo=False, is_cyclic=False)

    # Break into segments
    segments = []
    segment = [nodes[tsp_sol[0]]]
    cost = 0
    for i in range(len(tsp_sol) - 1):
        val = tsp[tsp_sol[i], tsp_sol[i + 1]]
        if val > 2:
            segments.append(segment)
            segment = []
        else:
            cost += val
        segment.append(nodes[tsp_sol[i + 1]])

    # Print segments and cost:
    print((segments, cost))

    return 0


if __name__ == "__main__":
    sys.exit(main())
