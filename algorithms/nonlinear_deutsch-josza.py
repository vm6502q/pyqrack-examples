# Nonlinear Deutsch-Josza
# By (Anthropic) Claude

import math
import random
import sys

from pyqrack import QrackStabilizer
from qiskit import QuantumCircuit


def toffoli_clifford_t(circ, c1, c2, t):
    circ.h(t)
    circ.cx(c2, t)
    circ.tdg(t)
    circ.cx(c1, t)
    circ.t(t)
    circ.cx(c2, t)
    circ.tdg(t)
    circ.cx(c1, t)
    circ.t(c2)
    circ.t(t)
    circ.h(t)
    circ.cx(c1, c2)
    circ.t(c1)
    circ.tdg(c2)
    circ.cx(c1, c2)


def n_toffoli(circ, controls, target, ancilla):
    if len(controls) == 1:
        circ.cx(controls[0], target)
        return
    if len(controls) == 2:
        toffoli_clifford_t(circ, controls[0], controls[1], target)
        return
    mid = ancilla.pop()
    toffoli_clifford_t(circ, controls[0], controls[1], mid)
    n_toffoli(circ, [mid] + controls[2:], target, ancilla)
    toffoli_clifford_t(circ, controls[0], controls[1], mid)
    ancilla.append(mid)


def nonlinear_balanced_oracle(circ, input_qubits, output_qubit, ancilla):
    """
    Random balanced oracle via random truth table.
    Exactly half of 2^n input patterns map to 1.
    Each 1-row is implemented as an n-controlled X on output_qubit,
    with input qubits flipped before/after to select the specific pattern.
    """
    n = len(input_qubits)
    n_pow = 1 << n
    all_inputs = list(range(n_pow))
    random.shuffle(all_inputs)
    ones = set(all_inputs[:n_pow // 2])

    for x in ones:
        # Flip inputs to match this pattern
        flip = []
        for bit_idx, q in enumerate(input_qubits):
            if not ((x >> bit_idx) & 1):
                circ.x(q)
                flip.append(q)

        n_toffoli(circ, input_qubits, output_qubit, ancilla[:])

        # Uncompute flips
        for q in flip:
            circ.x(q)


def deutsch_jozsa(circ, input_qubits, output_qubit, ancilla, oracle_fn):
    # Initialize output qubit to |->
    circ.x(output_qubit)
    circ.h(output_qubit)

    # Initialize input qubits to |+>
    for q in input_qubits:
        circ.h(q)

    # Apply oracle
    oracle_fn(circ, input_qubits, output_qubit, ancilla)

    # Final Hadamard on input qubits
    for q in input_qubits:
        circ.h(q)


def run_dj(n_input):
    # Layout: input qubits, output qubit, ancilla qubits
    n_ancilla = max(0, n_input - 2)
    n_total = n_input + 1 + n_ancilla

    input_qubits = list(range(n_input))
    output_qubit = n_input
    ancilla = list(range(n_input + 1, n_total))

    circ = QuantumCircuit(n_total)

    nonlinear_balanced_oracle_fn = lambda c, iq, oq, anc: nonlinear_balanced_oracle(
        c, iq, oq, anc
    )

    deutsch_jozsa(circ, input_qubits, output_qubit, ancilla, nonlinear_balanced_oracle_fn)

    sim = QrackStabilizer(n_total)
    sim.run_qiskit_circuit(circ, shots=0)
    result = sim.measure_shots(input_qubits, 1)[0]

    # For balanced oracle, result should be nonzero
    verdict = "BALANCED (correct)" if result != 0 else "CONSTANT (wrong)"
    print({
        "n_input": n_input,
        "measurement": result,
        "verdict": verdict,
    })

    return result != 0


def main():
    n_input = 4
    if len(sys.argv) > 1:
        n_input = int(sys.argv[1])

    trials = 10
    correct = sum(run_dj(n_input) for _ in range(trials))
    print(f"\n{correct}/{trials} correct for n={n_input} nonlinear balanced oracle")

    return 0


if __name__ == "__main__":
    sys.exit(main())
