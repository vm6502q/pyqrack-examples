# Simon's Algorithm
# Hidden subgroup problem: find s such that f(x) = f(x XOR s) for all x
# If s = 0, f is one-to-one; otherwise f is two-to-one
# By (Anthropic) Claude

import random
import sys
import numpy as np
from pyqrack import QrackStabilizer
from qiskit import QuantumCircuit, transpile


def build_simon_oracle(circ, input_qubits, output_qubits, s):
    """
    Build Simon's oracle: f(x) = f(x XOR s)
    Implementation: copy input to output, then XOR with s-masked input.
    This is Clifford (CNOT only) when s is fixed.
    """
    n = len(input_qubits)

    # Copy input to output
    for i in range(n):
        circ.cx(input_qubits[i], output_qubits[i])

    if s == 0:
        return

    # Find first set bit of s to use as control
    pivot = -1
    for i in range(n):
        if (s >> i) & 1:
            pivot = i
            break

    # XOR output with s whenever input[pivot] = 1
    for i in range(n):
        if (s >> i) & 1:
            circ.cx(input_qubits[pivot], output_qubits[i])


def simon_circuit(n, s):
    """Build full Simon's algorithm circuit."""
    num_qubits = 2 * n
    input_qubits = list(range(n))
    output_qubits = list(range(n, 2 * n))

    circ = QuantumCircuit(num_qubits)

    # Hadamard on input register
    for q in input_qubits:
        circ.h(q)

    # Oracle
    build_simon_oracle(circ, input_qubits, output_qubits, s)

    # Hadamard on input register again
    for q in input_qubits:
        circ.h(q)

    return circ, input_qubits


def solve_simon(equations, n):
    """
    Gaussian elimination over GF(2) to find s from collected equations.
    Each equation y satisfies y · s = 0 (mod 2).
    """
    # Build matrix
    matrix = []
    for y in equations:
        row = [(y >> i) & 1 for i in range(n)]
        matrix.append(row)

    matrix = np.array(matrix, dtype=np.int32)
    pivot_cols = []
    row_idx = 0

    for col in range(n):
        # Find pivot
        found = -1
        for r in range(row_idx, len(matrix)):
            if matrix[r, col] == 1:
                found = r
                break
        if found == -1:
            continue
        matrix[[row_idx, found]] = matrix[[found, row_idx]]
        for r in range(len(matrix)):
            if r != row_idx and matrix[r, col] == 1:
                matrix[r] = (matrix[r] + matrix[row_idx]) % 2
        pivot_cols.append(col)
        row_idx += 1

    # Free variables give candidate s
    all_cols = set(range(n))
    free_cols = list(all_cols - set(pivot_cols))

    if not free_cols:
        return 0  # s = 0, one-to-one function

    # Set first free variable to 1, solve for pivot variables
    s_bits = [0] * n
    s_bits[free_cols[0]] = 1

    for i, pc in enumerate(pivot_cols):
        if i < len(matrix):
            val = 0
            for fc in free_cols:
                val ^= matrix[i, fc] * s_bits[fc]
            s_bits[pc] = val % 2

    s = sum(b << i for i, b in enumerate(s_bits))
    return s


def run_simon(n, s=None):
    if s is None:
        s = random.randint(1, (1 << n) - 1)

    circ, input_qubits = simon_circuit(n, s)

    # Check if circuit is Clifford
    basis_gates_clifford = ["h", "cx", "x", "y", "z", "s", "sdg"]
    basis_gates_universal = ["rz", "h", "x", "y", "z", "sx", "sxdg", "s", "sdg", "t", "tdg", "cx", "cy", "cz", "swap", "iswap"]

    transpiled = transpile(circ, optimization_level=3, basis_gates=basis_gates_universal)
    ops = transpiled.count_ops()
    is_clifford = all(g in basis_gates_clifford for g in ops)

    print(f"n={n}, s={s}, circuit ops: {dict(ops)}, is_clifford={is_clifford}")

    # Collect n-1 linearly independent equations
    equations = []
    attempts = 0

    while len(equations) < n - 1 and attempts < 10 * n:
        attempts += 1
        sim = QrackStabilizer(2 * n)
        sim.run_qiskit_circuit(transpiled, shots=0)
        result = sum(sim.m(q) << i for i, q in enumerate(input_qubits))

        # Check linear independence via GF(2)
        candidate = equations + [result]
        mat = np.array([[(r >> i) & 1 for i in range(n)] for r in candidate], dtype=np.int32)
        rank = np.linalg.matrix_rank(mat)
        if rank == len(candidate):
            equations.append(result)

    s_found = solve_simon(equations, n)

    correct = s_found == s
    print(f"  s_found={s_found}, correct={correct}, equations collected={len(equations)}")
    return correct


def main():
    n = 16
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    trials = 10
    correct = sum(run_simon(n, s=None) for _ in range(trials))
    print(f"\n{correct}/{trials} correct for n={n} Simon's algorithm")

    return 0


if __name__ == "__main__":
    sys.exit(main())
