# Quimb-based sieve with Qrack's "automatic circuit elision" ACE
# by Daniel Strano (lead author and maintainer of Qrack)

# You likely want to set the following environment variables
# (and it's not possible to set them in a script-based way
# in a simple single Python script):

# QRACK_MAX_PAGING_QB / QRACK_MAX_CPU_QB - Sets a cap on largest entangled qubit subsystem size. For large circuits, set to half the total qubit count (or maybe 1/3, if 1/2 is too large, etc.).
# QRACK_QTENSORNETWORK_THRESHOLD_QB - Controls light-cone optimizations for memory and accuracy (which are slow, but only by a factor of ~n for n qubits). For fast (less-accurate) simulation, set much higher than the count of total qubits.
# QRACK_QUNIT_SEPARABILITY_THRESHOLD - Optionally, this rounds nearly-separable single qubits to total separable, in flight. That might help or hurt.
#     (A "golden" value for large, hard circuits seems to be (1 - 1 / sqrt(2)) / 2, approximately 0.1464466.)

import operator
import sys
from collections import Counter
from itertools import combinations

from pyqrack import QrackSimulator
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_quimb import quimb_circuit


def int_to_bitstring(integer, length, reverse):
    s = bin(integer)[2:].zfill(length)
    return s[::-1] if reverse else s


def run_qasm(file_in):
    qc = QuantumCircuit.from_qasm_file(file_in)
    basis_gates = QrackSimulator.get_qiskit_basis_gates()
    qc = transpile(qc, basis_gates=basis_gates)
    n_qubits = qc.num_qubits
    shots = n_qubits ** 3

    quimb_c = quimb_circuit(qc)

    sim = QrackSimulator(n_qubits, isTensorNetwork=False)
    sim.run_qiskit_circuit(qc, shots=0)

    highest_prob = sim.highest_prob_perm()
    print(f"ACE highest-probability dimension: {highest_prob}")

    experiment_counts = dict(Counter(sim.measure_shots(list(range(n_qubits)), shots)))
    # make sure the ACE argmax is present and dominant
    experiment_counts[highest_prob] = shots
    experiment_counts = sorted(experiment_counts.items(),
                               key=operator.itemgetter(1),
                               reverse=True)

    best_key = highest_prob
    best_prob = 0.0
    best_amp = 0.0 + 0.0j
    tot_prob = 0.0

    visited = set()

    def evaluate_key(key):
        """Evaluate amplitude / prob of a basis state if not yet visited."""
        nonlocal best_key, best_prob, best_amp, tot_prob
        if key in visited:
            return False
        visited.add(key)

        bitstr = int_to_bitstring(key, n_qubits, True)
        amp = complex(quimb_c.amplitude(bitstr, backend="jax"))
        prob = float((abs(amp) ** 2).real)

        tot_prob += prob
        if prob > best_prob:
            print(f"{key}: {prob}, {amp}")
            best_prob = prob
            best_amp = amp
            best_key = key
            return True

        return False

    # maximum Hamming radius around each candidate to explore
    MAX_RADIUS = 2  # 0 => just the key, 1 => key + all single-bit flips, etc.

    done = False
    improved = True

    for key, _cnt in experiment_counts:
        evaluate_key(key)
        if (1.0 - tot_prob) < best_prob:
            done = True
            break

    # Generate all neighbors of 'key' within Hamming distance MAX_RADIUS
    while improved and not done:
        improved = False
        for r in range(0, MAX_RADIUS + 1):
            for idxs in combinations(range(n_qubits), r):
                neighbor = key
                for i in idxs:
                    neighbor ^= (1 << i)  # flip bit i

                improved = evaluate_key(neighbor)

                if (1.0 - tot_prob) < best_prob:
                    done = True
                    break

                if improved:
                    break

            if done or improved:
                break

    rtl = int_to_bitstring(best_key, n_qubits, False)
    ltr = int_to_bitstring(best_key, n_qubits, True)
    print(f"Key integer: {best_key}")
    print(f"Right-to-left, least-to-most significant: {rtl}")
    print(f"Left-to-right, least-to-most significant: {ltr}")
    print(f"Probability: {best_prob}")
    print(f"Amplitude: {best_amp}")


def main():
    file_in = "qpe.qasm"
    if len(sys.argv) > 1:
        file_in = str(sys.argv[1])
    run_qasm(file_in)
    return 0


if __name__ == "__main__":
    sys.exit(main())
