# Quimb-based sieve with Qrack's "automatic circuit elision" ACE
# by Daniel Strano (lead author and maintainer of Qrack)

# You likely want to set the following environment variables
# (and it's not possible to set them in a script-based way
# in a simple single Python script):

# QRACK_MAX_PAGING_QB / QRACK_MAX_CPU_QB - Sets a cap on largest entangled qubit subsystem size. For large circuits, set to half the total qubit count (or maybe 1/3, if 1/2 is too large, etc.).
# QRACK_QTENSORNETWORK_THRESHOLD_QB - Controls light-cone optimizations for memory and accuracy (which are slow, but only by a factor of ~n for n qubits). For fast (less-accurate) simulation, set much higher than the count of total qubits.
# QRACK_QUNIT_SEPARABILITY_THRESHOLD - Optionally, this rounds nearly-separable single qubits to total separable, in flight. That might help or hurt.
#     (A "golden" value for large, hard circuits seems to be (1 - 1 / sqrt(2)) / 2, approximately 0.1464466.)

import json
import operator
import sys
import time

from collections import Counter

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile

from qiskit_quimb import quimb_circuit


def int_to_bitstring(integer, length, reverse):
    return (bin(integer)[2:].zfill(length))[::-1] if reverse else (bin(integer)[2:].zfill(length))


def run_qasm(file_in):
    start = time.perf_counter()
    # Well short of 2-hour challenge time limit:
    max_time = 60 * 60 * 1.5

    qc = QuantumCircuit.from_qasm_file(file_in)
    basis_gates = QrackSimulator.get_qiskit_basis_gates()
    qc = transpile(qc, basis_gates=basis_gates)
    n_qubits = qc.num_qubits
    shots = n_qubits ** 3

    quimb_c = quimb_circuit(qc)

    # print(quimb_c.psi)
    
    sim = QrackSimulator(n_qubits, isTensorNetwork=False)
    sim.run_qiskit_circuit(qc, shots=0)
    # print("ACE fidelity estimate: " + str(sim.get_unitary_fidelity()))
    highest_prob = sim.highest_prob_perm()
    print(f"ACE highest-probability dimension: {highest_prob}")
    experiment_counts = dict(Counter(sim.measure_shots(list(range(n_qubits)), shots)))
    experiment_counts[highest_prob] = shots
    experiment_counts = sorted(experiment_counts.items(), key=operator.itemgetter(1), reverse=True)
    experiment = None

    u_u =  1 / (1 << n_qubits)
    best_key = highest_prob
    best_prob = 0
    best_amp = 0
    for count_tuple in experiment_counts:
        key = count_tuple[0]
        amp = complex(quimb_c.amplitude(int_to_bitstring(key, n_qubits, True)), backend="jax"))
        prob = float((abs(amp) ** 2).real)
        if prob > best_prob:
            print(f"{key}: {prob}, {amp}")
            best_prob = prob
            best_amp = amp
            best_key = key
        current = time.perf_counter()
        if (current - start) > max_time:
            # We're running out of time: cut line and return the best we have.
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
