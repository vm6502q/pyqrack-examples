# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys

from collections import Counter

from pyqrack import QrackStabilizer

from qiskit import QuantumCircuit


def bench_qrack(n_qubits, hamming_n):
    # This is a "fully-connected" coupler random circuit.
    shots = hamming_n << 2
    lcv_range = range(n_qubits)
    all_bits = list(lcv_range)

    rz_count = n_qubits + 1
    rz_opportunities = n_qubits * n_qubits * 3
    rz_positions = []
    while len(rz_positions) < rz_count:
        rz_position = random.randint(0, rz_opportunities - 1)
        if rz_position in rz_positions:
            continue
        rz_positions.append(rz_position)

    qc = QuantumCircuit(n_qubits)
    gate_count = 0
    for d in range(n_qubits):
        # Single-qubit gates
        for i in lcv_range:
            # Single-qubit gates
            for _ in range(3):
                qc.h(i)
                s_count = random.randint(0, 3)
                if s_count & 1:
                    qc.z(i)
                if s_count & 2:
                    qc.s(i)
                if gate_count in rz_positions:
                    angle = random.uniform(0, math.pi / 2)
                    qc.rz(angle, i)
                gate_count = gate_count + 1

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            qc.cx(c, t)

        # Round to nearest Clifford circuit
        exp_shots = []
        for i in range(shots):
            experiment = QrackStabilizer(n_qubits)
            experiment.run_qiskit_circuit(qc, shots=0)
            exp_shots.append(experiment.m_all());
        experiment_counts = dict(Counter(exp_shots))

        aer_qc = qc.copy()
        aer_qc.save_statevector()
        control = AerSimulator(method="statevector")
        job = control.run(aer_qc)
        control_probs = Statevector(job.result().get_statevector()).probabilities()

        print(calc_stats(control_probs, experiment_counts, shots, d + 1, hamming_n))


def main():
    if len(sys.argv) < 2:
        raise RuntimeError(
            "Usage: python3 qs_fc_2n_plus_2_qiskit_validation.py [width] [hamming_n]"
        )

    n_qubits = 56
    hamming_n = 2048
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        hamming_n = int(sys.argv[2])

    # Run the benchmarks
    bench_qrack(n_qubits, hamming_n)

    return 0


if __name__ == "__main__":
    sys.exit(main())
