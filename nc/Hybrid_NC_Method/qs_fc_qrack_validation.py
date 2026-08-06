# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys

from collections import Counter

import numpy as np

from pyqrack import QrackSimulator, QrackStabilizer, Pauli

from qiskit import QuantumCircuit


def bench_qrack(n_qubits, magic, shots):
    # This is a "fully-connected" coupler random circuit.
    lcv_range = range(n_qubits)
    all_bits = list(lcv_range)
    mean = 1.0 / (1 << n_qubits)

    rz_count = magic
    rz_opportunities = n_qubits * n_qubits * 2
    rz_positions = []
    while len(rz_positions) < rz_count:
        rz_position = random.randint(0, rz_opportunities - 1)
        if rz_position in rz_positions:
            continue
        rz_positions.append(rz_position)

    qc = QuantumCircuit(n_qubits)
    control = QrackSimulator(n_qubits)
    gate_count = 0
    magic_count = 0
    for d in range(n_qubits):
        # Single-qubit gates
        for i in lcv_range:
            # Single-qubit gates
            for _ in range(2):
                qc.h(i)
                control.h(i)
                s_count = random.randint(0, 3)
                if s_count & 1:
                    qc.z(i)
                    control.z(i)
                if s_count & 2:
                    qc.s(i)
                    control.s(i)
                if gate_count in rz_positions:
                    angle = random.uniform(0, math.pi / 2)
                    qc.rz(angle, i)
                    control.r(Pauli.PauliZ, angle, i)
                    magic_count += 1
                gate_count = gate_count + 1

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            qc.cx(c, t)
            control.mcx([c], t)

        exp_shots = []
        exp_probs = {}
        sum_probs = 0.0
        i = 0
        while i < shots:
            experiment = QrackStabilizer(n_qubits)
            experiment.run_qiskit_circuit(qc, shots=0)
            s = experiment.m_all()
            if s in exp_shots:
                continue
            exp_shots.append(s)
            experiment = QrackSimulator(n_qubits, is_near_clifford_tableau_writer=True)
            experiment.run_qiskit_circuit(qc, shots=0)
            p = experiment.prob_perm(all_bits, [(s >> i) & 1 for i in range(n_qubits)])
            if p <= mean:
                continue
            i += 1
            exp_probs[s] = p
            sum_probs += p
        experiment_probs = { k: v / sum_probs for k, v in exp_probs.items() }

        control_probs = control.out_probs()

        print(calc_stats(control_probs, experiment_probs, shots, d + 1, magic_count))


def calc_stats(ideal_probs, probs, shots, depth, magic):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u = 1 / n_pow
    numer = 0
    denom = 0
    sum_hog_counts = 0
    experiment = [0] * n_pow
    for i in range(n_pow):
        exp = probs.get(i, 0)
        ideal = ideal_probs[i]
        count = exp * shots

        experiment[i] = count

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (exp - u_u)

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts += count

    hog_prob = sum_hog_counts / shots
    xeb = numer / denom

    return {
        "qubits": n,
        "depth": depth,
        "magic": magic,
        "xeb": float(xeb),
        "hog_prob": float(hog_prob)
    }


def main():
    n_qubits = 16
    magic = 17
    shots = 256
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        magic = int(sys.argv[2])
    else:
        magic = n_qubits + 1
    if len(sys.argv) > 3:
        shots = int(sys.argv[3])
    else:
        shots = n_qubits * n_qubits

    # Run the benchmarks
    bench_qrack(n_qubits, magic, shots)

    return 0


if __name__ == "__main__":
    sys.exit(main())
