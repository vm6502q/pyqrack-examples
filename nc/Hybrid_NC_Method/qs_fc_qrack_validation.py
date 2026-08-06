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
    aux = QrackSimulator(n_qubits, is_near_clifford_tableau_writer=True)
    gate_count = 0
    magic_count = 0
    for d in range(n_qubits):
        # Single-qubit gates
        for i in lcv_range:
            # Single-qubit gates
            for _ in range(2):
                qc.h(i)
                control.h(i)
                aux.h(i)
                s_count = random.randint(0, 3)
                if s_count & 1:
                    qc.z(i)
                    control.z(i)
                    aux.z(i)
                if s_count & 2:
                    qc.s(i)
                    control.s(i)
                    aux.s(i)
                if gate_count in rz_positions:
                    angle = random.uniform(0, math.pi / 2)
                    qc.rz(angle, i)
                    control.r(Pauli.PauliZ, angle, i)
                    aux.r(Pauli.PauliZ, angle, i)
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
            aux.mcx([c], t)
        
        exp_shots = []
        probs = {}
        i = 0
        while i < shots:
            experiment = QrackStabilizer(n_qubits)
            experiment.run_qiskit_circuit(qc, shots=0)
            s = experiment.m_all()
            if s in exp_shots:
                continue
            exp_shots.append(s)
            probs[s] = aux.prob_perm(all_bits, [(s >> j) & 1 for j in range(n_qubits)])
            i += 1
        experiment_probs = route_heavy_light(probs, mean)

        control_probs = control.out_probs()

        print(calc_stats(control_probs, experiment_probs, shots, d + 1, magic_count))


def route_heavy_light(prob_dict, u_u):
    """
    Split a {outcome: p} dict into (heavy, light) dicts centered on u_u.
    heavy: outcomes where p > u_u, values normalised to sum 1.
    light: outcomes where p < u_u, values (stored positive) normalised to sum 1.
    """
    heavy_raw = {}
    light_raw = {}
    for outcome, p in prob_dict.items():
        c = p - u_u
        if c > 0:
            heavy_raw[outcome] = c
        elif c < 0:
            light_raw[outcome] = -c          # store as positive

    s_h = sum(heavy_raw.values())
    s_l = sum(light_raw.values())
    heavy = {k: v / s_h for k, v in heavy_raw.items()} if s_h > 0 else {}
    light = {k: v / s_l for k, v in light_raw.items()} if s_l > 0 else {}
    return heavy, light


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
    probs_heavy, probs_light = probs
    for i in range(n_pow):
        exp = 0.5 * probs_heavy.get(i, 0)  +  0.5 * u_u * (1 - probs_light.get(i, 0))
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
