# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys

from pyqrack import QrackSimulator, Pauli

from qiskit import QuantumCircuit, transpile
from qiskit.providers.qrack import QStabilizerQasmSimulator


def bench_qrack(n_qubits, shots):
    # This is a "fully-connected" coupler random circuit.
    lcv_range = range(n_qubits)
    all_bits = list(lcv_range)
    mean = 1.0 / (1 << n_qubits)

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
                s_count = random.randint(0, 7)
                if s_count & 1:
                    qc.z(i)
                    control.z(i)
                if s_count & 2:
                    qc.s(i)
                    control.s(i)
                if s_count & 4:
                    qc.t(i)
                    control.t(i)
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
        
        experiment = QStabilizerQasmSimulator(n_qubits=n_qubits)
        aer_qc = qc.copy()
        aer_qc.measure_all()
        raw_counts = experiment.run(aer_qc, shots=shots).result().get_counts()
        counts = {int(k, 2): v for k, v in raw_counts.items()}

        control_probs = control.out_probs()

        print(calc_stats(control_probs, counts, shots, d + 1, magic_count, n_qubits))

def calc_stats(ideal_probs, counts, shots, depth, magic, n):
    n_pow = len(ideal_probs)
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    hog_prob = 0
    for b in range(n_pow):
        ideal = ideal_probs[b]
        exp = (counts.get(b, 0) / shots)

        ideal_centered = ideal - u_u
        denom += ideal_centered * ideal_centered
        numer += ideal_centered * (exp - u_u)

        if ideal > threshold:
            hog_prob += exp

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
    shots = 1024
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        shots = int(sys.argv[2])

    # Run the benchmarks
    bench_qrack(n_qubits, shots)

    return 0


if __name__ == "__main__":
    sys.exit(main())
