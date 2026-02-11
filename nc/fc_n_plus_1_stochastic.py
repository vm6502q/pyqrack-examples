# How good are Google's own "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)
# You probably want to set environment variable QRACK_MAX_PAGING_QB=-1.

import math
import random
import sys
import time

from qiskit import QuantumCircuit

from pyqrack import QrackSimulator, Pauli


def bench_qrack(n_qubits, use_rz):
    # This is a "fully-connected" coupler random circuit.
    lcv_range = range(n_qubits)
    all_bits = list(lcv_range)

    rz_count = (n_qubits + 1) << 1
    rz_opportunities = n_qubits * n_qubits * 3
    rz_positions = []
    while len(rz_positions) < rz_count:
        rz_position = random.randint(0, rz_opportunities - 1)
        if rz_position in rz_positions:
            continue
        rz_positions.append(rz_position)

    start = time.perf_counter()

    experiment = QrackSimulator(
        n_qubits,
        isTensorNetwork=False,
        isSchmidtDecompose=False,
        isStabilizerHybrid=True,
    )
    # Round to nearest Clifford circuit
    experiment.set_use_exact_near_clifford(False)

    qc = QuantumCircuit(n_qubits)
    gate_count = 0
    for d in range(n_qubits):
        # Single-qubit gates
        for i in lcv_range:
            for _ in range(3):
                experiment.h(i)
                # s_count = random.randint(0, 3)
                s_count = random.randint(0, 7)
                if s_count & 1:
                    experiment.z(i)
                if s_count & 2:
                    experiment.s(i)
                if gate_count in rz_positions:
                    if use_rz:
                      qc.rz(random.uniform(0, math.pi / 2), i)
                    elif s_count & 4:
                        qc.t(i)
                    else:
                        qc.tdg(i)
                gate_count = gate_count + 1

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            experiment.mcx([c], t)

        clone = experiment.clone()
        clone.m_all()

        print(
            {
                "qubits": n_qubits,
                "depth": d + 1,
                "seconds": time.perf_counter() - start,
            }
        )


def main():
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python3 fc_2n_plus_2.py [width] [use_rz]"
        )

    n_qubits = n_qubits = int(sys.argv[1])
    use_rz = False
    if len(sys.argv) > 2:
        use_rz = sys.argv[2] not in ["False", "0"]

    # Run the benchmarks
    bench_qrack(n_qubits, use_rz)


    return 0


if __name__ == "__main__":
    sys.exit(main())
