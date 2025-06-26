# Ising model Trotterization as interpreted by (OpenAI GPT) Elara
# You likely want to specify environment variable QRACK_MAX_PAGING_QB=28

import math
import sys
import time

from collections import Counter

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap

from pyqrack import QrackAceBackend
from qiskit.providers.qrack import AceQasmSimulator


def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if is_transpose else (row_len, col_len)


def trotter_step(circ, qubits, lattice_shape, J, h, dt):
    n_rows, n_cols = lattice_shape

    # First half of transverse field term
    for q in qubits:
        circ.rx(h * dt / 2, q)

    # Layered RZZ interactions (simulate 2D nearest-neighbor coupling)
    def add_rzz_pairs(pairs):
        for q1, q2 in pairs:
            circ.append(RZZGate(2 * J * dt), [q1, q2])

    # Layer 1: horizontal pairs (even rows)
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(0, n_cols - 1, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 2: horizontal pairs (odd rows)
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(1, n_cols - 1, 2)
    ]
    add_rzz_pairs(horiz_pairs)
    
    # horizontal wrap
    wrap_pairs = [(r*n_cols + (n_cols-1), r*n_cols) for r in range(n_rows)]
    add_rzz_pairs(wrap_pairs)

    # Layer 3: vertical pairs (even columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(0, n_rows - 1, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Layer 4: vertical pairs (odd columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(1, n_rows - 1, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # vertical wrap
    wrap_pairs = [((n_rows-1)*n_cols + c, c) for c in range(n_cols)]
    add_rzz_pairs(wrap_pairs)

    # Second half of transverse field term
    for q in qubits:
        circ.rx(h * dt / 2, q)

    return circ


def main():
    n_qubits = 56
    depth = 20
    shots = 32768
    long_range_columns = 4
    long_range_rows = 4
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        shots = int(sys.argv[3])
    else:
        shots = min(32768, 1 << (n_qubits + 2))
    if len(sys.argv) > 4:
        is_transpose = sys.argv[4] not in ["0", "False"]
    if len(sys.argv) > 5:
        long_range_columns = int(sys.argv[5])
    if len(sys.argv) > 6:
        long_range_rows = int(sys.argv[6])

    n_rows, n_cols = factor_width(n_qubits, is_transpose)
    J, h, dt = -1.0, 2.0, 0.25
    theta = 2 * math.pi / 9

    qc = QuantumCircuit(n_qubits)

    for q in range(n_qubits):
        qc.ry(theta, q)

    for _ in range(depth):
        trotter_step(qc, list(range(n_qubits)), (n_rows, n_cols), J, h, dt)

    experiment = QrackAceBackend(
        n_qubits, long_range_columns=long_range_columns, long_range_rows=long_range_rows
    )
    # We've achieved the dream: load balancing between discrete and integrated accelerators!
    # for sim_id in range(2, len(experiment.sim), 3):
    #     experiment.sim[sim_id].set_device(0)

    step = transpile(
        step,
        optimization_level=3,
        backend=AceQasmSimulator(
            n_qubits=n_qubits,
            long_range_columns=long_range_columns,
            long_range_rows=long_range_rows,
        ),
    )

    start = time.perf_counter()
    experiment.run_qiskit_circuit(qc)
    experiment_samples = experiment.measure_shots(list(range(n_qubits)), shots)
    seconds = time.perf_counter() - start

    magnetization = 0
    for sample in experiment_samples:
        for _ in range(n_qubits):
            magnetization += -1 if (sample & 1) else 1
            sample >>= 1
    magnetization /= shots * n_qubits

    print(
        {
            "width": n_qubits,
            "depth": depth,
            "magnetization": magnetization,
            "seconds": seconds,
        }
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
