# Nearest-neighbor RCS: Automatic circuit elision
#
# By Dan Strano and (Anthropic) Claude.

import math
import random
import sys
import time

from collections import Counter

from pyqrack import QrackSimulator
from qiskit.providers.qrack.backends import AceQasmSimulator
from qiskit import QuantumCircuit, transpile


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth):
    lcv_range = range(width)
    all_bits  = list(lcv_range)
    n_pow     = 1 << width
    shots     = 1 << min(10, width + 2)

    # -----------------------------------------------------------------------
    # Build circuit in Qiskit
    # -----------------------------------------------------------------------
    t_circ = time.perf_counter()
    qc = QuantumCircuit(width, width)

    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
            # Keep it Haar-random towards the poles:
            th = math.asin(th / math.pi)
            qc.u(th, ph, lm, i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            qc.cx(c, t)

    # -----------------------------------------------------------------------
    # Method: QrackAceBackend
    # -----------------------------------------------------------------------
    sim = AceQasmSimulator()
    qc = transpile(qc, backend=sim, optimization_level=3)
    qc = qc & qc.inverse()
    

    t_trans = time.perf_counter()
    print(f"transpile_seconds: {t_trans - t_circ:.4f}")

    logical_to_physical = qc.layout.final_index_layout()

    for logical_idx in range(width):
        physical_qubit = logical_to_physical[logical_idx]
        # Measure the exact physical wire into its designated classical bit
        qc.measure(physical_qubit, logical_idx)

    ace_str_counts = dict(sim.run(qc, shots=shots).result().get_counts())

    t_ace = time.perf_counter()
    print(f"ace_seconds: {t_ace - t_trans:.4f}")

    ace_counts = {}
    for s, count in ace_str_counts.items():
        ace_counts[int(s, 2)] = count
    hamming = 0
    for s, count in ace_counts.items():
        hamming += s.bit_count() * count
    hamming /= shots

    return {
        "width":              width,
        "depth":              depth,
        "fidelity":           ace_counts.get(0, 0) / shots,
        "hamming_weight":     hamming,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError("Usage: python3 fc_qiskit_qab_54.py [width] [depth]")
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    result = bench_qrack(width, depth)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
