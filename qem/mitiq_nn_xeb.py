# See "Error mitigation increases the effective quantum volume of quantum computers," https://arxiv.org/abs/2203.05489
#
# Mitiq is under the GPL 3.0.
# Hence, this example, as the entire work-in-itself, must be considered to be under GPL 3.0.
# See https://www.gnu.org/licenses/gpl-3.0.txt for details.

import math
import random
import statistics
import sys
import time

import numpy as np

from collections import Counter

from pyqrack import QrackSimulator, QrackAceBackend

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.qrack import AceQasmSimulator

from mitiq import zne
from mitiq.zne.scaling.folding import fold_gates_at_random
from mitiq.zne.inference import RichardsonFactory


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (row_len, col_len)


def cx(sim, q1, q2):
    sim.cx(q1, q2)


def cy(sim, q1, q2):
    sim.cy(q1, q2)


def cz(sim, q1, q2):
    sim.cz(q1, q2)


def acx(sim, q1, q2):
    sim.x(q1)
    sim.cx(q1, q2)
    sim.x(q1)


def acy(sim, q1, q2):
    sim.x(q1)
    sim.cy(q1, q2)
    sim.x(q1)


def acz(sim, q1, q2):
    sim.x(q1)
    sim.cz(q1, q2)
    sim.x(q1)


def swap(sim, q1, q2):
    sim.swap(q1, q2)


def iswap(sim, q1, q2):
    sim.iswap(q1, q2)


def iiswap(sim, q1, q2):
    sim.iswap(q1, q2)
    sim.iswap(q1, q2)
    sim.iswap(q1, q2)


def pswap(sim, q1, q2):
    sim.cz(q1, q2)
    sim.swap(q1, q2)


def mswap(sim, q1, q2):
    sim.swap(q1, q2)
    sim.cz(q1, q2)


def nswap(sim, q1, q2):
    sim.cz(q1, q2)
    sim.swap(q1, q2)
    sim.cz(q1, q2)


def calc_stats(ideal_probs, counts, shots):
    n_pow = len(ideal_probs)
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    hog_prob = 0
    for b in range(n_pow):
        ideal = ideal_probs[b]
        patch = (counts.get(b, 0) / shots)

        ideal_centered = ideal - u_u
        denom += ideal_centered * ideal_centered
        numer += ideal_centered * (patch - u_u)

        if ideal > threshold:
            hog_prob += patch

    xeb = numer / denom
    return xeb, hog_prob


def random_circuit(width, depth, lrc, lrr):
    # This is a "nearest-neighbor" coupler random circuit.
    lcv_range = range(width)
    all_bits = list(lcv_range)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    results = []

    qc = QuantumCircuit(width)
    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
            # Keep it Haar-random towards the poles:
            th = math.asin(th / math.pi)
            qc.u(th, ph, lm, i)

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
                temp_col = temp_col + (1 if (gate & 1) else 0)

                if temp_row < 0:
                    continue
                if temp_col < 0:
                    continue
                if temp_row >= row_len:
                    continue
                if temp_col >= col_len:
                    continue

                b1 = col * row_len + row
                b2 = temp_col * row_len + temp_row

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                g(qc, b1, b2)

    sim = AceQasmSimulator(n_qubits=width, long_range_columns=lrc, long_range_rows=lrr)
    qc = transpile(qc, backend=sim, optimization_level=2)

    return qc


# To take a [-1.0, 1.0] bounded interval to an unbounded one for OLS or Richardson extrapolation:
# Precise symmetric tanh/atanh limits provided by (Anthropic) Claude

_ATANH_CEIL = 18.714973875118524     # 0x1.2b708872320e2p+4
_ATANH_FLOOR = -_ATANH_CEIL          # exact: atanh is an exact odd function
                                     # under IEEE754 negation, verified directly

def atanh(z):
    if z >= 1.0:
        return _ATANH_CEIL
    if z <= -1.0:
        return _ATANH_FLOOR
    return max(_ATANH_FLOOR, min(_ATANH_CEIL, 0.5 * math.log((1 + z) / (1 - z))))


def tanh(x):
    if x >= _ATANH_CEIL:
        return 1.0
    if x <= _ATANH_FLOOR:
        return -1.0
    return math.tanh(x)


def execute(qc, n_qubits, shot_count, lrc, lrr, ideal_probs):
    qcm = qc.copy()
    qcm.measure_all()

    # -----------------------------------------------------------------------
    # Method: QrackAceBackend
    # -----------------------------------------------------------------------

    sim = AceQasmSimulator(n_qubits=n_qubits, long_range_columns=lrc, long_range_rows=lrr)
    qcm = qc.copy()
    qcm.measure_all()
    ace_str_counts = dict(sim.run(qcm, shots=shot_count).result().get_counts())
    ace_counts = {}
    for s, count in ace_str_counts.items():
        ace_counts[int(s, 2)] = count

    xeb_ace, hog_ace = calc_stats(ideal_probs, ace_counts, shot_count)

    return atanh(xeb_ace)


def main():
    if len(sys.argv) < 3:
        raise RuntimeError("Usage: python3 mitiq_nn_hamming_weight.py [width] [depth] [long_range_columns=4] [long_range_rows=4] [shots=4096]")

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    lrc = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    lrr = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    shots = int(sys.argv[5]) if len(sys.argv) > 5 else 4096

    qc = random_circuit(width, depth, lrc, lrr)

    # -----------------------------------------------------------------------
    # Ideal ground truth via QrackSimulator
    # -----------------------------------------------------------------------
    sim_ideal = QrackSimulator(width)
    sim_ideal.run_qiskit_circuit(qc, shots=0)
    ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
    del sim_ideal

    factory = RichardsonFactory(scale_factors=[1, 3, 5])
    ex = lambda circ: execute(qc, width, shots, lrc, lrr, ideal_probs)
    def scale(circ, scale_factor):
        return fold_gates_at_random(circ, scale_factor=scale_factor, fidelities={"single": 1.0, "double": 0.975})

    start = time.perf_counter()
    xeb = tanh(
        zne.execute_with_zne(qc, ex, scale_noise=scale, factory=factory)
    )
    end = time.perf_counter()

    print({"width": width, "depth": depth, "seconds": (end - start), "mitigated_xeb": float(xeb)})

    return 0


if __name__ == "__main__":
    sys.exit(main())
