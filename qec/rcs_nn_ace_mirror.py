# Orbifolded random circuit sampling
# How good are Google's own "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)
# (This is actually a different "elision" concept, but allow that it works.)

import math
import random
import sys
import time

from pyqrack import QrackAceBackend


def factor_width(width, reverse=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")
    row_len = width // col_len

    return (col_len, row_len) if reverse else (row_len, col_len)


def bench_qrack(width, depth, reverse):
    # This is a "nearest-neighbor" coupler random circuit.
    start = time.perf_counter()
    experiment = QrackAceBackend(width)

    lcv_range = range(width)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = (
        experiment.cx,
        experiment.cy,
        experiment.cz,
        experiment.acx,
        experiment.acy,
        experiment.acz,
    )

    row_len, col_len = factor_width(width, reverse)

    single = []
    double = []

    for _ in range(depth):
        # Single-qubit gates
        single_layer = []
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            experiment.u(th, ph, lm, i)
            single_layer.append((th, ph, lm, i))
        single.append(single_layer)

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        double_layer = []
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

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                g(b1, b2)
                double_layer.append((g, b1, b2))
        double.append(double_layer)

    # Mirror
    for single_layer, double_layer in zip(reversed(single), reversed(double)):
        for g in reversed(double_layer):
            g[0](g[1], g[2])
        for g in reversed(single_layer):
            experiment.u(-g[0], -g[2], -g[1], g[3])

    # Terminal measurement
    shots = 100
    experiment_samples = experiment.measure_shots(list(range(width)), shots)
    seconds = time.perf_counter() - start

    mirror_fidelity = 0
    hamming_weight = 0
    for sample in experiment_samples:
        success = True
        for _ in range(width):
            if sample & 1:
                success = False
                hamming_weight += 1
            sample >>= 1
        if success:
            mirror_fidelity += 1
    mirror_fidelity /= shots
    hamming_weight /= shots

    return seconds, mirror_fidelity, hamming_weight


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 rcs_nn_ace_mirror.py [width] [depth] [reverse row/column]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    reverse = False
    if len(sys.argv) > 3:
        reverse = sys.argv[3] not in ["0", "False"]

    # Run the benchmarks
    seconds, mirror_fidelity, hamming_weight = bench_qrack(width, depth, reverse)

    # Print the results
    print(
        {
            "width": width,
            "depth": depth,
            "seconds": seconds,
            "mirror_fidelity": mirror_fidelity,
            "hamming_weight": hamming_weight,
        }
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
