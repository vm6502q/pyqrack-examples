# Orbifolded random circuit sampling
# How good are Google's own "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)
# (This is actually a different "elision" concept, but allow that it works.)

import math
import random
import sys
import time

from pyqrack import QrackAceBackend


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")
    row_len = width // col_len

    return (row_len, col_len)


def bench_qrack(width, depth, lrc=4, lrr=4, sdrp=0.0):
    # This is a "nearest-neighbor" coupler random circuit.
    start = time.perf_counter()
    experiment = QrackAceBackend(width, long_range_columns=lrc, long_range_rows=lrr)
    experiment.set_sdrp(sdrp)

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

    row_len, col_len = factor_width(width)

    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            experiment.u(i, th, ph, lm)

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
                g(b1, b2)

    # Terminal measurement
    sample = experiment.m_all()
    seconds = time.perf_counter() - start

    return seconds, sample


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 nn_ace_time.py [width] [depth] [long_range_columns=4] [long_range_rows=4] [sdrp=0.1464466]")
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    lrc = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    lrr = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    sdrp  = float(sys.argv[5]) if len(sys.argv) > 5 else ((1 - 1 / math.sqrt(2)) / 2)
    result = bench_qrack(width, depth, lrc, lrr, sdrp)

    return 0


if __name__ == "__main__":
    sys.exit(main())
