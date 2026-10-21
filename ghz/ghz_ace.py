# ACE GHZ state preparation simulation

import math
import sys
import time

from collections import Counter

from pyqrack import QrackAceBackend


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (row_len, col_len)


def bench_qrack(n, lrc, lrr):
    shots = 8192
    row_len, col_len = factor_width(n)

    start = time.perf_counter()

    sim = QrackAceBackend(n, long_range_columns=lrc, long_range_rows=lrr)

    sim.h(0)
    for r in range(col_len):
        rin = r * row_len
        if r > 0:
            sim.mcx([rin - row_len], rin)
        for c in range(row_len - 1):
            sim.mcx([rin + c], rin + c + 1)

    ace_counts = dict(Counter(sim.measure_shots(list(range(n)), shots)))
    
    return {
        "n": n,
        "seconds": time.perf_counter() - start,
        "shots": shots,
        "counts": ace_counts
    }


def main():
    if len(sys.argv) < 2:
        raise RuntimeError(
            "Usage: python3 nn_qab.py [width] "
            "[long_range_columns=4] [long_range_rows=4] "
        )

    width = int(sys.argv[1])
    lrc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    lrr = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    result = bench_qrack(width, lrc, lrr)
    for k, v in result.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
