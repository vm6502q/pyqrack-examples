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

    sim.h(n - 1)
    for r in range(col_len):
        rin = r * row_len
        if r > 0:
            sim.mcx([(n - 1) - (rin - row_len)], (n - 1) - rin)
        fwd = row_len >> 1
        bwd = row_len - fwd
        for c in range(fwd):
            t = rin + c + 1
            sim.mcx([(n - 1) - (t - 1)], (n - 1) - t)
        sim.mcx([(n - 1) - rin], (n - 1)- (row_len + rin - 1))
        for c in range(1, bwd - 1):
            t = row_len + rin - (c + 1)
            sim.mcx([(n - 1) - (t + 1)], (n - 1) - t)

    ace_counts = dict(Counter(sim.measure_shots(list(range(n)), shots)))

    m = (1 << n) - 1
    freq_0 = ace_counts.get(0, 0)
    freq_max = ace_counts.get(m, 0)
    infidelity = 0
    for k, v in ace_counts.items():
        bc = k.bit_count()
        if bc == 0 or bc == n:
            continue
        bc <<= 1
        if bc > n:
            bc = (n << 1) - bc
        infidelity = v * bc
    infidelity /= n * shots
    
    return {
        "n": n,
        "seconds": time.perf_counter() - start,
        "shots": shots,
        "freq_0": freq_0,
        "freq_max": freq_max,
        "hamming_infidelity": infidelity,
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
