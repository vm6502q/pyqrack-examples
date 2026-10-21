# ACE GHZ state preparation simulation

import math
import sys
import time

import numpy as np

from collections import Counter

from pyqrack import QrackAceBackend


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (row_len, col_len)


def bench_qrack(n, lrc, lrr):
    shots = 16384
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
    freq_light = freq_0
    freq_heavy = freq_max
    light_i = 0
    heavy_i = 0
    for k, v in ace_counts.items():
        bc = k.bit_count()
        if bc == 0 or bc == n:
            continue
        bc <<= 1
        if bc > n:
            bc = (n << 1) - bc
            freq_heavy += v
            heavy_i = v * bc
        elif bc < n:
            freq_light += v
            light_i = v * bc
        else:
            # Balanced error
            v /= 2
            freq_heavy += v
            freq_light += v
            v *= bc
            heavy_i += v
            light_i += v

    freq_light /= shots
    freq_heavy /= shots
    heavy_i /= n * shots
    light_i /= n * shots
    if np.is_close(freq_light, 0) or np.is_close(freq_heavy, 0):
        fidelity = 0
    else:
        fidelity = 1.0 - 0.5 * (heavy_i / freq_heavy + light_i / freq_light)
        if freq_heavy > freq_light:
            balance = freq_light / freq_heavy
        else:
            balance = freq_heavy / freq_light
        fidelity *= balance
    
    return {
        "n": n,
        "seconds": time.perf_counter() - start,
        "shots": shots,
        "freq_0": freq_0,
        "freq_max": freq_max,
        "freq_light": freq_light,
        "freq_heavy": freq_heavy,
        "balanced_hamming_infidelity": 1 - fidelity,
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
