# ACE GHZ state preparation simulation

import math
import sys
import time

import numpy as np

from collections import Counter

from pyqrack import QrackAceBackend


def bench_qrack():
    n = 3
    shots = 8192

    start = time.perf_counter()

    sim = QrackAceBackend(n, long_range_columns=1, is_torus=False)

    sim.h(1)
    sim.mcx([1], 0)
    sim.mcx([1], 2)

    ace_counts = dict(Counter(sim.measure_shots(list(range(3)), shots)))

    freq_0 = ace_counts.get(0, 0)
    freq_max = ace_counts.get(7, 0)
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
            heavy_i += v * bc
        else:
            freq_light += v
            light_i += v * bc

    freq_light /= shots
    freq_heavy /= shots
    heavy_i /= n * shots
    light_i /= n * shots
    if np.isclose(freq_light, 0) or np.isclose(freq_heavy, 0):
        fidelity = 0
    else:
        fidelity = 1.0 - 0.5 * (heavy_i / freq_heavy + light_i / freq_light)
        if freq_heavy > freq_light:
            balance = freq_light / freq_heavy
        else:
            balance = freq_heavy / freq_light
        fidelity *= balance
    
    return {
        "seconds": time.perf_counter() - start,
        "shots": shots,
        "freq_0": freq_0,
        "freq_max": freq_max,
        "freq_light": freq_light,
        "freq_heavy": freq_heavy,
        "balanced_z": fidelity,
    }


def main():
    result = bench_qrack()
    for k, v in result.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
