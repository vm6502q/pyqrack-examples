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

    z_counts = Counter(sim.measure_shots(list(range(3)), shots))
    freq_0_z = z_counts.get(0, 0) / shots
    freq_max_z = z_counts.get(7, 0) / shots
    freq_viable_z = freq_0_z + freq_max_z
    freq_light_z = sum(v * (1 - 2 * (k.bit_count() & 1)) / shots if k.bit_count() < 2 else 0 for k, v in z_counts.items())
    odds_z = freq_light_z / (1 - freq_light_z)

    for q in range(3):
       sim.h(q)

    x_counts = Counter(sim.measure_shots(list(range(3)), shots))
    freq_0_x = z_counts.get(0, 0) / shots
    freq_max_x = z_counts.get(7, 0) / shots
    freq_viable_x = freq_0_z + freq_max_z
    freq_light_x = sum(v * (1 - 2 * (k.bit_count() & 1)) / shots if k.bit_count() < 2 else 0 for k, v in x_counts.items())
    odds_x = freq_light_x / (1 - freq_light_x)
    
    return {
        "seconds": time.perf_counter() - start,
        "shots": shots,
        "freq_0_z": freq_0_z,
        "freq_max_z": freq_max_z,
        "freq_viable_z": freq_viable_z,
        "light_to_heavy_odds_z": odds_z,
        "freq_0_x": freq_0_x,
        "freq_max_x": freq_max_x,
        "freq_viable_x": freq_viable_x,
        "light_to_heavy_odds_x": odds_x,
    }


def main():
    result = bench_qrack()
    for k, v in result.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
