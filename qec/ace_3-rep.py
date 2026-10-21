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

    for q in range(3):
       sim.h(q)

    x_counts = Counter(sim.measure_shots(list(range(3)), shots))
    freq_0_x = z_counts.get(0, 0) / shots
    freq_max_x = z_counts.get(7, 0) / shots
    freq_viable_x = freq_0_z + freq_max_z
    
    return {
        "seconds": time.perf_counter() - start,
        "shots": shots,
        "freq_0_z": freq_0_z,
        "freq_max_z": freq_max_z,
        "freq_viable_z": freq_viable_z,
        "freq_0_x": freq_0_x,
        "freq_max_x": freq_max_x,
        "freq_viable_x": freq_viable_x,
    }


def main():
    result = bench_qrack()
    for k, v in result.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
