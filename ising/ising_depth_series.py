# Ising model Trotterization as interpreted by (OpenAI GPT) Elara
# You likely want to specify environment variable QRACK_MAX_PAGING_QB=28

import math
import sys
import time

from collections import Counter

import numpy as np

from scipy.stats import distributions as dists

import matplotlib.pyplot as plt

from pyqrackising import get_tfim_hamming_distribution


def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if is_transpose else (row_len, col_len)


def main():
    n_qubits = 16
    depth = 40
    z = 4

    # Quantinuum settings
    J, h, dt = -1.0, 2.0, 0.125
    theta = math.pi / 18

    # Pure ferromagnetic
    # J, h, dt = -1.0, 0.0, 0.25
    # theta = 0

    # Pure transverse field
    # J, h, dt = 0.0, 2.0, 0.25
    # theta = -math.pi / 2

    # Critical point (symmetry breaking)
    # J, h, dt = -1.0, 1.0, 0.25
    # theta = -math.pi / 4

    if len(sys.argv) > 1:
        depth = int(sys.argv[1])
    if len(sys.argv) > 2:
        dt = float(sys.argv[2])

    depths = list(range(1, depth + 1))
    results = []
    magnetizations = {}

    n_rows, n_cols = factor_width(n_qubits, False)
    qubits = list(range(n_qubits))
    magnetizations = []

    start = time.perf_counter()
    for d in depths:
        t = d * dt

        bias = get_tfim_hamming_distribution(J=J, h=h, z=z, theta=theta, t=t, n_qubits=n_qubits)

        d_magnetization, d_sqr_magnetization = 0, 0
        for hamming_weight, value in enumerate(bias):
            m = 1.0 - 2 * hamming_weight / n_qubits
            d_magnetization += value * m
            d_sqr_magnetization += value * m * m

        seconds = time.perf_counter() - start

        results.append(
            {
                "width": n_qubits,
                "depth": d,
                "magnetization": float(d_magnetization),
                "square_magnetization": float(d_sqr_magnetization),
                "seconds": seconds,
            }
        )
        magnetizations.append(d_sqr_magnetization)
        print(results[-1])

    # Plotting (contributed by Elara, an OpenAI custom GPT)
    plt.figure(figsize=(14, 14))

    plt.plot(depths, magnetizations, marker='o', linestyle='-')

    plt.xlabel("step")
    plt.ylabel(r"$\langle Z^2_{tot} \rangle$")
    plt.title("Square Magnetization vs Trotter Depth")
    # plt.legend()
    plt.grid(True)
    plt.xticks(depths)
    # plt.ylim(0.05, 0.7)
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
