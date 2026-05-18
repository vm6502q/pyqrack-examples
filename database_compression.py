# Can we compress random data with Qrack?
#
# XEB-weighted Gray code encoding:
# Amplitudes are permuted before in_ket so that those with the highest
# XEB weight (largest |p_i - u| * |a_i|, i.e. furthest from uniform
# and largest magnitude) occupy the low-index positions.  After separate(),
# the lower-qubit subsystem is retained with higher fidelity — so the
# most XEB-significant amplitudes survive the lossy cut best.
# Gray code ensures small quantization errors (±1 bucket) flip exactly
# 1 bit, and those bits belong to the highest-weight amplitudes.
#
# By Dan Strano and (Anthropic) Claude.

import math
import os
import random
import sys

from pyqrack import QrackSimulator


# ---------------------------------------------------------------------------
# Gray code helpers
# ---------------------------------------------------------------------------

def to_gray(n):
    return n ^ (n >> 1)

def from_gray(n):
    mask = n >> 1
    while mask:
        n ^= mask
        mask >>= 1
    return n


# ---------------------------------------------------------------------------
# Fidelity metrics
# ---------------------------------------------------------------------------

def calc_stats(ideal_ket, split_ket):
    n_pow = len(ideal_ket)
    u_u = 1.0 / n_pow
    numer = 0.0
    denom = 0.0
    l2 = 0.0
    prob_diff = 0.0
    for i in range(n_pow):
        p_i = abs(ideal_ket[i]) ** 2
        q_i = abs(split_ket[i]) ** 2
        numer += (p_i - u_u) * (q_i - u_u)
        denom += (p_i - u_u) ** 2
        l2    += (p_i - q_i) ** 2
        prob_diff += abs(p_i - q_i)
    xeb = numer / denom if denom > 0 else 0.0
    return {
        'qubits':     int(round(math.log2(n_pow))),
        'xeb':        xeb,
        'l2_fidelity': 1.0 - math.sqrt(l2),
        'prob_diff':  prob_diff / n_pow,
    }


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def bench_qrack(width, p, b, w):
    levels   = 1 << w
    levels_m_1 = levels - 1
    n_entries = 1 << width
    u_u      = 1.0 / n_entries

    # Generate random data: 2*w bits per entry (Gray-coded bucket indices)
    data = [random.getrandbits(2 * w) for _ in range(n_entries)]

    # Step 1: build raw amplitudes from data (exact arithmetic, no rounding)
    raw_amps = []
    norm_sq  = 0.0
    for d in data:
        re_bits = from_gray(d & levels_m_1)
        im_bits = from_gray((d >> w) & levels_m_1)
        re = ((re_bits << 1) - levels_m_1) / levels
        im = ((im_bits << 1) - levels_m_1) / levels
        raw_amps.append(complex(re, im))
        norm_sq += re * re + im * im

    norm = math.sqrt(norm_sq)
    raw_amps = [a / norm for a in raw_amps]

    # Step 2: compute XEB weight for each amplitude position.
    # w_i = |p_i - u| * |a_i|  where p_i = |a_i|^2.
    # High weight => far from uniform AND large magnitude => most XEB-significant.
    xeb_weights = [abs(abs(a)**2 - u_u) * abs(a) for a in raw_amps]

    # Step 3: sort amplitudes by XEB weight DESCENDING.
    # High-weight amplitudes go to low indices (lower-qubit subsystem),
    # which survive the separate() cut with higher fidelity.
    # Track the permutation so we can invert it after decompression.
    order = sorted(range(n_entries), key=lambda i: xeb_weights[i], reverse=True)
    inv_order = [0] * n_entries
    for new_idx, old_idx in enumerate(order):
        inv_order[old_idx] = new_idx

    amps = [raw_amps[old_idx] for old_idx in order]

    # Step 4: load into Qrack, separate, compress
    sim = QrackSimulator(width)
    sim.in_ket(amps)
    sim.separate(list(range(width >> 1)))
    sim.lossy_out_to_file("lda.svtq", p=p, b=b)

    szd = w * n_entries / 4
    print(f"Saved {szd} bytes of data to lda.svtg.")
    szf = os.path.getsize("lda.svtq") + 8
    print(f"File size (plus normalization constant): {szf} bytes")
    print(f"Compression ratio: {szd / szf}")

    # Step 5: decompress and invert permutation
    sim.lossy_in_from_file("lda.svtq")
    e_amps_permuted = sim.out_ket()
    del sim

    # Invert the XEB-weight permutation to restore original amplitude ordering
    e_amps = [None] * n_entries
    for new_idx, old_idx in enumerate(order):
        e_amps[old_idx] = e_amps_permuted[new_idx]

    print(f"Fidelity statistics:")
    print(calc_stats(raw_amps, e_amps))


def main():
    width = 16
    if len(sys.argv) > 1:
        width = int(sys.argv[1])

    p = 6
    if len(sys.argv) > 2:
        p = int(sys.argv[2])

    b = 4
    if len(sys.argv) > 3:
        b = int(sys.argv[3])

    w = 12
    if len(sys.argv) > 4:
        w = int(sys.argv[4])

    bench_qrack(width, p, b, w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
