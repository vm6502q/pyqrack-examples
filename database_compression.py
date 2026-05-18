# Can we compress random data with Qrack?

import math
import os
import random
import sys

from pyqrack import QrackSimulator


def calc_fidelity(ideal_ket, split_ket):
    s = sum([x * y.conjugate() for x, y in zip(ideal_ket, split_ket)])
    return (s * s.conjugate()).real

def calc_stats(ideal_ket, split_ket):
    n_pow = len(ideal_ket)
    n = int(round(math.log2(n_pow)))
    u_u = 1 / n_pow
    numer = 0
    denom = 0
    hog_prob = 0
    sqr_diff = 0
    l2 = 0
    for i in range(n_pow):
        split = split_ket[i]
        ideal = ideal_ket[i]

        # L2 norm
        l2 += split * ideal.conjugate()

        split = (split * split.conjugate()).real
        ideal = (ideal * ideal.conjugate()).real

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (split - u_u)

        # L2 norm
        sqr_diff += (ideal - split) ** 2

    l2 = (l2 * l2.conjugate()).real
    xeb = numer / denom
    rss = math.sqrt(sqr_diff)

    return {
        "qubits": n,
        "xeb": float(xeb),
        "l2_fidelity": float(l2),
        "prob_diff": float(rss),
    }

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


# Encoding by (Anthropic) Claude
def bench_qrack(width, p, b, w):
    # width key qubits, no value qubit needed
    # Each of the 2^width amplitudes encodes 2*p bits of data
    sim = QrackSimulator(width)

    levels = (1 << w)
    levels_m_1 = levels - 1
    n_entries = 1 << width
    
    # Generate random data: 2*p bits per entry
    data = [random.getrandbits(2 * w) for _ in range(n_entries)]
    
    # Encode: map data bits to amplitude (re, im) coordinates
    # Each coordinate uses p bits -> levels uniformly spaced in [-1, 1]
    amps = []
    norm_sq = 0.0
    for d in range(n_entries):
        re_bits = from_gray(d & ((1 << w) - 1))        # low w bits
        im_bits = from_gray((d >> w) & ((1 << w) - 1)) # high w bits
        # Map to [-1, 1]: bucket centre
        re = ((re_bits << 1) - levels_m_1) / levels
        im = ((im_bits << 1) - levels_m_1) / levels
        amps.append(complex(re, im))
        norm_sq += re * re + im * im
    
    # Normalize (required for valid quantum state)
    norm = math.sqrt(norm_sq)
    amps = [a / norm for a in amps]
    
    # Load directly into simulator state
    sim.in_ket(amps)
    sim.separate(list(range(width >> 1)))
    sim.lossy_out_to_file("lda.svtq", p=p, b=b)

    szd = w * n_entries / 4
    print(f"Saved {szd} bytes of data to lda.svtg.")    
    szf = os.path.getsize("lda.svtq") + 8
    print(f"File size (plus normalization constant): {szf} bytes")
    print(f"Compression ratio: {szd / szf}")

    sim.lossy_in_from_file("lda.svtq")
    e_amps = sim.out_ket()
    del sim

    print("Fidelity statistics:")
    print(calc_stats(amps, e_amps))


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

    # Run the benchmarks
    bench_qrack(width, p, b, w)

    return 0


if __name__ == "__main__":
    sys.exit(main())
