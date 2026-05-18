# Can we compress random data with Qrack?

import math
import os
import random
import sys

from pyqrack import QrackSimulator


# ---------------------------------------------------------------------------
# Fidelity metrics
# ---------------------------------------------------------------------------

def calc_fidelity_ket(ideal_ket, split_ket):
    s = sum(x * y.conjugate() for x, y in zip(ideal_ket, split_ket))
    return (s * s.conjugate()).real


def hamming_fidelity(orig_bits, recovered_bits, n):
    if n == 0:
        return 1.0
    errors = sum(a != b for a, b in zip(orig_bits, recovered_bits))
    return 1.0 - errors / n, errors


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
# Encoding: 'gray' mode
# Gray-coded bucket indices packed sequentially (w bits re, w bits im per amp).
# A ±1 quantization error flips exactly 1 bit (vs up to w in plain binary).
# ---------------------------------------------------------------------------

def bits_to_amps_gray(bits, w, n_amps):
    levels = 1 << w
    step   = 2.0 / levels
    amps   = []
    norm_sq = 0.0
    bit_pos = 0
    total   = len(bits)

    for _ in range(n_amps):
        re_bucket = 0
        for b in range(w):
            bit = bits[bit_pos] if bit_pos < total else 0
            re_bucket |= bit << b
            bit_pos += 1
        im_bucket = 0
        for b in range(w):
            bit = bits[bit_pos] if bit_pos < total else 0
            im_bucket |= bit << b
            bit_pos += 1
        # Convert Gray-coded bucket index back to linear before quantizing
        re_linear = from_gray(re_bucket)
        im_linear = from_gray(im_bucket)
        re = -1.0 + (re_linear + 0.5) * step
        im = -1.0 + (im_linear + 0.5) * step
        amps.append(complex(re, im))
        norm_sq += re*re + im*im

    norm = math.sqrt(norm_sq)
    return [a / norm for a in amps], norm


def amps_to_bits_gray(amps, w, n_bits):
    levels = 1 << w
    step   = 2.0 / levels
    bits   = []

    for amp in amps:
        if len(bits) >= n_bits:
            break
        re_linear = int((amp.real + 1.0) / step)
        re_linear = max(0, min(levels - 1, re_linear))
        im_linear = int((amp.imag + 1.0) / step)
        im_linear = max(0, min(levels - 1, im_linear))
        re_gray = to_gray(re_linear)
        im_gray = to_gray(im_linear)
        for b in range(w):
            if len(bits) >= n_bits: break
            bits.append((re_gray >> b) & 1)
        for b in range(w):
            if len(bits) >= n_bits: break
            bits.append((im_gray >> b) & 1)

    return bits


# Encoding by (Anthropic) Claude
def bench_qrack(width, p, b, w):
    # width key qubits, no value qubit needed
    # Each of the 2^width amplitudes encodes 2*p bits of data
    sim = QrackSimulator(width)
    
    levels = 1 << w
    n_amps = 1 << width
    n_bits = 2 * w * n_amps
    bits_per_amp = 2 * w
    
    # Generate random data: 2*p bits per entry
    data = [random.getrandbits(2 * w) for _ in range(n_amps)]
    amps, norm = bits_to_amps_gray(data, w, n_amps)
    
    # Load directly into simulator state
    sim.in_ket(amps)
    sim.separate(list(range(width >> 1)))
    sim.lossy_out_to_file("lda.svtq", p=p, b=b)

    szd = w * n_amps / 4
    print(f"Saved {szd} bytes of data to lda.svtg.")    
    szf = os.path.getsize("lda.svtq") + 8
    print(f"File size (plus normalization constant): {szf} bytes")
    print(f"Compression ratio: {szd / szf}")

    sim.lossy_in_from_file("lda.svtq")
    e_amps = sim.out_ket()
    del sim
    
    fidelity = calc_fidelity_ket(amps, e_amps)
    print(f"Ket fidelity: {fidelity}")

    recovered_bits = amps_to_bits_gray(e_amps, w, n_bits)
    hf, errors = hamming_fidelity(data, recovered_bits, n_amps * 2 * w)
    print(f"Hamming fidelity:         {(n_bits - errors) / n_bits}  "
          f"({errors} bit errors / {n_bits} total bits)")

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

    w = 24
    if len(sys.argv) > 4:
        w = int(sys.argv[4])

    # Run the benchmarks
    bench_qrack(width, p, b, w)

    return 0


if __name__ == "__main__":
    sys.exit(main())
