# Can we compress random data with Qrack?
# Robust variant: Gray code + bit-plane interleaving for equalized Hamming fidelity.
#
# Encoding modes (set via --mode argument):
#   'gray'  : Gray-coded bucket indices, sequential packing (2x fewer bit errors
#             from ±1 quantization noise vs plain binary; default)
#   'sign'  : w=1 sign-only encoding — fully equalized, most robust, lower density
#   'plane' : Gray code + bit-plane interleaving — groups same-significance bits
#             together so burst errors affect one plane, not all bits of one amplitude
#
# Hamming fidelity = 1 - (hamming_distance / total_bits)
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

def calc_fidelity_ket(ideal_ket, split_ket):
    s = sum(x * y.conjugate() for x, y in zip(ideal_ket, split_ket))
    return (s * s.conjugate()).real


def hamming_fidelity(orig_bits, recovered_bits):
    n = len(orig_bits)
    if n == 0:
        return 1.0
    errors = sum(a != b for a, b in zip(orig_bits, recovered_bits))
    return 1.0 - errors / n, errors


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


# ---------------------------------------------------------------------------
# Encoding: 'sign' mode
# w=1: only the sign of each coordinate encodes data (1 bit re, 1 bit im).
# Most robust: a sign flip requires quantization error > amplitude magnitude
# (~1/sqrt(n_amps)), which is much larger than the typical quant step.
# Data density: 2 bits per amplitude (vs 2*w in gray mode).
# ---------------------------------------------------------------------------

def bits_to_amps_sign(bits, n_amps):
    """Encode 2 bits per amplitude via sign of re and im."""
    amps    = []
    norm_sq = 0.0
    bit_pos = 0
    total   = len(bits)
    # Fixed magnitude per coordinate; only sign carries data.
    # Use 1/sqrt(2) so each amplitude has unit magnitude before normalization.
    mag = 1.0 / math.sqrt(2.0)

    for _ in range(n_amps):
        re_bit = bits[bit_pos] if bit_pos < total else 0; bit_pos += 1
        im_bit = bits[bit_pos] if bit_pos < total else 0; bit_pos += 1
        re = mag if re_bit else -mag
        im = mag if im_bit else -mag
        amps.append(complex(re, im))
        norm_sq += re*re + im*im

    norm = math.sqrt(norm_sq)
    return [a / norm for a in amps], norm


def amps_to_bits_sign(amps, n_bits):
    bits = []
    for amp in amps:
        if len(bits) >= n_bits: break
        bits.append(1 if amp.real >= 0.0 else 0)
        if len(bits) >= n_bits: break
        bits.append(1 if amp.imag >= 0.0 else 0)
    return bits


# ---------------------------------------------------------------------------
# Encoding: 'plane' mode
# Gray code + bit-plane interleaving.
# Instead of packing all w bits of one amplitude together, group all amplitudes'
# bit-k together for each k in 0..w-1.  A quantization error in one amplitude
# affects one bit in one plane rather than w adjacent bits in the stream.
# Same data density as 'gray'; error rate per bit is identical, but error
# locality is improved (burst errors within one amplitude are spread across
# w different positions in the recovered bit stream).
# ---------------------------------------------------------------------------

def bits_to_amps_plane(bits, w, n_amps):
    """
    Bit-plane interleaved encoding.
    Layout: [bit0 of re for all amps] [bit0 of im for all amps]
            [bit1 of re for all amps] ... [bit(w-1) of im for all amps]
    Gray coded within each plane.
    """
    levels  = 1 << w
    step    = 2.0 / levels
    total   = len(bits)
    n_bits  = 2 * w * n_amps

    # Build (n_amps, w) arrays of re_gray_bits and im_gray_bits
    re_gray_bits = [[0]*w for _ in range(n_amps)]
    im_gray_bits = [[0]*w for _ in range(n_amps)]

    bit_pos = 0
    for plane in range(w):
        for i in range(n_amps):
            re_gray_bits[i][plane] = bits[bit_pos] if bit_pos < total else 0
            bit_pos += 1
        for i in range(n_amps):
            im_gray_bits[i][plane] = bits[bit_pos] if bit_pos < total else 0
            bit_pos += 1

    amps    = []
    norm_sq = 0.0
    for i in range(n_amps):
        re_gray = sum(re_gray_bits[i][b] << b for b in range(w))
        im_gray = sum(im_gray_bits[i][b] << b for b in range(w))
        re_lin  = from_gray(re_gray)
        im_lin  = from_gray(im_gray)
        re = -1.0 + (re_lin + 0.5) * step
        im = -1.0 + (im_lin + 0.5) * step
        amps.append(complex(re, im))
        norm_sq += re*re + im*im

    norm = math.sqrt(norm_sq)
    return [a / norm for a in amps], norm


def amps_to_bits_plane(amps, w, n_bits, n_amps):
    levels = 1 << w
    step   = 2.0 / levels

    re_gray_bits = []
    im_gray_bits = []
    for amp in amps:
        re_lin = max(0, min(levels-1, int((amp.real + 1.0) / step)))
        im_lin = max(0, min(levels-1, int((amp.imag + 1.0) / step)))
        rg = to_gray(re_lin)
        ig = to_gray(im_lin)
        re_gray_bits.append([(rg >> b) & 1 for b in range(w)])
        im_gray_bits.append([(ig >> b) & 1 for b in range(w)])

    bits    = []
    for plane in range(w):
        for i in range(n_amps):
            if len(bits) >= n_bits: break
            bits.append(re_gray_bits[i][plane])
        for i in range(n_amps):
            if len(bits) >= n_bits: break
            bits.append(im_gray_bits[i][plane])

    return bits[:n_bits]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, p, b, w, mode='gray', do_separate=True):
    n_amps = 1 << width

    if mode == 'sign':
        n_bits = 2 * n_amps          # 1 bit per coordinate
        bits_per_amp = 2
    else:
        n_bits = 2 * w * n_amps      # w bits per coordinate
        bits_per_amp = 2 * w

    orig_bits = [random.randint(0, 1) for _ in range(n_bits)]

    # Encode
    if mode == 'sign':
        amps, norm = bits_to_amps_sign(orig_bits, n_amps)
    elif mode == 'plane':
        amps, norm = bits_to_amps_plane(orig_bits, w, n_amps)
    else:  # 'gray'
        amps, norm = bits_to_amps_gray(orig_bits, w, n_amps)

    sim = QrackSimulator(width)
    sim.in_ket(amps)
    if do_separate:
        sim.separate(list(range(width >> 1)))
    sim.lossy_out_to_file("lda.svtq", p=p, b=b)

    szd = n_bits / 8
    szf = os.path.getsize("lda.svtq") + 8
    print(f"Mode:                     {mode}")
    print(f"Data size:                {szd:.0f} bytes")
    print(f"Compressed size (+ norm): {szf} bytes")
    print(f"Compression ratio:        {szd / szf:.4f}x")

    # Decompress
    sim2 = QrackSimulator(width)
    sim2.lossy_in_from_file("lda.svtq")
    e_amps = sim2.out_ket()
    del sim2

    ket_fidelity = calc_fidelity_ket(amps, e_amps)
    print(f"Inner-product fidelity:   {ket_fidelity:.6f}")

    # Recover bits
    if mode == 'sign':
        recovered_bits = amps_to_bits_sign(e_amps, n_bits)
    elif mode == 'plane':
        recovered_bits = amps_to_bits_plane(e_amps, w, n_bits, n_amps)
    else:
        recovered_bits = amps_to_bits_gray(e_amps, w, n_bits)

    hf, errors = hamming_fidelity(orig_bits, recovered_bits)
    print(f"Hamming fidelity:         {hf:.6f}  "
          f"({errors} bit errors / {n_bits} total bits)")

    # Per-bit-plane error breakdown (gray and plane modes)
    if mode in ('gray', 'plane') and w > 1:
        print(f"  Bit-plane error breakdown (plane 0 = MSB of Gray code):")
        plane_errors = [0] * w
        if mode == 'gray':
            # Bits are packed sequentially: [re_bit0..re_bitw-1, im_bit0..im_bitw-1] per amp
            for i in range(n_amps):
                for bit_k in range(w):
                    re_idx = i * 2 * w + bit_k
                    im_idx = i * 2 * w + w + bit_k
                    if re_idx < n_bits and orig_bits[re_idx] != recovered_bits[re_idx]:
                        plane_errors[bit_k] += 1
                    if im_idx < n_bits and orig_bits[im_idx] != recovered_bits[im_idx]:
                        plane_errors[bit_k] += 1
        else:  # plane
            for plane in range(w):
                base = plane * 2 * n_amps
                for j in range(2 * n_amps):
                    idx = base + j
                    if idx < n_bits and orig_bits[idx] != recovered_bits[idx]:
                        plane_errors[plane] += 1
        for k, err in enumerate(plane_errors):
            print(f"    plane {k} (Gray bit {k}): {err} errors")

    return {
        "mode":             mode,
        "compression_ratio": szd / szf,
        "ket_fidelity":     ket_fidelity,
        "hamming_fidelity": hf,
        "bit_errors":       errors,
        "total_bits":       n_bits,
    }


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
    mode = 'gray'
    if len(sys.argv) > 5:
        mode = sys.argv[5]
    do_separate = False
    if len(sys.argv) > 6:
        do_separate = sys.argv[6].lower() not in ('0', 'false', 'no')

    result = bench_qrack(width, p, b, w, mode, do_separate)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
