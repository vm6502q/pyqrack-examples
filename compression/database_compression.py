# Can we compress random data with Qrack?
# Variant: reconstruct original bit stream and measure Hamming-distance fidelity.
#
# Hamming fidelity = 1 - (hamming_distance / total_bits)
# i.e. the normalized fraction of bits that round-trip correctly.
#
# By Dan Strano and (Anthropic) Claude.

import math
import os
import random
import sys

from pyqrack import QrackSimulator


def calc_fidelity_ket(ideal_ket, split_ket):
    """Inner-product fidelity (original metric, kept for reference)."""
    s = sum(x * y.conjugate() for x, y in zip(ideal_ket, split_ket))
    return (s * s.conjugate()).real


def amps_to_bits(amps, w, n_bits):
    """
    Recover a bit stream from complex amplitudes.

    Each amplitude encodes w bits of re and w bits of im (LSB-first within
    each coordinate), packed sequentially.  Only the first n_bits bits are
    returned (the rest are padding introduced during encoding).

    Parameters
    ----------
    amps   : list of complex, length n_amps = 2^qubits
    w      : int, bits per amplitude coordinate
    n_bits : int, number of data bits to recover (= 2 * w * n_entries)
    """
    levels = 1 << w
    step   = 2.0 / levels   # [-1, 1] quantized in `levels` uniform steps

    bits = []
    for amp in amps:
        if len(bits) >= n_bits:
            break
        # Re coordinate
        re_bucket = int((amp.real + 1.0) / step)
        re_bucket = max(0, min(levels - 1, re_bucket))
        for b in range(w):
            if len(bits) >= n_bits:
                break
            bits.append((re_bucket >> b) & 1)
        # Im coordinate
        im_bucket = int((amp.imag + 1.0) / step)
        im_bucket = max(0, min(levels - 1, im_bucket))
        for b in range(w):
            if len(bits) >= n_bits:
                break
            bits.append((im_bucket >> b) & 1)

    return bits


def bits_to_amps(bits, w, n_amps):
    """
    Encode a bit stream into normalized complex amplitudes.

    Parameters
    ----------
    bits   : list of int (0/1)
    w      : int, bits per amplitude coordinate
    n_amps : int, number of complex amplitudes to produce (= 2^qubits)
    """
    levels = 1 << w
    step   = 2.0 / levels

    amps    = []
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

        re = -1.0 + (re_bucket + 0.5) * step
        im = -1.0 + (im_bucket + 0.5) * step
        amps.append(complex(re, im))
        norm_sq += re * re + im * im

    norm = math.sqrt(norm_sq)
    return [a / norm for a in amps], norm


def hamming_fidelity(orig_bits, recovered_bits):
    """
    Normalized Hamming fidelity: fraction of bits that match.

    Returns a value in [0, 1] where 1.0 = perfect reconstruction.
    """
    assert len(orig_bits) == len(recovered_bits), \
        f"Length mismatch: {len(orig_bits)} vs {len(recovered_bits)}"
    n = len(orig_bits)
    if n == 0:
        return 1.0
    errors = sum(a != b for a, b in zip(orig_bits, recovered_bits))
    return 1.0 - errors / n


def bench_qrack(width, p, b, w):
    """
    width : number of qubits (statevector has 2^width amplitudes)
    p     : TurboQuant block size exponent (block = 2^p amplitudes)
    b     : TurboQuant bits per quantized coordinate
    w     : data bits per amplitude coordinate (re and im separately)
            => 2*w data bits stored per amplitude
    """
    n_amps    = 1 << width       # number of complex amplitudes
    n_entries = n_amps           # one entry per amplitude
    n_bits    = 2 * w * n_amps   # total data bits encoded

    # Generate random bit stream
    orig_bits = [random.randint(0, 1) for _ in range(n_bits)]

    # Encode bits -> amplitudes -> QrackSimulator
    amps, norm = bits_to_amps(orig_bits, w, n_amps)

    sim = QrackSimulator(width)
    sim.in_ket(amps)
    # sim.separate(list(range(width >> 1)))
    sim.lossy_out_to_file("lda.svtq", p=p, b=b)

    # --- compression ratio ---
    szd = n_bits / 8             # data size in bytes (packed)
    szf = os.path.getsize("lda.svtq") + 8   # file + stored norm scalar
    print(f"Data size:                {szd:.0f} bytes")
    print(f"Compressed size (+ norm): {szf} bytes")
    print(f"Compression ratio:        {szd / szf:.4f}x")

    # Decompress
    sim2 = QrackSimulator(width)
    sim2.lossy_in_from_file("lda.svtq")
    e_amps = sim2.out_ket()
    del sim2

    # Inner-product fidelity (reference)
    ket_fidelity = calc_fidelity_ket(amps, e_amps)
    print(f"Inner-product fidelity:   {ket_fidelity:.6f}")

    # Recover bit stream from decompressed amplitudes
    recovered_bits = amps_to_bits(e_amps, w, n_bits)

    # Hamming fidelity
    hf = hamming_fidelity(orig_bits, recovered_bits)
    errors = sum(a != b for a, b in zip(orig_bits, recovered_bits))
    print(f"Hamming fidelity:         {hf:.6f}  "
          f"({errors} bit errors / {n_bits} total bits)")

    return {
        "compression_ratio": szd / szf,
        "ket_fidelity":      ket_fidelity,
        "hamming_fidelity":  hf,
        "bit_errors":        errors,
        "total_bits":        n_bits,
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

    w = 12
    if len(sys.argv) > 4:
        w = int(sys.argv[4])

    result = bench_qrack(width, p, b, w)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
