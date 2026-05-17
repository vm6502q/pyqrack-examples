# Can we compress random data with Qrack?

import math
import os
import random
import sys

from pyqrack import QrackSimulator


def calc_fidelity(ideal_ket, split_ket):
    s = sum([x * y.conjugate() for x, y in zip(ideal_ket, split_ket)])
    return (s * s.conjugate()).real


# Encoding by (Anthropic) Claude
def bench_qrack(width, p, b, w):
    # width key qubits, no value qubit needed
    # Each of the 2^width amplitudes encodes 2*p bits of data
    sim = QrackSimulator(width)
    
    levels = 1 << w  # quantization levels per coordinate
    n_entries = 1 << width
    
    # Generate random data: 2*p bits per entry
    data = [random.getrandbits(2 * w) for _ in range(n_entries)]
    
    # Encode: map data bits to amplitude (re, im) coordinates
    # Each coordinate uses p bits -> levels uniformly spaced in [-1, 1]
    amps = []
    norm_sq = 0.0
    for d in range(n_entries):
        re_bits = d & ((1 << w) - 1)        # low w bits
        im_bits = (d >> w) & ((1 << w) - 1) # high w bits
        # Map to [-1, 1]: bucket centre
        re = (re_bits + 0.5) / levels * 2.0 - 1.0
        im = (im_bits + 0.5) / levels * 2.0 - 1.0
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
    
    fidelity = calc_fidelity(amps, e_amps)
    print(f"Fidelity: {fidelity}")


def main():

    width = 16
    if len(sys.argv) > 1:
        width = int(sys.argv[1])

    p = 6
    if len(sys.argv) > 2:
        p = int(sys.argv[2])

    b = 3
    if len(sys.argv) > 3:
        b = int(sys.argv[3])

    w = 23
    if len(sys.argv) > 4:
        w = int(sys.argv[4])

    # Run the benchmarks
    bench_qrack(width, p, b, w)

    return 0


if __name__ == "__main__":
    sys.exit(main())
