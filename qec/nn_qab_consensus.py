# Nearest-neighbor RCS: Automatic circuit elision, with index-shifted-consensus
# across QrackAceBackend instances.
#
# By Dan Strano and (Anthropic) Claude.

import math
import random
import statistics
import sys
import time

from collections import Counter

import numpy as np
from pyqrack import QrackSimulator, QrackAceBackend


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (row_len, col_len)


# ---------------------------------------------------------------------------
# Index-shift consensus helpers
# ---------------------------------------------------------------------------
# ACE's own internal patch assignment is a fixed function of raw qubit
# INDEX alone (lq // row_length, lq % row_length -- verified directly
# against QrackAceBackend's source earlier in this session), independent
# of whatever topology a circuit imposes on those same indices. Shifting
# the index-to-(row,col) mapping per instance moves each instance's fixed
# patch boundaries to a different physical location relative to the SAME
# logical circuit, so the localized approximation error that concentrates
# near a boundary lands in a different place each time. This uses the
# SAME flatten convention verified against ACE's real source a few turns
# ago in nn_qab.py: b = col * row_len + row (not row * row_len + col,
# which was the OTHER script's now-known bug).

def shift_index(i, row_shift, col_shift, row_len, col_len):
    """Re-express flat index i's (row, col) position under a cumulative
    (row_shift, col_shift) offset, with modular wraparound on each
    dimension independently, then re-flatten via ACE's own convention.
    """
    row = i % row_len
    col = i // row_len
    shifted_row = (row + row_shift) % row_len
    shifted_col = (col + col_shift) % col_len
    return shifted_col * row_len + shifted_row


def build_shift_map(row_shift, col_shift, row_len, col_len, width):
    """orig_to_shifted[orig_i] = the actual physical qubit index a given
    instance should use to represent original circuit position orig_i.
    A true bijection on [0, width) for any integer shift, verified
    separately before this was written into the real script.
    """
    return [shift_index(i, row_shift, col_shift, row_len, col_len) for i in range(width)]


def unshift_sample(sample, orig_to_shifted):
    """Transform one measured outcome (bits indexed by an instance's
    SHIFTED qubit layout) back to the ORIGINAL circuit's index labeling,
    so bit position k means the same logical qubit across every instance
    and against ideal_probs, which was computed from the unshifted
    circuit.
    """
    result = 0
    for orig_i, shifted_i in enumerate(orig_to_shifted):
        bit = (sample >> shifted_i) & 1
        result |= bit << orig_i
    return result


# ---------------------------------------------------------------------------
# Gate wrappers
# ---------------------------------------------------------------------------

def u(sim, q, th, ph, lm):
    sim.u(q, th, ph, lm)


def cx(sim, q1, q2):
    sim.mcx([q1], q2)


def cy(sim, q1, q2):
    sim.mcy([q1], q2)


def cz(sim, q1, q2):
    sim.mcz([q1], q2)


def acx(sim, q1, q2):
    sim.macx([q1], q2)


def acy(sim, q1, q2):
    sim.macy([q1], q2)


def acz(sim, q1, q2):
    sim.macz([q1], q2)


def swap(sim, q1, q2):
    sim.swap(q1, q2)


def iswap(sim, q1, q2):
    sim.iswap(q1, q2)


def iiswap(sim, q1, q2):
    sim.adjiswap(q1, q2)


def pswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)


def mswap(sim, q1, q2):
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def nswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def run_circuit(sim, circ):
    for g in circ:
        g[0](sim, *g[1:])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats(ideal_probs, counts, shots):
    n_pow = len(ideal_probs)
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    hog_prob = 0
    for b in range(n_pow):
        ideal = ideal_probs[b]
        patch = (counts.get(b, 0) / shots)

        ideal_centered = ideal - u_u
        denom += ideal_centered * ideal_centered
        numer += ideal_centered * (patch - u_u)

        if ideal > threshold:
            hog_prob += patch

    xeb = numer / denom
    return xeb, hog_prob


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, lrc=4, lrr=4, sdrp=0.0):
    lcv_range    = range(width)
    all_bits     = list(lcv_range)
    n_inst       = 3
    n_pow        = 1 << width
    u_u          = 1.0 / n_pow
    shots        = 1 << min(10, width + 2)
    shots_per    = shots // n_inst
    shots        = shots_per * n_inst

    row_len, col_len = factor_width(width)

    # Per-instance cumulative index shift: instance k uses shift (k, k),
    # per the requested "shifting by one row and one column, iteratively
    # compounded for each subsequent instance." Precompute each instance's
    # full index-remapping once, up front.
    shift_maps = [
        build_shift_map(inst, inst, row_len, col_len, width) for inst in range(n_inst)
    ]

    # -----------------------------------------------------------------------
    # Build the circuit ONCE, in terms of the ORIGINAL (unshifted) qubit
    # indices -- this is the single logical circuit every instance is
    # meant to represent, just at a different physical index layout.
    # qc_logical holds gate tuples using ORIGINAL indices; each instance's
    # actual gate list is derived from this by remapping indices through
    # that instance's shift_map at dispatch time below, rather than
    # rebuilding the circuit from scratch per instance -- this guarantees
    # every instance really is representing the SAME logical circuit
    # (same random angles, same random 2-qubit gate choices, same
    # per-layer gate ORDER), differing only in index layout, not content.
    # -----------------------------------------------------------------------
    t_circ = time.perf_counter()
    qc_logical = []

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    for _ in range(depth):
        for i in lcv_range:
            th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
            # Keep it Haar-random towards the poles:
            th = math.pi + 2 * th * abs(math.cos(th))
            qc_logical.append((u, i, th, ph, lm))

        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        cl = []
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
                temp_col = temp_col + (1 if (gate & 1) else 0)

                if temp_row < 0:
                    temp_row = temp_row + row_len
                if temp_col < 0:
                    temp_col = temp_col + col_len
                if temp_row >= row_len:
                    temp_row = temp_row - row_len
                if temp_col >= col_len:
                    temp_col = temp_col - col_len

                b1 = col * row_len + row
                b2 = temp_col * row_len + temp_row

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                cl.append((g, b1, b2))

        random.shuffle(cl)
        qc_logical.extend(cl)

    # -----------------------------------------------------------------------
    # Ideal ground truth -- run directly against the ORIGINAL, unshifted
    # indices, exactly as ideal_probs' bit positions are meant to mean.
    # -----------------------------------------------------------------------
    sim_ideal = QrackSimulator(width)
    run_circuit(sim_ideal, qc_logical)
    ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
    del sim_ideal

    t_ideal = time.perf_counter()

    print(f"qrack_circuit_seconds: {t_ideal - t_circ}")

    # -----------------------------------------------------------------------
    # Method: ACE index-shifted consensus. Each instance runs the SAME
    # logical circuit (qc_logical, unchanged), but every gate's qubit
    # index is remapped through that instance's own shift_map before
    # being applied -- so ACE's fixed, index-based patch boundaries fall
    # at a different physical location relative to the circuit's actual
    # coupling structure in each instance. Measured samples are then
    # transformed BACK to the original index labeling via unshift_sample
    # before being pooled, so every instance's counts refer to the same
    # logical qubit at each bit position and remain directly comparable
    # to ideal_probs.
    # -----------------------------------------------------------------------
    ace_counts = []
    for inst in range(n_inst):
        shift_map = shift_maps[inst]

        # Remap every gate's qubit indices through this instance's shift,
        # preserving gate type, angle, and relative circuit order exactly.
        instance_circuit = []
        for gate_tuple in qc_logical:
            fn = gate_tuple[0]
            if fn is u:
                _, i, th, ph, lm = gate_tuple
                instance_circuit.append((u, shift_map[i], th, ph, lm))
            else:
                _, q1, q2 = gate_tuple
                instance_circuit.append((fn, shift_map[q1], shift_map[q2]))

        sim = QrackAceBackend(width, long_range_columns=lrc, long_range_rows=lrr)
        sim.set_sdrp(sdrp)
        run_circuit(sim, instance_circuit)

        raw_samples = sim.measure_shots(all_bits, shots_per)
        # Transform each measured sample back to the ORIGINAL circuit's
        # index labeling before pooling into the shared consensus counts.
        ace_counts.extend(unshift_sample(s, shift_map) for s in raw_samples)

    ace_counts = dict(Counter(ace_counts))

    xeb_ace, hog_ace = calc_stats(ideal_probs, ace_counts, shots)

    t_elapsed = time.perf_counter() - t_ideal

    return {
        "width":              width,
        "depth":              depth,
        "long_range_columns": lrc,
        "long_range_rows":    lrr,
        "sdrp":               sdrp,
        "depth":              depth,
        "xeb_ace":            xeb_ace,
        "hog_ace":            hog_ace,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 nn_qab_consensus.py [width] [depth] [long_range_columns=4] [long_range_rows=4] [sdrp=0.1464466]")
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    lrc = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    lrr = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    sdrp  = float(sys.argv[5]) if len(sys.argv) > 5 else ((1 - 1 / math.sqrt(2)) / 2)
    result = bench_qrack(width, depth, lrc, lrr, sdrp)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
