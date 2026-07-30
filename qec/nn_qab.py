# Nearest-neighbor RCS: Automatic circuit elision
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
from qiskit import QuantumCircuit


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (row_len, col_len)


def bulk_to_boundary_ratio(sim):
    """Empirically measured bulk-to-boundary ratio for an already-
    constructed QrackAceBackend, via its own _unpack() (same source of
    truth used internally, rather than re-deriving the geometry rules by
    hand). Returns float('inf') if there are zero boundary qubits (a
    valid, if edge-case, config), rather than raising on the divide.
    """
    n = sim.num_qubits()
    boundary = sum(1 for lq in range(n) if len(sim._unpack(lq)) > 1)
    bulk = n - boundary
    return bulk / boundary if boundary else float("inf")


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


# --- swap-family gates: native (QrackAceBackend.swap(), the _correct()-
# wrapped fast/sandwiched-shadow implementation) vs. cnot (manual 3-CNOT
# decomposition, going through the ordinary _cpauli-wrapped cx() path
# instead) -- two full sets of wrappers, selected between in bench_qrack()
# based on the swap_mode argument. Each _cnot variant mirrors the actual
# QrackAceBackend.swap()/iswap()/adjiswap() class-method gate sequences
# exactly, just using 3 explicit cx() calls in place of a single swap()
# call, so the two modes differ ONLY in how the swap itself is realized,
# not in the surrounding cz/s/adjs structure of the compound gates.

def swap_native(sim, q1, q2):
    sim.swap(q1, q2)


def swap_cnot(sim, q1, q2):
    if random.getrandbits(1):
        q1, q2 = q2, q1
    sim.mcx([q1], q2)
    sim.mcx([q2], q1)
    sim.mcx([q1], q2)


def iswap_native(sim, q1, q2):
    sim.iswap(q1, q2)


def iswap_cnot(sim, q1, q2):
    swap_cnot(sim, q1, q2)
    sim.mcz([q1], q2)
    sim.s(q1)
    sim.s(q2)


def iiswap_native(sim, q1, q2):
    sim.adjiswap(q1, q2)


def iiswap_cnot(sim, q1, q2):
    sim.adjs(q2)
    sim.adjs(q1)
    sim.mcz([q1], q2)
    swap_cnot(sim, q1, q2)


def pswap_native(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)


def pswap_cnot(sim, q1, q2):
    sim.mcz([q1], q2)
    swap_cnot(sim, q1, q2)


def mswap_native(sim, q1, q2):
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def mswap_cnot(sim, q1, q2):
    swap_cnot(sim, q1, q2)
    sim.mcz([q1], q2)


def nswap_native(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def nswap_cnot(sim, q1, q2):
    sim.mcz([q1], q2)
    swap_cnot(sim, q1, q2)
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

SWAP_RATIO_THRESHOLD = 7.0

def bench_qrack(width, depth, lrc=4, lrr=4, sdrp=0.0, swap_mode="auto"):
    if swap_mode not in ("auto", "swap", "cnot"):
        raise ValueError('swap_mode must be one of "auto", "swap", "cnot"')
    lcv_range = range(width)
    all_bits  = list(lcv_range)
    n_pow     = 1 << width
    shots     = 1 << min(10, width + 2)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]

    row_len, col_len = factor_width(width)

    sim = QrackAceBackend(width, long_range_columns=lrc, long_range_rows=lrr, is_torus=True)
    sim.set_sdrp(sdrp)

    ratio = bulk_to_boundary_ratio(sim)
    if swap_mode == "auto":
        resolved_swap_mode = "cnot" if ratio >= SWAP_RATIO_THRESHOLD else "swap"
    else:
        resolved_swap_mode = swap_mode

    if resolved_swap_mode == "swap":
        two_bit_gates = (
            swap_native, pswap_native, mswap_native, nswap_native,
            iswap_native, iiswap_native, cx, cy, cz, acx, acy, acz,
        )
    else:
        two_bit_gates = (
            swap_cnot, pswap_cnot, mswap_cnot, nswap_cnot,
            iswap_cnot, iiswap_cnot, cx, cy, cz, acx, acy, acz,
        )

    # -----------------------------------------------------------------------
    # Build circuit
    # -----------------------------------------------------------------------
    t_circ = time.perf_counter()
    qc = []

    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
            # Keep it Haar-random towards the poles:
            th = math.asin(th / math.pi)
            qc.append((u, i, th, ph, lm))

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
                temp_col = temp_col + (1 if (gate & 1) else 0)

                # Non-toroidal (is_torus=False) boundary handling, split
                # by axis rather than applied uniformly -- these two axes
                # do NOT behave the same way under is_torus=False, given
                # long_range_columns=2 is set explicitly below but
                # long_range_rows is left at its default (4):
                #
                # temp_row ranges over row_len (the LONG dimension), and
                # matches QrackAceBackend's "long_range_columns" axis
                # (verified: that parameter governs boundary density along
                # the same axis this script calls "row", both ranging over
                # row_len). long_range_columns=2 < row_len for any width
                # worth testing, so is_torus=False genuinely, actually
                # disables wraparound here -- skip (continue) is correct.
                #
                # temp_col ranges over col_len (the SHORT dimension), and
                # matches QrackAceBackend's "long_range_rows" axis, left at
                # its DEFAULT value of 4. is_torus=False only actually
                # disables wraparound on a given axis when
                # long_range_X < length_of_that_axis -- so for col_len<=4
                # (verified directly: true for col_len in {2,3,4}, false
                # for col_len>=5), long_range_rows=4 is NOT less than
                # col_len, meaning QrackAceBackend treats this entire short
                # dimension as one continuous interior run regardless of
                # is_torus. Skipping here would silently drop real
                # coupling gates that QrackAceBackend still, correctly,
                # treats as adjacent -- wrapping instead matches its
                # actual behavior. This holds for col_len<=4; for col_len>=5
                # this assumption would need revisiting (either passing
                # long_range_rows explicitly below, or reworking this check
                # to depend on it rather than a hardcoded default).
                if temp_row < 0:
                    continue
                if temp_row >= row_len:
                    continue
                if temp_col < 0:
                    if (row_len < 3) or (row_len <= lrr):
                        temp_col = temp_col + col_len
                    else:
                        continue
                if temp_col >= col_len:
                    if (row_len < 3) or (row_len <= lrr):
                        temp_col = temp_col - col_len
                    else:
                        continue

                b1 = col * row_len + row
                b2 = temp_col * row_len + temp_row

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                qc.append((g, b1, b2))

    # -----------------------------------------------------------------------
    # Method: QrackAceBackend
    # -----------------------------------------------------------------------
    sim = QrackAceBackend(width, long_range_columns=lrc, long_range_rows=lrr, is_torus=False)
    sim.set_sdrp(sdrp)
    run_circuit(sim, qc)
    ace_counts = dict(Counter(sim.measure_shots(all_bits, shots)))

    t_ace = time.perf_counter()
    print(f"ace_seconds: {t_ace - t_circ:.4f}")

    # -----------------------------------------------------------------------
    # Ideal ground truth via QrackSimulator
    # -----------------------------------------------------------------------
    sim_ideal = QrackSimulator(width)
    run_circuit(sim_ideal, qc)
    ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
    del sim_ideal

    t_ideal = time.perf_counter()
    print(f"ideal_seconds: {t_ideal - t_ace:.4f}")

    xeb_ace, hog_ace = calc_stats(ideal_probs, ace_counts, shots)

    return {
        "width":              width,
        "depth":              depth,
        "long_range_columns": lrc,
        "long_range_rows":    lrr,
        "sdrp":               sdrp,
        "bulk_to_boundary":   ratio,
        "swap_mode":          swap_mode,
        "resolved_swap_mode": resolved_swap_mode,
        "xeb_ace":            xeb_ace,
        "hog_ace":            hog_ace,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 nn_qab.py [width] [depth] "
            "[long_range_columns=4] [long_range_rows=4] "
            "[sdrp=0.1464466] [swap_mode=auto|swap|cnot]"
        )
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    lrc = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    lrr = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    sdrp  = float(sys.argv[5]) if len(sys.argv) > 5 else ((1 - 1 / math.sqrt(2)) / 2)
    swap_mode = sys.argv[6] if len(sys.argv) > 6 else "auto"
    result = bench_qrack(width, depth, lrc, lrr, sdrp, swap_mode)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
