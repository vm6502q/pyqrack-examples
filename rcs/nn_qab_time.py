# Orbifolded random circuit sampling
# How good are Google's own "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)
# (This is actually a different "elision" concept, but allow that it works.)

import math
import random
import sys
import time

from pyqrack import QrackAceBackend


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


SWAP_RATIO_THRESHOLD = 7.0

def bench_qrack(width, depth, lrc=4, lrr=4, swap_mode="auto", boundary_dev=-1):
    if swap_mode not in ("auto", "swap", "cnot"):
        raise ValueError('swap_mode must be one of "auto", "swap", "cnot"')

    lcv_range = range(width)

    # This is a "nearest-neighbor" coupler random circuit.
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]

    row_len, col_len = factor_width(width)

    sim = QrackAceBackend(width, long_range_columns=lrc, long_range_rows=lrr, is_torus=True)
    if boundary_dev > -1:
        sim.sim[sim._boundary_sim_id].set_device(boundary_dev)

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

    start = time.perf_counter()

    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
            # Keep it Haar-random towards the poles:
            th = math.asin(th / math.pi)
            sim.u(i, th, ph, lm)

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

                if temp_row < 0:
                    continue
                if temp_col < 0:
                    continue
                if temp_row >= row_len:
                    continue
                if temp_col >= col_len:
                    continue

                b1 = col * row_len + row
                b2 = temp_col * row_len + temp_row

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                g(sim, b1, b2)

    # (I might have a problem.)
    # for i in range(len(sim.sim)):
    #     s = sim.sim[i]
    #     s.lossy_out_to_file(f"s{i}.tqs")

    # Terminal measurement    
    sample = sim.m_all()
    seconds = time.perf_counter() - start
    print(f"{width} qb, {depth} circuit layers, {seconds} seconds. (Fidelity unknown.)")

    return seconds, sample


def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 84
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    lrc = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    lrr = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    swap_mode = sys.argv[5] if len(sys.argv) > 5 else "auto"
    boundary_dev = int(sys.argv[6]) if len(sys.argv) > 6 else -1
    bench_qrack(width, depth, lrc, lrr, swap_mode, boundary_dev)
    return 0

if __name__ == "__main__":
    sys.exit(main())
