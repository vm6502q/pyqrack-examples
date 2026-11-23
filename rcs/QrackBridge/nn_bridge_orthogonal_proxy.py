# Nearest-neighbor RCS on QrackBridge, validated AT SCALE by a K-patch
# (K = 2, 3, 4, ...) proxy whose cuts are ORTHOGONAL to QrackBridge's own patch seams.
#
# By Dan Strano and (Anthropic) Claude.
#
# Same circuit family and gate set as nn_qab.py / nn_qab_orthogonal_proxy.py.
# The difference from the ACE version is which axis each side cuts:
#
#   * Qubit index is b = col * row_len + row, with "row" the LONG grid coordinate
#     and "col" the SHORT one (see factor_width()).
#
#   * QrackBridge splits the qubit INDEX range into `patches` contiguous blocks.
#     A contiguous index block is a range of "col" values spanning every "row",
#     so every bridge seam is a "wall" across the SHORT coordinate. (If a patch
#     size is not a multiple of row_len, a seam steps once, partway through a
#     column; it still runs across every row.)
#
#   * The validation proxy therefore cuts the OTHER way: the row_len rows (long
#     coordinate) are split into K contiguous bands, with the torus wrap between
#     the last and first row also cut. Every coupler that crosses a band boundary
#     is removed from the circuit; nothing else changes (same random draw, same
#     order). Each proxy band holds every column, so it is NOT contiguous in
#     qubit index; its bits are gathered from the sample explicitly.
#
#   * The proxy circuit then factorizes into K independent bands, and its ideal
#     distribution is exactly p(x) = prod_k p_k(x_k). Each band is simulated
#     exactly on its own, so the ideal is computable for widths where 2^width
#     is not. K must not exceed row_len (the long grid dimension).
#
#   * Because the cuts are orthogonal, EVERY proxy band runs straight through
#     every bridge seam, and the bridge cannot line its patches up with the
#     proxy's. One honest caveat: a diagonal coupler (one that changes both row
#     and col) can cross a bridge seam AND a proxy boundary at once, and the
#     proxy drops it. The script counts bridge-seam couplers in the full and
#     proxy circuits and prints both, so how much seam traffic the proxy removes
#     is on the record for every run.
#
# The score is the same normalized XEB as nn_qab.py's calc_stats,
#     XEB = (N * sum_b f_b p_b - 1) / (N * sum_b p_b^2 - 1),   N = 2^width,
# evaluated in factorized form:  N p = prod_k (N_k p_k),  N sum p^2 = prod_k (N_k sum p_k^2).
#
# With --full (small widths only), the script ALSO runs QrackBridge on the
# unpatched circuit (the same draw, cut couplers included) and scores it
# against the full 2^width ideal, so you can show the proxy tracks the
# full-circuit XEB wherever both are computable.
#
# usage: python3 nn_bridge_orthogonal_proxy.py [width] [depth] [patches=2] [max_bonds=6] [proxy_patches=2] [--full] [--born] [--defer-bits=4] [--seed=N]
#
# --seed=N fixes the circuit draw, so A/B runs compare the same circuit.
# --defer-bits=N: per-layer deferred truncation, with each deferred block capped at N bond qubits of
# exact growth per cut (see run_layers_bridge). --defer-bits=0 restores eager, per-gate truncation.

import ctypes
import math
import random
import sys
import time

from collections import Counter
from importlib.metadata import version

import numpy as np
from pyqrack import QrackSimulator, QrackBridge
from pyqrack.qrack_system import Qrack


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (row_len, col_len)


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
    sim.mcz([q1], q2)
    sim.iswap(q1, q2)


def nswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.adjiswap(q1, q2)


def run_circuit(sim, circ):
    for g in circ:
        g[0](sim, *g[1:])


# ---------------------------------------------------------------------------
# Circuit: identical generator to nn_qab.py
# ---------------------------------------------------------------------------

def random_circuit(width, depth):
    lcv_range = range(width)
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    qc = []
    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(2))
            # Keep it Haar-random towards the poles:
            th = math.acos(random.uniform(-1, 1))
            qc.append((u, i, th, ph, lm))

        # Nearest-neighbor couplers:
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
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
                qc.append((g, b1, b2))

    return qc


# ---------------------------------------------------------------------------
# QrackBridge's own partition (mirrors QrackBridge._partition)
# ---------------------------------------------------------------------------

def bridge_patch_of(width, patches):
    base, rem = divmod(width, patches)
    sizes = [base + (1 if p >= patches - rem else 0) for p in range(patches)]
    owner = []
    for p, s in enumerate(sizes):
        owner += [p] * s
    return owner, sizes


def bridge_adjacent(pa, pb, patches, ring):
    if pa == pb or patches == 2:
        return True
    if abs(pa - pb) == 1:
        return True
    return ring and {pa, pb} == {0, patches - 1}


def check_bridge_layout(qc, width, patches, ring):
    """Every coupler must be in-patch or between chain (ring) neighbours, or the
    bridge is being asked for something its 1D layout does not provide."""
    owner, _ = bridge_patch_of(width, patches)
    for g in qc:
        if g[0] is u:
            continue
        pa, pb = owner[g[1]], owner[g[2]]
        if not bridge_adjacent(pa, pb, patches, ring):
            raise RuntimeError(
                f"coupler ({g[1]}, {g[2]}) joins bridge patches {pa} and {pb}, which are not chain neighbours. "
                "Use fewer patches, so each patch spans at least one full column of the grid "
                "(and 3+ patches wrap into a ring automatically)."
            )


def count_seam_couplers(qc, width, patches):
    owner, _ = bridge_patch_of(width, patches)
    return sum(1 for g in qc if g[0] is not u and owner[g[1]] != owner[g[2]])


# ---------------------------------------------------------------------------
# Per-layer deferred execution on QrackBridge
# ---------------------------------------------------------------------------
#
# Each layer of the generator is a run of single-qubit gates followed by
# couplers on DISJOINT qubit pairs, so a layer's couplers all commute: running
# the in-patch couplers first and the cross-patch ("bridged") couplers last is
# the identical circuit. The bridged couplers then run inside QrackBridge's
# deferred() block, which keeps patch tensors on the host and truncates once
# at the end of the block, instead of a full engine round trip (and truncation)
# after every bridged gate.
#
# Inside a deferred block, QBridge recompresses bonds only EXACTLY, so bonds
# grow with every bridged gate until the block closes. To bound memory, a
# layer's bridged couplers are split into consecutive blocks whose worst-case
# exact growth stays within `defer_bits` bond qubits on any one cut (a block
# always takes at least one gate). Upper bounds per gate, from how QBridge
# splits it: a single-control gate adds 1 bond qubit to the cut it crosses,
# swap/iswap/adjiswap add 2, and the composites pswap/mswap/nswap (controlled
# Z, then a swap-family gate) add 3. Peak bonds are therefore at most
# max_bonds + max(defer_bits, 3) per cut (logged as peak_bonds).
#
# defer_bits = 0 disables deferral: every gate runs in generator order, with
# QrackBridge's default eager truncation (the original behavior), for A/B runs.

GATE_BOND_BITS = {
    cx: 1, cy: 1, cz: 1, acx: 1, acy: 1, acz: 1,
    swap: 2, iswap: 2, iiswap: 2,
    pswap: 3, mswap: 3, nswap: 3,
}


def split_layers(circ):
    """Recover layers from a flat circuit: a new layer starts wherever a
    single-qubit gate follows a coupler (true of both the full and proxy circuits)."""
    layers = []
    prev_u = False
    for g in circ:
        is_u = g[0] is u
        if is_u and not prev_u:
            layers.append([])
        layers[-1].append(g)
        prev_u = is_u
    return layers


def cut_of(pa, pb, patches, ring):
    """Index of the bridge cut between adjacent patches pa != pb."""
    if patches == 2:
        return 0
    if ring and {pa, pb} == {0, patches - 1}:
        return patches - 1
    return min(pa, pb)


def run_layers_bridge(sim, circ, owner, patches, ring, defer_bits):
    if defer_bits <= 0:
        run_circuit(sim, circ)
        return 0
    blocks = 0
    for layer in split_layers(circ):
        local = [g for g in layer if (g[0] is u) or (owner[g[1]] == owner[g[2]])]
        bridged = [g for g in layer if (g[0] is not u) and (owner[g[1]] != owner[g[2]])]
        run_circuit(sim, local)
        block, cost = [], {}
        for g in bridged:
            c = cut_of(owner[g[1]], owner[g[2]], patches, ring)
            b = GATE_BOND_BITS[g[0]]
            if block and (cost.get(c, 0) + b) > defer_bits:
                with sim.deferred():
                    run_circuit(sim, block)
                blocks += 1
                block, cost = [], {}
            block.append(g)
            cost[c] = cost.get(c, 0) + b
        if block:
            with sim.deferred():
                run_circuit(sim, block)
            blocks += 1
    return blocks


# ---------------------------------------------------------------------------
# Orthogonal cut: bands of the LONG coordinate ("row")
# ---------------------------------------------------------------------------

def orthogonal_bands(width, k):
    """Split the long grid dimension into k contiguous bands of rows.
    Returns [sorted list of qubit indices in band k]."""
    row_len, col_len = factor_width(width)
    if k < 2:
        raise RuntimeError("proxy_patches must be at least 2.")
    if k > row_len:
        raise RuntimeError(
            f"width {width} factors to a {col_len} x {row_len} grid; "
            f"at most {row_len} orthogonal proxy patches are possible."
        )
    bounds = [(i * row_len) // k for i in range(k + 1)]
    bands = []
    for i in range(k):
        rows = range(bounds[i], bounds[i + 1])
        bands.append(sorted(col * row_len + row for col in range(col_len) for row in rows))
    return bands


def make_proxy(qc, bands):
    """Drop every coupler that crosses a band boundary (including the torus
    wrap); return (proxy circuit, [gates for each band, locally indexed], #removed)."""
    band_of = {}
    local = {}
    for i, qs in enumerate(bands):
        for j, q in enumerate(qs):
            band_of[q] = i
            local[q] = j

    proxy = []
    per_band = [[] for _ in bands]
    removed = 0
    for g in qc:
        if g[0] is u:
            i = band_of[g[1]]
            proxy.append(g)
            per_band[i].append((g[0], local[g[1]]) + tuple(g[2:]))
            continue
        i, j = band_of[g[1]], band_of[g[2]]
        if i != j:
            removed += 1
            continue
        proxy.append(g)
        per_band[i].append((g[0], local[g[1]], local[g[2]]))
    return proxy, per_band, removed


# ---------------------------------------------------------------------------
# Exact ideal of one band, straight into a numpy buffer
# ---------------------------------------------------------------------------

def ideal_probs(n, gates):
    sim = QrackSimulator(n)
    run_circuit(sim, gates)
    dt, ct = (np.float32, ctypes.c_float) if Qrack.fppow < 6 else (np.float64, ctypes.c_double)
    probs = np.empty(1 << n, dtype=dt)
    Qrack.qrack_lib.OutProbs(sim.sid, probs.ctypes.data_as(ctypes.POINTER(ct)))
    sim._throw_if_error()
    del sim
    return probs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats(ideal_probs, counts, shots):
    """nn_qab.py's estimator, on a full 2^n ideal (used only with --full)."""
    ideal = np.asarray(ideal_probs, dtype=np.float64)
    threshold = np.median(ideal)
    u_u = ideal.mean()
    ideal_centered = ideal - u_u
    denom = np.dot(ideal_centered, ideal_centered)
    keys = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    patch = np.fromiter(counts.values(), dtype=np.float64, count=len(counts)) / shots
    numer = np.dot(ideal_centered[keys], patch) - u_u * ideal_centered.sum()
    hog_prob = patch[ideal[keys] > threshold].sum()

    return numer / denom, hog_prob


def gather_bits(keys, qubits):
    """Local index of each sample within one band: bit j <- bit qubits[j]."""
    x = np.zeros(len(keys), dtype=np.uint64)
    one = np.uint64(1)
    for j, q in enumerate(qubits):
        x |= ((keys >> np.uint64(q)) & one) << np.uint64(j)
    return x.astype(np.int64)


def sum_of_squares(p, chunk=1 << 22):
    """sum(p^2), accumulated in float64 over fixed-size chunks: BLAS speed with
    no full-size float64 copy (the 2^n buffer may be float32)."""
    s = 0.0
    for i in range(0, p.size, chunk):
        c = p[i:i + chunk].astype(np.float64)
        s += np.dot(c, c)
    return s


def calc_stats_factorized(bands, per_band, counts, shots, median_draws=1 << 16):
    """The same estimator as calc_stats, for p(x) = prod_k p_k(x_k).

    Streams the bands: each band's exact 2^n_k distribution is computed, reduced
    to the three quantities the estimator needs (N_k sum p_k^2, N_k p_k at the
    sampled keys, N_k p_k at the HOG median draws), and freed before the next
    band is simulated. Peak memory is one band, not all of them.

    XEB is exact. HOG uses the median of the product distribution, which is
    estimated from `median_draws` uniformly random bitstrings (the median of
    2^width values is not computed exactly), so HOG is approximate.
    """
    keys = np.fromiter(counts.keys(), dtype=np.uint64, count=len(counts))
    freq = np.fromiter(counts.values(), dtype=np.float64, count=len(counts)) / shots

    sum_sq = 1.0
    scaled = np.ones(len(keys))
    rng = np.random.default_rng(12345)
    draws = np.ones(median_draws)
    for qubits, gates in zip(bands, per_band):
        p = ideal_probs(len(qubits), gates)
        n_pow = float(p.size)
        sum_sq *= n_pow * sum_of_squares(p)
        scaled *= n_pow * p[gather_bits(keys, qubits)].astype(np.float64)
        draws *= n_pow * p[rng.integers(0, p.size, median_draws)].astype(np.float64)
        del p

    xeb = (np.dot(freq, scaled) - 1.0) / (sum_sq - 1.0)
    hog_approx = freq[scaled > np.median(draws)].sum()

    return xeb, hog_approx


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bridge_counts(width, patches, max_bonds, truncation, circ, shots, defer_bits):
    ring = patches >= 3
    check_bridge_layout(circ, width, patches, ring)
    owner, _ = bridge_patch_of(width, patches)
    sim = QrackBridge(width, max_bonds=max_bonds, patches=patches, wrap=ring, truncation=truncation)
    blocks = run_layers_bridge(sim, circ, owner, patches, ring, defer_bits)
    counts = dict(Counter(sim.measure_shots(list(range(width)), shots)))
    # Fairness-critical settings and run statistics: logged, so every result
    # carries the exact configuration that produced it.
    _, sizes = bridge_patch_of(width, patches)
    stats = {
        "pyqrack": version("pyqrack"),
        "bridge_class": type(sim).__name__,
        "patches": patches,
        "patch_qubits": sizes,
        "wrap": ring,
        "max_bonds": max_bonds,
        "truncation": truncation,
        "defer_bits": defer_bits,
        "deferred_blocks": blocks,
        "fidelity_estimate": float(sim.fidelity_estimate),
        "truncations": int(sim.truncations),
        "peak_bonds": int(sim.peak_bonds),
        "peak_patch_qubits": int(sim.peak_patch_qubits),
    }
    del sim
    return counts, stats


def bench(width, depth, patches=2, max_bonds=6, k=2, full=False, truncation="greedy", defer_bits=4):
    if not 2 <= patches <= width:
        raise RuntimeError("need 2 <= patches <= width.")
    shots = 1 << min(13, width + 2)
    row_len, col_len = factor_width(width)
    bands = orthogonal_bands(width, k)

    t0 = time.perf_counter()
    qc = random_circuit(width, depth)
    proxy, per_band, removed = make_proxy(qc, bands)
    seam_full = count_seam_couplers(qc, width, patches)
    seam_proxy = count_seam_couplers(proxy, width, patches)
    t1 = time.perf_counter()
    print(f"circuit_seconds: {t1 - t0:.4f}  (grid {col_len} x {row_len}; {k} proxy patches of "
          f"{' + '.join(str(len(b)) for b in bands)} qubits; {removed} of {len(qc) - width * depth} couplers cut; "
          f"bridge-seam couplers {seam_full} full -> {seam_proxy} proxy)", flush=True)

    counts, stats = bridge_counts(width, patches, max_bonds, truncation, proxy, shots, defer_bits)
    t2 = time.perf_counter()
    print(f"bridge_proxy_seconds: {t2 - t1:.4f}", flush=True)
    for key, v in stats.items():
        print(f"  {key}: {v}")

    xeb_proxy, hog_proxy = calc_stats_factorized(bands, per_band, counts, shots)
    t3 = time.perf_counter()
    print(f"ideal_patches_and_stats_seconds: {t3 - t2:.4f}", flush=True)

    result = {
        "width": width,
        "depth": depth,
        "patches": patches,
        "max_bonds": max_bonds,
        "truncation": truncation,
        "defer_bits": defer_bits,
        "proxy_patches": k,
        "proxy_patch_qubits": [len(b) for b in bands],
        "couplers_cut": removed,
        "bridge_seam_couplers_full": seam_full,
        "bridge_seam_couplers_proxy": seam_proxy,
        "fidelity_estimate_proxy": stats["fidelity_estimate"],
        "xeb_bridge_proxy": float(xeb_proxy),
        "hog_bridge_proxy_approx": float(hog_proxy),
    }

    if full:
        counts_full, stats_full = bridge_counts(width, patches, max_bonds, truncation, qc, shots, defer_bits)
        t4 = time.perf_counter()
        print(f"bridge_full_seconds: {t4 - t3:.4f}", flush=True)
        sim_ideal = QrackSimulator(width)
        run_circuit(sim_ideal, qc)
        probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
        del sim_ideal
        xeb_full, hog_full = calc_stats(probs, counts_full, shots)
        print(f"ideal_full_seconds: {time.perf_counter() - t4:.4f}", flush=True)
        result["fidelity_estimate_full"] = stats_full["fidelity_estimate"]
        result["xeb_bridge_full"] = float(xeb_full)
        result["hog_bridge_full"] = float(hog_full)

    return result


def main():
    argv = sys.argv[1:]
    defer_bits = 0
    for a in argv:
        if a.startswith("--defer-bits="):
            defer_bits = int(a.split("=", 1)[1])
        elif a.startswith("--seed="):
            # Same seed, same circuit draw: for A/B runs (e.g. --defer-bits=0 vs 4).
            random.seed(int(a.split("=", 1)[1]))
    flags = {"--full", "--born"}
    args = [a for a in argv if (a not in flags) and not a.startswith(("--defer-bits=", "--seed="))]
    full = "--full" in argv
    truncation = "born" if "--born" in argv else "greedy"
    if len(args) < 2:
        raise RuntimeError(
            "Usage: python3 nn_bridge_orthogonal_proxy.py [width] [depth] "
            "[patches=2] [max_bonds=6] [proxy_patches=2] [--full] [--born] [--defer-bits=4] [--seed=N]"
        )
    width = int(args[0])
    depth = int(args[1])
    patches = int(args[2]) if len(args) > 2 else 2
    max_bonds = int(args[3]) if len(args) > 3 else 6
    k = int(args[4]) if len(args) > 4 else 2
    result = bench(width, depth, patches, max_bonds, k, full, truncation, defer_bits)
    for key, v in result.items():
        print(f"  {key}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
