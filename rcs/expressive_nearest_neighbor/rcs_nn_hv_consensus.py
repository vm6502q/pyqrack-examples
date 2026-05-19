# How good are Google's own "patch circuits" as a direct XEB approximation?
#
# This script runs the same nearest-neighbor random circuit TWICE, with
# genuinely orthogonal patch boundaries:
#   - "horizontal" patch: top rows vs bottom rows (contiguous qubit sets)
#   - "vertical"   patch: left cols vs right cols (non-contiguous qubit sets)
#
# XEB is computed BETWEEN the two patched simulators. Neither is a ground-truth
# full-state simulator — both are classical patch approximations of the same
# circuit with gate shadows at their respective boundaries.
#
# Memory note: for a grid of row_len x col_len qubits, the vertical patch
# splits into (col_len//2) and (col_len - col_len//2) columns per row.
# The larger patch has row_len * ceil(col_len/2) qubits. out_probs() on
# that patch requires 8 * 2^(patch_size) bytes. For n=54 (6x9) this is
# ~8GB for the 30-qubit patch. Prefer widths where max patch <= 24 qubits
# (e.g. n=42 for 6x7, n=35 for 5x7, n=30 for 5x6).
#
# By (Anthropic) Claude, working from Dan Strano's scripts and prompting.

import math
import random
import sys

import numpy as np

from pyqrack import QrackSimulator


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def factor_width(width):
    row_len = math.floor(math.sqrt(width))
    while ((width // row_len) * row_len) != width:
        row_len -= 1
    col_len = width // row_len
    if row_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")
    return row_len, col_len


def make_patches(width, row_len, col_len, axis):
    """
    Return (patch, local_idx) arrays of length `width`.
      patch[i]     : 0 or 1, which sub-simulator qubit i belongs to
      local_idx[i] : index of qubit i within its sub-simulator

    axis == 'horizontal': split by row  (contiguous — top rows / bottom rows)
    axis == 'vertical'  : split by col  (non-contiguous — left cols / right cols)
    """
    patch     = np.empty(width, dtype=np.int32)
    local_idx = np.empty(width, dtype=np.int32)
    if axis == 'horizontal':
        cut = (row_len >> 1) * col_len
        ctr = [0, 0]
        for i in range(width):
            p = 0 if i < cut else 1
            patch[i] = p; local_idx[i] = ctr[p]; ctr[p] += 1
    else:
        v_cols = col_len >> 1
        ctr = [0, 0]
        for i in range(width):
            p = 0 if (i % col_len) < v_cols else 1
            patch[i] = p; local_idx[i] = ctr[p]; ctr[p] += 1
    return patch, local_idx


def get_quadrants(width, patch_h, local_h, patch_v, local_v):
    """
    Return four lists of (h_local_bit, v_local_bit) pairs, one per quadrant:
      q00: H-patch0 ∩ V-patch0
      q01: H-patch0 ∩ V-patch1
      q10: H-patch1 ∩ V-patch0
      q11: H-patch1 ∩ V-patch1
    """
    q = [[], [], [], []]
    for i in range(width):
        q[int(patch_h[i]) * 2 + int(patch_v[i])].append(
            (int(local_h[i]), int(local_v[i])))
    return q[0], q[1], q[2], q[3]


# ---------------------------------------------------------------------------
# Gate shadow machinery — generalised to arbitrary patch membership
# ---------------------------------------------------------------------------

def _prob(sim, q, patch, local_idx):
    return sim[patch[q]].prob(local_idx[q])

def _x(sim, q, p, l):    sim[p[q]].x(l[q])
def _z(sim, q, p, l):    sim[p[q]].z(l[q])
def _h(sim, q, p, l):    sim[p[q]].h(l[q])
def _s(sim, q, p, l):    sim[p[q]].s(l[q])
def _adjs(sim, q, p, l): sim[p[q]].adjs(l[q])


def ct_pair_prob(sim, q1, q2, p, l):
    p1 = sim[p[q1]].prob(l[q1])
    p2 = sim[p[q2]].prob(l[q2])
    return (p2, q1) if p1 < p2 else (p1, q2)


def cz_shadow(sim, q1, q2, p, l, anti=False):
    if anti: _x(sim, q1, p, l)
    prob_max, t = ct_pair_prob(sim, q1, q2, p, l)
    if prob_max > 0.5: _z(sim, t, p, l)
    if anti: _x(sim, q1, p, l)


def cx_shadow(sim, c, t, p, l, anti=False):
    _h(sim, t, p, l); cz_shadow(sim, c, t, p, l, anti); _h(sim, t, p, l)


def cy_shadow(sim, c, t, p, l, anti=False):
    _adjs(sim, t, p, l); cx_shadow(sim, c, t, p, l, anti); _s(sim, t, p, l)


def swap_shadow(sim, q1, q2, p, l):
    cx_shadow(sim, q1, q2, p, l)
    cx_shadow(sim, q2, q1, p, l)
    cx_shadow(sim, q1, q2, p, l)


def _same(q1, q2, p): return p[q1] == p[q2]
def _lq(q, l):        return int(l[q])
def _pp(q, p):        return int(p[q])


def cx(sim, q1, q2, p, l):
    if not _same(q1,q2,p): cx_shadow(sim,q1,q2,p,l)
    else: sim[_pp(q1,p)].mcx([_lq(q1,l)],_lq(q2,l))

def cy(sim, q1, q2, p, l):
    if not _same(q1,q2,p): cy_shadow(sim,q1,q2,p,l)
    else: sim[_pp(q1,p)].mcy([_lq(q1,l)],_lq(q2,l))

def cz(sim, q1, q2, p, l):
    if not _same(q1,q2,p): cz_shadow(sim,q1,q2,p,l)
    else: sim[_pp(q1,p)].mcz([_lq(q1,l)],_lq(q2,l))

def acx(sim, q1, q2, p, l):
    if not _same(q1,q2,p): cx_shadow(sim,q1,q2,p,l,True)
    else: sim[_pp(q1,p)].macx([_lq(q1,l)],_lq(q2,l))

def acy(sim, q1, q2, p, l):
    if not _same(q1,q2,p): cy_shadow(sim,q1,q2,p,l,True)
    else: sim[_pp(q1,p)].macy([_lq(q1,l)],_lq(q2,l))

def acz(sim, q1, q2, p, l):
    if not _same(q1,q2,p): cz_shadow(sim,q1,q2,p,l,True)
    else: sim[_pp(q1,p)].macz([_lq(q1,l)],_lq(q2,l))

def swap(sim, q1, q2, p, l):
    if not _same(q1,q2,p): swap_shadow(sim,q1,q2,p,l)
    else: sim[_pp(q1,p)].swap(_lq(q1,l),_lq(q2,l))

def iswap(sim, q1, q2, p, l):
    if not _same(q1,q2,p):
        swap_shadow(sim,q1,q2,p,l); cz_shadow(sim,q1,q2,p,l)
        _s(sim,q1,p,l); _s(sim,q2,p,l)
    else: sim[_pp(q1,p)].iswap(_lq(q1,l),_lq(q2,l))

def iiswap(sim, q1, q2, p, l):
    if not _same(q1,q2,p):
        _s(sim,q1,p,l); _s(sim,q2,p,l)
        cz_shadow(sim,q1,q2,p,l); swap_shadow(sim,q1,q2,p,l)
    else: sim[_pp(q1,p)].adjiswap(_lq(q1,l),_lq(q2,l))

def pswap(sim, q1, q2, p, l):
    if not _same(q1,q2,p):
        cz_shadow(sim,q1,q2,p,l); swap_shadow(sim,q1,q2,p,l)
    else:
        pp=_pp(q1,p); l1=_lq(q1,l); l2=_lq(q2,l)
        sim[pp].mcz([l1],l2); sim[pp].swap(l1,l2)

def mswap(sim, q1, q2, p, l):
    if not _same(q1,q2,p):
        swap_shadow(sim,q1,q2,p,l); cz_shadow(sim,q1,q2,p,l)
    else:
        pp=_pp(q1,p); l1=_lq(q1,l); l2=_lq(q2,l)
        sim[pp].swap(l1,l2); sim[pp].mcz([l1],l2)

def nswap(sim, q1, q2, p, l):
    if not _same(q1,q2,p):
        cz_shadow(sim,q1,q2,p,l); swap_shadow(sim,q1,q2,p,l); cz_shadow(sim,q1,q2,p,l)
    else:
        pp=_pp(q1,p); l1=_lq(q1,l); l2=_lq(q2,l)
        sim[pp].mcz([l1],l2); sim[pp].swap(l1,l2); sim[pp].mcz([l1],l2)


TWO_BIT_GATES = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz


# Single-simulator gate wrappers for the ideal (non-patched) circuit
def _cx_i(sim, q1, q2):   sim.mcx([q1], q2)
def _cy_i(sim, q1, q2):   sim.mcy([q1], q2)
def _cz_i(sim, q1, q2):   sim.mcz([q1], q2)
def _acx_i(sim, q1, q2):  sim.macx([q1], q2)
def _acy_i(sim, q1, q2):  sim.macy([q1], q2)
def _acz_i(sim, q1, q2):  sim.macz([q1], q2)
def _sw_i(sim, q1, q2):   sim.swap(q1, q2)
def _isw_i(sim, q1, q2):  sim.iswap(q1, q2)
def _iisw_i(sim, q1, q2): sim.adjiswap(q1, q2)
def _psw_i(sim, q1, q2):  sim.mcz([q1], q2); sim.swap(q1, q2)
def _msw_i(sim, q1, q2):  sim.swap(q1, q2);  sim.mcz([q1], q2)
def _nsw_i(sim, q1, q2):  sim.mcz([q1], q2); sim.swap(q1, q2); sim.mcz([q1], q2)
TWO_BIT_GATES_IDEAL = _sw_i, _psw_i, _msw_i, _nsw_i, _isw_i, _iisw_i, \
                      _cx_i, _cy_i, _cz_i, _acx_i, _acy_i, _acz_i


# ---------------------------------------------------------------------------
# Circuit runner
# ---------------------------------------------------------------------------

def run_circuit(width, depth, row_len, col_len, patch, local_idx, rng_state):
    random.setstate(rng_state)
    n0 = int(np.sum(patch == 0))
    n1 = int(np.sum(patch == 1))
    sim = [QrackSimulator(n0), QrackSimulator(n1)]
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]

    for _ in range(depth):
        for i in range(width):
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            sim[patch[i]].u(local_idx[i], th, ph, lm)

        gate = gateSequence.pop(0)
        gateSequence.append(gate)

        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row + (1 if (gate & 2) else -1)
                temp_col = col + (1 if (gate & 1) else 0)
                if temp_row < 0:        temp_row += row_len
                if temp_col < 0:        temp_col += col_len
                if temp_row >= row_len: temp_row -= row_len
                if temp_col >= col_len: temp_col -= col_len
                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col
                if (b1 == b2) or (b1 >= width) or (b2 >= width):
                    continue
                g = random.choice(TWO_BIT_GATES)
                g(sim, b1, b2, patch, local_idx)

    return [sim[0].out_probs(), sim[1].out_probs()]


def run_circuit_ideal(width, depth, row_len, col_len, rng_state):
    """Full statevector simulation — no patch boundary, ground truth."""
    random.setstate(rng_state)
    sim = QrackSimulator(width)
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]

    for _ in range(depth):
        for i in range(width):
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            sim.u(i, th, ph, lm)

        gate = gateSequence.pop(0)
        gateSequence.append(gate)

        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row + (1 if (gate & 2) else -1)
                temp_col = col + (1 if (gate & 1) else 0)
                if temp_row < 0:        temp_row += row_len
                if temp_col < 0:        temp_col += col_len
                if temp_row >= row_len: temp_row -= row_len
                if temp_col >= col_len: temp_col -= col_len
                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col
                if (b1 == b2) or (b1 >= width) or (b2 >= width):
                    continue
                g = random.choice(TWO_BIT_GATES_IDEAL)
                g(sim, b1, b2)

    return np.asarray(sim.out_probs(), dtype=np.float64)


# ---------------------------------------------------------------------------
# XEB — exact via quadrant tensor contraction
# ---------------------------------------------------------------------------

def _build_tensor_a(arr, qa, qb):
    """
    T[k_qa, k_qb] where k_qa indexes qa-bits of the distribution index (a_local).
    Vectorised using np.add.at.
    """
    na, nb = len(qa), len(qb)
    ia = np.arange(len(arr), dtype=np.int64)
    ka = np.zeros(len(arr), dtype=np.int64)
    kb = np.zeros(len(arr), dtype=np.int64)
    for k, (al, bl) in enumerate(qa): ka |= ((ia >> al) & 1) << k
    for k, (al, bl) in enumerate(qb): kb |= ((ia >> al) & 1) << k
    T = np.zeros((1 << na, 1 << nb), dtype=np.float64)
    np.add.at(T, (ka, kb), arr)
    return T


def _build_tensor_b(arr, qa, qb):
    """
    U[k_qa, k_qb] where indices come from b_local bits (second element of pairs).
    """
    na, nb = len(qa), len(qb)
    jc = np.arange(len(arr), dtype=np.int64)
    ka = np.zeros(len(arr), dtype=np.int64)
    kb = np.zeros(len(arr), dtype=np.int64)
    for k, (al, bl) in enumerate(qa): ka |= ((jc >> bl) & 1) << k
    for k, (al, bl) in enumerate(qb): kb |= ((jc >> bl) & 1) << k
    U = np.zeros((1 << na, 1 << nb), dtype=np.float64)
    np.add.at(U, (ka, kb), arr)
    return U


def calc_xeb(probs_a, probs_b, n, q00, q01, q10, q11, depth, label_a, label_b):
    """
    Exact XEB via quadrant tensor contraction.

    cross = sum_i P_A(i)*P_B(i)
          = einsum('ij,kl,ik,jl', T, S, U, V)
    where:
      T[k00,k01] = a0 marginalised to quadrant bits
      S[k10,k11] = a1 marginalised to quadrant bits
      U[k00,k10] = b0 marginalised to quadrant bits
      V[k01,k11] = b1 marginalised to quadrant bits

    Contraction path (optimal for equal quadrant sizes):
      Step 1: IL[k00,k11] = sum_{k01} T[k00,k01] * V[k01,k11]  (T @ V)
      Step 2: IK[k00,k10] = sum_{k11} IL[k00,k11] * S[k10,k11].T  (IL @ S.T)
      Step 3: cross = sum_{k00,k10} IK[k00,k10] * U[k00,k10]
    """
    u_u = 1.0 / (1 << n)

    a0 = np.asarray(probs_a[0], dtype=np.float64)
    a1 = np.asarray(probs_a[1], dtype=np.float64)
    b0 = np.asarray(probs_b[0], dtype=np.float64)
    b1 = np.asarray(probs_b[1], dtype=np.float64)

    # denom = dot(a0,a0)*dot(a1,a1) - 2u + u^2*2^n
    denom = (float(np.dot(a0, a0)) * float(np.dot(a1, a1))
             - 2.0 * u_u + u_u * u_u * (1 << n))

    # Build quadrant tensors
    T = _build_tensor_a(a0, q00, q01)   # [k00, k01]
    S = _build_tensor_a(a1, q10, q11)   # [k10, k11]
    U = _build_tensor_b(b0, q00, q10)   # [k00, k10]
    V = _build_tensor_b(b1, q01, q11)   # [k01, k11]

    # Exact contraction: einsum('ij,kl,ik,jl', T, S, U, V)
    # Step 1: IL = T @ V  [k00, k11]
    IL = T @ V
    # Step 2: IK = IL @ S.T  [k00, k10]
    IK = IL @ S.T
    # Step 3: cross = sum(IK * U)
    cross = float(np.sum(IK * U))

    numer = cross - 2.0 * u_u + u_u * u_u * (1 << n)
    xeb   = numer / denom if denom != 0.0 else float("nan")

    return {
        "control":    label_a,
        "experiment": label_b,
        "qubits":     n,
        "depth":      depth,
        "xeb":        float(xeb),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth):
    row_len, col_len = factor_width(width)

    patch_h, local_h = make_patches(width, row_len, col_len, 'horizontal')
    patch_v, local_v = make_patches(width, row_len, col_len, 'vertical')

    n0_h = int(np.sum(patch_h == 0)); n1_h = int(np.sum(patch_h == 1))
    n0_v = int(np.sum(patch_v == 0)); n1_v = int(np.sum(patch_v == 1))
    max_patch = max(n0_h, n1_h, n0_v, n1_v)

    print(f"Grid: {row_len} x {col_len}")
    print(f"Horizontal split: {n0_h} + {n1_h} qubits (top/bottom rows)")
    print(f"Vertical   split: {n0_v} + {n1_v} qubits (left/right cols)")
    print(f"Largest patch: {max_patch} qubits ({2**max_patch * 8 / 1e6:.0f} MB for out_probs)")

    if np.array_equal(patch_h, patch_v):
        print("WARNING: horizontal and vertical splits are identical on this grid!")
        print("Suggest: n=42 (6x7), n=35 (5x7), or n=30 (5x6).")
        return

    if max_patch > 28:
        gb = 2**max_patch * 8 / 1e9
        print(f"WARNING: largest patch requires {gb:.1f} GB for out_probs(). "
              f"This will likely OOM. Suggest n<=42.")
        return

    q00_hv, q01_hv, q10_hv, q11_hv = get_quadrants(width, patch_h, local_h, patch_v, local_v)
    q00_vh, q01_vh, q10_vh, q11_vh = get_quadrants(width, patch_v, local_v, patch_h, local_h)

    rng_state = random.getstate()
    probs_h = run_circuit(width, depth, row_len, col_len, patch_h, local_h, rng_state)
    probs_v = run_circuit(width, depth, row_len, col_len, patch_v, local_v, rng_state)

    print(calc_xeb(probs_h, probs_v, width,
                   q00_hv, q01_hv, q10_hv, q11_hv,
                   depth, "horizontal", "vertical"))

    print(calc_xeb(probs_v, probs_h, width,
                   q00_vh, q01_vh, q10_vh, q11_vh,
                   depth, "vertical", "horizontal"))


# ---------------------------------------------------------------------------
# Consensus XEB — patch-agreement rank estimation
# ---------------------------------------------------------------------------

def _expand_probs(probs_pair, n, patch, local_idx):
    """
    Expand a pair of subsystem probability vectors back to the full 2^n space
    using the separability assumption: P(i) = p0[local_0(i)] * p1[local_1(i)].

    This is the same factorisation the XEB quadrant contraction uses, but
    materialised as a full 2^n vector for consensus scoring.
    Only feasible when n <= 28 or so.
    """
    p0 = np.asarray(probs_pair[0], dtype=np.float64)
    p1 = np.asarray(probs_pair[1], dtype=np.float64)
    n_pow = 1 << n

    # Build local index arrays for each patch
    idx0 = np.zeros(n_pow, dtype=np.int64)
    idx1 = np.zeros(n_pow, dtype=np.int64)
    ctr  = [0, 0]
    # Recompute local indices from patch array
    for q in range(n):
        bit_pos = np.arange(n_pow, dtype=np.int64)
        bit_q   = (bit_pos >> q) & 1
        p_q     = int(patch[q])
        l_q     = int(local_idx[q])
        if p_q == 0:
            idx0 |= bit_q << l_q
        else:
            idx1 |= bit_q << l_q

    return p0[idx0] * p1[idx1]


def calc_xeb_consensus(probs_h, probs_v, n, patch_h, local_h, patch_v, local_v,
                       q00_hv, q01_hv, q10_hv, q11_hv,
                       q00_vh, q01_vh, q10_vh, q11_vh,
                       depth):
    """
    Consensus XEB using patch-agreement rank estimation.

    For each bitstring i, estimate its probability as the geometric mean of
    the H-patch and V-patch separable approximations:
        P_consensus(i) = sqrt(P_H(i) * P_V(i))

    XEB is computed with:
      - "ideal" = one patch distribution (which sets the XEB denominator/threshold)
      - "experiment" = consensus probability
    averaged over both choices of which patch is "ideal".

    This replaces MPS amplitude queries with the inter-patch agreement signal:
    bitstrings where both orthogonal approximations assign high probability are
    the classical analog of "heavy outputs" — they are robust to the specific
    choice of patch boundary.
    """
    u_u   = 1.0 / (1 << n)
    n_pow = 1 << n

    a0_h = np.asarray(probs_h[0], dtype=np.float64)
    a1_h = np.asarray(probs_h[1], dtype=np.float64)
    a0_v = np.asarray(probs_v[0], dtype=np.float64)
    a1_v = np.asarray(probs_v[1], dtype=np.float64)

    # Expand both patch distributions to full 2^n probability vectors
    P_H = _expand_probs(probs_h, n, patch_h, local_h)
    P_V = _expand_probs(probs_v, n, patch_v, local_v)

    # Consensus: geometric mean of the two patch estimates
    # Clamp to avoid sqrt(0) issues at zero-probability entries
    P_cons = np.sqrt(np.maximum(P_H * P_V, 0.0))
    # Renormalise (geometric mean doesn't sum to 1 in general)
    cons_sum = P_cons.sum()
    if cons_sum > 0:
        P_cons /= cons_sum

    # XEB with H as ideal, consensus as experiment
    h_c    = P_H - u_u
    cons_c = P_cons - u_u
    v_c    = P_V - u_u

    denom_h  = float(np.dot(h_c, h_c))
    denom_v  = float(np.dot(v_c, v_c))

    xeb_h = float(np.dot(h_c, cons_c)) / denom_h if denom_h > 0 else float('nan')
    xeb_v = float(np.dot(v_c, cons_c)) / denom_v if denom_v > 0 else float('nan')

    # HOG probability: fraction of consensus weight on above-median-P_H outputs
    threshold_h = float(np.median(P_H))
    threshold_v = float(np.median(P_V))
    hog_h = float(P_cons[P_H > threshold_h].sum())
    hog_v = float(P_cons[P_V > threshold_v].sum())

    # Agreement score: cosine similarity between P_H and P_V in probability space
    # High = patches agree on rank; low = patches disagree
    agreement = float(np.dot(P_H, P_V) / (
        np.sqrt(np.dot(P_H, P_H)) * np.sqrt(np.dot(P_V, P_V)) + 1e-30))

    return {
        "qubits":      n,
        "depth":       depth,
        "xeb_h_ideal": xeb_h,
        "xeb_v_ideal": xeb_v,
        "xeb_mean":    (xeb_h + xeb_v) / 2.0,
        "hog_h":       hog_h,
        "hog_v":       hog_v,
        "agreement":   agreement,
    }


def bench_qrack_consensus(width, depth):
    """
    Run the H/V patch circuits and compute consensus XEB.
    Replaces MPS amplitude queries with inter-patch rank agreement.
    """
    row_len, col_len = factor_width(width)

    patch_h, local_h = make_patches(width, row_len, col_len, 'horizontal')
    patch_v, local_v = make_patches(width, row_len, col_len, 'vertical')

    n0_h = int(np.sum(patch_h == 0)); n1_h = int(np.sum(patch_h == 1))
    n0_v = int(np.sum(patch_v == 0)); n1_v = int(np.sum(patch_v == 1))
    max_patch = max(n0_h, n1_h, n0_v, n1_v)

    print(f"Grid: {row_len} x {col_len}")
    print(f"Horizontal split: {n0_h} + {n1_h} qubits (top/bottom rows)")
    print(f"Vertical   split: {n0_v} + {n1_v} qubits (left/right cols)")
    print(f"Largest patch: {max_patch} qubits ({2**max_patch * 8 / 1e6:.0f} MB for out_probs)")

    if np.array_equal(patch_h, patch_v):
        print("WARNING: horizontal and vertical splits are identical on this grid!")
        print("Suggest: n=42 (6x7), n=35 (5x7), or n=30 (5x6).")
        return

    if max_patch > 28:
        gb = 2**max_patch * 8 / 1e9
        print(f"WARNING: largest patch requires {gb:.1f} GB. Suggest n<=42.")
        return

    q00_hv, q01_hv, q10_hv, q11_hv = get_quadrants(
        width, patch_h, local_h, patch_v, local_v)
    q00_vh, q01_vh, q10_vh, q11_vh = get_quadrants(
        width, patch_v, local_v, patch_h, local_h)

    rng_state = random.getstate()
    probs_h = run_circuit(width, depth, row_len, col_len, patch_h, local_h, rng_state)
    probs_v = run_circuit(width, depth, row_len, col_len, patch_v, local_v, rng_state)
    ideal_probs = run_circuit_ideal(width, depth, row_len, col_len, rng_state)

    # Original patch-vs-patch XEB (kept for comparison)
    print("--- Patch vs patch XEB ---")
    print(calc_xeb(probs_h, probs_v, width,
                   q00_hv, q01_hv, q10_hv, q11_hv,
                   depth, "horizontal", "vertical"))
    print(calc_xeb(probs_v, probs_h, width,
                   q00_vh, q01_vh, q10_vh, q11_vh,
                   depth, "vertical", "horizontal"))

    # Consensus XEB (patch vs patch)
    print("--- Consensus XEB ---")
    consensus_result = calc_xeb_consensus(
        probs_h, probs_v, width,
        patch_h, local_h, patch_v, local_v,
        q00_hv, q01_hv, q10_hv, q11_hv,
        q00_vh, q01_vh, q10_vh, q11_vh,
        depth)
    print(consensus_result)

    # Consensus vs ideal — the honest comparison
    P_H = _expand_probs(probs_h, width, patch_h, local_h)
    P_V = _expand_probs(probs_v, width, patch_v, local_v)
    P_cons = np.sqrt(np.maximum(P_H * P_V, 0.0))
    cons_sum = P_cons.sum()
    if cons_sum > 0:
        P_cons /= cons_sum

    u_u      = 1.0 / (1 << width)
    ideal_c  = ideal_probs - u_u
    cons_c   = P_cons - u_u
    denom_i  = float(np.dot(ideal_c, ideal_c))
    xeb_ci   = float(np.dot(ideal_c, cons_c)) / denom_i if denom_i > 0 else float('nan')
    hog_ci   = float(P_cons[ideal_probs > float(np.median(ideal_probs))].sum())

    print("--- Consensus vs ideal ---")
    print({
        "qubits":              width,
        "depth":               depth,
        "xeb_consensus_ideal": xeb_ci,
        "hog_consensus_ideal": hog_ci,
    })


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 rcs_nn_qrack_validation_hv.py [width] [depth]\n"
            "Recommended widths: 42 (6x7), 35 (5x7), 30 (5x6)"
        )
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    bench_qrack_consensus(width, depth)
    return 0


if __name__ == "__main__":
    sys.exit(main())

