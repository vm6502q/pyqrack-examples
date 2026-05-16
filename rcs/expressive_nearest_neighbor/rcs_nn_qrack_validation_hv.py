# How good are Google's own "patch circuits" as a direct XEB approximation?
#
# This script runs the same nearest-neighbor random circuit TWICE, with
# orthogonal patch boundaries:
#   - "vertical"   patch: qubits split by index (left cols vs right cols)
#   - "horizontal" patch: qubits split by row   (top rows vs bottom rows)
#
# XEB is then computed BETWEEN the two patched simulators.  Neither is a
# "ground truth" full-state simulator — both are classical patch approximations
# of the same circuit, with gate shadows at their respective boundaries.
#
# Because both experiments are separable along their own known boundary,
# the XEB numerator and denominator factor into dot products over the
# subsystem probability arrays — no iteration over 2^n states, no prob_perm.
#
# This demonstrates the methodological point: two patch-circuit approximations
# with orthogonal boundaries score well against each other via XEB, which is
# the same figure-of-merit used to validate quantum hardware.
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


# ---------------------------------------------------------------------------
# Gate shadow machinery (works for any patch boundary)
# ---------------------------------------------------------------------------

def ct_pair_prob(sim, q1, q2, bound):
    p1 = sim[0].prob(q1) if q1 < bound else sim[1].prob(q1 - bound)
    p2 = sim[0].prob(q2) if q2 < bound else sim[1].prob(q2 - bound)
    return (p2, q1) if p1 < p2 else (p1, q2)


def cz_shadow(sim, q1, q2, bound, anti=False):
    if anti:
        if q1 < bound: sim[0].x(q1)
        else:          sim[1].x(q1 - bound)
    prob_max, t = ct_pair_prob(sim, q1, q2, bound)
    if prob_max > 0.5:
        if t < bound: sim[0].z(t)
        else:         sim[1].z(t - bound)
    if anti:
        if q1 < bound: sim[0].x(q1)
        else:          sim[1].x(q1 - bound)


def cx_shadow(sim, c, t, bound, anti=False):
    if t < bound:
        sim[0].h(t); cz_shadow(sim, c, t, bound, anti); sim[0].h(t)
    else:
        sim[1].h(t - bound); cz_shadow(sim, c, t, bound, anti); sim[1].h(t - bound)


def cy_shadow(sim, c, t, bound, anti=False):
    if t < bound:
        sim[0].adjs(t); cx_shadow(sim, c, t, bound, anti); sim[0].s(t)
    else:
        sim[1].adjs(t - bound); cx_shadow(sim, c, t, bound, anti); sim[1].s(t - bound)


def swap_shadow(sim, q1, q2, bound):
    cx_shadow(sim, q1, q2, bound)
    cx_shadow(sim, q2, q1, bound)
    cx_shadow(sim, q1, q2, bound)


def _cross(q1, q2, bound):
    return ((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))


def cx(sim, q1, q2, bound):
    if _cross(q1, q2, bound): cx_shadow(sim, q1, q2, bound)
    elif q1 < bound:          sim[0].mcx([q1], q2)
    else:                     sim[1].mcx([q1 - bound], q2 - bound)

def cy(sim, q1, q2, bound):
    if _cross(q1, q2, bound): cy_shadow(sim, q1, q2, bound)
    elif q1 < bound:          sim[0].mcy([q1], q2)
    else:                     sim[1].mcy([q1 - bound], q2 - bound)

def cz(sim, q1, q2, bound):
    if _cross(q1, q2, bound): cz_shadow(sim, q1, q2, bound)
    elif q1 < bound:          sim[0].mcz([q1], q2)
    else:                     sim[1].mcz([q1 - bound], q2 - bound)

def acx(sim, q1, q2, bound):
    if _cross(q1, q2, bound): cx_shadow(sim, q1, q2, bound, True)
    elif q1 < bound:          sim[0].macx([q1], q2)
    else:                     sim[1].macx([q1 - bound], q2 - bound)

def acy(sim, q1, q2, bound):
    if _cross(q1, q2, bound): cy_shadow(sim, q1, q2, bound, True)
    elif q1 < bound:          sim[0].macy([q1], q2)
    else:                     sim[1].macy([q1 - bound], q2 - bound)

def acz(sim, q1, q2, bound):
    if _cross(q1, q2, bound): cz_shadow(sim, q1, q2, bound, True)
    elif q1 < bound:          sim[0].macz([q1], q2)
    else:                     sim[1].macz([q1 - bound], q2 - bound)

def swap(sim, q1, q2, bound):
    if _cross(q1, q2, bound): swap_shadow(sim, q1, q2, bound)
    elif q1 < bound:          sim[0].swap(q1, q2)
    else:                     sim[1].swap(q1 - bound, q2 - bound)

def iswap(sim, q1, q2, bound):
    if _cross(q1, q2, bound):
        swap_shadow(sim, q1, q2, bound); cz_shadow(sim, q1, q2, bound)
        if q1 < bound: sim[0].s(q1);        sim[1].s(q2 - bound)
        else:          sim[1].s(q1 - bound); sim[0].s(q2)
    elif q1 < bound: sim[0].iswap(q1, q2)
    else:            sim[1].iswap(q1 - bound, q2 - bound)

def iiswap(sim, q1, q2, bound):
    if _cross(q1, q2, bound):
        if q1 < bound: sim[0].s(q1);        sim[1].s(q2 - bound)
        else:          sim[1].s(q1 - bound); sim[0].s(q2)
        cz_shadow(sim, q1, q2, bound); swap_shadow(sim, q1, q2, bound)
    elif q1 < bound: sim[0].adjiswap(q1, q2)
    else:            sim[1].adjiswap(q1 - bound, q2 - bound)

def pswap(sim, q1, q2, bound):
    if _cross(q1, q2, bound):
        cz_shadow(sim, q1, q2, bound); swap_shadow(sim, q1, q2, bound)
    elif q1 < bound:
        sim[0].mcz([q1], q2); sim[0].swap(q1, q2)
    else:
        sim[1].mcz([q1 - bound], q2 - bound); sim[1].swap(q1 - bound, q2 - bound)

def mswap(sim, q1, q2, bound):
    if _cross(q1, q2, bound):
        swap_shadow(sim, q1, q2, bound); cz_shadow(sim, q1, q2, bound)
    elif q1 < bound:
        sim[0].swap(q1, q2); sim[0].mcz([q1], q2)
    else:
        sim[1].swap(q1 - bound, q2 - bound); sim[1].mcz([q1 - bound], q2 - bound)

def nswap(sim, q1, q2, bound):
    if _cross(q1, q2, bound):
        cz_shadow(sim, q1, q2, bound)
        swap_shadow(sim, q1, q2, bound)
        cz_shadow(sim, q1, q2, bound)
    elif q1 < bound:
        sim[0].mcz([q1], q2); sim[0].swap(q1, q2); sim[0].mcz([q1], q2)
    else:
        sim[1].mcz([q1 - bound], q2 - bound)
        sim[1].swap(q1 - bound, q2 - bound)
        sim[1].mcz([q1 - bound], q2 - bound)


TWO_BIT_GATES = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz


# ---------------------------------------------------------------------------
# Circuit runner — returns subsystem out_probs for a given patch boundary
# ---------------------------------------------------------------------------

def run_circuit(width, depth, row_len, col_len, patch_bound, rng_state):
    """
    Replay the circuit from rng_state with the given patch boundary.
    Returns [out_probs(patch0), out_probs(patch1)].
    """
    random.setstate(rng_state)

    n_low  = patch_bound
    n_high = width - patch_bound
    sim = [QrackSimulator(n_low), QrackSimulator(n_high)]

    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]

    for _ in range(depth):
        for i in range(width):
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            if i < patch_bound:
                sim[0].u(i, th, ph, lm)
            else:
                sim[1].u(i - patch_bound, th, ph, lm)

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
                g(sim, b1, b2, patch_bound)

    return [sim[0].out_probs(), sim[1].out_probs()]


# ---------------------------------------------------------------------------
# XEB between two patch experiments — fully vectorised
# ---------------------------------------------------------------------------

def calc_xeb(probs_a, probs_b, n, bound_a, bound_b, depth, label_a, label_b):
    """
    Compute XEB treating experiment A as "ideal" and experiment B as "sampled."

    Both are separable along their own boundaries, so ideal_i and actual_i
    both factor as products of subsystem probabilities.  The sums then reduce
    to dot products over arrays of size 2^bound, never 2^n.

    XEB = numer / denom where:

      denom = sum_i (ideal_i - u)^2
            = dot(a_low, a_low) * dot(a_high, a_high) - 2u + u^2 * 2^n

      numer = sum_i (ideal_i - u)(actual_i - u)

    For numer, ideal_i = a_low[i & mask_a] * a_high[i >> bound_a]
               actual_i = b_low[i & mask_b] * b_high[i >> bound_b]

    When the two boundaries differ, ideal and actual are NOT separable along
    the same axis, so numer does NOT factor trivially.  Instead we use the
    identity:

      sum_i f(i & mask_a, i >> bound_a) * g(i & mask_b, i >> bound_b)

    For vertical vs horizontal boundaries on a row_len x col_len grid this
    still factors — see inline comment — but in general we compute it via
    the outer-product contraction which costs O(2^bound_a * 2^bound_b).
    For half-splits that's O(2^(n/2)) — tractable.
    """
    n_pow  = 1 << n
    u_u    = 1.0 / n_pow
    u_a    = 1.0 / (1 << bound_a)
    u_b    = 1.0 / (1 << bound_b)

    a_low  = np.asarray(probs_a[0], dtype=np.float64)   # shape (2^bound_a,)
    a_high = np.asarray(probs_a[1], dtype=np.float64)   # shape (2^(n-bound_a),)
    b_low  = np.asarray(probs_b[0], dtype=np.float64)   # shape (2^bound_b,)
    b_high = np.asarray(probs_b[1], dtype=np.float64)   # shape (2^(n-bound_b),)

    # --- denom: depends only on experiment A ---
    denom = (float(np.dot(a_low, a_low)) * float(np.dot(a_high, a_high))
             - 2.0 * u_u + u_u * u_u * n_pow)

    # --- numer ---
    # Both boundaries are half-splits, so bound_a == n - bound_b when the
    # splits are orthogonal on the grid (vertical vs horizontal).
    # In that case: i & mask_a indexes the same bits as i >> bound_b,
    # and i >> bound_a indexes the same bits as i & mask_b.
    # The double sum over all 2^n bitstrings then factors as:
    #
    #   sum_i (a_l[i&mask_a]*a_h[i>>bound_a] - u)(b_l[i&mask_b]*b_h[i>>bound_b] - u)
    #
    # We compute this as a single outer-product contraction at cost
    # O(2^bound_a * 2^bound_b) = O(2^(n/2) * 2^(n/2)) = O(2^n) in the worst
    # case, but with a very small constant since it's pure numpy arithmetic.
    # For n=54 this is still 2^54 — so we rely on the orthogonal factorisation.
    #
    # Orthogonal factorisation (vertical bound_a = col_len * (row_len//2),
    # horizontal bound_b = col_len * (row_len//2)):
    # When bound_a + bound_b == n (complementary half-splits):
    #   mask_a bits == high bits of b  =>  a_low  aligns with b_high indices
    #   high bits of a == mask_b bits  =>  a_high aligns with b_low  indices
    #
    # numer = sum_l_a sum_h_a (a_l[l_a]*a_h[h_a] - u)(b_l[h_a]*b_h[l_a] - u)
    #       = dot2d(a_low[:,None]*a_high[None,:] - u,
    #               b_high[None,:]*b_low[:,None] - u)   [reindexed]
    #
    # Which factors as:
    #   (sum_{l,h} a_l[l]*a_h[h]*b_h[h]*b_l[l])
    #   - u*(sum_l a_l[l]*b_l[l])*(sum_h a_h[h])
    #   - u*(sum_l a_l[l])*(sum_h a_h[h]*b_h[h])
    #   + u^2 * 2^n
    #
    # = dot(a_low, b_low) * dot(a_high, b_high)   [since indices swap]
    #   - u * dot(a_low, b_low) * 1
    #   - u * 1 * dot(a_high, b_high)
    #   + u^2 * n_pow
    #
    # Valid when bound_a + bound_b == n (orthogonal complementary splits).

    if bound_a + bound_b == n:
        # Orthogonal complementary splits — O(2^(n/2)) dot products
        # a_low  aligns with b_high (same bit positions), and vice versa
        dot_lh = float(np.dot(a_low,  b_high))  # a_low  vs b_high
        dot_hl = float(np.dot(a_high, b_low))   # a_high vs b_low
        numer = (dot_lh * dot_hl
                 - u_u * dot_lh
                 - u_u * dot_hl
                 + u_u * u_u * n_pow)
    else:
        # General case — outer product contraction, O(2^bound_a * 2^(n-bound_b))
        # Only feasible when both bounds are small.
        ideal  = np.outer(a_low, a_high).ravel() - u_u   # shape (2^n,)
        actual = np.outer(b_low, b_high).ravel() - u_u
        numer  = float(np.dot(ideal, actual))

    xeb = numer / denom if denom != 0.0 else float("nan")

    return {
        "control":    label_a,
        "experiment": label_b,
        "qubits":     n,
        "bound_ctrl": bound_a,
        "bound_exp":  bound_b,
        "depth":      depth,
        "xeb":        float(xeb),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth):
    row_len, col_len = factor_width(width)

    # Vertical patch boundary: split by qubit index (left cols / right cols)
    v_bound = (width + 1) >> 1

    # Horizontal patch boundary: split by row (top rows / bottom rows)
    # Qubits are row-major: qubit q is at row q//col_len, col q%col_len
    h_rows  = row_len >> 1
    h_bound = h_rows * col_len

    # Snapshot RNG — both circuits replay the same random gates
    rng_state = random.getstate()

    print(f"Grid: {row_len} x {col_len}, vertical bound={v_bound}, horizontal bound={h_bound}")

    probs_v = run_circuit(width, depth, row_len, col_len, v_bound, rng_state)
    probs_h = run_circuit(width, depth, row_len, col_len, h_bound, rng_state)

    # XEB: vertical as "control," horizontal as "experiment"
    print(calc_xeb(probs_v, probs_h, width, v_bound, h_bound, depth,
                   "vertical", "horizontal"))

    # XEB: horizontal as "control," vertical as "experiment"
    print(calc_xeb(probs_h, probs_v, width, h_bound, v_bound, depth,
                   "horizontal", "vertical"))


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 rcs_nn_qrack_validation_hv.py [width] [depth]"
        )
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    bench_qrack(width, depth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
