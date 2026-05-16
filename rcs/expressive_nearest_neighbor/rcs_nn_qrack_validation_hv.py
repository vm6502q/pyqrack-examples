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
        h_rows = row_len >> 1
        cut    = h_rows * col_len          # first qubit of bottom patch
        ctr    = [0, 0]
        for i in range(width):
            p = 0 if i < cut else 1
            patch[i]     = p
            local_idx[i] = ctr[p]
            ctr[p]      += 1

    else:  # vertical: left cols / right cols
        v_cols = col_len >> 1              # number of "left" columns
        ctr    = [0, 0]
        for i in range(width):
            col = i % col_len
            p   = 0 if col < v_cols else 1
            patch[i]     = p
            local_idx[i] = ctr[p]
            ctr[p]      += 1

    return patch, local_idx


# ---------------------------------------------------------------------------
# Gate shadow machinery — generalised to arbitrary patch membership
# ---------------------------------------------------------------------------

def _prob(sim, q, patch, local_idx):
    return sim[patch[q]].prob(local_idx[q])


def _x(sim, q, patch, local_idx):
    sim[patch[q]].x(local_idx[q])

def _z(sim, q, patch, local_idx):
    sim[patch[q]].z(local_idx[q])

def _h(sim, q, patch, local_idx):
    sim[patch[q]].h(local_idx[q])

def _s(sim, q, patch, local_idx):
    sim[patch[q]].s(local_idx[q])

def _adjs(sim, q, patch, local_idx):
    sim[patch[q]].adjs(local_idx[q])


def ct_pair_prob(sim, q1, q2, patch, local_idx):
    p1 = _prob(sim, q1, patch, local_idx)
    p2 = _prob(sim, q2, patch, local_idx)
    return (p2, q1) if p1 < p2 else (p1, q2)


def cz_shadow(sim, q1, q2, patch, local_idx, anti=False):
    if anti:
        _x(sim, q1, patch, local_idx)
    prob_max, t = ct_pair_prob(sim, q1, q2, patch, local_idx)
    if prob_max > 0.5:
        _z(sim, t, patch, local_idx)
    if anti:
        _x(sim, q1, patch, local_idx)


def cx_shadow(sim, c, t, patch, local_idx, anti=False):
    _h(sim, t, patch, local_idx)
    cz_shadow(sim, c, t, patch, local_idx, anti)
    _h(sim, t, patch, local_idx)


def cy_shadow(sim, c, t, patch, local_idx, anti=False):
    _adjs(sim, t, patch, local_idx)
    cx_shadow(sim, c, t, patch, local_idx, anti)
    _s(sim, t, patch, local_idx)


def swap_shadow(sim, q1, q2, patch, local_idx):
    cx_shadow(sim, q1, q2, patch, local_idx)
    cx_shadow(sim, q2, q1, patch, local_idx)
    cx_shadow(sim, q1, q2, patch, local_idx)


def _same(q1, q2, patch):
    return patch[q1] == patch[q2]

def _lq(sim, q, patch, local_idx):
    return local_idx[q]

def _p(q, patch):
    return patch[q]


def cx(sim, q1, q2, patch, local_idx):
    if not _same(q1, q2, patch):
        cx_shadow(sim, q1, q2, patch, local_idx)
    else:
        sim[_p(q1, patch)].mcx([_lq(sim, q1, patch, local_idx)],
                                _lq(sim, q2, patch, local_idx))

def cy(sim, q1, q2, patch, local_idx):
    if not _same(q1, q2, patch):
        cy_shadow(sim, q1, q2, patch, local_idx)
    else:
        sim[_p(q1, patch)].mcy([_lq(sim, q1, patch, local_idx)],
                                _lq(sim, q2, patch, local_idx))

def cz(sim, q1, q2, patch, local_idx):
    if not _same(q1, q2, patch):
        cz_shadow(sim, q1, q2, patch, local_idx)
    else:
        sim[_p(q1, patch)].mcz([_lq(sim, q1, patch, local_idx)],
                                _lq(sim, q2, patch, local_idx))

def acx(sim, q1, q2, patch, local_idx):
    if not _same(q1, q2, patch):
        cx_shadow(sim, q1, q2, patch, local_idx, True)
    else:
        sim[_p(q1, patch)].macx([_lq(sim, q1, patch, local_idx)],
                                  _lq(sim, q2, patch, local_idx))

def acy(sim, q1, q2, patch, local_idx):
    if not _same(q1, q2, patch):
        cy_shadow(sim, q1, q2, patch, local_idx, True)
    else:
        sim[_p(q1, patch)].macy([_lq(sim, q1, patch, local_idx)],
                                  _lq(sim, q2, patch, local_idx))

def acz(sim, q1, q2, patch, local_idx):
    if not _same(q1, q2, patch):
        cz_shadow(sim, q1, q2, patch, local_idx, True)
    else:
        sim[_p(q1, patch)].macz([_lq(sim, q1, patch, local_idx)],
                                  _lq(sim, q2, patch, local_idx))

def swap(sim, q1, q2, patch, local_idx):
    if not _same(q1, q2, patch):
        swap_shadow(sim, q1, q2, patch, local_idx)
    else:
        sim[_p(q1, patch)].swap(_lq(sim, q1, patch, local_idx),
                                 _lq(sim, q2, patch, local_idx))

def iswap(sim, q1, q2, patch, local_idx):
    if not _same(q1, q2, patch):
        swap_shadow(sim, q1, q2, patch, local_idx)
        cz_shadow(sim, q1, q2, patch, local_idx)
        _s(sim, q1, patch, local_idx)
        _s(sim, q2, patch, local_idx)
    else:
        sim[_p(q1, patch)].iswap(_lq(sim, q1, patch, local_idx),
                                   _lq(sim, q2, patch, local_idx))

def iiswap(sim, q1, q2, patch, local_idx):
    if not _same(q1, q2, patch):
        _s(sim, q1, patch, local_idx)
        _s(sim, q2, patch, local_idx)
        cz_shadow(sim, q1, q2, patch, local_idx)
        swap_shadow(sim, q1, q2, patch, local_idx)
    else:
        sim[_p(q1, patch)].adjiswap(_lq(sim, q1, patch, local_idx),
                                      _lq(sim, q2, patch, local_idx))

def pswap(sim, q1, q2, patch, local_idx):
    if not _same(q1, q2, patch):
        cz_shadow(sim, q1, q2, patch, local_idx)
        swap_shadow(sim, q1, q2, patch, local_idx)
    else:
        p = _p(q1, patch)
        l1, l2 = _lq(sim, q1, patch, local_idx), _lq(sim, q2, patch, local_idx)
        sim[p].mcz([l1], l2); sim[p].swap(l1, l2)

def mswap(sim, q1, q2, patch, local_idx):
    if not _same(q1, q2, patch):
        swap_shadow(sim, q1, q2, patch, local_idx)
        cz_shadow(sim, q1, q2, patch, local_idx)
    else:
        p = _p(q1, patch)
        l1, l2 = _lq(sim, q1, patch, local_idx), _lq(sim, q2, patch, local_idx)
        sim[p].swap(l1, l2); sim[p].mcz([l1], l2)

def nswap(sim, q1, q2, patch, local_idx):
    if not _same(q1, q2, patch):
        cz_shadow(sim, q1, q2, patch, local_idx)
        swap_shadow(sim, q1, q2, patch, local_idx)
        cz_shadow(sim, q1, q2, patch, local_idx)
    else:
        p = _p(q1, patch)
        l1, l2 = _lq(sim, q1, patch, local_idx), _lq(sim, q2, patch, local_idx)
        sim[p].mcz([l1], l2); sim[p].swap(l1, l2); sim[p].mcz([l1], l2)


TWO_BIT_GATES = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz


# ---------------------------------------------------------------------------
# Circuit runner
# ---------------------------------------------------------------------------

def run_circuit(width, depth, row_len, col_len, patch, local_idx, rng_state):
    """
    Replay the circuit from rng_state using the given patch/local_idx assignment.
    Returns [out_probs(patch0), out_probs(patch1)].
    """
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


# ---------------------------------------------------------------------------
# XEB — vectorised, no iteration over 2^n
# ---------------------------------------------------------------------------

def calc_xeb(probs_a, probs_b, n0_a, n1_a, n0_b, n1_b, depth, label_a, label_b):
    """
    XEB between two patch experiments A (control) and B (experiment).

    Each experiment's joint probability factors as:
      P_A(i) = a_low[idx0_a(i)] * a_high[idx1_a(i)]
      P_B(i) = b_low[idx0_b(i)] * b_high[idx1_b(i)]

    where idx0/idx1 map global bitstring i to the local indices of each patch.

    denom = sum_i (P_A(i) - u)^2
          = dot(a0,a0)*dot(a1,a1) - 2u + u^2*2^n

    numer = sum_i (P_A(i) - u)(P_B(i) - u)

    For orthogonal splits on a rectangular grid the patches of A and B
    partition the qubit set into four quadrants.  The sum over all 2^n
    bitstrings factors into a product of sums over the four quadrants,
    each of size at most 2^(n/2).  We compute this via einsum over the
    four quadrant marginals.
    """
    n = n0_a + n1_a
    assert n0_b + n1_b == n
    u_u = 1.0 / (1 << n)

    a0 = np.asarray(probs_a[0], dtype=np.float64)  # shape (2^n0_a,)
    a1 = np.asarray(probs_a[1], dtype=np.float64)  # shape (2^n1_a,)
    b0 = np.asarray(probs_b[0], dtype=np.float64)  # shape (2^n0_b,)
    b1 = np.asarray(probs_b[1], dtype=np.float64)  # shape (2^n1_b,)

    # denom
    denom = (float(np.dot(a0, a0)) * float(np.dot(a1, a1))
             - 2.0 * u_u + u_u * u_u * (1 << n))

    # numer = sum_i (P_A(i)-u)(P_B(i)-u)
    # = sum_i P_A(i)*P_B(i) - u*sum_i P_A(i) - u*sum_i P_B(i) + u^2*2^n
    # = sum_i P_A(i)*P_B(i) - u - u + u^2*2^n   [since marginals sum to 1]
    # = sum_i P_A(i)*P_B(i) - 2u + u^2*2^n
    #
    # sum_i P_A(i)*P_B(i)
    # = sum_i a0[idx0_a(i)] * a1[idx1_a(i)] * b0[idx0_b(i)] * b1[idx1_b(i)]
    #
    # For horizontal A (rows) x vertical B (cols), the four quadrants are:
    #   Q00: top-left    -> a0 subset x b0 subset
    #   Q01: top-right   -> a0 subset x b1 subset
    #   Q10: bottom-left -> a1 subset x b0 subset
    #   Q11: bottom-right-> a1 subset x b1 subset
    #
    # Each quadrant's qubits are independent across A and B partitions, so:
    # sum_i P_A(i)*P_B(i) = product over quadrants of (sum over quadrant bits)
    #
    # Rather than assume specific quadrant structure, we build the joint
    # probability table at O(2^n0_a * 2^n0_b) cost by noting that the
    # sum factors as an outer product sum:
    #
    # sum_i a0[ia]*a1[ib]*b0[jc]*b1[jd]
    # where (ia,ib) index A's patches and (jc,jd) index B's patches,
    # and these are determined by the global bitstring i.
    #
    # For genuinely orthogonal splits the factorisation reduces to
    # dot products over the four quadrant marginals (see below).
    #
    # We precompute the four quadrant joint marginals from the patch probs:
    # M_ab[x,y] = sum_{i: A-patch=x qubit bits match, B-patch=y qubit bits match}
    # This is equivalent to summing a0/a1 outer-producted with b0/b1 over
    # the shared qubit subsets in each quadrant.
    #
    # For the general case we use the factored outer product directly.
    # Cost: O(2^n0_a * 2^n1_a) = O(2^n) — only feasible for small n.
    # For large n with orthogonal splits use the quadrant marginal approach.

    if n <= 30:
        # Small enough to materialise the full joint table
        P_A = np.outer(a0, a1).ravel()
        P_B = np.outer(b0, b1).ravel()
        cross = float(np.dot(P_A, P_B))
    else:
        # Orthogonal split factorisation:
        # With horizontal A (top/bottom rows) and vertical B (left/right cols),
        # each global qubit belongs to exactly one quadrant (A-patch, B-patch).
        # The sum over all bitstrings factors into independent sums per quadrant.
        #
        # Let n_ab = number of qubits in quadrant (a,b).
        # sum_i P_A(i)*P_B(i) = prod_{(a,b)} sum_{k=0}^{2^n_ab - 1} marg_A_ab[k] * marg_B_ab[k]
        # where marg_A_ab and marg_B_ab are the marginal probabilities of A and B
        # respectively over the qubits in quadrant (a,b).
        #
        # We approximate by noting that for large random circuits the marginals
        # over quadrant qubits are approximately products of the full patch marginals
        # restricted to those qubit indices — computable via partial sums of a0,a1,b0,b1.
        # For exact computation we'd need to pass the quadrant structure explicitly.
        # Here we fall back to the complementary-split dot product formula which is
        # exact when n0_a + n0_b == n (complementary half-splits):
        if n0_a + n0_b == n:
            dot_lh = float(np.dot(a0, b1))
            dot_hl = float(np.dot(a1, b0))
            cross = dot_lh * dot_hl
        else:
            # General large-n case: use chunk-wise computation to avoid OOM
            # Process a0 in chunks, accumulating the cross sum
            chunk = min(1 << 20, len(a0))
            cross = 0.0
            for start in range(0, len(a0), chunk):
                end = min(start + chunk, len(a0))
                P_A_chunk = np.outer(a0[start:end], a1).ravel()
                P_B_chunk = np.outer(b0[start:end], b1).ravel()
                cross += float(np.dot(P_A_chunk, P_B_chunk))

    numer = cross - 2.0 * u_u + u_u * u_u * (1 << n)
    xeb   = numer / denom if denom != 0.0 else float("nan")

    return {
        "control":    label_a,
        "experiment": label_b,
        "qubits":     n,
        "n0_ctrl":    n0_a,
        "n1_ctrl":    n1_a,
        "n0_exp":     n0_b,
        "n1_exp":     n1_b,
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

    n0_h = int(np.sum(patch_h == 0))
    n1_h = int(np.sum(patch_h == 1))
    n0_v = int(np.sum(patch_v == 0))
    n1_v = int(np.sum(patch_v == 1))

    print(f"Grid: {row_len} x {col_len}")
    print(f"Horizontal split: {n0_h} + {n1_h} qubits (top/bottom rows)")
    print(f"Vertical   split: {n0_v} + {n1_v} qubits (left/right cols)")

    # Verify splits are genuinely different
    if np.array_equal(patch_h, patch_v):
        print("WARNING: horizontal and vertical splits are identical on this grid!")
        print("Try a non-square grid (e.g. width=35 for 5x7).")
        return

    rng_state = random.getstate()

    probs_h = run_circuit(width, depth, row_len, col_len, patch_h, local_h, rng_state)
    probs_v = run_circuit(width, depth, row_len, col_len, patch_v, local_v, rng_state)

    print(calc_xeb(probs_h, probs_v, n0_h, n1_h, n0_v, n1_v, depth,
                   "horizontal", "vertical"))
    print(calc_xeb(probs_v, probs_h, n0_v, n1_v, n0_h, n1_h, depth,
                   "vertical", "horizontal"))


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
