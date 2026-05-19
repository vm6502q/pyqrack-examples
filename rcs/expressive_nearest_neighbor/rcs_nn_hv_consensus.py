# Nearest-neighbor RCS with H/V patch consensus.
#
# Runs the same nearest-neighbor random circuit with two orthogonal patch
# boundaries (horizontal: top/bottom rows; vertical: left/right cols).
# The symmetrized outer-product density matrix diagonal
#   p_dm[i] = Re(psi_H[i] * psi_V[i].conj())
# is used to sieve heavy-output candidates and estimate their probabilities.
# XEB and HOG are computed against a full ideal simulator for ground truth.
#
# Memory note: max patch size should be <= 28 qubits (out_probs fits in RAM).
# Recommended widths: 42 (6x7), 35 (5x7), 30 (5x6).
#
# By Dan Strano and (Anthropic) Claude.

import math
import random
import sys
import time
from itertools import combinations

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


def expand_probs(probs_pair, width, patch, local_idx):
    """Expand subsystem prob pair to full 2^width vector (separability assumption)."""
    p0    = np.asarray(probs_pair[0], dtype=np.float64)
    p1    = np.asarray(probs_pair[1], dtype=np.float64)
    n_pow = 1 << width
    idx0  = np.zeros(n_pow, dtype=np.int64)
    idx1  = np.zeros(n_pow, dtype=np.int64)
    for q in range(width):
        bit_q = (np.arange(n_pow, dtype=np.int64) >> q) & 1
        if patch[q] == 0:
            idx0 |= bit_q << int(local_idx[q])
        else:
            idx1 |= bit_q << int(local_idx[q])
    return p0[idx0] * p1[idx1]


# ---------------------------------------------------------------------------
# Gate shadow machinery
# ---------------------------------------------------------------------------

def _x(sim, q, p, l):    sim[p[q]].x(l[q])
def _z(sim, q, p, l):    sim[p[q]].z(l[q])
def _h(sim, q, p, l):    sim[p[q]].h(l[q])
def _s(sim, q, p, l):    sim[p[q]].s(l[q])
def _adjs(sim, q, p, l): sim[p[q]].adjs(l[q])

def ct_pair_prob(sim, q1, q2, p, l):
    p1 = sim[p[q1]].prob(l[q1]); p2 = sim[p[q2]].prob(l[q2])
    return (p2, q1) if p1 < p2 else (p1, q2)

def cz_shadow(sim, q1, q2, p, l, anti=False):
    if anti: _x(sim, q1, p, l)
    prob_max, t = ct_pair_prob(sim, q1, q2, p, l)
    if prob_max > 0.5: _z(sim, t, p, l)
    if anti: _x(sim, q1, p, l)

def cx_shadow(sim, c, t, p, l, anti=False):
    _h(sim,t,p,l); cz_shadow(sim,c,t,p,l,anti); _h(sim,t,p,l)

def cy_shadow(sim, c, t, p, l, anti=False):
    _adjs(sim,t,p,l); cx_shadow(sim,c,t,p,l,anti); _s(sim,t,p,l)

def swap_shadow(sim, q1, q2, p, l):
    cx_shadow(sim,q1,q2,p,l); cx_shadow(sim,q2,q1,p,l); cx_shadow(sim,q1,q2,p,l)

def _same(q1,q2,p): return p[q1]==p[q2]
def _lq(q,l):       return int(l[q])
def _pp(q,p):       return int(p[q])

def cx(sim,q1,q2,p,l):
    if not _same(q1,q2,p): cx_shadow(sim,q1,q2,p,l)
    else: sim[_pp(q1,p)].mcx([_lq(q1,l)],_lq(q2,l))
def cy(sim,q1,q2,p,l):
    if not _same(q1,q2,p): cy_shadow(sim,q1,q2,p,l)
    else: sim[_pp(q1,p)].mcy([_lq(q1,l)],_lq(q2,l))
def cz(sim,q1,q2,p,l):
    if not _same(q1,q2,p): cz_shadow(sim,q1,q2,p,l)
    else: sim[_pp(q1,p)].mcz([_lq(q1,l)],_lq(q2,l))
def acx(sim,q1,q2,p,l):
    if not _same(q1,q2,p): cx_shadow(sim,q1,q2,p,l,True)
    else: sim[_pp(q1,p)].macx([_lq(q1,l)],_lq(q2,l))
def acy(sim,q1,q2,p,l):
    if not _same(q1,q2,p): cy_shadow(sim,q1,q2,p,l,True)
    else: sim[_pp(q1,p)].macy([_lq(q1,l)],_lq(q2,l))
def acz(sim,q1,q2,p,l):
    if not _same(q1,q2,p): cz_shadow(sim,q1,q2,p,l,True)
    else: sim[_pp(q1,p)].macz([_lq(q1,l)],_lq(q2,l))
def swap(sim,q1,q2,p,l):
    if not _same(q1,q2,p): swap_shadow(sim,q1,q2,p,l)
    else: sim[_pp(q1,p)].swap(_lq(q1,l),_lq(q2,l))
def iswap(sim,q1,q2,p,l):
    if not _same(q1,q2,p):
        swap_shadow(sim,q1,q2,p,l); cz_shadow(sim,q1,q2,p,l)
        _s(sim,q1,p,l); _s(sim,q2,p,l)
    else: sim[_pp(q1,p)].iswap(_lq(q1,l),_lq(q2,l))
def iiswap(sim,q1,q2,p,l):
    if not _same(q1,q2,p):
        _s(sim,q1,p,l); _s(sim,q2,p,l)
        cz_shadow(sim,q1,q2,p,l); swap_shadow(sim,q1,q2,p,l)
    else: sim[_pp(q1,p)].adjiswap(_lq(q1,l),_lq(q2,l))
def pswap(sim,q1,q2,p,l):
    if not _same(q1,q2,p): cz_shadow(sim,q1,q2,p,l); swap_shadow(sim,q1,q2,p,l)
    else:
        pp=_pp(q1,p); l1=_lq(q1,l); l2=_lq(q2,l)
        sim[pp].mcz([l1],l2); sim[pp].swap(l1,l2)
def mswap(sim,q1,q2,p,l):
    if not _same(q1,q2,p): swap_shadow(sim,q1,q2,p,l); cz_shadow(sim,q1,q2,p,l)
    else:
        pp=_pp(q1,p); l1=_lq(q1,l); l2=_lq(q2,l)
        sim[pp].swap(l1,l2); sim[pp].mcz([l1],l2)
def nswap(sim,q1,q2,p,l):
    if not _same(q1,q2,p):
        cz_shadow(sim,q1,q2,p,l); swap_shadow(sim,q1,q2,p,l); cz_shadow(sim,q1,q2,p,l)
    else:
        pp=_pp(q1,p); l1=_lq(q1,l); l2=_lq(q2,l)
        sim[pp].mcz([l1],l2); sim[pp].swap(l1,l2); sim[pp].mcz([l1],l2)

TWO_BIT_GATES = swap,pswap,mswap,nswap,iswap,iiswap,cx,cy,cz,acx,acy,acz

# Single-sim wrappers for ideal circuit
def _cx_i(s,a,b):   s.mcx([a],b)
def _cy_i(s,a,b):   s.mcy([a],b)
def _cz_i(s,a,b):   s.mcz([a],b)
def _acx_i(s,a,b):  s.macx([a],b)
def _acy_i(s,a,b):  s.macy([a],b)
def _acz_i(s,a,b):  s.macz([a],b)
def _sw_i(s,a,b):   s.swap(a,b)
def _isw_i(s,a,b):  s.iswap(a,b)
def _iisw_i(s,a,b): s.adjiswap(a,b)
def _psw_i(s,a,b):  s.mcz([a],b); s.swap(a,b)
def _msw_i(s,a,b):  s.swap(a,b);  s.mcz([a],b)
def _nsw_i(s,a,b):  s.mcz([a],b); s.swap(a,b); s.mcz([a],b)
TWO_BIT_GATES_IDEAL = _sw_i,_psw_i,_msw_i,_nsw_i,_isw_i,_iisw_i,\
                      _cx_i,_cy_i,_cz_i,_acx_i,_acy_i,_acz_i


# ---------------------------------------------------------------------------
# Circuit runners
# ---------------------------------------------------------------------------

def run_patch_circuit(width, depth, row_len, col_len, patch, local_idx, rng_state):
    random.setstate(rng_state)
    n0  = int(np.sum(patch == 0)); n1 = int(np.sum(patch == 1))
    sim = [QrackSimulator(n0), QrackSimulator(n1)]
    gateSequence = [0,3,2,1,2,1,0,3]
    for _ in range(depth):
        for i in range(width):
            th=random.uniform(0,2*math.pi); ph=random.uniform(0,2*math.pi); lm=random.uniform(0,2*math.pi)
            sim[patch[i]].u(local_idx[i],th,ph,lm)
        gate = gateSequence.pop(0); gateSequence.append(gate)
        for row in range(1,row_len,2):
            for col in range(col_len):
                tr=row+(1 if(gate&2)else -1); tc=col+(1 if(gate&1)else 0)
                if tr<0: tr+=row_len
                if tc<0: tc+=col_len
                if tr>=row_len: tr-=row_len
                if tc>=col_len: tc-=col_len
                b1=row*row_len+col; b2=tr*row_len+tc
                if (b1==b2)or(b1>=width)or(b2>=width): continue
                g=random.choice(TWO_BIT_GATES); g(sim,b1,b2,patch,local_idx)
    return sim[0].out_ket(), sim[1].out_ket()


def run_ideal_circuit(width, depth, row_len, col_len, rng_state):
    random.setstate(rng_state)
    sim = QrackSimulator(width)
    gateSequence = [0,3,2,1,2,1,0,3]
    for _ in range(depth):
        for i in range(width):
            th=random.uniform(0,2*math.pi); ph=random.uniform(0,2*math.pi); lm=random.uniform(0,2*math.pi)
            sim.u(i,th,ph,lm)
        gate = gateSequence.pop(0); gateSequence.append(gate)
        for row in range(1,row_len,2):
            for col in range(col_len):
                tr=row+(1 if(gate&2)else -1); tc=col+(1 if(gate&1)else 0)
                if tr<0: tr+=row_len
                if tc<0: tc+=col_len
                if tr>=row_len: tr-=row_len
                if tc>=col_len: tc-=col_len
                b1=row*row_len+col; b2=tr*row_len+tc
                if (b1==b2)or(b1>=width)or(b2>=width): continue
                g=random.choice(TWO_BIT_GATES_IDEAL); g(sim,b1,b2)
    return np.asarray(sim.out_probs(), dtype=np.float64)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats(ideal_probs, split_probs):
    n_pow = len(ideal_probs)
    u_u   = 1.0 / n_pow
    p = np.asarray(ideal_probs, dtype=np.float64)
    q = np.asarray(split_probs, dtype=np.float64)
    p_c = p - u_u; q_c = q - u_u
    denom = float(np.dot(p_c, p_c))
    xeb   = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    hog   = float(q[p > float(np.median(p))].sum())
    return xeb, hog


def calc_stats_sparse(ideal_probs, exp_probs_sparse, n_pow):
    u_u   = 1.0 / n_pow
    model = 0.5
    exp_dense = np.zeros(n_pow, dtype=np.float64)
    for k, v in exp_probs_sparse.items():
        exp_dense[k] = v
    exp_mixed = (1.0 - model) * exp_dense + model * u_u
    p_c   = ideal_probs - u_u; q_c = exp_mixed - u_u
    denom = float(np.dot(p_c, p_c))
    xeb   = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    hog   = float(exp_mixed[ideal_probs > float(np.median(ideal_probs))].sum())
    return xeb, hog


def sieve_and_score(ideal_probs, p_dm, n_pow, n_candidates, u_u):
    """Sieve from p_dm, build sparse estimate, return xeb_sparse and hog_sparse."""
    top_idx    = np.argpartition(p_dm, -n_candidates)[-n_candidates:]
    top_idx    = top_idx[np.argsort(p_dm[top_idx])[::-1]]
    bottom_idx = np.argpartition(p_dm,  n_candidates)[:n_candidates]
    bottom_idx = bottom_idx[np.argsort(p_dm[bottom_idx])]

    exp_h = {int(i): float(p_dm[i]) for i in top_idx}
    s = sum(exp_h.values()); exp_h = {k: v/s for k,v in exp_h.items()}

    exp_l = {int(i): max(0.0, float(2*u_u - p_dm[i])) for i in bottom_idx}
    s = sum(exp_l.values())
    exp_l = {k: max(0.0, 2*u_u - v/s) for k,v in exp_l.items()} if s > 0 else exp_l

    exp_all = {k: exp_h.get(k,0) + exp_l.get(k,0)
               for k in set(exp_h)|set(exp_l)}
    s = sum(exp_all.values())
    exp_all = {k: v/s for k,v in exp_all.items()}

    return calc_stats_sparse(ideal_probs, exp_all, n_pow)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth):
    row_len, col_len = factor_width(width)
    n_pow        = 1 << width
    n_candidates = width ** 2
    u_u          = 1.0 / n_pow

    patch_h, local_h = make_patches(width, row_len, col_len, 'horizontal')
    patch_v, local_v = make_patches(width, row_len, col_len, 'vertical')

    max_patch = max(int(np.sum(patch_h==0)), int(np.sum(patch_h==1)),
                    int(np.sum(patch_v==0)), int(np.sum(patch_v==1)))

    print(f"Grid: {row_len} x {col_len}")
    print(f"Largest patch: {max_patch} qubits ({2**max_patch*8/1e6:.0f} MB)")

    if np.array_equal(patch_h, patch_v):
        print("WARNING: H and V splits are identical. Try n=42,35,30.")
        return
    if max_patch > 28:
        print(f"WARNING: {2**max_patch*8/1e9:.1f} GB needed. Try n<=42.")
        return

    rng_state = random.getstate()
    t_start   = time.perf_counter()

    # Ideal ground truth
    ideal_probs = run_ideal_circuit(width, depth, row_len, col_len, rng_state)

    # Two patch circuits — return kets (complex) for density matrix
    ket_h0, ket_h1 = run_patch_circuit(width, depth, row_len, col_len, patch_h, local_h, rng_state)
    ket_v0, ket_v1 = run_patch_circuit(width, depth, row_len, col_len, patch_v, local_v, rng_state)

    t_elapsed = time.perf_counter() - t_start

    # Expand to full probability vectors
    probs_h = (np.asarray(ket_h0, dtype=np.complex128),
               np.asarray(ket_h1, dtype=np.complex128))
    probs_v = (np.asarray(ket_v0, dtype=np.complex128),
               np.asarray(ket_v1, dtype=np.complex128))

    P_H = expand_probs([np.abs(probs_h[0])**2, np.abs(probs_h[1])**2],
                       width, patch_h, local_h)
    P_V = expand_probs([np.abs(probs_v[0])**2, np.abs(probs_v[1])**2],
                       width, patch_v, local_v)

    # Phase-canonicalize kets to common gauge before density matrix
    # Use P_H + P_V mean field to find gauge index
    mean_p    = (P_H + P_V) / 2.0
    gauge_idx = int(np.argmax(mean_p > u_u))

    # Reconstruct full kets from subsystem kets
    def full_ket(ket_pair, patch, local_idx):
        k0 = np.asarray(ket_pair[0], dtype=np.complex128)
        k1 = np.asarray(ket_pair[1], dtype=np.complex128)
        idx0 = np.zeros(n_pow, dtype=np.int64)
        idx1 = np.zeros(n_pow, dtype=np.int64)
        for q in range(width):
            bit_q = (np.arange(n_pow, dtype=np.int64) >> q) & 1
            if patch[q] == 0: idx0 |= bit_q << int(local_idx[q])
            else:              idx1 |= bit_q << int(local_idx[q])
        return k0[idx0] * k1[idx1]

    psi_h = full_ket((ket_h0, ket_h1), patch_h, local_h)
    psi_v = full_ket((ket_v0, ket_v1), patch_v, local_v)

    # Phase canonicalize
    for psi in (psi_h, psi_v):
        ref = psi[gauge_idx]
        if abs(ref) > 1e-30:
            psi /= (ref / abs(ref))

    # Symmetrized outer-product density matrix diagonal
    p_dm = (psi_h * psi_v.conj() + psi_v * psi_h.conj()).real / 2.0
    p_dm = np.maximum(p_dm, 0.0)
    dm_sum = p_dm.sum()
    if dm_sum > 0:
        p_dm /= dm_sum

    # Sieve from density matrix
    xeb_sparse, hog_sparse = sieve_and_score(ideal_probs, p_dm, n_pow, n_candidates, u_u)

    # Full comparisons
    xeb_dm,  hog_dm  = calc_stats(ideal_probs, p_dm)
    xeb_h,   hog_h   = calc_stats(ideal_probs, P_H)
    xeb_v,   hog_v   = calc_stats(ideal_probs, P_V)
    xeb_avg, hog_avg = calc_stats(ideal_probs, (P_H + P_V) / 2.0)

    return {
        "width":        width,
        "depth":        depth,
        "seconds":      t_elapsed,
        "xeb_sparse":   xeb_sparse,
        "hog_sparse":   hog_sparse,
        "xeb_dm":       xeb_dm,
        "hog_dm":       hog_dm,
        "xeb_avg_hv":   xeb_avg,
        "hog_avg_hv":   hog_avg,
        "xeb_h":        xeb_h,
        "hog_h":        hog_h,
        "xeb_v":        xeb_v,
        "hog_v":        hog_v,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 rcs_nn_qrack_validation_hv_consensus.py [width] [depth]\n"
            "Recommended widths: 42 (6x7), 35 (5x7), 30 (5x6)"
        )
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    result = bench_qrack(width, depth)
    if result:
        for k, v in result.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
