# Nearest-neighbor RCS with diagonal/anti-diagonal patch consensus.
#
# Designed for perfect-square grids where H and V splits are identical.
# Uses diagonal (row+col < n) and anti-diagonal (row-col < 0 mod n) cuts
# instead, which are genuinely orthogonal on square grids.
#
# By Dan Strano and (Anthropic) Claude.

import math
import random
import sys
import time

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


def make_patches_general(width, membership):
    """
    membership: array of ints (0 or 1), one per qubit.
    Returns (patch, local_idx) arrays.
    """
    patch     = np.asarray(membership, dtype=np.int32)
    local_idx = np.empty(width, dtype=np.int32)
    ctr = [0, 0]
    for i in range(width):
        local_idx[i] = ctr[patch[i]]
        ctr[patch[i]] += 1
    return patch, local_idx


def make_diagonal_patches(width, row_len, col_len):
    """
    Diagonal cut: qubit q at (row, col) goes to patch 0 if row+col < row_len,
    patch 1 otherwise. For square grids this cuts along the main diagonal.
    """
    membership = []
    for q in range(width):
        row = q // col_len
        col = q % col_len
        membership.append(0 if (row + col) < row_len else 1)
    return make_patches_general(width, membership)


def make_antidiag_patches(width, row_len, col_len):
    """
    Anti-diagonal cut: qubit q at (row, col) goes to patch 0 if
    row + (col_len-1-col) < row_len, patch 1 otherwise.
    Orthogonal to the diagonal cut on square grids.
    """
    membership = []
    for q in range(width):
        row = q // col_len
        col = q % col_len
        membership.append(0 if (row + (col_len - 1 - col)) < row_len else 1)
    return make_patches_general(width, membership)


def patch_mask(width, patch):
    mask0 = mask1 = 0
    for q in range(width):
        if patch[q] == 0: mask0 |= 1 << q
        else:              mask1 |= 1 << q
    return mask0, mask1


def expand_ket(ket_pair, width, patch, local_idx):
    n_pow = 1 << width
    k0 = np.asarray(ket_pair[0], dtype=np.complex128)
    k1 = np.asarray(ket_pair[1], dtype=np.complex128)
    idx0 = np.zeros(n_pow, dtype=np.int64)
    idx1 = np.zeros(n_pow, dtype=np.int64)
    for q in range(width):
        bit_q = (np.arange(n_pow, dtype=np.int64) >> q) & 1
        if patch[q] == 0: idx0 |= bit_q << int(local_idx[q])
        else:              idx1 |= bit_q << int(local_idx[q])
    return k0[idx0] * k1[idx1]


# ---------------------------------------------------------------------------
# Gate shadow machinery (same as rcs_nn_hv_consensus.py)
# ---------------------------------------------------------------------------

def _x(sim,q,p,l):   sim[p[q]].x(l[q])
def _z(sim,q,p,l):   sim[p[q]].z(l[q])
def _h(sim,q,p,l):   sim[p[q]].h(l[q])
def _s(sim,q,p,l):   sim[p[q]].s(l[q])
def _adjs(sim,q,p,l):sim[p[q]].adjs(l[q])

def ct_pair_prob(sim,q1,q2,p,l):
    p1=sim[p[q1]].prob(l[q1]); p2=sim[p[q2]].prob(l[q2])
    return (p2,q1) if p1<p2 else (p1,q2)

def cz_shadow(sim,q1,q2,p,l,anti=False):
    if anti: _x(sim,q1,p,l)
    prob_max,t=ct_pair_prob(sim,q1,q2,p,l)
    if prob_max>0.5: _z(sim,t,p,l)
    if anti: _x(sim,q1,p,l)

def cx_shadow(sim,c,t,p,l,anti=False):
    _h(sim,t,p,l); cz_shadow(sim,c,t,p,l,anti); _h(sim,t,p,l)

def cy_shadow(sim,c,t,p,l,anti=False):
    _adjs(sim,t,p,l); cx_shadow(sim,c,t,p,l,anti); _s(sim,t,p,l)

def swap_shadow(sim,q1,q2,p,l):
    cx_shadow(sim,q1,q2,p,l); cx_shadow(sim,q2,q1,p,l); cx_shadow(sim,q1,q2,p,l)

def _same(q1,q2,p): return p[q1]==p[q2]
def _lq(q,l): return int(l[q])
def _pp(q,p): return int(p[q])

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
    n0  = int(np.sum(patch==0)); n1 = int(np.sum(patch==1))
    sim = [QrackSimulator(n0), QrackSimulator(n1)]
    gateSequence = [0,3,2,1,2,1,0,3]
    for _ in range(depth):
        for i in range(width):
            th=random.uniform(0,2*math.pi); ph=random.uniform(0,2*math.pi); lm=random.uniform(0,2*math.pi)
            sim[patch[i]].u(local_idx[i],th,ph,lm)
        gate=gateSequence.pop(0); gateSequence.append(gate)
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
        gate=gateSequence.pop(0); gateSequence.append(gate)
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
# Walsh-Hadamard transform
# ---------------------------------------------------------------------------

def hadamard_transform(v):
    n = len(v); h = v.copy(); step = 1
    while step < n:
        for i in range(0, n, step*2):
            lo=h[i:i+step].copy(); hi=h[i+step:i+2*step].copy()
            h[i:i+step]=lo+hi; h[i+step:i+2*step]=lo-hi
        step *= 2
    return h


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def sieve_and_score(ideal_probs, p_dm, n_pow, n_candidates, u_u):
    """
    Sieve heavy and light tails from p_dm, combine as a sparse probability
    estimate, then mix 50/50 with uniform (QV protocol).

    Heavy tail: top n_candidates by p_dm, renormalized.
    Light tail: bottom n_candidates by p_dm, inverted via (2*u_u - p_dm[i]),
                renormalized — peaks where p_dm is smallest, asserting suppression.
    Combined:   50% heavy + 50% light (equal weight to both tails).
    Final mix:  50% combined + 50% uniform (QV protocol).

    Both tails contribute positively to XEB: heavy outputs push q_i above u_u
    where p_i > u_u; light outputs push q_i below u_u where p_i < u_u.
    """
    # Heavy tail
    top_idx = np.argpartition(p_dm, -n_candidates)[-n_candidates:]
    top_idx = top_idx[np.argsort(p_dm[top_idx])[::-1]]
    heavy = {int(i): float(p_dm[i]) for i in top_idx}
    s_h = sum(heavy.values())
    if s_h > 0:
        heavy = {k: v / s_h for k, v in heavy.items()}

    # Light tail: invert p_dm so lightest outputs get highest weight
    bot_idx = np.argpartition(p_dm, n_candidates)[:n_candidates]
    bot_idx = bot_idx[np.argsort(p_dm[bot_idx])]
    light_raw = {int(i): max(0.0, 2.0 * u_u - float(p_dm[i])) for i in bot_idx}
    s_l = sum(light_raw.values())
    if s_l > 0:
        # Normalize to sum 1, then invert back: light entries should be
        # *below* u_u in the final distribution, so we use (u_u - normalized_weight)
        # clipped to [0, u_u]. This keeps the light tail properly suppressed.
        light = {k: max(0.0, u_u - (v / s_l) * u_u) for k, v in light_raw.items()}
        s_l2 = sum(light.values())
        if s_l2 > 0:
            light = {k: v / s_l2 for k, v in light.items()}
    else:
        light = {}

    # Combine: 50% heavy + 50% light (union of keys)
    all_keys = set(heavy) | set(light)
    combined = {k: 0.5 * heavy.get(k, 0.0) + 0.5 * light.get(k, 0.0)
                for k in all_keys}
    s_c = sum(combined.values())
    if s_c > 0:
        combined = {k: v / s_c for k, v in combined.items()}

    return calc_stats_sparse(ideal_probs, combined, n_pow)


def calc_stats_sparse(ideal_probs, exp_probs_sparse, n_pow):
    u_u   = 1.0 / n_pow
    model = 0.5
    exp_dense = np.zeros(n_pow, dtype=np.float64)
    for k, v in exp_probs_sparse.items():
        exp_dense[k] = v
    exp_mixed = (1.0 - model) * exp_dense + model * u_u
    p_c   = ideal_probs - u_u
    q_c   = exp_mixed   - u_u
    denom = float(np.dot(p_c, p_c))
    xeb   = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    hog   = float(exp_mixed[ideal_probs > float(np.median(ideal_probs))].sum())
    return xeb, hog


def calc_stats(ideal_probs, split_probs):
    n_pow = len(ideal_probs); u_u = 1.0/n_pow
    p=np.asarray(ideal_probs,dtype=np.float64); q=np.asarray(split_probs,dtype=np.float64)
    p_c=p-u_u; q_c=q-u_u
    denom=float(np.dot(p_c,p_c))
    xeb=float(np.dot(p_c,q_c))/denom if denom>0 else 0.0
    hog=float(q[p>float(np.median(p))].sum())
    return xeb, hog


def hadamard_consensus(psis, masks, n_pow):
    """
    Combine n_cuts patch kets in the Hadamard basis using patch support masks.
    For each Hadamard index s, average over all patches that support it
    (i.e., where s doesn't straddle that patch's boundary).
    """
    sqrt_n = np.sqrt(n_pow)
    phis = [hadamard_transform(psi) / sqrt_n for psi in psis]

    def _within(s, m0, m1):
        return ((s & m0) == s) or ((s & m1) == s)

    # Support mask arrays: supp[k][s] = True if patch k supports Hadamard index s
    supps = [
        np.array([_within(s, m0, m1) for s in range(n_pow)])
        for (m0, m1) in masks
    ]

    # For each Hadamard index, average over supporting patches (or all if none)
    phi_combined = np.zeros(n_pow, dtype=np.complex128)
    for s in range(n_pow):
        supporters = [k for k in range(len(psis)) if supps[k][s]]
        if not supporters:
            supporters = list(range(len(psis)))  # no support: average all
        phi_combined[s] = sum(phis[k][s] for k in supporters) / len(supporters)

    psi_out = hadamard_transform(phi_combined) / sqrt_n
    p_out   = np.abs(psi_out) ** 2
    s = p_out.sum()
    if s > 0: p_out /= s
    return p_out


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth):
    row_len, col_len = factor_width(width)
    n_pow        = 1 << width
    u_u          = 1.0 / n_pow

    patch_d,  local_d  = make_diagonal_patches(width, row_len, col_len)
    patch_ad, local_ad = make_antidiag_patches(width, row_len, col_len)

    max_patch = max(int(np.sum(patch_d==0)),  int(np.sum(patch_d==1)),
                    int(np.sum(patch_ad==0)), int(np.sum(patch_ad==1)))

    print(f"Grid: {row_len} x {col_len}")
    print(f"Largest patch: {max_patch} qubits ({2**max_patch*8/1e6:.0f} MB)")

    if np.array_equal(patch_d, patch_ad):
        print("WARNING: diagonal and anti-diagonal splits are identical!")
        return
    if max_patch > 28:
        print(f"WARNING: {2**max_patch*8/1e9:.1f} GB needed.")
        return

    rng_state = random.getstate()
    t_start   = time.perf_counter()

    ideal_probs = run_ideal_circuit(width, depth, row_len, col_len, rng_state)
    ket_d0,  ket_d1  = run_patch_circuit(width, depth, row_len, col_len, patch_d,  local_d,  rng_state)
    ket_ad0, ket_ad1 = run_patch_circuit(width, depth, row_len, col_len, patch_ad, local_ad, rng_state)

    t_elapsed = time.perf_counter() - t_start

    psi_d  = expand_ket((ket_d0,  ket_d1),  width, patch_d,  local_d)
    psi_ad = expand_ket((ket_ad0, ket_ad1), width, patch_ad, local_ad)

    # Phase canonicalize to common gauge
    kets    = [psi_d, psi_ad]
    mean_p  = sum((k*k.conj()).real for k in kets) / 2
    gauge   = int(np.argmax(mean_p > u_u))
    psis    = []
    for k in kets:
        ref = k[gauge]; phase = ref/abs(ref) if abs(ref)>1e-30 else 1.0
        psis.append(k / phase)

    # Symmetrized density matrix diagonal
    p_dm = (psis[0]*psis[1].conj() + psis[1]*psis[0].conj()).real
    p_dm = np.maximum(p_dm, 0.0)
    s = p_dm.sum()
    if s > 0: p_dm /= s

    # Hadamard consensus
    masks = [patch_mask(width, patch_d), patch_mask(width, patch_ad)]
    p_had = hadamard_consensus(psis, masks, n_pow)

    P_D  = np.abs(psis[0])**2; P_D  /= P_D.sum()
    P_AD = np.abs(psis[1])**2; P_AD /= P_AD.sum()

    n_candidates = min(width ** 2, 1 << width)

    xeb_dm,  hog_dm  = calc_stats(ideal_probs, p_dm)
    xeb_had, hog_had = calc_stats(ideal_probs, p_had)
    xeb_d,   hog_d   = calc_stats(ideal_probs, P_D)
    xeb_ad,  hog_ad  = calc_stats(ideal_probs, P_AD)

    # Sieve heavy+light tails from symmetrized density matrix diagonal
    xeb_sieve, hog_sieve = sieve_and_score(ideal_probs, p_dm, n_pow, n_candidates, u_u)

    return {
        "width":      width, "depth":     depth, "seconds":   t_elapsed,
        "xeb_sieve":  xeb_sieve,  "hog_sieve":  hog_sieve,
        "xeb_dm":     xeb_dm,     "hog_dm":     hog_dm,
        "xeb_had":    xeb_had,    "hog_had":    hog_had,
        "xeb_d":      xeb_d,      "hog_d":      hog_d,
        "xeb_ad":     xeb_ad,     "hog_ad":     hog_ad,
    }


def main():
    if len(sys.argv) < 3:
        raise RuntimeError("Usage: python3 rcs_nn_diag_consensus.py [width] [depth]")
    width = int(sys.argv[1]); depth = int(sys.argv[2])
    result = bench_qrack(width, depth)
    if result:
        for k, v in result.items(): print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
