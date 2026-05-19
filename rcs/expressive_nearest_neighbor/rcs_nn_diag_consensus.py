# Nearest-neighbor RCS with diagonal/anti-diagonal patch consensus.
# Sampling-based — no statevector materialization beyond the ideal
# ground-truth simulator (present only for small-scale validation).
#
# Two patch circuits (diagonal and anti-diagonal cuts) each contribute
# measure_shots samples.  Sample counts proxy for probability; routing
# by count vs mean_count mirrors the u_u split in the MPS/ACE hybrid.
#
# By Dan Strano and (Anthropic) Claude.

import math
import random
import sys
import time
from collections import Counter

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
    patch     = np.asarray(membership, dtype=np.int32)
    local_idx = np.empty(width, dtype=np.int32)
    ctr = [0, 0]
    for i in range(width):
        local_idx[i] = ctr[patch[i]]
        ctr[patch[i]] += 1
    return patch, local_idx


def make_diagonal_patches(width, row_len, col_len):
    membership = []
    for q in range(width):
        row = q // col_len; col = q % col_len
        membership.append(0 if (row + col) < row_len else 1)
    return make_patches_general(width, membership)


def make_antidiag_patches(width, row_len, col_len):
    membership = []
    for q in range(width):
        row = q // col_len; col = q % col_len
        membership.append(0 if (row + (col_len - 1 - col)) < row_len else 1)
    return make_patches_general(width, membership)


# ---------------------------------------------------------------------------
# Gate shadow machinery
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

def run_patch_circuit_shots(width, depth, row_len, col_len,
                             patch, local_idx, rng_state, n_shots):
    random.setstate(rng_state)
    n0  = int(np.sum(patch==0)); n1 = int(np.sum(patch==1))
    sim = [QrackSimulator(n0), QrackSimulator(n1)]
    gateSequence = [0,3,2,1,2,1,0,3]
    for _ in range(depth):
        for i in range(width):
            th, ph, lm = (random.uniform(0,2*math.pi) for _ in range(3))
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
    # Sample from each subsystem independently; combine into full bitstring
    shots0 = sim[0].measure_shots(list(range(n0)), n_shots)
    shots1 = sim[1].measure_shots(list(range(n1)), n_shots)
    # Reconstruct full-width integer outcomes from subsystem samples
    full_shots = []
    for s0, s1 in zip(shots0, shots1):
        outcome = 0
        for q in range(width):
            if patch[q] == 0:
                bit = (int(s0) >> int(local_idx[q])) & 1
            else:
                bit = (int(s1) >> int(local_idx[q])) & 1
            outcome |= bit << q
        full_shots.append(outcome)
    return full_shots


def run_ideal_circuit(width, depth, row_len, col_len, rng_state):
    random.setstate(rng_state)
    sim = QrackSimulator(width)
    gateSequence = [0,3,2,1,2,1,0,3]
    for _ in range(depth):
        for i in range(width):
            th, ph, lm = (random.uniform(0,2*math.pi) for _ in range(3))
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
# Statistics
# ---------------------------------------------------------------------------

def calc_stats(ideal_probs, split_probs):
    n_pow = len(ideal_probs); u_u = 1.0/n_pow
    p = np.asarray(ideal_probs, dtype=np.float64)
    q = np.asarray(split_probs, dtype=np.float64)
    p_c = p-u_u; q_c = q-u_u
    denom = float(np.dot(p_c,p_c))
    xeb   = float(np.dot(p_c,q_c))/denom if denom>0 else 0.0
    hog   = float(q[p>float(np.median(p))].sum())
    return xeb, hog


def calc_stats_sparse(ideal_probs, exp_probs_sparse, n_pow):
    u_u = 1.0/n_pow; model = 0.5
    exp_dense = np.zeros(n_pow, dtype=np.float64)
    for k, v in exp_probs_sparse.items():
        exp_dense[k] = v
    exp_mixed = (1.0-model)*exp_dense + model*u_u
    p_c = ideal_probs-u_u; q_c = exp_mixed-u_u
    denom = float(np.dot(p_c,p_c))
    xeb   = float(np.dot(p_c,q_c))/denom if denom>0 else 0.0
    hog   = float(exp_mixed[ideal_probs>float(np.median(ideal_probs))].sum())
    return xeb, hog


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth):
    row_len, col_len = factor_width(width)
    n_pow   = 1 << width
    u_u     = 1.0 / n_pow
    n_shots = width ** 3

    patch_d,  local_d  = make_diagonal_patches(width, row_len, col_len)
    patch_ad, local_ad = make_antidiag_patches(width, row_len, col_len)

    max_patch = max(int(np.sum(patch_d==0)),  int(np.sum(patch_d==1)),
                    int(np.sum(patch_ad==0)), int(np.sum(patch_ad==1)))

    print(f"Grid: {row_len} x {col_len}, largest patch: {max_patch} qubits")

    if np.array_equal(patch_d, patch_ad):
        print("WARNING: diagonal and anti-diagonal splits are identical!")
        return
    if max_patch > 28:
        print(f"WARNING: {2**max_patch*8/1e9:.1f} GB needed for out_probs.")

    rng_state = random.getstate()
    t_start   = time.perf_counter()

    # Ideal ground truth (small-scale validation only)
    ideal_probs = run_ideal_circuit(width, depth, row_len, col_len, rng_state)

    # Two patch circuits — sampling only
    counts = Counter()
    for patch, local_idx in [(patch_d, local_d), (patch_ad, local_ad)]:
        shots = run_patch_circuit_shots(
            width, depth, row_len, col_len, patch, local_idx, rng_state, n_shots)
        counts.update(shots)

    t_elapsed = time.perf_counter() - t_start

    # Route by count vs mean_count
    total_shots = 2 * n_shots
    mean_count  = total_shots * u_u

    heavy_raw = {}; light_raw = {}
    for outcome, cnt in counts.items():
        if cnt > mean_count:
            heavy_raw[outcome] = float(cnt)
        else:
            light_raw[outcome] = max(0.0, 2.0 * mean_count - cnt)

    s_h = sum(heavy_raw.values())
    heavy = {k: v/s_h for k,v in heavy_raw.items()} if s_h > 0 else {}

    s_l = sum(light_raw.values())
    if s_l > 0:
        light = {k: max(0.0, u_u - (v/s_l)*u_u) for k,v in light_raw.items()}
        s_l2  = sum(light.values())
        light = {k: v/s_l2 for k,v in light.items()} if s_l2 > 0 else {}
    else:
        light = {}

    n_nonempty = (1 if heavy else 0) + (1 if light else 0)
    if n_nonempty == 0:
        combined = {}
    else:
        w = 1.0/n_nonempty
        all_keys = set(heavy)|set(light)
        combined = {k: w*heavy.get(k,0.0)+w*light.get(k,0.0) for k in all_keys}
        s_c = sum(combined.values())
        if s_c > 0:
            combined = {k: v/s_c for k,v in combined.items()}

    xeb_sieve, hog_sieve = calc_stats_sparse(ideal_probs, combined, n_pow)

    # -----------------------------------------------------------------------
    # ACE direct probability via prob_perm — scalable, no out_probs.
    # Each patch circuit contributes a separable joint probability:
    # p(outcome) = prob_perm(patch0 qubits) * prob_perm(patch1 qubits).
    # Average over both patch configurations (1/2 marginal weight each).
    # -----------------------------------------------------------------------
    def make_patch_sim(patch, local_idx):
        random.setstate(rng_state)
        n0 = int(np.sum(patch==0)); n1 = int(np.sum(patch==1))
        sim = [QrackSimulator(n0), QrackSimulator(n1)]
        gateSequence = [0,3,2,1,2,1,0,3]
        for _ in range(depth):
            for i in range(width):
                th, ph, lm = (random.uniform(0,2*math.pi) for _ in range(3))
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
        return sim

    sim_d  = make_patch_sim(patch_d,  local_d)
    sim_ad = make_patch_sim(patch_ad, local_ad)

    def patch_prob(sim_pair, patch, local_idx, outcome):
        # Separable joint probability via prob_perm on each subsystem
        q0 = [int(local_idx[q]) for q in range(width) if patch[q] == 0]
        c0 = [(outcome >> q) & 1  for q in range(width) if patch[q] == 0]
        q1 = [int(local_idx[q]) for q in range(width) if patch[q] == 1]
        c1 = [(outcome >> q) & 1  for q in range(width) if patch[q] == 1]
        p0 = sim_pair[0].prob_perm(q0, c0) if q0 else 1.0
        p1 = sim_pair[1].prob_perm(q1, c1) if q1 else 1.0
        return p0 * p1

    ace_sparse = {}
    for outcome in counts:
        p_avg = 0.5 * patch_prob(sim_d,  patch_d,  local_d,  outcome) +                 0.5 * patch_prob(sim_ad, patch_ad, local_ad, outcome)
        if p_avg > 0:
            ace_sparse[outcome] = p_avg
    for s in sim_d:  del s
    for s in sim_ad: del s

    xeb_ace, hog_ace = calc_stats_sparse(ideal_probs, ace_sparse, n_pow)

    # -----------------------------------------------------------------------
    # Equal mixture of sieve and ACE prob_perm distributions.
    # If errors are largely uncorrelated, the mixture outperforms either alone.
    # -----------------------------------------------------------------------
    all_mix_keys = set(combined) | set(ace_sparse)
    mixed = {k: 0.5 * combined.get(k, 0.0) + 0.5 * ace_sparse.get(k, 0.0)
             for k in all_mix_keys}
    s_mix = sum(mixed.values())
    if s_mix > 0:
        mixed = {k: v / s_mix for k, v in mixed.items()}
    xeb_mix, hog_mix = calc_stats_sparse(ideal_probs, mixed, n_pow)

    return {
        "width":    width, "depth":   depth,
        "n_unique": len(counts),
        "n_heavy":  len(heavy), "n_light": len(light),
        "seconds":  t_elapsed,
        "xeb_sieve":   xeb_sieve,   "hog_sieve":   hog_sieve,
        "xeb_ace_avg": xeb_ace,     "hog_ace_avg": hog_ace,
        "xeb_mix":     xeb_mix,     "hog_mix":     hog_mix,
    }


def main():
    if len(sys.argv) < 3:
        raise RuntimeError("Usage: python3 rcs_nn_diag_consensus.py [width] [depth]")
    width=int(sys.argv[1]); depth=int(sys.argv[2])
    result = bench_qrack(width, depth)
    if result:
        for k, v in result.items(): print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
