#!/usr/bin/env python3
# ace_seam_phase5.py -- 5-qubit PHASE-FLIP repetition code across an ACE seam,
# with per-round correction.
#
# QrackAceBackend, user API only (h, s, adjs, u, x, z, cx, m, prob). No
# injected noise: simulator noise=0, SDRP=0. The only error source is the
# backend's inherent cross-patch seam channel. Z-basis information is
# already carried exactly by the seam; X-basis information is not, so this
# code protects X.
#
# Layout: 5x6 torus, long_range_columns=2 -> seam column 2 (left patch
# sim 0, right patch sim 1, crossbar sim 2).
#
#            col0   col1    col2 (seam)   col3   col4
#    row0            E2      A_top         .      .
#    row1            A_L     s_top         A_R    .
#    row2    REF     p_L     s_mid         p_R    .
#    row3            E0      s_bot         E1     .
#    row4            .       A_bot         E3     .
#
#   data (the "plus"): p_L, s_top, s_mid, s_bot, p_R
#   --init random : logical |+_L>=|+++++> or |-_L>=|-----> (random per shot)
#   --init bell   : REF Bell-paired with the logical X value:
#                   |0>_REF |+++++> + |1>_REF |----->   (a genuine logical
#                   superposition/entanglement, not a classical bit)
#   checks X(p_L)X(s_mid), X(s_top)X(s_mid), X(s_mid)X(s_bot), X(s_mid)X(p_R)
#     via ancilla |+> -> CX onto both data -> H -> m -> reset.
#   E0..E3: spectator qubits the data exchange gates with.
#
# Seam traffic per round: --depth random CX gates from the backend's own
# logical coupling map over data + spectators. Each qubit's ideal X value is
# tracked as alpha*b XOR beta (b = logical bit), i.e. a Pauli frame
# (X-frame CX(c,t): c ^= t). Gates that would move b out of a data qubit
# are skipped so the code space is preserved. (Mirror circuits, U then
# U^-1, cancel the seam error exactly, so they can't be used as workload.)
#
# Per-round correction (--correct):
#   none     : checks measured, nothing applied (decode at the end only)
#   syndrome : decode THIS round's detection events (min-weight on the star)
#              and apply Z to the flagged data immediately. Physical.
#   prob     : the backend's own pseudo-repetition-code heuristic, layered
#              one level up: the five data are treated as replicas of one
#              qubit, exactly as correct() treats a boundary qubit's
#              physical replicas. Non-collapsing prob() peeks at each copy's
#              X marginal (frame-corrected) -> decisive high/low/undecided
#              vote -> Z-flip decisive dissenters -> rotate still-separable
#              copies onto the consensus X direction. Uses prob(), so it is
#              a simulator affordance (like correct() itself), not hardware.
#   both     : syndrome, then prob.
#   (--reps k measures each check k times per round and acts only on
#    detections present in all k: measured check-readout error is ~19% per
#    check vs ~4% data error per round, so single-shot syndromes miscorrect.)
#
# Baseline: s_mid alone, no code, with its traffic EXPOSURE-MATCHED to the
# paired encoded shot (as many gates touch the bare s_mid, round by round,
# as touched the encoded s_mid).
#
# Output: logical vs bare error at each round count in --scan, plus a fit of
# error PER ROUND from P(R) = (1 - (1 - 2 eps)^R) / 2.

import argparse
import itertools
import math
import random

from pyqrack import QrackAceBackend

ROWS, COLS = 5, 6


def lq(r, c):
    return r * COLS + c


REF = lq(2, 0)
P_L, S_TOP, S_MID, S_BOT, P_R = lq(2, 1), lq(1, 2), lq(2, 2), lq(3, 2), lq(2, 3)
DATA = [P_L, S_TOP, S_MID, S_BOT, P_R]
CHECKS = [(0, 2), (1, 2), (2, 3), (2, 4)]
ANC = {
    "seam": [lq(1, 1), lq(0, 2), lq(4, 2), lq(1, 3)],
    "patch": [lq(1, 1), lq(0, 1), lq(4, 3), lq(1, 3)],
}
SPECT = {
    "seam": [lq(3, 1), lq(3, 3), lq(0, 1), lq(4, 3)],
    "patch": [lq(3, 1), lq(3, 3), lq(1, 0), lq(2, 4)],
}

EPS = 2**-11  # decisiveness margin around 1/2, as in correct()
ROT_EPS = (1.0 - 1.0 / math.sqrt(2)) / 2  # correct()'s separability gate

# min-weight correction table for the star: syndrome tuple -> data flips
_TABLE = {}
for _e in itertools.product((0, 1), repeat=5):
    _s = tuple(_e[i] ^ _e[j] for i, j in CHECKS)
    if _s not in _TABLE or sum(_e) < sum(_TABLE[_s]):
        _TABLE[_s] = _e


def new_backend(o):
    return QrackAceBackend(
        ROWS * COLS,
        long_range_columns=2,
        long_range_rows=4,
        is_torus=True,
        is_gpu=bool(o.gpu),
        use_crossbars=bool(o.crossbars),
        is_error_detection=bool(o.detect),
        is_boundary_repetition_code=bool(o.authority),
    )


_CMAP = {}


def allowed_pairs(o, qubits):
    key = (o.crossbars, o.detect, o.authority, tuple(qubits))
    if key not in _CMAP:
        cm = set(new_backend(o).get_logical_coupling_map())
        _CMAP[key] = [(a, b) for a in qubits for b in qubits if (a, b) in cm]
    return _CMAP[key]


def traffic(b, pairs, depth, data, fr, touch_target=None):
    done = touches = 0
    while (done < depth) if touch_target is None else (touches < touch_target):
        c, t = random.choice(pairs)
        if c in data and (t in data or fr[t][0]):
            continue
        b.cx(c, t)
        if S_MID in (c, t):
            touches += 1
        fr[c][0] ^= fr[t][0]
        fr[c][1] ^= fr[t][1]
        done += 1
    return touches


def bloch(b, q):
    # non-collapsing Bloch vector via prob() in three bases
    z = 1 - 2 * b.prob(q)
    b.h(q)
    x = 1 - 2 * b.prob(q)
    b.h(q)
    b.adjs(q)
    b.h(q)
    y = 1 - 2 * b.prob(q)
    b.h(q)
    b.s(q)
    return x, y, z


def prob_correct(b, fr):
    # correct()'s cascade, one level up. Frame-corrected X marginal of each
    # copy: p = P(X readout == 1) XOR-ed by its known frame offset beta.
    vecs = [bloch(b, q) for q in DATA]
    p = []
    for q, (x, _, _) in zip(DATA, vecs):
        p1 = (1 - x) / 2
        p.append(1 - p1 if fr[q][1] else p1)
    high = sum(1 for v in p if v > 0.5 + EPS)
    low = sum(1 for v in p if v < 0.5 - EPS)
    if high == low:
        return  # no decisive majority: leave everything alone
    maj = int(high > low)
    for k, q in enumerate(DATA):
        x, y, z = vecs[k]
        decisive_wrong = (p[k] > 0.5 + EPS) if maj == 0 else (p[k] < 0.5 - EPS)
        if decisive_wrong:
            b.z(q)  # flip the copy's X value
            x, y = -x, -y
        r = math.sqrt(x * x + y * y + z * z)
        if 1 - r <= ROT_EPS:
            # still separable: rotate onto the consensus X direction
            tx = 1 - 2 * (maj ^ fr[q][1])  # +1 -> |+>, -1 -> |->
            th_c = math.atan2(math.sqrt(x * x + y * y), z)
            ph_c = math.atan2(y, x)
            b.u(q, -th_c, 0.0, -ph_c)  # current direction -> |0>
            b.u(q, math.pi / 2, 0.0 if tx > 0 else math.pi, 0.0)  # |0> -> target


def run_shot(o, rounds, encoded, match=None):
    b = new_backend(o)
    data = DATA if encoded else [S_MID]
    spect = SPECT[o.anc]
    fr = {}
    if o.init == "bell":
        b.h(REF)
        b.cx(REF, data[0])
        for q in data[1:]:
            b.cx(data[0], q) if not encoded else None
        if encoded:
            b.cx(P_L, S_MID)
            b.cx(S_MID, S_TOP)
            b.cx(S_MID, S_BOT)
            b.cx(S_MID, P_R)
        for q in data:
            b.h(q)
        logical = None
    else:
        logical = random.randint(0, 1)
        for q in data:
            if logical:
                b.x(q)
            b.h(q)
    for q in data:
        fr[q] = [1, 0]
    for q in spect:
        v = random.randint(0, 1)
        if v:
            b.x(q)
        b.h(q)
        fr[q] = [0, v]
    pairs = allowed_pairs(o, data + spect)

    touches = []
    for r in range(rounds):
        touches.append(traffic(b, pairs, o.depth, data, fr, None if match is None else match[r]))
        if not encoded:
            continue
        if o.correct in ("syndrome", "both", "none"):
            # --reps k: measure every check k times; a detection counts only
            # if it fires on all k repetitions (filters check-readout error,
            # which is ~5x the per-round data error on this channel)
            det = [1] * 4
            for _rep in range(o.reps):
                for k, ((i, j), a) in enumerate(zip(CHECKS, ANC[o.anc])):
                    b.h(a)
                    b.cx(a, DATA[i])
                    b.cx(a, DATA[j])
                    b.h(a)
                    s = int(b.m(a))
                    if s:
                        b.x(a)
                    det[k] &= s ^ fr[DATA[i]][1] ^ fr[DATA[j]][1]
            if o.correct != "none":
                for k, flip in enumerate(_TABLE[tuple(det)]):
                    if flip:
                        b.z(DATA[k])
        if o.correct in ("prob", "both"):
            prob_correct(b, fr)

    order = list(data)
    random.shuffle(order)
    out = {}
    for q in order:
        b.h(q)
        out[q] = int(b.m(q)) ^ fr[q][1]
    bits = [out[q] for q in data]
    if logical is None:
        logical = int(b.m(REF))
    return bits, logical, touches


def eps_per_round(p, rounds):
    if rounds == 0 or p >= 0.5:
        return float("nan") if p >= 0.5 else 0.0
    return (1 - (1 - 2 * p) ** (1.0 / rounds)) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=600)
    ap.add_argument("--scan", type=str, default="1,2,4,8", help="round counts")
    ap.add_argument("--depth", type=int, default=4, help="random CX gates per round")
    ap.add_argument("--correct", choices=["none", "syndrome", "prob", "both"], default="syndrome")
    ap.add_argument("--reps", type=int, default=1, help="check repetitions per round (AND-filtered)")
    ap.add_argument("--init", choices=["random", "bell"], default="random")
    ap.add_argument("--anc", choices=["seam", "patch"], default="seam")
    ap.add_argument("--crossbars", type=int, default=1)
    ap.add_argument("--detect", type=int, default=1)
    ap.add_argument("--authority", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    o = ap.parse_args()
    if o.seed is not None:
        random.seed(o.seed)

    print("config:", vars(o))
    print(f"{'R':>3} {'bare':>15} {'logical':>15} {'eps/rd bare':>12} {'eps/rd log':>11}  verdict")
    n = o.shots
    for R in [int(x) for x in o.scan.split(",")]:
        lerr = berr = 0
        for _ in range(n):
            bits, lg, touches = run_shot(o, R, True)
            lerr += int(sum(bits) > 2) != lg
            bb, blg, _ = run_shot(o, R, False, match=touches)
            berr += bb[0] != blg
        pl, pb = lerr / n, berr / n
        sl = math.sqrt(pl * (1 - pl) / n)
        sb = math.sqrt(pb * (1 - pb) / n)
        sig = 2 * math.sqrt(sl**2 + sb**2) + 1e-12
        v = "GAIN" if pb - pl > sig else ("LOSS" if pl - pb > sig else "n.s.")
        print(
            f"{R:>3} {pb:.4f}+/-{sb:.4f} {pl:.4f}+/-{sl:.4f} "
            f"{eps_per_round(pb, R):>12.4f} {eps_per_round(pl, R):>11.4f}  {v}"
        )


if __name__ == "__main__":
    main()
