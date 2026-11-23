#!/usr/bin/env python3
# ace_seam_phase5.py -- 5-qubit PHASE-FLIP repetition code across an ACE seam.
#
# QrackAceBackend, user API only (h, x, cx, m). No injected noise: simulator
# noise=0, SDRP=0. The only error source is the backend's inherent
# cross-patch seam channel. Z-basis information is already carried exactly
# by the seam; X-basis information is not, so this code protects X.
#
# Layout: 5x6 torus, long_range_columns=2 -> seam column 2 (left patch
# sim 0, right patch sim 1, crossbar sim 2).
#
#            col0   col1    col2 (seam)   col3   col4
#    row0            E2      A_top         .      .
#    row1            A_L     s_top         A_R    .
#    row2            p_L     s_mid         p_R    .
#    row3            E0      s_bot         E1     .
#    row4            .       A_bot         E3     .
#
#   data (the "plus"): p_L, s_top, s_mid, s_bot, p_R
#   logical |+_L> = |+++++>, |-_L> = |----->  (random each shot)
#   checks X(p_L)X(s_mid), X(s_top)X(s_mid), X(s_mid)X(s_bot), X(s_mid)X(p_R),
#     each measured by an ancilla in |+> that CXs onto both data, then H, m.
#   E0..E3: spectator qubits (random X eigenstates) that the data talk to.
#
# Seam traffic ("workload"): between check rounds, --depth random CX gates
# drawn from the backend's own logical coupling map over data + spectators.
# (Mirror circuits were tried first: U then U^-1 cancels the seam error
# exactly, so they expose the code to nothing.) The ideal X value of every
# qubit is tracked classically as alpha*b XOR beta (b = the logical bit), a
# Pauli frame: in the X frame CX(c,t) does c ^= t. Gates that would move b
# OUT of a data qubit (data-data CX, or a data control on a spectator that
# currently carries b) are skipped, so the code space is preserved and
# checks/readout are compared against the frame's predicted values.
#
# Comparisons (same workload statistics):
#   per-copy physical : each data qubit's raw X readout in the encoded run
#   bare              : s_mid alone, no code, same spectators, with its
#                       traffic EXPOSURE-MATCHED to the paired encoded shot:
#                       exactly as many gates touch the bare s_mid, round by
#                       round, as touched the encoded s_mid
#   logical           : majority vote / syndrome-assisted decode
#
# Gain = logical error < bare error (and < per-copy physical error).

import argparse
import itertools
import math
import random

from pyqrack import QrackAceBackend

ROWS, COLS = 5, 6


def lq(r, c):
    return r * COLS + c


P_L, S_TOP, S_MID, S_BOT, P_R = lq(2, 1), lq(1, 2), lq(2, 2), lq(3, 2), lq(2, 3)
DATA = [P_L, S_TOP, S_MID, S_BOT, P_R]
NAMES = ["p_L", "s_top", "s_mid", "s_bot", "p_R"]
CHECKS = [(0, 2), (1, 2), (2, 3), (2, 4)]
ANC = {
    "seam": [lq(1, 1), lq(0, 2), lq(4, 2), lq(1, 3)],
    "patch": [lq(1, 1), lq(0, 1), lq(4, 3), lq(1, 3)],
}
SPECT = {
    "seam": [lq(3, 1), lq(3, 3), lq(0, 1), lq(4, 3)],
    "patch": [lq(3, 1), lq(3, 3), lq(1, 0), lq(2, 4)],
}


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


def prep_x(b, q, bit):
    if bit:
        b.x(q)
    b.h(q)


TOUCH = {"enc": 0, "bare": 0}


def traffic(b, pairs, depth, data, fr, tag, touch_target=None):
    # fr[q] = [alpha, beta]; X-frame CX(c,t): c ^= t.
    # Encoded runs: `depth` gates. Bare runs: keep going until s_mid has been
    # touched exactly `touch_target` times (exposure matched to the paired
    # encoded shot's s_mid, round by round). Returns s_mid touch count.
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
    TOUCH[tag] += touches
    return touches


def run_shot(o, encoded, match=None):
    b = new_backend(o)
    logical = random.randint(0, 1)
    data = DATA if encoded else [S_MID]
    spect = SPECT[o.anc]
    fr = {}
    for q in data:
        prep_x(b, q, logical)
        fr[q] = [1, 0]
    for q in spect:
        v = random.randint(0, 1)
        prep_x(b, q, v)
        fr[q] = [0, v]
    pairs = allowed_pairs(o, data + spect)

    hist = []
    touches = []
    for r in range(o.rounds):
        touches.append(
            traffic(b, pairs, o.depth, data, fr, "enc" if encoded else "bare",
                    None if match is None else match[r])
        )
        if encoded and o.checks:
            synd = []
            for (i, j), a in zip(CHECKS, ANC[o.anc]):
                b.h(a)
                b.cx(a, DATA[i])
                b.cx(a, DATA[j])
                b.h(a)
                s = int(b.m(a))
                if s:
                    b.x(a)  # reset to |0>
                # detection event = measured parity vs frame-predicted parity
                synd.append(s ^ fr[DATA[i]][1] ^ fr[DATA[j]][1])
            hist.append(synd)

    order = list(data)
    random.shuffle(order)
    out = {}
    for q in order:
        b.h(q)
        out[q] = int(b.m(q)) ^ fr[q][1]  # frame-corrected
    return [out[q] for q in data], logical, hist, touches


def decode(bits, hist):
    # Min-weight: data flips making the readout unanimous, plus disagreements
    # between the last measured syndrome and the readout-implied one.
    if not hist:
        return int(sum(bits) > 2)
    last = hist[-1]  # detection events (0 = matches frame)
    implied = [bits[i] ^ bits[j] for i, j in CHECKS]
    best = None
    for e in itertools.product((0, 1), repeat=5):
        c = [bits[k] ^ e[k] for k in range(5)]
        if len(set(c)) != 1:
            continue
        w = sum(e) + sum(implied[k] ^ last[k] for k in range(4))
        if best is None or w < best[0]:
            best = (w, c[0])
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--depth", type=int, default=6, help="random CX gates per round")
    ap.add_argument("--checks", type=int, default=1, help="0 = no syndrome rounds (final vote only)")
    ap.add_argument("--anc", choices=["seam", "patch"], default="seam")
    ap.add_argument("--crossbars", type=int, default=1)
    ap.add_argument("--detect", type=int, default=1)
    ap.add_argument("--authority", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    o = ap.parse_args()
    if o.seed is not None:
        random.seed(o.seed)

    n = o.shots
    phys = [0] * 5
    maj = dec = bare = 0
    fires = [0] * 4
    for _ in range(n):
        bits, lg, hist, touches = run_shot(o, encoded=True)
        for k in range(5):
            phys[k] += bits[k] != lg
        maj += int(sum(bits) > 2) != lg
        dec += decode(bits, hist) != lg
        for h in hist:
            for k in range(4):
                fires[k] += h[k]
        bb, blg, _, _ = run_shot(o, encoded=False, match=touches)
        bare += bb[0] != blg

    def f(c):
        p = c / n
        return f"{p:.4f} +/- {(p * (1 - p) / n) ** 0.5:.4f}"

    print("config:", vars(o))
    print("per-copy X error:     ", ", ".join(f"{NAMES[k]} {phys[k] / n:.4f}" for k in range(5)))
    print("check fire rate:      ", [round(c / max(1, n * o.rounds), 4) for c in fires])
    print(f"traffic gates on s_mid per shot: encoded {TOUCH['enc'] / n:.2f}, bare {TOUCH['bare'] / n:.2f}")
    print(f"bare seam qubit:       {f(bare)}")
    print(f"logical (majority):    {f(maj)}")
    print(f"logical (decoded):     {f(dec)}")
    lo, lb = dec / n, bare / n
    sig = 2 * math.sqrt(lo * (1 - lo) / n + lb * (1 - lb) / n) + 1e-12
    if lb - lo > sig:
        v = f"GAIN: bare/logical = {lb / max(lo, 1.0 / n):.2f}x (>2 sigma)"
    elif lo - lb > sig:
        v = "LOSS (logical > bare, >2 sigma)"
    else:
        v = "NO SIGNIFICANT DIFFERENCE (2 sigma)"
    print("verdict:", v)


if __name__ == "__main__":
    main()
