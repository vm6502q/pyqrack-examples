# Is the matter side of ghost/matter pair creation thermal?  (Scrambled-not-lost check, on Qrack.)
#
# By (Anthropic) Claude and Dan Strano.
#
# A) One bosonic mode pair, K = kappa (a^dag b^dag + a b), from |0,0>. Untruncated, this is two-mode squeezing:
#    |Psi> = sum_n c_n |n>_ghost |n>_matter with |c_n|^2 = tanh^(2n)(r)/cosh^2(r), r = kappa t: the matter side alone is
#    EXACTLY thermal (geometric occupation), and the global state stays pure. Here Qrack evolves it with the
#    constraint-exact Givens Trotterization, truncated to 2^n quanta per mode, and we read the Schmidt coefficients
#    |c_n|^2 straight from the amplitudes of |n,n> (no density matrix). Thermality tests:
#      - log-ratio ln(p_{n+1}/p_n) constant in n  (= -w/T)
#      - entanglement entropy S = -sum p ln p equals the thermal entropy at the same mean occupation
#        S_th(N) = (N+1)ln(N+1) - N ln N  (a Gibbs state is the maximum-entropy state at fixed <N>)
# B) Several fermion mode pairs (one qubit per particle mode), pair rotation exp(-i r_k (X X - Y Y)/2) per mode, as
#    a mass-monopole pair vertex would give. Each mode alone is trivially Fermi-Dirac at SOME temperature; a spectrum
#    thermal at ONE T_c needs tan^2 r_k = exp(-E_k/T_c). Uniform coupling (same r for every mode) is compared with that.
#
# Usage: python3 wdw_thermality_qrack.py

import numpy as np
from pyqrack import Pauli, QrackSimulator


def givens(sim, sys_q, x, y, theta):
    """exp(-i theta (|x><y| + |y><x|)) on basis states x, y of the register sys_q (no extra controls)."""
    diff = [q for q in sys_q if ((x ^ y) >> q) & 1]
    k = diff[0]
    for dq in diff[1:]:
        sim.mcx([k], dq)
    xr = x
    for dq in diff[1:]:
        if (xr >> k) & 1:
            xr ^= 1 << dq
    ctrls = [q for q in sys_q if q != k]
    perm = 0
    for i, q in enumerate(ctrls):
        if (xr >> q) & 1:
            perm |= 1 << i
    c, s = np.cos(theta), np.sin(theta)
    sim.ucmtrx(ctrls, [c, -1j * s, -1j * s, c], k, perm)
    for dq in diff[1:][::-1]:
        sim.mcx([k], dq)


def squeeze_pair(n, r, steps_per_unit=40):   # more steps: single-precision round-off wins
    d = 2 ** n
    moves = [(i + d * j, (i + 1) + d * (j + 1), np.sqrt((i + 1) * (j + 1))) for i in range(d - 1) for j in range(d - 1)]
    sys_q = list(range(2 * n))
    sim = QrackSimulator(2 * n)
    steps = max(1, int(np.ceil(r * steps_per_unit)))
    h = r / steps
    for _ in range(steps):
        for x, y, amp in moves + moves[::-1]:
            givens(sim, sys_q, x, y, amp * h / 2)
    psi = np.array(sim.out_ket())
    p = np.array([abs(psi[m + d * m]) ** 2 for m in range(d)])    # Schmidt coefficients |c_n|^2
    leak = 1 - p.sum()                                             # weight outside the N_a = N_b sector (should be 0)
    return p, leak


def S_thermal(N):
    return (N + 1) * np.log(N + 1) - (N * np.log(N) if N > 0 else 0.)


def part_a():
    print("A) bosonic ghost/matter mode pair: is the matter side thermal?")
    print(f"{'qubits/mode':>11s} {'r=kappa t':>9s} {'<N>':>7s} {'T/w (p1/p0)':>11s} {'ln-ratio spread':>15s} "
          f"{'S/S_thermal':>11s} {'sector leak':>11s}   (untruncated: spread 0, S/S_th = 1)")
    for n in [2, 3, 4]:
        for r in [0.3, 0.6, 1.0]:
            p, leak = squeeze_pair(n, r)
            N = np.sum(np.arange(len(p)) * p)
            nz = p > 1e-4
            lr = np.diff(np.log(p[nz]))
            T = -1 / lr[0]
            S = -np.sum(p[p > 1e-300] * np.log(p[p > 1e-300]))
            # reference: exact evolution of the SAME truncated generator (no Trotter, double precision)
            from scipy.linalg import expm
            d = 2 ** n; aa = np.diag(np.sqrt(np.arange(1, d)), 1); I = np.eye(d)
            K = np.kron(I, aa).T @ np.kron(aa, I).T + np.kron(aa, I) @ np.kron(I, aa)
            v = expm(-1j * K * r)[:, 0]; pe = np.array([abs(v[m + d * m]) ** 2 for m in range(d)])
            me = pe > 1e-10
            print(f"{n:11d} {r:9.2f} {N:7.3f} {T:11.4f} {np.ptp(lr):15.4f} {S / S_thermal(N):11.4f} {leak:11.1e}"
                  f"   exact-truncated ln-ratio spread (excl. top rung) {np.ptp(np.diff(np.log(pe[me]))[:-1]):.1e}")
    r = 0.6
    print(f"    untruncated r={r}: <N> = sinh^2 r = {np.sinh(r)**2:.3f}, T/w = -1/ln(tanh^2 r) = {-1/np.log(np.tanh(r)**2):.4f}")


def part_b():
    print("\nB) fermion mode pairs (energies E_k in units of T_c): one temperature across modes?")
    E = np.array([1.0, 2.0, 3.5, 5.0, 7.0])
    nm = len(E)
    for label, r in [("uniform coupling, r_k = 0.6", np.full(nm, 0.6)),
                     ("horizon-type, tan^2 r_k = exp(-E_k/T_c)", np.arctan(np.exp(-E / 2)))]:
        sim = QrackSimulator(2 * nm)          # qubit 2k: ghost-partner mode, 2k+1: matter mode
        for k in range(nm):
            # exp(-i r (XX - YY)/2) = rotation in span{|00>,|11>} by angle r
            sim.exp([Pauli.PauliX, Pauli.PauliX], -r[k] / 2, [2 * k, 2 * k + 1])
            sim.exp([Pauli.PauliY, Pauli.PauliY], r[k] / 2, [2 * k, 2 * k + 1])
        occ = np.array([sim.prob(2 * k + 1) for k in range(nm)])
        Tk = E / np.log(1 / occ - 1)          # temperature each mode would need, Fermi-Dirac occ = 1/(e^{E/T}+1)
        print(f"   {label:42s} occupations {np.round(occ, 4)}  ->  per-mode T/T_c {np.round(Tk, 3)}")


if __name__ == "__main__":
    part_a()
    part_b()
