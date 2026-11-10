# Wheeler-DeWitt / minisuperspace toy for the ghost-condensate picture, on Qrack.
#
# By (Anthropic) Claude and Dan Strano.
#
# Minisuperspace: two modes of one wavenumber k.
#   ghost mode  a : the scalar ("monopole") mode of the gravitational field, WRONG-sign energy  -w a^dag a
#   matter mode b : the positive-rest-mass quanta it condenses into,                           +w b^dag b
# Their monopole coupling creates/annihilates them in pairs (back-to-back momenta +k, -k):
#   K = kappa (a^dag b^dag + a b).
# Constraints (from global conservation, the "outside perspective on time"):
#   Hamiltonian (Wheeler-DeWitt):  C_E = w (N_b - N_a),   C_E |Psi> = 0
#   momentum:                      C_P = k (N_b - N_a) + P0 (here the pair term conserves it; P0 = 0 default)
# K commutes with both constraints (pairs carry -w + w = 0 energy and -k + k = 0 momentum), even after truncating
# each mode to 2^n Fock states, so Qrack can check the constraints exactly while K drives "condensation".
#
# Time is recovered relationally (Page & Wootters 1983): a clock register |t> is entangled with the system,
#   |Psi> = 2^(-c/2) sum_t |t> (x) exp(-i K t dt) |0_a, 0_b>,
# a single timeless state satisfying the constraints. Conditioning on a clock reading t gives the "dynamics":
# pair number grows, ghost energy goes negative, matter energy goes positive, and they cancel exactly at every t.
# Truncation at 2^n quanta per mode plays the role of the condensate cutoff: the runaway saturates and recurs.
#
# Usage: python3 wdw_ghost_qrack.py [n_mode_qubits=3] [clock_qubits=4] [kappa*dt=0.15] [trotter_steps_per_unit=16] [sector m=0]
#   m > 0 starts with m matter quanta: the nonzero (but conserved) constraint sector N_b - N_a = m, E_total = m*w.

import itertools
import sys

import numpy as np
from pyqrack import Pauli, QrackSimulator

PAULIS = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
ENUM = {"X": Pauli.PauliX, "Y": Pauli.PauliY, "Z": Pauli.PauliZ}


def fock_ops(n):
    d = 2 ** n
    a = np.diag(np.sqrt(np.arange(1, d)), 1)          # truncated annihilator
    return a, np.diag(np.arange(d))


def pauli_decompose(H, nq):
    """H (2^nq square, little-endian: qubit 0 = least significant bit) -> [(coeff, 'string over qubits 0..nq-1')]."""
    terms = []
    for s in itertools.product("IXYZ", repeat=nq):
        M = np.array([[1]])
        for ch in reversed(s):                          # qubit nq-1 is the most significant (leftmost kron)
            M = np.kron(M, PAULIS[ch])
        c = np.trace(M.conj().T @ H).real / 2 ** nq
        if abs(c) > 1e-12 and any(ch != "I" for ch in s):
            terms.append((c, "".join(s)))
    return terms


def main():
    a_ = sys.argv[1:]
    n = int(a_[0]) if len(a_) > 0 else 3
    c = int(a_[1]) if len(a_) > 1 else 4
    kdt = float(a_[2]) if len(a_) > 2 else 0.15
    spu = int(a_[3]) if len(a_) > 3 else 16
    m0 = int(a_[4]) if len(a_) > 4 else 0       # momentum/energy sector: initial matter quanta, N_b - N_a = m0

    a, N = fock_ops(n)
    I = np.eye(2 ** n)
    A, B = np.kron(I, a), np.kron(a, I)                 # register order: a on qubits [0,n), b on [n,2n)
    Na, Nb = np.kron(I, N), np.kron(N, I)
    K = A.conj().T @ B.conj().T + B @ A                 # kappa = 1; time in units of 1/kappa
    C = Nb - Na                                         # WdW / momentum constraint (w = k = 1)
    assert np.allclose(K @ C - C @ K, 0)

    # K = sum over pair transitions |i,j> <-> |i+1,j+1> with amplitude sqrt((i+1)(j+1)). Each transition is a
    # two-level (Givens) rotation between basis states of EQUAL constraint value, so every Trotter factor commutes
    # with C exactly: the constraint holds to machine precision for any step size (a Pauli-string Trotterization
    # does not: individual Pauli terms leak out of the physical sector).
    d = 2 ** n
    moves = [(i + d * j, (i + 1) + d * (j + 1), np.sqrt((i + 1) * (j + 1))) for i in range(d - 1) for j in range(d - 1)]
    sys_q = list(range(2 * n))
    clk_q = list(range(2 * n, 2 * n + c))
    sim = QrackSimulator(2 * n + c)
    for bit in range(n):
        if (m0 >> bit) & 1:
            sim.x(n + bit)
    for q in clk_q:
        sim.h(q)

    def givens(x, y, theta, cq):
        """exp(-i theta (|x><y| + |y><x|)) on the system, controlled on clock qubit cq."""
        diff = [q for q in sys_q if ((x ^ y) >> q) & 1]
        k = diff[0]
        for dq in diff[1:]:
            sim.mcx([k], dq)                            # now x, y differ only in bit k
        xr = x
        for dq in diff[1:]:
            if (xr >> k) & 1:
                xr ^= 1 << dq
        ctrls = [q for q in sys_q if q != k] + [cq]
        perm = 0
        for i, q in enumerate(ctrls[:-1]):
            if (xr >> q) & 1:
                perm |= 1 << i
        perm |= 1 << (len(ctrls) - 1)
        cs, sn = np.cos(theta), np.sin(theta)
        sim.ucmtrx(ctrls, [cs, -1j * sn, -1j * sn, cs], k, perm)
        for dq in diff[1:][::-1]:
            sim.mcx([k], dq)

    # controlled exp(-i K dt 2^j) on clock bit j: second-order (symmetric) Trotter over the Givens factors
    for j, cq in enumerate(clk_q):
        tau = kdt * 2 ** j
        steps = max(1, int(np.ceil(tau * spu)))
        h = tau / steps
        for _ in range(steps):
            for x, y, amp in moves + moves[::-1]:
                givens(x, y, amp * h / 2, cq)

    psi = np.array(sim.out_ket()).reshape(2 ** c, 2 ** (2 * n))   # [clock, system]
    # global (timeless) checks
    Csys = np.kron(np.eye(2 ** c), C)
    flat = psi.reshape(-1)
    print(f"modes: 2^{n} Fock states each; clock: 2^{c} ticks of kappa*dt = {kdt}; sector N_b - N_a = {m0}")
    print(f"global constraint <(C - m0)^2> = {np.vdot(flat, ((Csys - m0 * np.eye(len(Csys))) @ (Csys - m0 * np.eye(len(Csys)))) @ flat).real:.2e}   (Wheeler-DeWitt / momentum constraint: 0)")
    print(f"\n{'clock t':>7s} {'kappa t':>8s} {"<N_matter>":>10s} {'<E_ghost>':>10s} {'<E_matter>':>10s} {'<E_total>':>10s} "
          f"{'exact <N>':>10s}")
    from scipy.linalg import expm
    for t in range(2 ** c):
        v = psi[t] * np.sqrt(2 ** c)                    # condition on clock reading t
        Eg = -np.vdot(v, Na @ v).real
        Em = np.vdot(v, Nb @ v).real
        ex = expm(-1j * K * kdt * t)[:, m0 * d]               # exact (untrotterized) check from |0,0>
        print(f"{t:7d} {kdt * t:8.2f} {Em:10.4f} {Eg:10.4f} {Em:10.4f} {Eg + Em:10.4f} {np.vdot(ex, Nb @ ex).real:10.4f}")
    print("\nE in units of the mode frequency w. Pairs are created from the constraint-satisfying vacuum at no net")
    print("energy (ghost -w, matter +w per pair); the truncation (condensate cutoff) bounds and recycles the growth.")


if __name__ == "__main__":
    main()
