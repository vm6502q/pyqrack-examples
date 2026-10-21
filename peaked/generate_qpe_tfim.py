# QPE generation
# Produced by Elara (OpenAI custom GPT)

import math
import sys


epsilon = math.pi * sys.float_info.epsilon


def qpe_ising_z_qasm(num_counting_qubits: int,
                     num_system_qubits: int,
                     k: int = 1,
                     j_coupling: float = 1.0) -> str:
    """
    Generate OpenQASM 2.0 for QPE with a multi-qubit Ising-Z phase source.

    Hamiltonian:
        H_Z = J * sum_{i=0}^{L-1} Z_i

    Unitery:
        U = exp(-i * tau * H_Z)

    We choose tau such that, on the eigenstate |1...1>,
        U |1...1> = exp(i 2π k / 2^t) |1...1>

    Layout:
        q[0..t-1]           : counting register
        q[t..t+L-1]         : system qubits (Ising chain)
        c[0..t-1]           : classical bits for counting measurement

    Parameters
    ----------
    num_counting_qubits : int
        t, number of phase-estimation counting qubits.
    num_system_qubits : int
        L, number of system qubits in the Ising-Z source.
    k : int
        Integer phase index; output should be the t-bit binary of k (mod 2^t).
    j_coupling : float
        Coupling J in H_Z. Only enters via tau; we pick tau so that
        J * L * tau = 2π k / 2^t (up to sign).
    """

    t = num_counting_qubits
    L = num_system_qubits
    n_qubits = t + L

    # We want: eigenphase phi = k / 2^t
    # For state |1...1>, eigenvalue of H_Z is -J * L (since Z|1> = -|1>)
    # So we choose tau such that:
    #   exp(-i * tau * (-J*L)) = exp(i * 2π * k / 2^t)
    # -> tau * J * L = 2π * k / 2^t
    tau = (2.0 * math.pi * k) / (j_coupling * L * (2 ** t))

    # Each single-qubit term contributes exp(-i * tau * J * Z)
    # On |1>, Z = -1, so eigenphase per qubit is +tau * J
    # Total phase = L * tau * J as above.

    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        f'qreg q[{n_qubits}];',
        f'creg c[{t}];',
        '',
        '// Prepare system register in |1...1> eigenstate of H_Z',
    ]

    # X on all system qubits q[t..t+L-1] to make |1...1>
    for s in range(L):
        lines.append(f'x q[{t + s}];')
    lines.append('')

    # Hadamards on counting register
    lines.append('// Hadamards on counting register')
    for j in range(t):
        lines.append(f'h q[{j}];')
    lines.append('')

    # Controlled-U^{2^j} layers with U = exp(-i tau H_Z)
    # Implemented as controlled-Rz on each system qubit
    lines.append('// Controlled-U^{2^j} with U = exp(-i tau * J * sum Z_i)')

    # angle per system qubit for one application of U:
    # U_single = exp(-i tau * J * Z) = Rz(2 * tau * J) up to global phase
    base_theta = 2.0 * tau * j_coupling

    for j in range(t):
        power = 2 ** j
        theta = base_theta * power
        half = theta / 2.0

        if abs(half) < epsilon:
            # AQFT, or at least below system precision
            continue

        lines.append(f'// Layer j={j}: applying controlled Rz({theta}) on each system qubit')

        for s in range(L):
            target = t + s
            # CRz(theta) decomposition using u1 and cx:
            # Rz(half) -- CX -- Rz(-half) -- CX (control on q[j], target on q[target])
            lines.append(f'u1({half}) q[{target}];')
            lines.append(f'cx q[{j}], q[{target}];')
            lines.append(f'u1({-half}) q[{target}];')
            lines.append(f'cx q[{j}], q[{target}];')

    lines.append('')
    lines.append('// Inverse QFT on counting register')

    # Optional: reverse counting qubit order for canonical output
    # for j in range(t // 2):
    #    lines.append(f'swap q[{j}], q[{t - 1 - j}];')

    # Inverse QFT: for j = 0..t-1
    for j in range(t):
        for m in range(j):
            angle = -math.pi / (2 ** (j - m))
            half = angle / 2.0
            if abs(half) < epsilon:
                # AQFT, or at least below system precision
                continue
            lines.append(f'// inverse QFT phase: m={m}, j={j}, angle={angle}')
            lines.append(f'u1({half}) q[{j}];')
            lines.append(f'cx q[{m}], q[{j}];')
            lines.append(f'u1({-half}) q[{j}];')
            lines.append(f'cx q[{m}], q[{j}];')
        lines.append(f'h q[{j}];')

    # lines.append('')
    # lines.append('// Measure counting register')
    # for j in range(t):
    #     lines.append(f'measure q[{j}] -> c[{j}];')

    return "\n".join(lines)


def main():
    t = 12
    L = 24
    k = 100
    j = 1.0
    file_out = "qpe.qasm"
    if len(sys.argv) > 1:
        t = int(sys.argv[1])
    if len(sys.argv) > 2:
        L = int(sys.argv[2])
    if len(sys.argv) > 3:
        k = int(sys.argv[3])
    if len(sys.argv) > 4:
        j = float(sys.argv[4])
    if len(sys.argv) > 5:
        file_out = str(sys.argv[3])

    qasm_str = qpe_ising_z_qasm(t, L, k=k, j_coupling=j)
    with open(file_out, "w") as f:
        f.write(qasm_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
