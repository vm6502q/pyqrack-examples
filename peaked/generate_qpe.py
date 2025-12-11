# QPE generation
# Produced by Elara (OpenAI custom GPT)

import math
import sys

def qpe_qasm(num_counting_qubits: int, k: int = 1) -> str:
    """
    Generate an OpenQASM 2.0 circuit for textbook QPE with:
      - num_counting_qubits t
      - 1 system qubit
      - phase phi = k / 2^t, so the outcome should be the t-bit binary of k.

    Qubit layout:
      q[0..t-1]  : counting register (LSB at q[0])
      q[t]       : system qubit, prepared in |1> eigenstate of U = Rz(2*pi*phi)

    Returns
    -------
    qasm : str
        QASM source code as a string.
    """

    t = num_counting_qubits
    n_qubits = t + 1

    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        f'qreg q[{n_qubits}];',
        f'creg c[{t}];',
        '',
        '// Prepare system qubit in |1> (eigenstate of Rz)',
        f'x q[{t}];',
        '',
        '// Hadamards on counting register',
    ]

    # Hadamards on counting qubits
    for j in range(t):
        lines.append(f'h q[{j}];')

    lines.append('')
    lines.append('// Controlled-U^{2^j} with U = Rz(2*pi*k/2^t) on q[t]')

    # Phase phi = k / 2^t
    denom = 2 ** t
    for j in range(t):
        power = 2 ** j
        theta = 2 * math.pi * k * power / denom  # angle for Rz in this layer

        # Implement controlled-Rz(theta) using two CNOTs and Rz splits:
        # CRz(theta) = Rz(theta/2) -- CX -- Rz(-theta/2) -- CX
        half = theta / 2.0

        lines.append(f'// layer j={j}: theta = {theta}')
        lines.append(f'u1({half}) q[{t}];')
        lines.append(f'cx q[{j}], q[{t}];')
        lines.append(f'u1({-half}) q[{t}];')
        lines.append(f'cx q[{j}], q[{t}];')

    lines.append('')
    lines.append('// Inverse QFT on counting register')

    # Optional: reverse qubit order to match standard QPE output
    for j in range(t // 2):
        lines.append(f'swap q[{j}], q[{t-1-j}];')

    # Inverse QFT: for j = 0..t-1
    #   for m = 0..j-1: apply controlled phase -pi/2^{j-m} from m to j
    #   then H on j
    for j in range(t):
        # controlled phase rotations
        for m in range(j):
            angle = -math.pi / (2 ** (j - m))
            # Implement controlled-phase via CRz on target j
            half = angle / 2.0
            lines.append(f'// inverse QFT phase: m={m}, j={j}, angle={angle}')
            lines.append(f'u1({half}) q[{j}];')
            lines.append(f'cx q[{m}], q[{j}];')
            lines.append(f'u1({-half}) q[{j}];')
            lines.append(f'cx q[{m}], q[{j}];')

        lines.append(f'h q[{j}];')

    lines.append('')
    lines.append('// Measure counting register')
    for j in range(t):
        lines.append(f'measure q[{j}] -> c[{j}];')

    return "\n".join(lines)


def main():
    n = 35
    k = 5
    file_out = "qpe.qasm"
    if len(sys.argv) > 1:
        power = int(sys.argv[1])
    if len(sys.argv) > 2:
        power = int(sys.argv[2])
    if len(sys.argv) > 3:
        file_out = str(sys.argv[3])

    qasm_str = qpe_qasm(n, k=k)
    with open(file_out, "w") as f:
        f.write(qasm_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
