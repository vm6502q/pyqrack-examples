import sys

from collections import Counter
from itertools import product

import numpy as np

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile

import quimb as qu
from qiskit_quimb import quimb_circuit


def int_to_bitstring(integer, length, reverse):
    return (bin(integer)[2:].zfill(length))[::-1] if reverse else (bin(integer)[2:].zfill(length))


def run_qasm(file_in, file_out):
    n = 3

    qc_orig = QuantumCircuit.from_qasm_file(file_in)
    basis_gates = ['h', 's', 't', 'rx', 'ry', 'rz', 'cx', 'cy', 'cz', 'swap', 'iswap', 'ccx', 'ccz', 'cswap', 'u3']
    qc_quimb = transpile(qc_orig, basis_gates=basis_gates)
    quimb_circ = quimb_circuit(qc_quimb)
    bitstring = "0010001100101011000011101100011010"
    print(f"Bit string: {bitstring}")
    best_prob = abs(quimb_circ.amplitude(bitstring)) ** 2
    print(f"Probability: {best_prob}")

def main():
    file_in = "qft.qasm"
    file_out = "out.json"
    if len(sys.argv) > 1:
        file_in = str(sys.argv[1])
    if len(sys.argv) > 2:
        file_out = str(sys.argv[2])

    run_qasm(file_in, file_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
