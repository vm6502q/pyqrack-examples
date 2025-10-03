import heapq
import math
import random
import sys

import psutil

from collections import Counter

import numpy as np

import quimb.tensor as qtn

from pyqrackising import convert_quimb_tree_to_tsp, tsp_symmetric


def generate_qv_circuit(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    lcv_range = range(width)
    all_bits = list(lcv_range)

    circ = qtn.Circuit(width)
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            circ.apply_gate('U3', th, ph, lm, i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            circ.apply_gate('CNOT', c, t)

    return circ


def contract_single(tn):
    # Contract to the right
    for tensor in tn.tensors:
        if len(tensor.inds) > 2:
            continue
        right_inds = set(tensor.inds)
        if hasattr(tensor, 'left_inds') and tensor.left_inds is not None:
            right_inds -= set(tensor.left_inds)
        for idx in right_inds:
            if idx in tn.ind_map and len(tn.ind_map[idx]) == 2:
                tn.contract_ind(idx)

    # Contract dangling indices (1-edge tensors)
    while True:
        num_tensors = len(tn.tensors)
        for tensor in tn.tensors[:]:  # Use a c1opy of the list to avoid mutation issues
            if len(tensor.inds) == 1:
                tn.contract_ind(tensor.inds[0])
        if num_tensors == len(tn.tensors):
            break


# Produced with a ton of help from Elara, the custom OpenAI GPT (and more generally)
def max_amplitude_beam_search(tn, phys_inds, beam_width=4):
    # Each beam entry is a tuple: (neg_amp2, bitstring_so_far, network_so_far)
    # We negate amp2 to use Python's min-heap as a max-heap
    beam = [(-1.0, [], tn.copy())]

    for qi in phys_inds:
        new_beam = []

        for neg_amp2, bitstr, partial_tn in beam:
            for bit in (0, 1):
                # Project this bit
                arr = [1, 0] if bit == 0 else [0, 1]
                proj = qtn.Tensor(data=arr, inds=(qi,))
                test_tn = partial_tn.copy()
                test_tn.add_tensor(proj)

                cntrct = test_tn.contract(all, optimize='auto-hq')
                amp2 = (np.abs(cntrct.norm()) if isinstance(cntrct, qtn.Tensor) else np.abs(cntrct)) ** 2

                new_beam.append((-amp2, bitstr + [bit], test_tn))

        # Keep only top `beam_width` candidates
        beam = heapq.nsmallest(beam_width, new_beam, key=lambda x: x[0])

    # Return the best-scoring final result
    best_neg_amp2, best_bits, final_tn = beam[0]
    final_amp = final_tn.contract(all, optimize='auto-hq')

    return tuple(best_bits), final_amp


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_qiskit_validation.py [width] [depth]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Generate the circuit
    print("Generating circuit...")
    qc = generate_qv_circuit(width, depth)
    quimb_tn = qc.psi
    # Contract single-qubit gates.
    print("Pruning leaves...")
    contract_single(quimb_tn)
    # Convert to TSP
    print("Converting to tsp matrix...")
    tsp, _nodes = convert_quimb_tree_to_tsp(quimb_tn)
    # Isolate unique tags:
    nodes = []
    i = 0
    to_remove = []
    for n0 in _nodes:
        n = n0.copy()
        for n1 in _nodes:
            if n0 != n1:
                n = (n ^ n1) & n
        if len(n) == 0:
            to_remove.append(i)
        else:
            nodes.append(n)
        i += 1
    _nodes = None
    for rc in reversed(to_remove):
        tsp = np.delete(tsp, rc, axis=0)
        tsp = np.delete(tsp, rc, axis=1)

    # Solve TSP
    print("Solving TSP...")
    tsp_sol, raw_cost = tsp_symmetric(tsp, monte_carlo=False, is_cyclic=False)

    # Break into segments
    id_tag_set = set()
    id_tag_set.update(nodes[tsp_sol[0]])
    segments = []
    segment = [frozenset(nodes[tsp_sol[0]])]
    cost = 0
    for i in range(len(tsp_sol) - 1):
        val = tsp[tsp_sol[i], tsp_sol[i + 1]]
        if val > 2:
            segments.append(segment)
            segment = []
        else:
            cost += val
        segment.append(frozenset(nodes[tsp_sol[i + 1]]))
        id_tag_set.update(nodes[tsp_sol[i + 1]])
    segments.sort(key=len)

    # Print segments and cost:
    print("Optimal contraction path segments and cost:")
    print((segments, cost))

    itemsize = quimb_tn.tensors[0].data.itemsize

    keys = [p.copy() for p in segments]
    tag_to_index = {}
    tag_to_inds = {}
    for i, t in enumerate(quimb_tn.tensors):
        uid = frozenset(set(t.tags) & id_tag_set)
        tag_to_index[uid] = i
        result_bytes = itemsize
        for ix in t.inds:
            for iy in quimb_tn.ind_map.get(ix, 2):
                result_bytes *= iy
        tag_to_inds[uid] = result_bytes

    i = 0
    while len(keys[i]) == 1:
        keys[i] = tag_to_index[keys[i][0]]

    byte_count = itemsize << 1
    is_more = True
    while is_more:
        is_more = False
        n_keys = []
        for path in keys:
            if len(path) < 2:
                continue

            key = path[0]
            n_key = [key]
            for i in range(len(path) - 1):
                o_key = path[i + 1]

                # Manual product of dimensions
                result_bytes = tag_to_inds[key] * tag_to_inds[o_key]

                if result_bytes > byte_count:
                    # Reset tags to start new path
                    n_key.append(key)
                    key = o_key
                    is_more = True
                    continue

                # Contract safely
                tag_to_inds[key] = result_bytes
                contracted_index = tag_to_index[o_key]
                path.append((tag_to_index[key], contracted_index))
                for key, value in tag_to_index.items():
                    if value >= contracted_index:
                        tag_to_index[key] -= 1

            if n_key[-1] != key:
                n_key.append(key)
            n_keys.append(n_key)

        keys = n_keys
        byte_count <<= 1

    print("Contraction path:")
    print(path)

    print("Contracting...")
    quimb_tn.contract(optimize=tuple(path))

    print("Contraction result:")
    print(quimb_tn)
    quimb_tn.draw()

    quimb_tn = qtn.TensorNetwork(quimb_tn)
    print("Best guess for highest-probability bit string:")
    print(max_amplitude_beam_search(quimb_tn, [f"k{qi}" for qi in range(width)])[0])

    return 0


if __name__ == "__main__":
    sys.exit(main())
