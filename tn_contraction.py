import math
import random
import sys

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


def safe_contract_between(tn, tags1, tags2):
    tensors1 = tn.select(tags1).tensors
    tensors2 = tn.select(tags2).tensors
    if not tensors1 or not tensors2:
        print(f"[WARN] Skipping contraction between {tags1} and {tags2} (missing tensors)")
        return
    t1_tag = next(iter(tensors1[0].tags))
    t2_tag = next(iter(tensors2[0].tags))
    tn.contract_between(t1_tag, t2_tag)


# Produced with a ton of help from Elara, the custom OpenAI GPT (and more generally)
def max_amplitude_via_output_inds(tn, phys_inds):
    bitstr = []
    working_tn = tn.copy()

    for qi in phys_inds:
        best_bit = None
        best_amp2 = -1.0

        for bit in (0, 1):
            arr = [1, 0] if bit == 0 else [0, 1]
            proj = qtn.Tensor(data=arr, inds=(qi,))

            test_tn = working_tn.copy()
            test_tn.add_tensor(proj)

            try:
                amp2 = test_tn.contract(all, get='norm')**2
            except Exception:
                amp2 = 0  # fallback in case of numerical instability

            if amp2 > best_amp2:
                best_amp2 = amp2
                best_bit = bit

        bitstr.append(best_bit)

        # Fix this bit in the main network
        arr = [1, 0] if best_bit == 0 else [0, 1]
        proj = qtn.Tensor(data=arr, inds=(qi,))
        working_tn.add_tensor(proj)

    # Final contraction for amplitude (if desired)
    final_amp = working_tn.contract(all)

    return tuple(bitstr), final_amp


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_qiskit_validation.py [width] [depth]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Generate the circuit
    qc = generate_qv_circuit(width, depth)
    quimb_tn = qc.psi
    # Contract single-qubit gates.
    contract_single(quimb_tn)
    # Convert to TSP
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
    tsp_sol, raw_cost = tsp_symmetric(tsp, monte_carlo=False, is_cyclic=False)

    # Break into segments
    segments = []
    segment = [nodes[tsp_sol[0]]]
    cost = 0
    for i in range(len(tsp_sol) - 1):
        val = tsp[tsp_sol[i], tsp_sol[i + 1]]
        if val > 2:
            segments.append(segment)
            segment = []
        else:
            cost += val
        segment.append(nodes[tsp_sol[i + 1]])
    segments.sort(key=len)

    # Print segments and cost:
    print("Optimal contraction path segments and cost:")
    print((segments, cost))

    print("Contracting...")
    MAX_BYTES = 1 << 20 # 1 MB
    for path in segments:
        if len(path) < 2:
            continue
        tags = set(path[0])
        for i in range(len(path) - 1):
            n_tags = path[i + 1]

            # Get tensors
            tensors1 = quimb_tn.select(tags).tensors
            tensors2 = quimb_tn.select(n_tags).tensors
            if not tensors1 or not tensors2:
                print(f"[WARN] Skipping contraction between {tags} and {n_tags} (missing tensors)")
                continue

            t1 = tensors1[0]
            t2 = tensors2[0]

            # Estimate memory usage of contraction
            # Union of indices = resulting tensor indices
            result_inds = set(t1.inds) | set(t2.inds)

            # Manual product of dimensions
            dtype_size = 4  # for float32
            result_bytes = dtype_size
            for ix in result_inds:
                for iy in quimb_tn.ind_map.get(ix, 2):
                    result_bytes *= iy
                    if result_bytes > MAX_BYTES:
                        break
                if result_bytes > MAX_BYTES:
                    break

            if result_bytes > MAX_BYTES:
                # print(f"[SKIP] Exceeded maximum contraction size.")
                tags = set(n_tags)  # Reset tags to start new path
                continue

            # Contract safely
            safe_contract_between(quimb_tn, tags, n_tags)
            tags = tags.union(n_tags)

    contract_single(quimb_tn)
    print("Contraction result:")
    print(quimb_tn)
    quimb_tn.draw()

    quimb_tn = qtn.TensorNetwork(quimb_tn)
    print("Best guess for highest-probability bit string:")
    print(max_amplitude_via_output_inds(quimb_tn, [f"k{qi}" for qi in range(width)])[0])

    return 0


if __name__ == "__main__":
    sys.exit(main())
