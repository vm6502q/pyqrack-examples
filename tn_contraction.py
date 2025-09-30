import math
import random
import sys

import numpy as np

from qiskit_quimb import quimb_circuit
import quimb.tensor as qtn

from pyqrackising import convert_quimb_tree_to_tsp, tsp_symmetric


def generate_qv_circuit(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    lcv_range = range(width)
    all_bits = list(lcv_range)

    circ = qtn.Circuit(width)
    contraction_set = []
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            tag = f"U3_d{d}_q{i}"
            circ.apply_gate('U3', th, ph, lm, i, tags=tag)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            tag = f"CNOT_d{d}_c{c}_t{t}"
            circ.apply_gate('CNOT', (c, t), tags=tag)
            contraction_set.append((f"U3_d{d}_q{c}", tag))
            contraction_set.append((f"U3_d{d}_q{t}", tag))

    circ = circ.psi
    for tags in contraction_set:
        circ.contract_between(tags[0], tags[1])

    return circ


def safe_contract_between(tn, tags1, tags2):
    tensors1 = tn.select(tags1).tensors
    tensors2 = tn.select(tags2).tensors
    if not tensors1 or not tensors2:
        print(f"[WARN] Skipping contraction between {tags1} and {tags2} (missing tensors)")
        return
    t1_tag = next(iter(tensors1[0].tags))
    t2_tag = next(iter(tensors2[0].tags))
    tn.contract_between(t1_tag, t2_tag)


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_qiskit_validation.py [width] [depth]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Generate the circuit
    quimb_tn = generate_qv_circuit(width, depth)
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
    MAX_BYTES = 16 * 2**30  # 16 GiB
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
            result_shape = [quimb_tn.ind_map.get(ix, 2) for ix in result_inds]  # default dimension 2
            result_bytes = int(np.prod(np.array(result_shape))) * t1.data.itemsize

            if result_bytes > MAX_BYTES:
                print(f"[SKIP] Est. contraction size {result_bytes / 2**30:.2f} GiB > limit ({MAX_BYTES / 2**30:.0f} GiB)")
                tags = set(n_tags)  # Reset tags to start new path
                continue

            # Contract safely
            safe_contract_between(quimb_tn, tags, n_tags)
            tags = tags.union(n_tags)

    print("Contraction result:")
    print(quimb_tn)

    return 0


if __name__ == "__main__":
    sys.exit(main())
