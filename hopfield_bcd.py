# Train a Hopfield neural network (associative memory) to associate base-2 integers to their respective representations with base-10 "nibbles" (also historically called "binary-coded decimal," "BCD")

import math
import random
import sys
import time

from pyqrack import QrackSimulator, QrackNeuron


def main():
    start = time.perf_counter()

    width_i = 4
    width_o = 5
    sim = QrackSimulator(width_i + width_o, isTensorNetwork=False)
    i_range = range(width_i)
    o_range = range(width_o)
    pow_width = 1 << width_i
    i_mask = pow_width - 1
    o_mask = ((1 << width_o) - 1) << width_i
    eta = 0.5

    neurons = []
    for b in o_range:
        neurons.append(QrackNeuron(sim, i_range, b + width_i))

    for p in range(pow_width):
        # Prepare input.
        sim.reset_all()
        for b in i_range:
            if (p >> b) & 1:
                sim.x(b)

        # Train on output.
        comp = (p % 10) | ((p // 10) << 4)
        for b in o_range:
            neurons[b].learn_permutation(eta, ((comp >> b) & 1))

    sim.reset_all()
    train_time = time.perf_counter() - start

    # with open(f"hopfield_{width}.csv", "w") as f:
    #     for b in i_range:
    #         f.write(str(neurons[b].get_angles()))

    start = time.perf_counter()

    for p in range(pow_width):
        # Prepare input.
        sim.reset_all()
        for b in i_range:
            if (p >> b) & 1:
                sim.x(b)

        # Predict output.
        for n in neurons:
            n.predict()

        result = sim.m_all()

        print("Input", result & i_mask, "produces output", (result & o_mask) >> width_i)

    predict_time = time.perf_counter() - start

    print(width_i, "qubits input; Train: ", train_time, "seconds, Predict:" , predict_time, "seconds.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
