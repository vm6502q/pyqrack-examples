# Train a Hopfield neural network (associative memory) to associate integers with their respective "two's complement" (equivalent to multiplying a common signed integer by -1)

import math
import random
import sys
import time

from pyqrack import QrackSimulator, QrackNeuron


def main():
    start = time.perf_counter()

    width = 4
    if len(sys.argv) > 1:
        width = int(sys.argv[1])
    width_x2 = width << 1
    sim = QrackSimulator(width_x2, isTensorNetwork=False)
    i_range = range(width)
    o_range = range(width, width_x2)
    pow_width = 1 << width
    i_mask = pow_width - 1
    o_mask = i_mask << width
    eta = 0.5

    neurons = []
    for b in o_range:
        neurons.append(QrackNeuron(sim, i_range, b))

    for p in range(pow_width):
        # Prepare input.
        sim.reset_all()
        for b in i_range:
            if (p >> b) & 1:
                sim.x(b)

        # Train on output.
        comp = ((~p) + 1) & i_mask
        for b in i_range:
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

        print("Input", result & i_mask, "produces output", (result & o_mask) >> width)

    predict_time = time.perf_counter() - start

    print(
        width,
        "qubits input; Train: ",
        train_time,
        "seconds, Predict:",
        predict_time,
        "seconds.",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
