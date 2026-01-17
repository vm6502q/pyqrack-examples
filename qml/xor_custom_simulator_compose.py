# What if we want to modify the layer simulator, such as to compose it for multiple layers?
# This examples conveys the general idea of how to do so.

import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pyqrack import QrackNeuronTorchLayer, QrackSimulator

# XOR data
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])

Y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]])

# Model
class QrackXORNet(nn.Module):
    def __init__(self):
        super(QrackXORNet, self).__init__()
        self.q = QrackNeuronTorchLayer(2, 1, hidden_qubits=0, lowest_combo_count=2, highest_combo_count=2)

        # We compose 3 additional qubits (which could be from another layer, for example).
        self.q.simulator.compose(QrackSimulator(3), [3, 4, 5])
        # Reset to |0> so we clear the output 50/50 superposition initialization.
        self.q.simulator.reset_all()
        # Set up an Ansatz in the input qubits (Bell pair, in our case).
        self.q.simulator.h(2)
        self.q.simulator.macx([2], 3)
        # Initialize any output qubits to 50/50 superposition (or whatever else is desired).
        self.q.simulator.h(4)
        # Remap the input qubits.
        self.q.input_indices = [2, 3]
        # Remap the output qubits.
        self.q.output_indices = [4]
        # We also need to update the mappings in any and all neurons.
        self.q.neurons[0].neuron.set_qubit_ids(self.q.input_indices, self.q.output_indices[0])
        # (This has just been an example that adds nothing but successfully remaps to produce the expected output.)

        self.readout = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        x = self.q(x)
        x = x - x.mean()
        return torch.sigmoid(self.readout(x))

model = QrackXORNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training loop
for epoch in range(2000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/2000], Loss: {loss.item():.4f}")

# Evaluation
with torch.no_grad():
    print("XOR predictors:\n", X)
    predictions = model(X).round()
    print("XOR predictions:\n", predictions)
