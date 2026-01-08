import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pyqrack import QrackNeuronTorchLayer

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
        half_pi = math.pi / 2
        self.q = QrackNeuronTorchLayer(2, 1, hidden_qubits=0, lowest_combo_count=2, highest_combo_count=2, parameters=[-half_pi, half_pi, half_pi, -half_pi])

    def forward(self, x):
        x = self.q(x)
        return x

model = QrackXORNet()
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
# for epoch in range(1000):
#     optimizer.zero_grad()
#     outputs = model(X)
#     loss = criterion(outputs, Y)
#     loss.backward()
#     optimizer.step()
#    
#     if epoch % 100 == 0:
#         print(f"Epoch [{epoch}/1000], Loss: {loss.item():.4f}")

# Evaluation
with torch.no_grad():
    print("XOR predictors:\n", X)
    predictions = model(X).round()
    print("XOR predictions:\n", predictions)
