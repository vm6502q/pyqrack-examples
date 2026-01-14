import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# NOT data
X = torch.tensor([[0.],
                  [1.]])

Y = torch.tensor([[1.],
                  [0.]])

# Model
class NOTNet(nn.Module):
    def __init__(self):
        super(NOTNet, self).__init__()
        self.readout = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.readout(x))

model = NOTNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/1000], Loss: {loss.item():.4f}")

# Evaluation
with torch.no_grad():
    print("NOT predictors:\n", X)
    predictions = model(X).round()
    print("NOT predictions:\n", predictions)
