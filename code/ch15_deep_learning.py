"""
Python & AI in Asset Management
Chapter 15 · Deep Learning for Cross-Sectional and Panel Data

(c) Dr. Yves J. Hilpisch
AI-Powered by GPT 5.1
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

# %% cell 4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("seaborn-v0_8")
plt.rcParams.update({"font.family": "serif", "figure.dpi": 300})

DATA_PATH = Path("data/pyaiam_eod.csv")
if not DATA_PATH.exists():
    DATA_PATH = "https://hilpisch.com/pyaiam_eod.csv"

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# %% cell 6
prices = pd.read_csv(DATA_PATH,
parse_dates=["Date"]).set_index("Date").sort_index().ffill()

# %% cell 8
panel = prices.ffill().pct_change().dropna()
features = panel.rolling(20).mean().stack().dropna()
labels = panel.shift(-1).stack().reindex(features.index).dropna()
features = features.loc[labels.index]
X = torch.from_numpy(features.values.astype(np.float32)).unsqueeze(-1)
y = torch.from_numpy(labels.values.astype(np.float32))
train_size = int(0.8 * len(X))
train_ds = TensorDataset(X[:train_size], y[:train_size])
val_ds = TensorDataset(X[train_size:], y[train_size:])
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512)

# %% cell 10
class MLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# %% cell 12
def train(model, epochs=5):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_loader:
                val_losses.append(loss_fn(model(xb), yb).item())
        print(f"Epoch {epoch+1}, val loss {np.mean(val_losses):.5f}")

train(model, epochs=5)

# %% cell 14
class AutoEncoder(nn.Module):
    def __init__(self, latent=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return latent, recon

auto = AutoEncoder()
opt = torch.optim.Adam(auto.parameters(), lr=1e-3)

# %% cell 16
for epoch in range(5):
    auto.train()
    epoch_loss = 0
    for xb, _ in train_loader:
        opt.zero_grad()
        _, recon = auto(xb)
        loss = loss_fn(recon.squeeze(-1), xb.squeeze(-1))
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, loss {epoch_loss / len(train_loader):.5f}")

with torch.no_grad():
    latent_factors, _ = auto(X)
latent_factors[:5]
