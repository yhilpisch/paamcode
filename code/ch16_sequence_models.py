"""
Python & AI in Asset Management
Chapter 16 · Sequence Models and Temporal Deep Learning

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
log_rets = np.log(prices / prices.shift(1)).dropna()["AAPL"]
window = 30
X_seq = []
y_seq = []
for i in range(len(log_rets) - window - 1):
    X_seq.append(log_rets.iloc[i : i + window].values)
    y_seq.append(log_rets.iloc[i + window + 1])
X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)
X_seq = torch.from_numpy(X_seq).unsqueeze(-1)
y_seq = torch.from_numpy(y_seq)
train_len = int(len(X_seq) * 0.8)
train_loader = DataLoader(TensorDataset(X_seq[:train_len], y_seq[:train_len]),
batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(X_seq[train_len:], y_seq[train_len:]),
batch_size=256)

# %% cell 8
class ReturnLSTM(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        last = h[:, -1, :]
        return self.out(last).squeeze(-1)

lstm_model = ReturnLSTM()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# %% cell 10
def train_seq(model, epochs=5):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_losses = [loss_fn(model(xb), yb).item() for xb, yb in val_loader]
        print(f"Epoch {epoch+1}, val loss {np.mean(val_losses):.6f}")

train_seq(lstm_model, epochs=5)

# %% cell 12
class SimpleTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.input_proj(x)
        encoded = self.encoder(h)
        last = encoded[:, -1, :]
        return self.out(last).squeeze(-1)

transformer = SimpleTransformer()
opt_trans = torch.optim.Adam(transformer.parameters(), lr=1e-3)

# %% cell 14
for epoch in range(3):
    transformer.train()
    for xb, yb in train_loader:
        opt_trans.zero_grad()
        loss = loss_fn(transformer(xb), yb)
        loss.backward()
        opt_trans.step()
    transformer.eval()
    with torch.no_grad():
        val_losses = [loss_fn(transformer(xb), yb).item() for xb, yb in val_loader]
    print(f"Epoch {epoch+1}, val loss {np.mean(val_losses):.6f}")
