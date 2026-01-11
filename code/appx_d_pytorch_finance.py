"""
Python & AI in Asset Management
Appendix D · PyTorch for Finance

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
parse_dates=['Date']).set_index('Date').sort_index().ffill()
log_rets = np.log(prices['AAPL'] / prices['AAPL'].shift(1)).dropna()
X_np = log_rets.shift(1).dropna().values.astype(np.float32).reshape(-1, 1)
y_np = log_rets.reindex(log_rets.shift(1).dropna().index).values.astype(np.float32)
X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np)
train_ds = TensorDataset(X, y)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# %% cell 8
model = nn.Sequential(nn.Linear(1, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

for epoch in range(5):
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb).squeeze(-1)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}, loss {epoch_loss / len(train_loader):.6f}')
