"""
Python & AI in Asset Management
Chapter 19 · Unsupervised Learning and Representation Learning

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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# %% cell 6
prices = pd.read_csv(DATA_PATH,
parse_dates=["Date"]).set_index("Date").sort_index().ffill()

# %% cell 8
prices = pd.read_csv(DATA_PATH,
parse_dates=["Date"]).set_index("Date").sort_index().ffill()
log_rets = np.log(prices / prices.shift(1)).dropna()
window = 60
feature_panel = pd.concat(
    {"momentum": prices.pct_change(window),
     "volatility": log_rets.rolling(window).std()},
    axis=1,
).dropna()
latest = feature_panel.iloc[-1].unstack(0)
scaler = StandardScaler()
X = scaler.fit_transform(latest)
assets = latest.index
latest.head()

# %% cell 10
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X)
clusters = pd.Series(labels, index=assets, name="cluster")
clusters

# %% cell 12
silhouette = silhouette_score(X, labels)
silhouette

# %% cell 14
pca = PCA(n_components=2)
components = pca.fit_transform(X)
fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(
    components[:, 0],
    components[:, 1],
    c=labels,
    cmap="coolwarm",
    s=160,
)
for idx, name in enumerate(assets):
    ax.annotate(
        name,
        (components[idx, 0], components[idx, 1]),
        textcoords="offset points",
        xytext=(6, 6),
    )
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Clusters in PCA Space")
plt.show()

# %% cell 16
cluster_summary = latest.assign(cluster=labels).groupby("cluster").agg(["mean",
"std"])
cluster_summary
