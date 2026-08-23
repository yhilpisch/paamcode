"""
Generate a simple 2D clustering visualization for Chapter 16.

The script uses scikit-learn's ``make_blobs`` to create a toy dataset with
three clusters and then applies k-means clustering. The resulting scatter
plot shows points colored by cluster label and is saved as a PNG for
inclusion in the LaTeX sources.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def main() -> None:
    # Configure matplotlib for book-style figures
    plt.style.use("seaborn-v0_8")
    rcParams["figure.figsize"] = (6.0, 4.0)
    rcParams["figure.dpi"] = 300
    rcParams["axes.grid"] = True
    rcParams["lines.linewidth"] = 1.0

    # Generate a toy dataset with three clusters in 2D
    X, _ = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=0.6,
        random_state=0,
    )

    # Fit k-means and predict cluster labels
    km = KMeans(n_clusters=3, n_init=10, random_state=0)
    labels = km.fit_predict(X)

    # Plot the clusters
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap="tab10",
        s=15,
        edgecolor="none",
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Toy 2D clusters from k-means")

    # Add cluster centers for illustration
    centers = km.cluster_centers_
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c="black",
        s=40,
        marker="x",
        linewidths=1.0,
        label="Cluster centers",
    )
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig("figures/ch16_kmeans_clusters.png", bbox_inches="tight")


if __name__ == "__main__":
    main()

