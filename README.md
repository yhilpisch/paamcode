# Python and AI for Asset Management — Code & Notebooks

<p align="right">
  <img src="https://hilpisch.com/tpq_logo_bic.png" alt="The Python Quants" width="25%">
</p>

This repository contains the Jupyter notebooks and Python scripts that accompany the *Python and AI for Asset Management* class and book in the CPF Program. The material covers the eight parts of the main text:

- Part I — Foundations of Asset Management and Quantitative Methods  
- Part II — Classical Asset Management Theory and Practice  
- Part III — Risk Measures and Portfolio Risk Management
- Part IV — Financial Data Science for Asset Management
- Part V — Machine Learning for Asset Management
- Part VI — Reinforcement and Unsupervised Learning
- Part VII — From Research to Production: Risk, Governance, and Infrastructure  
- Part VIII — LLMs, Agents, and Modern AI in Asset Management

The notebooks combine narrative, mathematics, and Python to reproduce central examples from the class, while the scripts provide focused, reusable implementations for figures, diagnostics, and numerical experiments.

## Book

The material follows the structure of the book *Python and AI for Asset Management* by Dr. Yves J. Hilpisch.

<p align="center">
  <img src="https://hilpisch.com/cpf_logo.png" alt="CPF Program" width="35%">
</p>

## Structure

- `notebooks/` — chapter and appendix notebooks (`chXX_*.ipynb`, `appx_*.ipynb`) that bring together concepts, code, and plots.
- `code/` — standalone Python modules and helper scripts used for figures, simulations, and risk and performance calculations.
- `data/` — source CSV datasets required by the notebooks and scripts (for example, `pyaiam_eod.csv`); generated artifacts such as Parquet files are intentionally excluded from sync.

See the `README.md` files inside `notebooks/` and `code/` for concise per-file overviews.

## Usage

The notebooks are designed to run in a standard scientific Python environment (or in Google Colab) with the usual stack:

- Python 3.11+  
- `numpy`, `pandas`, `matplotlib`  
- `scikit-learn`, `statsmodels` (selected examples)
- `torch` (deep learning, sequence models, and RL examples)

Run the scripts under `code/` as standalone programs from the repository root. Reusable functions can be imported from scripts that expose an explicit function interface.

## Disclaimer

This repository and its contents are provided for educational and illustrative purposes only and come without any warranty or guarantees of any kind — express or implied. Use at your own risk. The authors and The Python Quants GmbH are not responsible for any direct or indirect damages, losses, or issues arising from the use of this code. Do not use the provided examples for critical decision‑making, financial transactions, medical advice, or production deployments without rigorous review, testing, and validation.

Some examples may reference third‑party libraries, datasets, services, or application programming interfaces that are subject to their own licenses and terms; you are responsible for ensuring compliance.

## Contact

- Email: [team@tpq.io](mailto:team@tpq.io)  
- Linktree: [linktr.ee/dyjh](https://linktr.ee/dyjh)  
- CPF Program: [python-for-finance.com](https://python-for-finance.com)  
- The AI Engineer: [theaiengineer.dev](https://theaiengineer.dev)  
- The Crypto Engineer: [thecryptoengineer.dev](https://thecryptoengineer.dev)
