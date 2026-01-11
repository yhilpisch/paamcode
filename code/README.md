# Code Overview

This folder contains the Python scripts and small helper modules used throughout *Python and AI for Asset Management*. The scripts are organised by chapter and appendix and are designed to be readable and reusable.

## Chapters

- `ch01_asset_management_basics.py` — toy universe setup, basic portfolio operations, and simple back-of-the-envelope calculations.
- `ch02_math_stat_preliminaries.py` — statistical summaries, covariance estimation, and illustrative plots for the math and statistics refresher.
- `ch03_python_infrastructure.py` — project skeleton helpers, file‑system utilities, and environment checks for the research workflow.
- `ch04_mean_variance.py` — mean–variance portfolio construction, efficient frontier calculations, and simple constraint examples.
- `ch05_capm_factor_models.py` — single‑ and multi‑factor regressions, beta estimation, and factor return diagnostics.
- `ch06_black_litterman.py` — Black–Litterman posterior calculations, including views, confidences, and implied equilibrium returns.
- `ch07_risk_measures.py` — volatility, Value-at-Risk (VaR), Expected Shortfall (ES), and drawdown calculations for portfolios.
- `ch08_risk_decomposition.py` — risk contributions, risk‑parity weights, and marginal risk diagnostics.
- `ch09_active_risk_management.py` — active risk, information ratio, and tracking error calculations, plus simple overlay strategies.
- `ch10_data_engineering.py` — feature engineering utilities for end‑of‑day data, including rolling statistics and factor panels.
- `ch11_performance_backtesting.py` — vectorised backtest loops, performance metrics, and equity curve generation.
- `ch12_ml_workflow.py` — helpers for train/validation/test splits, cross‑validation, and feature scaling.
- `ch13_linear_glm.py` — wrappers around linear and logistic models for cross‑sectional and panel‑style problems.
- `ch14_trees_ensembles.py` — tree‑based models, ensembles, and feature importance extraction.
- `ch15_deep_learning.py` — deep cross‑sectional and panel architectures implemented in PyTorch.
- `ch16_sequence_models.py` — sequence models such as recurrent networks and temporal convolutions for return and risk prediction.
- `ch17_rl_foundations.py` — Markov decision process utilities and simple policy evaluation examples.
- `ch18_rl_algorithms.py` — tabular and function‑approximation reinforcement learning algorithms.
- `ch19_unsupervised_representation.py` — clustering and dimensionality‑reduction helpers for feature and state representations.
- `ch20_model_risk_explainability.py` — tooling for model inventories, diagnostics, and explainability metrics.
- `ch21_tech_stack_deployment.py` — deployment helpers, configuration examples, and monitoring hooks.
- `ch22_llms_assistants.py` — abstractions for LLM‑based coding and research assistants.
- `ch23_llms_agents_value_chain.py` — agentic orchestration utilities that connect LLMs to the asset management value chain.

## Appendices

- `appx_b_numpy_pandas.py` — NumPy and pandas recipes for time‑series and panel operations.
- `appx_c_sklearn_cheatsheet.py` — small scikit‑learn examples aligned with the cheat‑sheet appendix.
- `appx_d_pytorch_finance.py` — PyTorch utilities, simple network blocks, and training loops for financial data.
- `appx_f_practical_tools.py` — project, testing, and profiling helpers used in the practical tools appendix.
- `appx_g_repo_colab.py` — utilities for working with the companion repository and Colab notebooks.

## Figure Scripts

The `figures/` subfolder contains small scripts that regenerate selected matplotlib figures used in the slides and main text, such as:

- `ch03_equity_curve.py` — plots a stylised equity curve for a simple backtest.
- `ch07_aapl_price.py` — price trajectory and log‑return histogram for a single stock.
- `ch07_aapl_logret_hist.py` — histogram and kernel density estimate of log‑returns.
- `ch16_kmeans_clusters.py` — cluster visualisation for a stylised feature space.

Each script is designed so that you can import its functions into your own projects or run it directly from the command line to reproduce the corresponding figure.

