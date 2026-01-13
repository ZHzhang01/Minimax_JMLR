# Minimax_JMLR


# Minimax Learning with Negative Controls — Experiments & Replication

This repository contains **simulation and real-data experiments** accompanying our work on **causal inference under unmeasured confounding using negative controls**, based on **minimax learning of bridge functions**.  
The code reproduces the **MSE, bias, variance, point-wise estimation, confidence interval** reported in the paper, including **permutation-based robustness checks** for negative control variable selection.

---

## Contents

- `real_world_6_variable_permutation/`  
  Real-data experiments with **all 6 permutations** of selecting 2 negative control treatments (Z) and 2 negative control outcomes (W) from 4 candidate proxies.

- `result/`  
  Saved numerical outputs (`.txt`) for simulation experiments (power, bias, variance).

- `*_syn_*.py / *.txt`  
  Synthetic data experiments under different sample sizes (`n=400,1200`) and nonlinear transformations (`sin`, `power`).

- `real_world_ex_layer_*.ipynb`  
  Neural-network–based minimax bridge function estimation on real data with different architectures.

- `Background of real data.ipynb`  
  Significance and comparison analysis.

---

## Method Summary

We estimate causal effects under **unmeasured confounding** using **negative controls**, by:
- Learning **outcome and action bridge functions** via **minimax strategys**
- Allowing **non-unique bridge functions** (no completeness assumption)
- Supporting **linear, RKHS, and neural network...** hypothesis classes
- Evaluating robustness across **proxy permutations**

The estimators include:
- IPW / REG / DR (with and without stabilizers)
- Other baselines

---

## Running Guide

All experiments can be run **directly in Python 3.13.5**.


