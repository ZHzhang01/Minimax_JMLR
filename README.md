


# Experiments

# Causal Inference Under Unmeasured Confounding with Negative Controls: A Minimax Learning Approach

This repository contains **simulation and real-data experiments** accompanying our work on **causal inference under unmeasured confounding using negative controls**, based on **minimax learning of bridge functions**.  
The code reproduces the **MSE, point-wise estimation, bias and variance** reported in the paper, including **permutation-based robustness checks** for negative control variable selection.

---

## Contents

- `real_world_6_variable_permutation/`  
  For Table 14 in our main text. Real-data experiments with **all 6 permutations** of selecting 2 negative control treatments (Z) and 2 negative control outcomes (W) from 4 candidate proxies.

- `syn_*.py` and `result/syn_*.txt`  
  For Table 2-7 in our main text. Synthetic data experiments (MSE, bias, variance) under different sample sizes (`n=400,1200`) and nonlinear transformations (`sin`, `power`).

- `background of real data.ipynb`  
  Significance and comparison analysis upon real data.

- `baseline_2sls(moment-based)`  
  For Table 10 and Table 13 in our main text. 2SLS baseline.

- `real_world_ex_layer_*.ipynb`  
  For Table 8-9 in our main text. Neural-network–based minimax bridge function estimation on real data with different architectures.

  
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


