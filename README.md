# Reproducing *Causal Inference Under Unmeasured Confounding with Negative Controls: A Minimax Learning Approach*

This repository contains standalone Jupyter notebooks for reproducing the simulation and real-data experiments in the paper by Nathan Kallus, Xiaojie Mao, Masatoshi Uehara, and Zhiheng Zhang.

The paper studies causal estimation when important confounders are unobserved but negative-control variables are available. Its estimators learn outcome and action bridge functions through minimax objectives, support flexible function classes such as neural networks and RKHS critics, and combine the learned bridges through IPW, regression, and doubly robust estimating equations.

## What is included

Each notebook is self-contained: it includes the data-generating process or a frozen copy of the raw RHC data, preprocessing, estimator implementations, uncertainty calculations, formatting, and integrity checks. The notebooks do not import project-local modules or read cached result files.

| Notebook | Reproduced analysis |
| --- | --- |
| `Table_02.ipynb` | Table 2: synthetic experiment, `g(t) = sin(t)`, `n = 400` |
| `Table_03.ipynb` | Table 3: synthetic experiment, `g(t) = sin(t)`, `n = 1200` |
| `Table_04.ipynb` | Table 4: synthetic experiment, `g(t) = t^3`, `n = 400` |
| `Table_05.ipynb` | Table 5: synthetic experiment, `g(t) = t^3`, `n = 1200` |
| `Table_06.ipynb` | Table 6: no-unmeasured-confounding ablation, `g(t) = sin(t)`, `n = 400` |
| `Table_07.ipynb` | Table 7: no-unmeasured-confounding ablation, `g(t) = t^3`, `n = 400` |
| `Table_08.ipynb` | Table 8: main right-heart-catheterization (RHC) analysis |
| `Table_09.ipynb` | Table 9: neural-network width/depth sensitivity |
| `Table_10.ipynb` | Table 10: linear moment-based basis sensitivity |
| `Table_13.ipynb` | Table 13: six proxy permutations with the moment-based linear baseline |
| `Table_14.ipynb` | Table 14: six proxy permutations with the closed-form linear baseline |
| `Reviewer_X_Omission_Stress_test.ipynb` | Tables 15-16: RHC covariate-omission stress test with 28 full-$X$, nested-omission, and no-$X$ scenarios |

## Environment

The notebook metadata records Python 3.13.5. Install the scientific Python packages used by the notebooks in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyterlab ipython numpy pandas torch scikit-learn linearmodels matplotlib
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead.

## Running the reproductions

Start Jupyter from the repository root:

```bash
jupyter lab
```

Open one notebook and choose **Kernel -> Restart Kernel and Run All Cells**. The last cell prints the freshly computed table and a line beginning with `VERIFIED:` when all numerical and qualitative checks pass.

To execute a notebook non-interactively without overwriting the source file:

```bash
mkdir -p executed
jupyter nbconvert \
  --to notebook \
  --execute Table_02.ipynb \
  --output-dir executed \
  --ExecutePreprocessor.timeout=-1
```

Replace `Table_02.ipynb` with the notebook you want to run. Running notebooks individually is recommended because the neural minimax experiments are computationally intensive.

## Experiment details

### Synthetic experiments: Tables 2-7

Tables 2-5 use nonlinear, high-dimensional data-generating processes with sample sizes 400 and 1200 and transformations `sin(t)` and `t^3`. They compare the proposed IPW, REG, and DR minimax estimators, with and without stabilizers, against linear and unconfoundedness-based baselines. Each notebook runs 1,000 Monte Carlo replications and reports normalized MSE, squared bias, and variance.

Tables 6-7 repeat the study after removing unmeasured confounding while retaining the nonlinear, high-dimensional structure. These ablations test whether the negative-control estimators remain competitive when standard ignorability holds.

### RHC analysis: Tables 8-10, 13-14

The real-data notebooks analyze the SUPPORT right-heart-catheterization study. The treatment indicates RHC within 24 hours of admission, and the outcome is 30-day survival. Following the paper, the main proxy allocation is:

- negative-control actions: `Z = (pafi1, paco21)`
- negative-control outcomes: `W = (ph1, hema1)`

Table 8 compares neural minimax DR estimators with linear closed-form and moment-based bridge estimators. Table 9 changes neural-network width and depth. Table 10 changes the linear moment basis by adding interaction terms, quadratic terms, or both. Tables 13-14 enumerate all six ways to allocate two of the four proxy variables to `Z` and the remaining two to `W`, highlighting robustness to proxy assignment.

The RHC data used by these analyses are frozen and compressed inside the notebooks, so no external data download or project-local data path is required.

### Reviewer covariate-omission sensitivity analysis: Tables 15-16

`Reviewer_X_Omission_Stress_test.ipynb` removes complete observed baseline-variable blocks from `X` and treats them as pseudo-unobserved confounders. The 49 meaningful raw-variable blocks are ranked in two deterministic ways: by joint treatment/outcome predictive strength and by conditional predictive strength for the four proxies. For each ranking, the notebook omits the top 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, and 45 blocks. A final union scenario removes all 49 blocks and leaves no observed `X`. Including the full-`X` reference, Tables 15-16 report 28 scenarios, with every estimator refitted from the raw data under the corresponding reduced covariate set.

## Reproducibility and validation

- Run each notebook in a fresh kernel; do not rely only on the saved display output.
- Fixed seeds are set for Python, NumPy, and PyTorch.
- The synthetic notebooks verify the MSE decomposition and qualitative estimator comparisons.
- The RHC notebooks verify point estimate, standard error, and confidence interval consistency; permutation notebooks also check every proxy allocation.
- The final cells report the corrected full rerun. These freshly recomputed values can differ from the originally published table values, which is why the integrity checks and complete end-to-end code are included.
- Tables 2-7 use 1,000 replications, and the RHC neural estimators use cross-fitting and hundreds of training epochs. Full execution can take substantial time on CPU-only machines.

## Citation

```bibtex
@article{kallus2022causal,
  title   = {Causal Inference Under Unmeasured Confounding with Negative Controls: A Minimax Learning Approach},
  author  = {Kallus, Nathan and Mao, Xiaojie and Uehara, Masatoshi and Zhang, Zhiheng},
  journal = {Journal of Machine Learning Research},
  volume  = {23},
  pages   = {1--100},
  year    = {2022}
}
```


