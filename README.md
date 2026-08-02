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
| `Reviewer_X_Omission_Sensitivity.ipynb` | Additional RHC sensitivity analysis that treats selected observed covariates as pseudo-unobserved confounders |

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

### Reviewer covariate-omission sensitivity analysis

`Reviewer_X_Omission_Sensitivity.ipynb` removes observed baseline covariates from `X` and treats them as pseudo-unobserved confounders. It includes proxy-safe nested omissions, strong-confounder nested omissions, a clinical-severity stress block, diagnostics for covariate balance and overlap, and comparisons with standard unconfoundedness-based baselines.

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



## Response to the reviewer

> **Response:** We thank the reviewer for clarifying this concern. The proposed covariate-omission analysis is informative when it is interpreted as a stress test rather than as a formal test of proxy validity. We have now added this analysis.
>
> Specifically, we keep the four proxy variables and their original allocation, $Z=(\texttt{pafi1},\texttt{paco21}),\qquad W=(\texttt{ph1},\texttt{hema1})$, fixed throughout the analysis. We remove complete raw-variable blocks from $X$ before imputation, dummy encoding, and standardization, and then refit every estimator from the raw data under each reduced $X$ specification. The estimator settings, sample splits, and random seeds are held fixed across omission scenarios. Thus, the reported changes are attributable to the information removed from $X$, rather than to changes in preprocessing, proxy allocation, or estimator tuning.
>
> We consider three complementary families of omissions.
>
> 1. **Proxy-safe nested omissions.** Among covariates whose conditional association with the four proxies is below the median, we rank variables by their joint treatment- and outcome-predictive strength and omit the top 1, 3, and 5 variables. The resulting sets are $\{\texttt{dnr1}\}$, $\{\texttt{dnr1},\texttt{ninsclas},\texttt{transhx}\}$, and $\{\texttt{dnr1},\texttt{ninsclas},\texttt{transhx},\texttt{cat2},\texttt{urin1}\}$. These are our primary sensitivity scenarios because they remove variables that are relevant to treatment and outcome while minimizing disruption to the empirical proxy structure.
>
> 2. **Strong-confounder nested omissions.** We rank all baseline covariates by joint treatment- and outcome-predictive strength and omit the top 1, 3, and 5 variables: $\{\texttt{surv2md1}\}$, $\{\texttt{surv2md1},\texttt{dnr1},\texttt{aps1}\}$, and $\{\texttt{surv2md1},\texttt{dnr1},\texttt{aps1},\texttt{cat1},\texttt{ca}\}$. These deliberately adversarial scenarios treat strongly prognostic variables as pseudo-unobserved confounders.
>
> 3. **Clinical-severity stress omission.** We jointly omit eight baseline severity variables: $\{\texttt{surv2md1},\texttt{aps1},\texttt{scoma1},\texttt{meanbp1},\texttt{hrt1},\texttt{resp1},\texttt{temp1},\texttt{dnr1}\}$.
>
> To quantify whether the omissions substantially alter the empirical proxy structure, we additionally report the cross-validated $R^2$ for predicting the four proxies from the retained $X$. It is 0.2556 with the full $X$, and remains 0.2556, 0.2545, and 0.2520 under the three proxy-safe omissions. Thus, the primary omission sequence removes up to five raw covariates while changing proxy predictability by no more than 0.0036. In contrast, the $R^2$ decreases to 0.2428, 0.2131, and 0.1665 under the strong-confounder omissions. We report this distinction explicitly because the latter scenarios intentionally push the analysis toward settings in which proxy informativeness may deteriorate.
>
> The results are presented in two tables and two corresponding coefficient plots, organized as $X$ omission analogues of Tables 13 and 14. In each plot, the red dashed line denotes the estimator’s full $X$ estimate, the black line denotes zero, and the horizontal bars are 95% confidence intervals.
>
> The proposed minimax DR estimators are directionally and inferentially stable throughout the analysis. Across the full $X$ specification and all seven omission scenarios:
>
> - both DR and DR(sta) produce negative point estimates in all 8 settings;
> - all 16 corresponding 95% confidence intervals remain entirely below zero;
> - the DR estimates range from $-0.0774$ to $-0.0303$, with a maximum absolute shift from the full $X$ estimate of 0.0245;
> - the DR(sta) estimates range from $-0.0793$ to $-0.0291$, with a maximum absolute shift of 0.0258; and
> - their standard errors remain within relatively narrow ranges: 0.0099–0.0122 for DR and 0.0104–0.0118 for DR(sta).
>
> The proxy-safe results are particularly relevant to the reviewer’s proposed diagnostic. After omitting one, three, and five variables, the DR estimates are $-0.0303$, $-0.0413$, and $-0.0383$, respectively, and the DR(sta) estimates are $-0.0388$, $-0.0291$, and $-0.0379$. Every confidence interval remains below zero. Thus, the substantive conclusion that RHC reduces 30-day survival is not driven by retaining the exact original set of 68 observed covariates.
>
> The contrast with the linear bridge baselines becomes pronounced under the more demanding omissions. The original linear moment-based estimator has only 4 of 8 confidence intervals entirely below zero and a maximum absolute shift of 0.6797. For example, its estimate changes from $-0.0828$ under the full $X$ to $-0.7625$ with standard error 2.6770 after omitting three strong covariates, and changes sign to $+0.0833$ after omitting five. The original linear closed-form estimator similarly has only 4 of 8 confidence intervals below zero and a maximum absolute shift of 0.4786; after the five-variable strong omission, it changes from $-0.0860$ to $+0.3925$, with standard error 0.8721.
>
> Adding interaction terms, quadratic terms, or both does not uniformly resolve this instability. Some individual linear specifications are relatively stable under the proxy-safe omissions, and we do not claim that the proposed estimator has the smallest numerical change in every individual row. Rather, the relevant distinction is uniformity: none of the linear specifications maintains the same combination of sign stability, bounded shifts, and confidence intervals below zero across the full set of omissions. This pattern is also apparent in the two coefficient plots: the proposed DR estimates remain concentrated on the negative side of zero, whereas several linear panels require much wider horizontal scales and contain sign reversals or very wide confidence intervals.
>
> Accordingly, we describe these results as a stress test, not as a validation test. Nevertheless, they provide the practically useful evidence requested by the reviewer: the substantive conclusion from the proposed minimax DR estimators is robust to several reasonable reductions of $X$, including omissions specifically designed to preserve empirical proxy predictability, and it remains stable even under substantially more adversarial omissions. At the same time, the deterioration of some linear estimators illustrates their greater dependence on a particular low-dimensional bridge specification and on well-conditioned linear moment equations.
>
> We have revised the manuscript to state both conclusions clearly.
