# OD — Optimal Differentiation / Strategic Conformity Analysis

Replicates and extends **Miller (2013)** — studies the optimal strategy position for Fortune 1000 firms across 6 strategy dimensions. Fits single-variable, additive, and full quadratic OLS models with two-way fixed effects and clustered standard errors, then solves a box-constrained quadratic program (QP) to find each model's optimal strategy vector.

---

## Repository Structure

```
OD/
├── OD.py                    # Core QuadraticODAnalyzer class
├── panel_od_analysis.py     # Empirical pipeline (real Fortune 1000 data)
├── simulate_od.py           # Simulation study (synthetic data)
├── od_panel_1962_2025.csv   # Panel data 1962–2025, ~27K firm-years
├── output/
│   ├── roa/                 # All tables for ROA outcome
│   └── tobinq/              # All tables for Tobin's Q outcome
├── OD.pdf                   # Target tables from the paper
└── 2013_miller.pdf          # Source paper (Miller 2013)
```

---

## Setup

**Conda environment:** `ambu` (Python 3.9, Apple M4 arm64)

```bash
conda activate ambu
```

**Solvers required:**

| Solver | Status | Notes |
|--------|--------|-------|
| Gurobi | Active (expires 2027-03-27, v12.0.1) | Primary solver, `NonConvex=2` for indefinite QP |
| CPLEX  | Fallback | `optimalitytarget = optimal_global` |
| MOSEK  | Fallback (expires 2027-03-26) | `~/mosek/mosek.lic`; rejects indefinite objectives |

---

## The Three Models

All three models fit the outcome (ROA or Tobin's Q) as a function of the 6 strategy variables. In the empirical pipeline, all models include firm and year fixed effects absorbed via alternating projections and two-way clustered standard errors.

### Model 1 — Individual (Single-Variable)

Fit a separate quadratic for each dimension $k$ independently:

$$y = \alpha + \beta_1 x_k + \beta_2 x_k^2 + \varepsilon, \quad k = 1, \ldots, 6$$

Optimize each dimension independently; combine the 6 per-dimension optima into one joint strategy vector.

### Model 2 — Additive

One joint regression with linear and squared terms, no cross terms:

$$y = \alpha + \sum_{k=1}^{6} \left( \beta_k x_k + \gamma_k x_k^2 \right) + \varepsilon$$

Optimized jointly (separable across dimensions).

### Model 3 — Full Quadratic

All linear, squared, and pairwise cross terms:

$$y = \alpha + \sum_{k=1}^{6} \left( \beta_k x_k + \gamma_k x_k^2 \right) + \sum_{i < j} \delta_{ij}\, x_i x_j + \varepsilon$$

Optimized jointly via a box-constrained QP (potentially non-convex due to cross terms — requires Gurobi/CPLEX).

### Optimality Gap

The **full quadratic model is the benchmark** — the best estimated surface.
For empirical results, gaps are relative to the full model's estimated optimum.
For simulation, gaps are relative to the *true* noiseless optimum.

$$\text{gap} = f^{\ast}_{M3} - f\!\left(x^{\ast}_{\text{model}}\right)$$

$$\text{gap}_{\%} = \frac{\text{gap}}{\left| f^{\ast}_{M3} \right|} \times 100$$

---

## Running the Empirical Pipeline

Produces all output tables (Tables 1–7 from the paper) for ROA and Tobin's Q.

```bash
conda run -n ambu python panel_od_analysis.py \
  --data od_panel_1962_2025.csv \
  --output output \
  --joint-opt-solver gurobi
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `/Volumes/T7 Shield/.../final_panel.csv` | Path to panel CSV |
| `--output` | `/Volumes/T7 Shield/.../output/panel_od` | Output directory |
| `--joint-opt-solver` | `gurobi` | Solver: `gurobi`, `cplex`, `mosek`, `scipy` |

**Output files per outcome** (`output/roa/` and `output/tobinq/`):

| File | Contents |
|------|----------|
| `sample_summary.csv` | Descriptive statistics (Table 1) |
| `single_variable_main.csv` | Per-variable quadratic fits with individual optima (Table 2) |
| `additive_main_coefficients.csv` | Additive model coefficients (Table 3) |
| `full_main_coefficients.csv` | Full quadratic coefficients (Table 4) |
| `additive_main_wald.csv` | Wald test for additive model (Table 5) |
| `full_main_all_terms_wald.csv` | Wald test, all terms (Table 6) |
| `full_main_cross_terms_wald.csv` | Wald test, cross terms only (Table 6) |
| `optimal_comparison.csv` | Optimal strategy vectors + optimality gaps (Table 7) |
| `robustness_summary.csv` | Robustness checks |

**Key columns in `optimal_comparison.csv`:**

| Column | Meaning |
|--------|---------|
| `additive_optimum` / `full_optimum` | Optimal strategy value per variable |
| `additive_surface_optimum` | Estimated ROA/Q at M2 optimum |
| `full_surface_optimum` | Estimated ROA/Q at M3 optimum (benchmark) |
| `full_surface_at_single_combined_optimum` | Full surface value at M1 combined x* |
| `full_surface_at_additive_optimum` | Full surface value at M2 x* |
| `m1_gap` / `m1_gap_pct` | M1 optimality gap (absolute / %) |
| `m2_gap` / `m2_gap_pct` | M2 optimality gap (absolute / %) |

**Empirical results:**

| Outcome | M3 Optimum | M2 Gap | M2 % Gap |
|---------|-----------|--------|----------|
| ROA | **0.2941** | 0.172 | 58.5% |
| Tobin's Q | **4.8391** | 3.396 | 70.2% |

The large gaps show that cross-term interactions between strategy dimensions are essential — the additive and individual models substantially underestimate the achievable optimum.

---

## Running the Simulation

Generates synthetic data from a known quadratic generator, fits all three models, and compares the *true* (noiseless) revenue at each model's optimal solution against the true optimum.

### Quick start

```bash
conda run -n ambu python simulate_od.py
```

No arguments needed. All configuration is edited directly at the top of `simulate_od.py`.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K` | `6` | Number of strategy dimensions |
| `N` | `100` | Sample size (smaller → noisier estimates → larger gaps) |
| `NOISE_STD` | `4.0` | Std dev of Gaussian noise added to observed outcome |
| `BOUNDS` | `[(0.0, 1.0)] * 6` | Box constraints for each dimension |
| `TRUE_INTERCEPT` | `5.0` | Baseline revenue level (ensures true optimum is clearly positive) |
| `TRUE_BDIAG` | `[-4, -3, -5, -4.5, -3.5, -4.2]` | Per-dimension quadratic curvature (must be negative) |
| `TARGET_OPT` | `[0.40, 0.60, 0.50, 0.45, 0.65, 0.35]` | Target interior optima for each dimension |
| `SEED` | `42` | Random seed for reproducibility |

To tune the difficulty of the estimation problem:
- **Increase gaps**: reduce `N` (e.g. 50–100) or increase `NOISE_STD` (e.g. 3.0–5.0)
- **Decrease gaps**: increase `N` (e.g. 500–1000) or reduce `NOISE_STD` (e.g. 0.5–1.0)
- **Strengthen cross-term interactions**: widen the `rng.uniform` range for `cross_vals` (default `[-2, 2]`)

### True generator design

The noiseless revenue function has the form:

$$f(x) = c + \mathbf{a}^\top x + x^\top B x$$

where:
- $c$ = `TRUE_INTERCEPT` (shifts baseline revenue above zero)
- $\mathbf{a}$ is set so the unconstrained per-dimension optimum lands at `TARGET_OPT`
- $B$ has **negative diagonal** (`TRUE_BDIAG`) → each $x_k$ slice is individually concave, guaranteeing interior optima in $[0, 1]$
- $B$ has **random off-diagonal** cross terms $\sim \text{Uniform}[-2, 2]$ → couples dimensions and makes the full Hessian indefinite
- The true global optimum $x^\ast$ is found exactly by Gurobi (`NonConvex=2`) on the noiseless surface

### What the simulation measures

Each model is estimated from the noisy sample, its optimal strategy $\hat{x}^\ast$ is found by QP, and the **true** (noiseless) revenue $f(\hat{x}^\ast)$ is evaluated and compared against $f(x^\ast)$:

| Source of gap | Affected models |
|--------------|----------------|
| Omitted cross terms (model misspecification) | M1, M2 |
| Noisy coefficient estimates (finite sample) | M1, M2, M3 |
| Overfitting from too many parameters | M3 (when N is small) |

### Output files (written to `output/`)

| File | Contents |
|------|----------|
| `sim_comparison.csv` | True revenue + absolute and % optimality gap per model |
| `sim_dim_optima.csv` | Per-dimension optimal values: truth vs M1 vs M2 vs M3 |

### Example results (N=100, noise_std=4.0)

| Model | True Revenue at $\hat{x}^\ast$ | % Gap |
|-------|-------------------------------|-------|
| True Optimum | 14.016 | — |
| M3 Full Quadratic | 11.537 | 17.7% |
| M2 Additive | 10.608 | 24.3% |
| M1 Individual | 10.532 | 24.9% |

With small N and high noise, M3 overfits the 15 cross-term coefficients (28 parameters from 100 observations), so its estimated surface leads Gurobi to the wrong region of the box. M2 and M1 are more robust here because they have fewer parameters to misestimate. This illustrates the **bias-variance tradeoff in strategy optimization**: a richer model does not always yield better decisions under limited, noisy data.

---

## Core Class: `QuadraticODAnalyzer` (OD.py)

```python
from OD import QuadraticODAnalyzer

analyzer = QuadraticODAnalyzer(K=6, bounds=[(0.0, 1.0)] * 6, random_state=42)

# Simulate or load data
X, y = analyzer.simulate_data(n_samples=500, A=..., B=..., intercept=5.0, noise_std=1.5)
# or: analyzer.load_data(X_df, y_series)

# Fit three models
m1 = analyzer.fit_single_variable_quadratics()    # dict keyed by dim index
m2 = analyzer.fit_additive_quadratic()             # dict with optimal_p, optimal_value
m3 = analyzer.fit_full_quadratic(prefer_gurobi=True)  # dict with optimal_p, solver_used

# Evaluate true objective at any point (simulation only)
true_opt_x, true_opt_val, solver = analyzer.solve_true_optimum()
true_val_at_m2 = analyzer.evaluate_true_objective(m2["optimal_p"])
```

**Solver chain** (tried in order): Gurobi → CPLEX → MOSEK → scipy multistart

---

## 6 Strategy Variables

| Variable | Description |
|----------|-------------|
| `rnd_intensity` | R&D spending / sales |
| `capital_intensity` | Net PPE / sales |
| `leverage` | Long-term debt / assets |
| `adv_intensity` | Advertising / sales |
| `div_policy` | Dividends / earnings |
| `unsystematic_risk` | Residual return variance |

---

## Reference

Miller, D. (2013). *Technological diversity, related diversification, and firm performance*. Strategic Management Journal.
