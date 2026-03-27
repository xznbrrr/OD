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
Fit a separate quadratic for each dimension independently:

$$y \sim \alpha + \beta_1 x_k + \beta_2 x_k^2 + \varepsilon \quad \text{for each } k$$

Optimize each dimension independently; combine the 6 per-dimension optima into one joint strategy vector.

### Model 2 — Additive
One joint regression with linear and squared terms, no cross terms:

$$y \sim \alpha + \sum_k \beta_k x_k + \sum_k \gamma_k x_k^2 + \varepsilon$$

Optimized jointly (separable across dimensions).

### Model 3 — Full Quadratic
All linear, squared, and pairwise cross terms:

$$y \sim \alpha + \sum_k \beta_k x_k + \sum_k \gamma_k x_k^2 + \sum_{i < j} \delta_{ij} x_i x_j + \varepsilon$$

Optimized jointly via a box-constrained QP (potentially non-convex due to cross terms — requires Gurobi/CPLEX).

### Optimality Gap
The **full quadratic model is the benchmark** — the best estimated surface.
For empirical results, gaps are relative to the full model's estimated optimum.
For simulation, gaps are relative to the *true* noiseless optimum.

$$\text{gap} = f^*_{\text{M3}} - f(x^*_{\text{model}}) \qquad \%\text{gap} = \frac{\text{gap}}{|f^*_{\text{M3}}|} \times 100$$

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

Generates synthetic data from a known quadratic generator, fits all three models, and compares the *true* (noiseless) outcome at each model's optimal solution.

```bash
conda run -n ambu python simulate_od.py
```

No arguments needed. Key configuration at the top of `simulate_od.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `K` | 6 | Number of strategy dimensions |
| `N` | 100 | Sample size |
| `NOISE_STD` | 4.0 | Gaussian noise on observed outcome |
| `BOUNDS` | `[0, 1]^6` | Box constraints for optimization |
| `TRUE_INTERCEPT` | 5.0 | Baseline revenue level (true optimum clearly above 0) |

**True generator design:**
- Diagonal quadratic terms (`B_ii < 0`): each dimension is individually concave, guaranteeing interior optima in `[0, 1]`
- Cross terms (`c_ij ~ Uniform[−2, 2]`): couple dimensions, making the full Hessian indefinite
- True optimum solved exactly by Gurobi on the noiseless surface

**Output files** (written to `output/`):

| File | Contents |
|------|----------|
| `sim_comparison.csv` | True outcome + absolute/% optimality gap per model |
| `sim_dim_optima.csv` | Per-dimension optimal values for each model vs truth |

**Example results** (N=100, noise_std=4.0):

| Model | True ROA at x* | % Gap vs True Opt |
|-------|---------------|-------------------|
| True Optimum | 14.016 | — |
| M3 Full Quadratic | 11.537 | 17.7% |
| M2 Additive | 10.608 | 24.3% |
| M1 Individual | 10.532 | 24.9% |

With small N and high noise, M3 overfits the cross terms (28 parameters from 100 observations), producing larger gaps than M2. This demonstrates the bias-variance tradeoff in strategy optimization: a more flexible model does not always yield better decisions under noisy, limited data.

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
