# CLAUDE.md — Project Memory for OD

## Project Identity
- **Name**: OD — Optimal Differentiation / Strategic Conformity Analysis
- **Location**: `~/OR/OD/`
- **Conda env**: `ambu` (Python 3.x, Apple M4 Pro arm64)
- **Run pipeline**: `conda run -n ambu python panel_od_analysis.py`
- **GitHub**: `git@github.com:xznbrrr/OD.git` (remote set to SSH, not HTTPS)

## Purpose
Replicates and extends Miller (2013) — studies strategic conformity of Fortune 1000 firms using 6 strategy variables. Fits single-variable, additive, and full quadratic OLS models, then solves a box-constrained QP to find optimal strategy positions.

## Key Files
- `OD.py` — core `QuadraticODAnalyzer` class; regression + QP optimization
- `panel_od_analysis.py` — main pipeline producing all output CSVs (Tables 1–7)
- `od_panel_1962_2025.csv` — panel data 1962–2025, ~27K firm-years
- `output/` — generated CSVs per outcome (roa, tobinq); not committed
- `OD.pdf` — target tables to reproduce
- `2013_miller.pdf` — source paper

## 6 Core Variables (CORE_VARS)
`rnd_intensity`, `capital_intensity`, `leverage`, `adv_intensity`, `div_policy`, `unsystematic_risk`

## Regression
- Two-way clustered FE: firm + year fixed effects absorbed via alternating projections (`absorb_fixed_effects`)
- Clustered SE via `cov_cluster_2groups`

## QP Optimization
- Problem: `maximize intercept + A^T p + p^T B p  s.t.  lb ≤ p ≤ ub`
- B is **indefinite** (OLS regression) — requires a **global** non-convex QP solver
- Solver chain (in order): **Gurobi** → **CPLEX** → **MOSEK** → scipy multistart
- CLI default solver: `--joint-opt-solver gurobi`
- `prefer_gurobi = True` when any of gurobi/cplex/mosek is selected (triggers the chain)

## Solver Notes

### Gurobi
- License **renewed 2026-03-27**, valid until **2027-03-27** (academic, version 12.0.1)
- Uses `NonConvex=2` parameter for indefinite QP
- **Currently the preferred/default solver** — use `--joint-opt-solver gurobi`

### CPLEX
- `optimalitytarget = optimal_global` converts to MIQP for global solution
- Status codes 101/102 = optimal
- Fallback if Gurobi unavailable

### MOSEK
- License at `~/mosek/mosek.lic` (valid until 2027-03-26, version 11)
- **CRITICAL**: `MOSEK_LICENSE_FILE` must be set at shell level, not inside Python
  - Permanent: `conda env config vars set MOSEK_LICENSE_FILE=/Users/yuezhao/mosek/mosek.lic -n ambu`
  - Or prefix: `MOSEK_LICENSE_FILE=/Users/yuezhao/mosek/mosek.lic conda run -n ambu python ...`
- MOSEK **rejects non-NSD objectives** (`err_obj_q_not_nsd`) — indefinite B from OLS will fail
- Version 11.1.10 installed from official arm64 package (Apple M4 = aarch64)

## Git
- Remote: `git@github.com:xznbrrr/OD.git` (SSH)
- SSH key at `~/.ssh/id_rsa` authenticates as `skywalker6174`
- `output/` data CSVs are not tracked

## Confirmed Results (Tables recovered from OD.pdf)
- ROA full quadratic optimum: **0.294073**
- Tobin's q full quadratic optimum: **4.839133**
- Full model > additive model optimum due to cross-term interactions
