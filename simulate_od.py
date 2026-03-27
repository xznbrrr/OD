"""
simulate_od.py — OD Simulation Study

Uses QuadraticODAnalyzer (OD.py) for all fitting and QP solving (Gurobi chain).

True generator:
  - Quadratic in all K dimensions
  - CONCAVE per-dimension: diagonal B terms are negative, interior optima guaranteed
  - Cross-term interactions couple dimensions (can make full Hessian indefinite)
  - Large positive intercept so revenue (true optimum) is clearly above zero

Three models estimated from N=100 noisy observations:
  M1 — Individual: K separate univariate quadratics, optima combined into one vector
  M2 — Additive:   joint linear + squared (no cross terms)
  M3 — Full quad:  all linear + squared + pairwise cross terms

Optimality gap = true_opt_value - true_f(model_x*)
             % = gap / true_opt_value * 100
"""

import numpy as np
import pandas as pd
from itertools import combinations

from OD import QuadraticODAnalyzer

# ── Config ─────────────────────────────────────────────────────────────────────
SEED      = 42
K         = 6
N         = 100           # small sample → noisy coefficient estimates
NOISE_STD = 4.0           # high noise → poor cross-term recovery, large gaps
BOUNDS    = [(0.0, 1.0)] * K
DIM_NAMES = [f"x{i+1}" for i in range(K)]

rng = np.random.default_rng(SEED)

# ── True Generator Parameters ──────────────────────────────────────────────────
# Diagonal B: negative → concave per-dimension, interior optima well inside [0,1]
TRUE_BDIAG = np.array([-4.0, -3.0, -5.0, -4.5, -3.5, -4.2])

# Target unconstrained per-dim optima (marginal, ignoring cross terms)
TARGET_OPT = np.array([0.40, 0.60, 0.50, 0.45, 0.65, 0.35])

# Linear coefficients chosen so xi* = -ai/(2*bii) = TARGET_OPT[i]
TRUE_A = -2.0 * TRUE_BDIAG * TARGET_OPT   # all positive

# Cross terms: larger magnitudes → stronger coupling between dimensions
# Makes M2 (which ignores cross terms) pay a larger price
cross_pairs = list(combinations(range(K), 2))
cross_vals  = rng.uniform(-2.0, 2.0, len(cross_pairs))

TRUE_B = np.diag(TRUE_BDIAG.copy())
for idx, (i, j) in enumerate(cross_pairs):
    TRUE_B[i, j] = cross_vals[idx] / 2.0   # x'Bx cross contribution = cross_vals[idx]*xi*xj
    TRUE_B[j, i] = cross_vals[idx] / 2.0

# Large positive intercept: ensures revenue (true optimum) is clearly above zero
TRUE_INTERCEPT = 5.0


# ── Simulate data ──────────────────────────────────────────────────────────────
analyzer = QuadraticODAnalyzer(K=K, bounds=BOUNDS, random_state=SEED)
X, y = analyzer.simulate_data(
    n_samples   = N,
    A           = TRUE_A,
    B           = TRUE_B,
    intercept   = TRUE_INTERCEPT,
    noise_std   = NOISE_STD,
)

# ── True optimum (noiseless, via Gurobi) ──────────────────────────────────────
true_opt_x, true_opt_val, true_solver = analyzer.solve_true_optimum()

# ── Fit three models ───────────────────────────────────────────────────────────
m1_fits = analyzer.fit_single_variable_quadratics()
m2_fit  = analyzer.fit_additive_quadratic()
m3_fit  = analyzer.fit_full_quadratic(prefer_gurobi=True)

# ── Model 1: combine all individual per-dimension optima into one vector ───────
# Each dimension is optimised independently from its own univariate regression.
m1_opt_x   = np.array([m1_fits[k]["optimal_point"] for k in range(K)])
m1_true_val = analyzer.evaluate_true_objective(m1_opt_x)

# ── Model 2: additive (joint, separable) ──────────────────────────────────────
m2_opt_x    = m2_fit["optimal_p"]
m2_true_val = analyzer.evaluate_true_objective(m2_opt_x)

# ── Model 3: full quadratic (joint, Gurobi) ───────────────────────────────────
m3_opt_x    = m3_fit["optimal_p"]
m3_true_val = analyzer.evaluate_true_objective(m3_opt_x)


# ── Optimality gap helpers ────────────────────────────────────────────────────
def opt_gap(model_true_val: float, true_opt: float):
    """Returns (absolute_gap, percentage_gap)."""
    abs_gap = true_opt - model_true_val
    pct_gap = abs_gap / abs(true_opt) * 100.0
    return abs_gap, pct_gap


m1_abs, m1_pct = opt_gap(m1_true_val, true_opt_val)
m2_abs, m2_pct = opt_gap(m2_true_val, true_opt_val)
m3_abs, m3_pct = opt_gap(m3_true_val, true_opt_val)


# ── Printing ───────────────────────────────────────────────────────────────────
W = 70

def sep(c="─"):
    print(c * W)

def header(title):
    sep("═")
    print(f" {title}")
    sep()


print()
header(f"OD SIMULATION  |  K={K}  N={N}  noise_std={NOISE_STD}")

print("\nTRUE GENERATOR:  f(x) = intercept + a·x + x·B·x")
print(f"  Intercept   : {TRUE_INTERCEPT:.2f}")
print(f"  B diagonal  : {TRUE_BDIAG}  (all negative → per-dim concave)")
print(f"  Cross terms : {len(cross_pairs)} pairs, values ∈ [{cross_vals.min():.3f}, {cross_vals.max():.3f}]")
print(f"  Marginal x* : {TARGET_OPT}  (unconstrained per-dim)")

print()
print("CROSS-TERM COEFFICIENTS (c_ij on x_i·x_j):")
for k, (i, j) in enumerate(cross_pairs):
    end = "\n" if (k + 1) % 4 == 0 else "   "
    print(f"  x{i+1}·x{j+1}: {cross_vals[k]:+.3f}", end=end)
print()

header("TRUE OPTIMUM  (solver: Gurobi on noiseless generator)")
print(f"  x*          = {true_opt_x.round(4)}")
print(f"  True f(x*)  = {true_opt_val:.6f}")

# ── Model 1 ────────────────────────────────────────────────────────────────────
header("MODEL 1  —  Individual dimension quadratics  (6 × univariate OLS)")
print(f"  {'Dim':<6} {'True b_ii':>10} {'Est b_ii':>10} {'True a_i':>10} "
      f"{'Est a_i':>10} {'True x*_i':>10} {'Est x*_i':>10}  R²")
sep("-")
for k in range(K):
    s   = m1_fits[k]
    c   = s["coefficients"]
    print(f"  {DIM_NAMES[k]:<6} {TRUE_BDIAG[k]:>10.4f} {c['quadratic']:>10.4f} "
          f"{TRUE_A[k]:>10.4f} {c['linear']:>10.4f} "
          f"{true_opt_x[k]:>10.4f} {m1_opt_x[k]:>10.4f}  {s['r2']:.4f}")
print(f"\n  Combined x* = {m1_opt_x.round(4)}")
print(f"  True f(x*_M1)      = {m1_true_val:.6f}")
print(f"  Optimality gap     = {m1_abs:.6f}   ({m1_pct:.3f}%)")

# ── Model 2 ────────────────────────────────────────────────────────────────────
header("MODEL 2  —  Additive: linear + squared, no cross terms")
print(f"  R² = {m2_fit['r2']:.6f}")
print(f"\n  {'Dim':<6} {'Est linear':>12} {'Est quadratic':>15} {'Est x*':>10}")
sep("-")
a2, d2 = m2_fit["A"], m2_fit["diag_quadratic"]
for k in range(K):
    print(f"  {DIM_NAMES[k]:<6} {a2[k]:>12.4f} {d2[k]:>15.4f} {m2_opt_x[k]:>10.4f}")
print(f"\n  Joint x*           = {m2_opt_x.round(4)}")
print(f"  Est. optimum       = {m2_fit['optimal_value']:.6f}  (evaluated on fitted model)")
print(f"  True f(x*_M2)      = {m2_true_val:.6f}")
print(f"  Optimality gap     = {m2_abs:.6f}   ({m2_pct:.3f}%)")

# ── Model 3 ────────────────────────────────────────────────────────────────────
header(f"MODEL 3  —  Full quadratic  (solver: {m3_fit['solver_used']})")
print(f"  R² = {m3_fit['r2']:.6f}   N_params = {1 + 2*K + len(cross_pairs)}")

print(f"\n  Cross-term recovery (estimated vs true c_ij):")
B3    = m3_fit["B"]
for k, (i, j) in enumerate(cross_pairs):
    est_c  = B3[i, j] * 2.0      # B stores c/2, true c = cross_vals[k]
    true_c = cross_vals[k]
    end = "\n" if (k + 1) % 2 == 0 else "   "
    print(f"    x{i+1}·x{j+1}: est={est_c:+.4f}  true={true_c:+.4f}", end=end)
print()

print(f"\n  Joint x*           = {m3_opt_x.round(4)}")
print(f"  Est. optimum       = {m3_fit['optimal_value']:.6f}  (evaluated on fitted model)")
print(f"  True f(x*_M3)      = {m3_true_val:.6f}")
print(f"  Optimality gap     = {m3_abs:.6f}   ({m3_pct:.3f}%)")

# ── Summary comparison ─────────────────────────────────────────────────────────
header("SUMMARY  —  Optimality Gaps (True ROA at Each Model's x*)")
print(f"  {'Model':<32} {'True f(x*)':>12} {'Abs Gap':>12} {'% Gap':>10}")
sep("-")
rows = [
    ("True Optimum",                true_opt_val, 0.0,    0.0),
    ("M1  Individual (combined)",   m1_true_val,  m1_abs, m1_pct),
    ("M2  Additive (no cross)",     m2_true_val,  m2_abs, m2_pct),
    ("M3  Full Quadratic",          m3_true_val,  m3_abs, m3_pct),
]
for label, val, ab, pct in rows:
    ab_str  = f"{ab:.6f}" if ab > 0 else "—"
    pct_str = f"{pct:.3f}%" if pct > 0 else "—"
    print(f"  {label:<32} {val:>12.6f} {ab_str:>12} {pct_str:>10}")
sep("═")

# ── Per-dimension table ────────────────────────────────────────────────────────
print("\nPER-DIMENSION OPTIMAL VALUES")
sep()
print(f"  {'Dim':<6} {'True x*':>10} {'M1 x*':>10} {'M2 x*':>10} {'M3 x*':>10}")
sep("-")
for k in range(K):
    print(f"  {DIM_NAMES[k]:<6} {true_opt_x[k]:>10.4f} {m1_opt_x[k]:>10.4f}"
          f" {m2_opt_x[k]:>10.4f} {m3_opt_x[k]:>10.4f}")
sep()

# ── Save ───────────────────────────────────────────────────────────────────────
comparison = pd.DataFrame([
    dict(model="true_optimum",       true_roa=true_opt_val, abs_gap=0.0,    pct_gap=0.0),
    dict(model="M1_individual",      true_roa=m1_true_val,  abs_gap=m1_abs, pct_gap=m1_pct),
    dict(model="M2_additive",        true_roa=m2_true_val,  abs_gap=m2_abs, pct_gap=m2_pct),
    dict(model="M3_full_quadratic",  true_roa=m3_true_val,  abs_gap=m3_abs, pct_gap=m3_pct),
])
dim_table = pd.DataFrame({
    "dim"        : DIM_NAMES,
    "true_b_diag": TRUE_BDIAG,
    "true_a"     : TRUE_A,
    "true_opt_x" : true_opt_x.round(6),
    "m1_opt_x"   : m1_opt_x.round(6),
    "m2_opt_x"   : m2_opt_x.round(6),
    "m3_opt_x"   : m3_opt_x.round(6),
})
comparison.to_csv("output/sim_comparison.csv", index=False)
dim_table.to_csv("output/sim_dim_optima.csv",  index=False)
print(f"\nSaved: output/sim_comparison.csv  |  output/sim_dim_optima.csv")
