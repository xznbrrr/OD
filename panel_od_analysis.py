import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, pearsonr, spearmanr, t as student_t
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_cluster_2groups

from OD import QuadraticODAnalyzer


CORE_VARS = [
    "x_rnd",
    "x_capint",
    "x_lev",
    "x_adv",
    "x_div",
    "x_risk",
]
CORE_VAR_LABELS = {
    "x_rnd": "R&D nonconformity",
    "x_capint": "Capital intensity nonconformity",
    "x_lev": "Leverage nonconformity",
    "x_adv": "Advertising nonconformity",
    "x_div": "Dividend nonconformity",
    "x_risk": "Unsystematic risk nonconformity",
}
CONTROL_VARS = [
    "firm_size",
    "firm_age",
    "beta",
    "firm_diversification",
]
OUTCOME_SPECIFIC_CONTROLS = {
    "roa": ["industry_median_roa"],
    "tobinq": ["industry_median_tobinq"],
}
DEFAULT_OUTCOMES = ["roa", "tobinq"]
LOWER_Q = 0.01
UPPER_Q = 0.99


@dataclass
class AbsorbedOLSResult:
    model_name: str
    spec_name: str
    outcome: str
    fe_spec: str
    cluster_mode_requested: str
    cluster_mode_used: str
    coefficient_table: pd.DataFrame
    params: pd.Series
    covariance: np.ndarray
    design_columns: List[str]
    n_obs: int
    n_firms: int
    n_years: int
    r2_within: float
    condition_number: float
    warnings: List[str]


def winsorize_series(series: pd.Series, lower: float = LOWER_Q, upper: float = UPPER_Q) -> pd.Series:
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def bh_adjust(pvalues: Sequence[float]) -> np.ndarray:
    if len(pvalues) == 0:
        return np.array([])
    _, qvals, _, _ = multipletests(np.asarray(pvalues, dtype=float), method="fdr_bh")
    return qvals


def _group_means(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        values_2d = values[:, None]
    else:
        values_2d = values

    n_groups = int(codes.max()) + 1
    sums = np.zeros((n_groups, values_2d.shape[1]), dtype=float)
    np.add.at(sums, codes, values_2d)

    counts = np.bincount(codes, minlength=n_groups).astype(float)[:, None]
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    expanded = means[codes]
    return expanded[:, 0] if values.ndim == 1 else expanded


def absorb_fixed_effects(
    y: np.ndarray,
    x: np.ndarray,
    groups: Sequence[np.ndarray],
    max_iter: int = 200,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    y_tilde = np.asarray(y, dtype=float).reshape(-1)
    x_tilde = np.asarray(x, dtype=float).copy()

    encoded_groups = [pd.factorize(group, sort=False)[0] for group in groups]

    for _ in range(max_iter):
        max_change = 0.0
        for codes in encoded_groups:
            y_mean = _group_means(y_tilde, codes)
            x_mean = _group_means(x_tilde, codes)
            y_tilde = y_tilde - y_mean
            x_tilde = x_tilde - x_mean
            max_change = max(max_change, float(np.max(np.abs(y_mean))), float(np.max(np.abs(x_mean))))
        if max_change < tol:
            break

    return y_tilde, x_tilde


def _safe_t_pvalues(tvalues: np.ndarray, dof: int) -> np.ndarray:
    dof = max(int(dof), 1)
    return 2.0 * student_t.sf(np.abs(tvalues), df=dof)


def _build_coefficient_table(
    params: np.ndarray,
    covariance: np.ndarray,
    columns: Sequence[str],
    dof: int,
) -> pd.DataFrame:
    variances = np.clip(np.diag(covariance), a_min=0.0, a_max=None)
    std_errors = np.sqrt(variances)

    with np.errstate(divide="ignore", invalid="ignore"):
        tvalues = np.divide(params, std_errors, out=np.full_like(params, np.nan, dtype=float), where=std_errors > 0)
    pvalues = _safe_t_pvalues(tvalues, dof)

    return pd.DataFrame(
        {
            "term": list(columns),
            "coef": params,
            "std_error": std_errors,
            "t_value": tvalues,
            "p_value": pvalues,
        }
    )


def _wald_test(
    params: pd.Series,
    covariance: np.ndarray,
    design_columns: Sequence[str],
    terms: Sequence[str],
) -> Dict[str, float]:
    idx = [design_columns.index(term) for term in terms if term in design_columns]
    if not idx:
        return {"df": 0, "chi2": np.nan, "p_value": np.nan}

    beta = params.iloc[idx].to_numpy()
    cov_subset = covariance[np.ix_(idx, idx)]
    try:
        inv_cov = np.linalg.pinv(cov_subset)
        stat = float(beta.T @ inv_cov @ beta)
        p_value = float(chi2.sf(stat, len(idx)))
    except np.linalg.LinAlgError:
        stat = np.nan
        p_value = np.nan

    return {"df": len(idx), "chi2": stat, "p_value": p_value}


def pairwise_correlations(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    records = []
    for i, var1 in enumerate(columns):
        for var2 in columns[i + 1 :]:
            subset = df[[var1, var2]].dropna()
            pearson_corr, pearson_pvalue = pearsonr(subset[var1], subset[var2])
            spearman_corr, spearman_pvalue = spearmanr(subset[var1], subset[var2])
            records.append(
                {
                    "var1": var1,
                    "var2": var2,
                    "n_obs": subset.shape[0],
                    "pearson_corr": pearson_corr,
                    "pearson_pvalue": pearson_pvalue,
                    "spearman_corr": spearman_corr,
                    "spearman_pvalue": spearman_pvalue,
                }
            )

    result = pd.DataFrame(records)
    result["bh_qvalue"] = bh_adjust(result["pearson_pvalue"].to_numpy())
    return result


def _pairwise_to_matrix(
    pairwise_df: pd.DataFrame,
    variables: Sequence[str],
    value_col: str,
    diagonal: float,
) -> pd.DataFrame:
    matrix = pd.DataFrame(diagonal, index=variables, columns=variables, dtype=float)
    for _, row in pairwise_df.iterrows():
        matrix.loc[row["var1"], row["var2"]] = row[value_col]
        matrix.loc[row["var2"], row["var1"]] = row[value_col]
    return matrix


def _create_lagged_columns(df: pd.DataFrame, group_col: str, columns: Sequence[str]) -> pd.DataFrame:
    df = df.sort_values([group_col, "year"]).copy()
    grouped = df.groupby(group_col, sort=False)
    for col in columns:
        df[f"lag_{col}"] = grouped[col].shift(1)
    return df


def prepare_outcome_sample(
    df: pd.DataFrame,
    outcome: str,
    winsorize: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    required = ["gvkey", "year", "sic2", outcome] + CORE_VARS + _outcome_controls(outcome)
    required = [col for col in required if col in df.columns]
    sample = _create_lagged_columns(df[required].copy(), "gvkey", CORE_VARS)
    lag_cols = [f"lag_{var}" for var in CORE_VARS]
    sample = sample.dropna(subset=[outcome] + lag_cols).copy()

    if winsorize:
        sample[outcome] = winsorize_series(sample[outcome])
        for col in lag_cols:
            sample[col] = winsorize_series(sample[col])

    centers = {}
    bounds = {}
    centered_bounds = {}
    for col in lag_cols:
        center = float(sample[col].mean())
        centered_col = f"c_{col}"
        sample[centered_col] = sample[col] - center
        centers[col] = center
        bounds[col] = (float(sample[col].min()), float(sample[col].max()))
        centered_bounds[centered_col] = (bounds[col][0] - center, bounds[col][1] - center)

    sample["sic2"] = sample["sic2"].fillna(-1).astype(int).astype(str)
    sample["industry_year"] = sample["sic2"] + "_" + sample["year"].astype(int).astype(str)

    metadata = {
        "outcome": outcome,
        "n_obs": int(sample.shape[0]),
        "n_firms": int(sample["gvkey"].nunique()),
        "n_years": int(sample["year"].nunique()),
        "lag_columns": lag_cols,
        "centered_columns": [f"c_{col}" for col in lag_cols],
        "centers": centers,
        "bounds": bounds,
        "centered_bounds": centered_bounds,
        "winsorized": winsorize,
    }
    return sample, metadata


def _build_square_and_cross_terms(df: pd.DataFrame, centered_columns: Sequence[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = df.copy()
    square_terms = []
    cross_terms = []

    for col in centered_columns:
        sq_name = f"{col}_sq"
        df[sq_name] = df[col] ** 2
        square_terms.append(sq_name)

    for i, left in enumerate(centered_columns):
        for right in centered_columns[i + 1 :]:
            cross_name = f"{left}__x__{right}"
            df[cross_name] = df[left] * df[right]
            cross_terms.append(cross_name)

    return df, square_terms, cross_terms


def _single_variable_design(df: pd.DataFrame, variable: str) -> Tuple[pd.DataFrame, List[str]]:
    lag_col = f"lag_{variable}"
    sq_col = f"{lag_col}_sq"
    design = df.copy()
    design[sq_col] = design[lag_col] ** 2
    return design, [lag_col, sq_col]


def _fit_absorbed_ols(
    df: pd.DataFrame,
    outcome: str,
    regressors: Sequence[str],
    absorb_cols: Sequence[str],
    cluster_mode: str,
    model_name: str,
    spec_name: str,
) -> AbsorbedOLSResult:
    work_df = df.dropna(subset=[outcome] + list(regressors) + list(absorb_cols)).copy()

    y = work_df[outcome].to_numpy(dtype=float)
    x = work_df[list(regressors)].to_numpy(dtype=float)
    y_tilde, x_tilde = absorb_fixed_effects(y, x, [work_df[col].to_numpy() for col in absorb_cols])

    scale = np.nanstd(x_tilde, axis=0)
    keep_mask = np.isfinite(scale) & (scale > 1e-12)
    dropped_terms = [term for term, keep in zip(regressors, keep_mask) if not keep]
    kept_terms = [term for term, keep in zip(regressors, keep_mask) if keep]
    x_tilde = x_tilde[:, keep_mask]

    model = sm.OLS(y_tilde, x_tilde, hasconst=False)
    fitted = model.fit()

    warnings_list = []
    if dropped_terms:
        warnings_list.append(f"Dropped collinear/absorbed terms: {', '.join(dropped_terms)}")

    if cluster_mode == "two_way":
        try:
            covariance, _, _ = cov_cluster_2groups(
                fitted,
                work_df["gvkey"].to_numpy(),
                work_df["year"].to_numpy(),
            )
            cluster_mode_used = "two_way"
            dof = min(work_df["gvkey"].nunique(), work_df["year"].nunique()) - 1
        except Exception as exc:
            covariance = cov_cluster(fitted, work_df["gvkey"].to_numpy())
            cluster_mode_used = "firm_only_fallback"
            dof = work_df["gvkey"].nunique() - 1
            warnings_list.append(f"Two-way clustering failed; fell back to firm clustering: {exc}")
    elif cluster_mode == "firm_only":
        covariance = cov_cluster(fitted, work_df["gvkey"].to_numpy())
        cluster_mode_used = "firm_only"
        dof = work_df["gvkey"].nunique() - 1
    else:
        raise ValueError(f"Unknown cluster_mode: {cluster_mode}")

    coefficient_table = _build_coefficient_table(fitted.params, covariance, kept_terms, dof)

    fitted_values = fitted.fittedvalues
    ss_res = float(np.sum((y_tilde - fitted_values) ** 2))
    ss_tot = float(np.sum((y_tilde - y_tilde.mean()) ** 2))
    r2_within = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot

    condition_number = float(np.linalg.cond(x_tilde)) if x_tilde.size else np.nan

    return AbsorbedOLSResult(
        model_name=model_name,
        spec_name=spec_name,
        outcome=outcome,
        fe_spec=" + ".join(absorb_cols),
        cluster_mode_requested=cluster_mode,
        cluster_mode_used=cluster_mode_used,
        coefficient_table=coefficient_table,
        params=pd.Series(fitted.params, index=kept_terms),
        covariance=covariance,
        design_columns=kept_terms,
        n_obs=int(work_df.shape[0]),
        n_firms=int(work_df["gvkey"].nunique()),
        n_years=int(work_df["year"].nunique()),
        r2_within=r2_within,
        condition_number=condition_number,
        warnings=warnings_list,
    )


def solve_1d_quadratic_optimum(beta1: float, beta2: float, bounds: Tuple[float, float]) -> Tuple[float, float, bool]:
    lb, ub = bounds
    x_star, value_star = QuadraticODAnalyzer._solve_1d_quadratic(
        intercept=0.0,
        b1=beta1,
        b2=beta2,
        lb=lb,
        ub=ub,
    )
    hit_bound = bool(np.isclose(x_star, lb) or np.isclose(x_star, ub))
    return x_star, value_star, hit_bound


def optimize_quadratic_surface(
    intercept: float,
    a: np.ndarray,
    b: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    prefer_gurobi: bool = False,
    n_starts: int = 30,
    require_solver: Optional[str] = None,
) -> Tuple[np.ndarray, float, List[bool], str]:
    analyzer = QuadraticODAnalyzer(K=len(a), bounds=list(bounds), random_state=42)
    p_star, value_star, solver_used = analyzer._solve_box_qp(
        intercept=intercept,
        A=np.asarray(a, dtype=float),
        B=np.asarray(b, dtype=float),
        bounds=list(bounds),
        prefer_gurobi=prefer_gurobi,
        n_starts=n_starts,
        scipy_method="L-BFGS-B",
    )
    if require_solver and solver_used != require_solver:
        raise RuntimeError(f"Expected solver {require_solver}, but used {solver_used}")
    hit_bounds = [
        bool(np.isclose(value, lb) or np.isclose(value, ub))
        for value, (lb, ub) in zip(p_star, bounds)
    ]
    return p_star, float(value_star), hit_bounds, solver_used


def _surface_from_centered_result(
    result: AbsorbedOLSResult,
    centered_columns: Sequence[str],
    square_terms: Sequence[str],
    cross_terms: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    a = np.zeros(len(centered_columns), dtype=float)
    b = np.zeros((len(centered_columns), len(centered_columns)), dtype=float)
    position = {col: idx for idx, col in enumerate(centered_columns)}

    for col in centered_columns:
        if col in result.params.index:
            a[position[col]] = result.params[col]

    for sq_col in square_terms:
        base = sq_col[:-3]
        if sq_col in result.params.index and base in position:
            idx = position[base]
            b[idx, idx] = result.params[sq_col]

    for cross_col in cross_terms:
        left, right = cross_col.split("__x__")
        if cross_col in result.params.index and left in position and right in position:
            i = position[left]
            j = position[right]
            value = result.params[cross_col] / 2.0
            b[i, j] = value
            b[j, i] = value

    return a, b


def _evaluate_surface(intercept: float, a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    return float(intercept + p @ a + p @ b @ p)


def _outcome_controls(outcome: str) -> List[str]:
    return CONTROL_VARS + OUTCOME_SPECIFIC_CONTROLS[outcome]


def _split_available_columns(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[List[str], List[str]]:
    available = [col for col in columns if col in df.columns]
    missing = [col for col in columns if col not in df.columns]
    return available, missing


def _safe_vif(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    matrix = df[list(columns)].to_numpy(dtype=float)
    rows = []
    for idx, col in enumerate(columns):
        try:
            vif = float(variance_inflation_factor(matrix, idx))
        except Exception:
            vif = np.inf
        rows.append({"variable": col, "vif": vif})
    return pd.DataFrame(rows)


def _serialize_warnings(model_results: Iterable[AbsorbedOLSResult]) -> pd.DataFrame:
    rows = []
    for result in model_results:
        for warning in result.warnings:
            rows.append(
                {
                    "outcome": result.outcome,
                    "model_name": result.model_name,
                    "spec_name": result.spec_name,
                    "warning": warning,
                }
            )
    return pd.DataFrame(rows)


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_matrix(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def _conformity_index_analysis(
    df: pd.DataFrame,
    outcome: str,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Quadratic FE regression of outcome on conformity_index (Miller 2013 aggregate)."""
    if "conformity_index" not in df.columns:
        return None
    required = ["gvkey", "year", outcome, "conformity_index"]
    sub = df[[c for c in required if c in df.columns]].dropna().copy()
    sub = _create_lagged_columns(sub, "gvkey", ["conformity_index"])
    sub = sub.dropna(subset=[outcome, "lag_conformity_index"]).copy()
    if len(sub) < 50:
        return None
    sub[outcome] = winsorize_series(sub[outcome])
    sub["lag_conformity_index"] = winsorize_series(sub["lag_conformity_index"])
    sub["lag_conformity_index_sq"] = sub["lag_conformity_index"] ** 2
    regressors = ["lag_conformity_index", "lag_conformity_index_sq"]
    result = _fit_absorbed_ols(
        sub,
        outcome=outcome,
        regressors=regressors,
        absorb_cols=["gvkey", "year"],
        cluster_mode="two_way",
        model_name="conformity_quadratic",
        spec_name="main",
    )
    beta1 = result.params.get("lag_conformity_index", np.nan)
    beta2 = result.params.get("lag_conformity_index_sq", np.nan)
    bounds = (float(sub["lag_conformity_index"].min()), float(sub["lag_conformity_index"].max()))
    x_star, val_star, hit_bound = solve_1d_quadratic_optimum(beta1, beta2, bounds)
    coef_map = result.coefficient_table.set_index("term")
    summary_row = {
        "outcome": outcome,
        "variable": "conformity_index",
        "n_obs": result.n_obs,
        "n_firms": int(sub["gvkey"].nunique()),
        "n_years": int(sub["year"].nunique()),
        "r2_within": result.r2_within,
        "cluster_mode_used": result.cluster_mode_used,
        "coef_linear": beta1,
        "se_linear": coef_map.loc["lag_conformity_index", "std_error"] if "lag_conformity_index" in coef_map.index else np.nan,
        "p_linear": coef_map.loc["lag_conformity_index", "p_value"] if "lag_conformity_index" in coef_map.index else np.nan,
        "coef_quadratic": beta2,
        "se_quadratic": coef_map.loc["lag_conformity_index_sq", "std_error"] if "lag_conformity_index_sq" in coef_map.index else np.nan,
        "p_quadratic": coef_map.loc["lag_conformity_index_sq", "p_value"] if "lag_conformity_index_sq" in coef_map.index else np.nan,
        "optimal_x": x_star,
        "optimal_surface_value": val_star,
        "optimal_hits_bound": hit_bound,
    }
    return pd.DataFrame([summary_row]), result.coefficient_table


def run_outcome_analysis(
    df: pd.DataFrame,
    outcome: str,
    output_dir: Path,
    joint_opt_solver: str = "scipy",
) -> Dict[str, Path]:
    sample, metadata = prepare_outcome_sample(df, outcome=outcome, winsorize=True)
    lag_cols = metadata["lag_columns"]
    centered_cols = metadata["centered_columns"]

    pairwise = pairwise_correlations(sample, lag_cols)
    pearson_matrix = _pairwise_to_matrix(pairwise, lag_cols, "pearson_corr", diagonal=1.0)
    pearson_pvalue_matrix = _pairwise_to_matrix(pairwise, lag_cols, "pearson_pvalue", diagonal=0.0)
    spearman_matrix = _pairwise_to_matrix(pairwise, lag_cols, "spearman_corr", diagonal=1.0)
    vif_table = _safe_vif(sample, centered_cols)

    single_rows = []
    model_results: List[AbsorbedOLSResult] = []

    for variable in CORE_VARS:
        design_df, regressors = _single_variable_design(sample, variable)
        result = _fit_absorbed_ols(
            design_df,
            outcome=outcome,
            regressors=regressors,
            absorb_cols=["gvkey", "year"],
            cluster_mode="two_way",
            model_name=f"single_{variable}",
            spec_name="main",
        )
        model_results.append(result)

        beta1 = result.params.get(f"lag_{variable}", np.nan)
        beta2 = result.params.get(f"lag_{variable}_sq", np.nan)
        x_star, value_star, hit_bound = solve_1d_quadratic_optimum(beta1, beta2, metadata["bounds"][f"lag_{variable}"])

        coef_map = result.coefficient_table.set_index("term")
        single_rows.append(
            {
                "outcome": outcome,
                "variable": variable,
                "coef_linear": beta1,
                "coef_quadratic": beta2,
                "se_linear": coef_map.loc[f"lag_{variable}", "std_error"] if f"lag_{variable}" in coef_map.index else np.nan,
                "p_linear": coef_map.loc[f"lag_{variable}", "p_value"] if f"lag_{variable}" in coef_map.index else np.nan,
                "se_quadratic": coef_map.loc[f"lag_{variable}_sq", "std_error"] if f"lag_{variable}_sq" in coef_map.index else np.nan,
                "p_quadratic": coef_map.loc[f"lag_{variable}_sq", "p_value"] if f"lag_{variable}_sq" in coef_map.index else np.nan,
                "n_obs": result.n_obs,
                "r2_within": result.r2_within,
                "cluster_mode_used": result.cluster_mode_used,
                "fe_spec": result.fe_spec,
                "optimal_x": x_star,
                "optimal_surface_value": value_star,
                "optimal_hits_bound": hit_bound,
            }
        )

    additive_df, square_terms, _ = _build_square_and_cross_terms(sample, centered_cols)
    additive_regressors = list(centered_cols) + list(square_terms)
    additive_main = _fit_absorbed_ols(
        additive_df,
        outcome=outcome,
        regressors=additive_regressors,
        absorb_cols=["gvkey", "year"],
        cluster_mode="two_way",
        model_name="additive_quadratic",
        spec_name="main",
    )
    model_results.append(additive_main)

    full_df, square_terms, cross_terms = _build_square_and_cross_terms(sample, centered_cols)
    full_regressors = list(centered_cols) + list(square_terms) + list(cross_terms)
    full_main = _fit_absorbed_ols(
        full_df,
        outcome=outcome,
        regressors=full_regressors,
        absorb_cols=["gvkey", "year"],
        cluster_mode="two_way",
        model_name="full_quadratic",
        spec_name="main",
    )
    model_results.append(full_main)

    additive_a, additive_b = _surface_from_centered_result(additive_main, centered_cols, square_terms, [])
    full_a, full_b = _surface_from_centered_result(full_main, centered_cols, square_terms, cross_terms)
    centered_bounds = [metadata["centered_bounds"][col] for col in centered_cols]

    prefer_gurobi = joint_opt_solver in ("gurobi", "mosek", "cplex")
    required_solver = None

    additive_p_centered, additive_value, additive_hit_bounds, additive_solver_used = optimize_quadratic_surface(
        intercept=0.0,
        a=additive_a,
        b=additive_b,
        bounds=centered_bounds,
        prefer_gurobi=prefer_gurobi,
        n_starts=30,
        require_solver=required_solver,
    )
    full_p_centered, full_value, full_hit_bounds, full_solver_used = optimize_quadratic_surface(
        intercept=0.0,
        a=full_a,
        b=full_b,
        bounds=centered_bounds,
        prefer_gurobi=prefer_gurobi,
        n_starts=40,
        require_solver=required_solver,
    )

    centers = np.array([metadata["centers"][col[2:]] for col in centered_cols], dtype=float)
    additive_p_raw = additive_p_centered + centers
    full_p_raw = full_p_centered + centers

    full_at_additive = _evaluate_surface(0.0, full_a, full_b, additive_p_centered)

    # M1 combined: collect each variable's individual optimum, convert to centered space,
    # then evaluate the full quadratic surface at that joint point.
    m1_p_raw = np.array([row["optimal_x"] for row in single_rows], dtype=float)
    m1_p_centered = m1_p_raw - centers
    full_at_single_combined = _evaluate_surface(0.0, full_a, full_b, m1_p_centered)
    additive_wald = _wald_test(additive_main.params, additive_main.covariance, additive_main.design_columns, additive_regressors)
    full_all_wald = _wald_test(full_main.params, full_main.covariance, full_main.design_columns, full_regressors)
    full_cross_wald = _wald_test(full_main.params, full_main.covariance, full_main.design_columns, cross_terms)

    optimal_comparison = pd.DataFrame(
        {
            "variable": CORE_VARS,
            "additive_optimum": additive_p_raw,
            "full_optimum": full_p_raw,
            "difference_full_minus_additive": full_p_raw - additive_p_raw,
            "additive_hits_bound": additive_hit_bounds,
            "full_hits_bound": full_hit_bounds,
        }
    )
    optimal_comparison["outcome"] = outcome
    optimal_comparison["additive_surface_optimum"] = additive_value
    optimal_comparison["full_surface_optimum"] = full_value
    optimal_comparison["full_surface_at_additive_optimum"] = full_at_additive
    optimal_comparison["full_surface_gain_vs_additive_optimum"] = full_value - full_at_additive
    optimal_comparison["additive_solver_used"] = additive_solver_used
    optimal_comparison["full_solver_used"] = full_solver_used

    # Optimality gaps: full model is the benchmark (best estimated surface).
    # Gap = full_surface_optimum - full_surface_at_model_x*
    # %Gap = gap / |full_surface_optimum| * 100
    _full_opt = full_value
    optimal_comparison["full_surface_at_single_combined_optimum"] = full_at_single_combined
    optimal_comparison["m1_gap"] = _full_opt - full_at_single_combined
    optimal_comparison["m1_gap_pct"] = (_full_opt - full_at_single_combined) / abs(_full_opt) * 100
    optimal_comparison["m2_gap"] = _full_opt - full_at_additive
    optimal_comparison["m2_gap_pct"] = (_full_opt - full_at_additive) / abs(_full_opt) * 100

    robustness_rows = []
    extra_warning_rows = []
    expected_controls = _outcome_controls(outcome)
    available_controls, missing_controls = _split_available_columns(sample, expected_controls)

    if missing_controls:
        missing_controls_msg = ", ".join(missing_controls)
        extra_warning_rows.append(
            {
                "outcome": outcome,
                "model_name": "control_variables",
                "spec_name": "robust_controls",
                "warning": f"Missing control variables omitted from robust_controls: {missing_controls_msg}",
            }
        )

    robustness_configs = [
        {
            "spec_name": "robust_firm_cluster",
            "absorb_cols": ["gvkey", "year"],
            "cluster_mode": "firm_only",
            "extra_regressors": [],
        },
        {
            "spec_name": "robust_industry_year_fe",
            "absorb_cols": ["gvkey", "industry_year"],
            "cluster_mode": "firm_only",
            "extra_regressors": [],
        },
    ]

    robustness_configs.append(
        {
            "spec_name": "robust_controls",
            "absorb_cols": ["gvkey", "year"],
            "cluster_mode": "two_way",
            "extra_regressors": available_controls,
        }
    )

    for config in robustness_configs:
        additive_regressors_r = additive_regressors + config["extra_regressors"]
        full_regressors_r = full_regressors + config["extra_regressors"]

        additive_result = _fit_absorbed_ols(
            additive_df,
            outcome=outcome,
            regressors=additive_regressors_r,
            absorb_cols=config["absorb_cols"],
            cluster_mode=config["cluster_mode"],
            model_name="additive_quadratic",
            spec_name=config["spec_name"],
        )
        full_result = _fit_absorbed_ols(
            full_df,
            outcome=outcome,
            regressors=full_regressors_r,
            absorb_cols=config["absorb_cols"],
            cluster_mode=config["cluster_mode"],
            model_name="full_quadratic",
            spec_name=config["spec_name"],
        )
        model_results.extend([additive_result, full_result])

        additive_wald_r = _wald_test(
            additive_result.params,
            additive_result.covariance,
            additive_result.design_columns,
            [term for term in additive_regressors if term in additive_result.design_columns],
        )
        full_cross_wald_r = _wald_test(
            full_result.params,
            full_result.covariance,
            full_result.design_columns,
            [term for term in cross_terms if term in full_result.design_columns],
        )

        robustness_rows.append(
            {
                "outcome": outcome,
                "spec_name": config["spec_name"],
                "model_name": "additive_quadratic",
                "n_obs": additive_result.n_obs,
                "r2_within": additive_result.r2_within,
                "cluster_mode_used": additive_result.cluster_mode_used,
                "fe_spec": additive_result.fe_spec,
                "condition_number": additive_result.condition_number,
                "wald_df": additive_wald_r["df"],
                "wald_chi2": additive_wald_r["chi2"],
                "wald_p_value": additive_wald_r["p_value"],
            }
        )
        robustness_rows.append(
            {
                "outcome": outcome,
                "spec_name": config["spec_name"],
                "model_name": "full_quadratic",
                "n_obs": full_result.n_obs,
                "r2_within": full_result.r2_within,
                "cluster_mode_used": full_result.cluster_mode_used,
                "fe_spec": full_result.fe_spec,
                "condition_number": full_result.condition_number,
                "wald_df": full_cross_wald_r["df"],
                "wald_chi2": full_cross_wald_r["chi2"],
                "wald_p_value": full_cross_wald_r["p_value"],
            }
        )

    summary_rows = [
        {
            "outcome": outcome,
            "n_obs": metadata["n_obs"],
            "n_firms": metadata["n_firms"],
            "n_years": metadata["n_years"],
            "winsorized": metadata["winsorized"],
        }
    ]

    output_paths = {}
    outcome_dir = output_dir / outcome
    outcome_dir.mkdir(parents=True, exist_ok=True)

    _save_dataframe(pd.DataFrame(summary_rows), outcome_dir / "sample_summary.csv")
    _save_dataframe(pairwise, outcome_dir / "correlations_long.csv")
    _save_matrix(pearson_matrix, outcome_dir / "pearson_corr_matrix.csv")
    _save_matrix(pearson_pvalue_matrix, outcome_dir / "pearson_pvalue_matrix.csv")
    _save_matrix(spearman_matrix, outcome_dir / "spearman_corr_matrix.csv")
    _save_dataframe(vif_table, outcome_dir / "centered_vif.csv")
    _save_dataframe(pd.DataFrame(single_rows), outcome_dir / "single_variable_main.csv")
    _save_dataframe(additive_main.coefficient_table, outcome_dir / "additive_main_coefficients.csv")
    _save_dataframe(full_main.coefficient_table, outcome_dir / "full_main_coefficients.csv")
    _save_dataframe(pd.DataFrame([additive_wald]), outcome_dir / "additive_main_wald.csv")
    _save_dataframe(pd.DataFrame([full_all_wald]), outcome_dir / "full_main_all_terms_wald.csv")
    _save_dataframe(pd.DataFrame([full_cross_wald]), outcome_dir / "full_main_cross_terms_wald.csv")
    _save_dataframe(optimal_comparison, outcome_dir / "optimal_comparison.csv")
    _save_dataframe(pd.DataFrame(robustness_rows), outcome_dir / "robustness_summary.csv")

    conf_result = _conformity_index_analysis(df, outcome)
    if conf_result is not None:
        conf_summary, conf_coefs = conf_result
        _save_dataframe(conf_summary, outcome_dir / "conformity_index_main.csv")
        _save_dataframe(conf_coefs, outcome_dir / "conformity_index_coefficients.csv")

    if all(additive_hit_bounds):
        extra_warning_rows.append(
            {
                "outcome": outcome,
                "model_name": "additive_quadratic",
                "spec_name": "main",
                "warning": "Additive optimum hits the support boundary for all six variables.",
            }
        )
    if all(full_hit_bounds):
        extra_warning_rows.append(
            {
                "outcome": outcome,
                "model_name": "full_quadratic",
                "spec_name": "main",
                "warning": "Full optimum hits the support boundary for all six variables.",
            }
        )
    if full_main.condition_number > 1e6:
        extra_warning_rows.append(
            {
                "outcome": outcome,
                "model_name": "full_quadratic",
                "spec_name": "main",
                "warning": f"Full model condition number is high ({full_main.condition_number:.2e}).",
            }
        )
    if not vif_table.empty and float(vif_table["vif"].replace(np.inf, np.nan).max()) > 10:
        extra_warning_rows.append(
            {
                "outcome": outcome,
                "model_name": "multicollinearity",
                "spec_name": "main",
                "warning": "At least one centered main-term VIF exceeds 10.",
            }
        )

    warning_df = _serialize_warnings(model_results)
    if extra_warning_rows:
        warning_df = pd.concat([warning_df, pd.DataFrame(extra_warning_rows)], ignore_index=True)
    if warning_df.empty:
        warning_df = pd.DataFrame(columns=["outcome", "model_name", "spec_name", "warning"])
    _save_dataframe(warning_df, outcome_dir / "warnings.csv")

    summary_payload = {
        "outcome": outcome,
        "sample_summary": summary_rows[0],
        "data_compatibility": {
            "expected_controls": expected_controls,
            "available_controls": available_controls,
            "missing_controls": missing_controls,
        },
        "joint_opt_solver_requested": joint_opt_solver,
        "warnings": warning_df.to_dict("records"),
        "main_models": {
            "additive": {
                "n_obs": additive_main.n_obs,
                "r2_within": additive_main.r2_within,
                "cluster_mode_used": additive_main.cluster_mode_used,
                "solver_used": additive_solver_used,
                "joint_wald": additive_wald,
                "optimal_point": {var: float(val) for var, val in zip(CORE_VARS, additive_p_raw)},
                "optimal_surface_value": float(additive_value),
            },
            "full": {
                "n_obs": full_main.n_obs,
                "r2_within": full_main.r2_within,
                "cluster_mode_used": full_main.cluster_mode_used,
                "solver_used": full_solver_used,
                "all_terms_wald": full_all_wald,
                "cross_terms_wald": full_cross_wald,
                "optimal_point": {var: float(val) for var, val in zip(CORE_VARS, full_p_raw)},
                "optimal_surface_value": float(full_value),
                "full_surface_at_additive_optimum": float(full_at_additive),
            },
        },
    }
    with (outcome_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2, ensure_ascii=True)

    output_paths["outcome_dir"] = outcome_dir
    return output_paths


def run_panel_od_analysis(
    data_path: Path,
    output_dir: Path,
    outcomes: Optional[Sequence[str]] = None,
    joint_opt_solver: str = "scipy",
    max_year: Optional[int] = None,
) -> Dict[str, Path]:
    outcomes = list(outcomes or DEFAULT_OUTCOMES)
    df = pd.read_csv(data_path)
    if max_year is not None:
        df = df[df["year"] <= max_year].copy()
    output_dir.mkdir(parents=True, exist_ok=True)

    created_paths = {}
    for outcome in outcomes:
        created_paths[outcome] = run_outcome_analysis(
            df,
            outcome=outcome,
            output_dir=output_dir,
            joint_opt_solver=joint_opt_solver,
        )["outcome_dir"]
    return created_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-data panel OD analysis.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/Volumes/T7 Shield/Program/OD Program/data/final_panel.csv"),
        help="Path to the panel dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Volumes/T7 Shield/Program/OD Program/output/panel_od"),
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--outcomes",
        nargs="+",
        default=DEFAULT_OUTCOMES,
        help="Outcome variables to analyze.",
    )
    parser.add_argument(
        "--joint-opt-solver",
        choices=["gurobi", "mosek", "cplex"],
        default="cplex",
        help="Solver used for additive and full-model joint optimum calculations.",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=None,
        help="Exclude observations with year > max_year (e.g. 2019 for pre-pandemic sample).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    created_paths = run_panel_od_analysis(
        args.data,
        args.output_dir,
        args.outcomes,
        joint_opt_solver=args.joint_opt_solver,
        max_year=args.max_year,
    )
    for outcome, path in created_paths.items():
        print(f"{outcome}: {path}")


if __name__ == "__main__":
    main()
