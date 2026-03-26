import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class QuadraticODAnalyzer:
    """
    Analyze quadratic-form models of the form

        y = intercept + A^T p + p^T B p + epsilon

    with p in R^K subject to box bounds:
        p_k in [lb_k, ub_k].

    It supports:
    1. Simulating data
    2. Loading external data
    3. Fitting K separate 1D quadratic OLS models
    4. Fitting an additive quadratic OLS model:
           y ~ const + sum_k p_k + sum_k p_k^2
    5. Fitting a full quadratic OLS model:
           y ~ const + sum_k p_k + sum_k p_k^2 + sum_{i<j} p_i p_j
    6. Solving for the optimal p under each model
    7. Comparing optimality gaps relative to the full quadratic model

    Notes
    -----
    - For the full quadratic optimization, this class tries gurobipy first if available.
      If gurobipy is not available, it falls back to scipy with multi-start local search.
    - Only the symmetric part of B matters in p^T B p. The class symmetrizes B internally.
    """

    def __init__(
        self,
        K: int,
        bounds: Optional[List[Tuple[float, float]]] = None,
        random_state: Optional[int] = None,
    ):
        self.K = int(K)
        self.rng = np.random.default_rng(random_state)

        if bounds is None:
            self.bounds = [(0.0, 1.0) for _ in range(self.K)]
        else:
            if len(bounds) != self.K:
                raise ValueError("bounds must have length K")
            self.bounds = [(float(lb), float(ub)) for lb, ub in bounds]

        self.X: Optional[np.ndarray] = None  # shape: (n, K)
        self.y: Optional[np.ndarray] = None  # shape: (n,)
        self.feature_names = [f"p{k+1}" for k in range(self.K)]

        # True DGP parameters if simulated
        self.true_intercept: Optional[float] = None
        self.true_A: Optional[np.ndarray] = None
        self.true_B: Optional[np.ndarray] = None
        self.true_noise_std: Optional[float] = None

        # Fitted models
        self.single_models: Dict[int, Dict] = {}
        self.additive_model: Optional[Dict] = None
        self.full_model: Optional[Dict] = None

    # ---------------------------------------------------------------------
    # Data handling
    # ---------------------------------------------------------------------
    def simulate_data(
        self,
        n_samples: int,
        A: np.ndarray,
        B: np.ndarray,
        intercept: float = 0.0,
        noise_std: float = 0.1,
        sampler: str = "uniform",
        custom_sampler: Optional[Callable[[int, List[Tuple[float, float]], np.random.Generator], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate p and y from:
            y = intercept + A^T p + p^T B p + epsilon

        Parameters
        ----------
        n_samples : int
            Number of observations.
        A : np.ndarray
            Shape (K,), linear coefficients.
        B : np.ndarray
            Shape (K, K), quadratic interaction matrix. Will be symmetrized internally.
        intercept : float
            Intercept term.
        noise_std : float
            Standard deviation of Gaussian noise.
        sampler : str
            "uniform" or "truncated_normal".
        custom_sampler : callable, optional
            Custom sampler returning X of shape (n_samples, K).

        Returns
        -------
        X, y : np.ndarray, np.ndarray
        """
        A = np.asarray(A, dtype=float).reshape(-1)
        B = np.asarray(B, dtype=float)

        if A.shape[0] != self.K:
            raise ValueError("A must have shape (K,)")
        if B.shape != (self.K, self.K):
            raise ValueError("B must have shape (K, K)")

        B = 0.5 * (B + B.T)

        if custom_sampler is not None:
            X = custom_sampler(n_samples, self.bounds, self.rng)
        else:
            X = self._sample_p(n_samples, sampler=sampler)

        quad_term = np.einsum("ni,ij,nj->n", X, B, X)
        noise = self.rng.normal(loc=0.0, scale=noise_std, size=n_samples)
        y = intercept + X @ A + quad_term + noise

        self.X = X
        self.y = y

        self.true_intercept = float(intercept)
        self.true_A = A.copy()
        self.true_B = B.copy()
        self.true_noise_std = float(noise_std)

        return X, y

    def load_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, List[float]],
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Load external data.

        Parameters
        ----------
        X : array-like, shape (n, K)
        y : array-like, shape (n,)
        feature_names : optional list of length K
        """
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=float)
            self.feature_names = list(X.columns)
        else:
            X_arr = np.asarray(X, dtype=float)
            if feature_names is not None:
                self.feature_names = feature_names

        y_arr = np.asarray(y, dtype=float).reshape(-1)

        if X_arr.ndim != 2 or X_arr.shape[1] != self.K:
            raise ValueError(f"X must have shape (n, {self.K})")
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        self.X = X_arr
        self.y = y_arr

        # Clear true DGP if external data loaded
        self.true_intercept = None
        self.true_A = None
        self.true_B = None
        self.true_noise_std = None

    def get_dataframe(self) -> pd.DataFrame:
        """Return the current dataset as a DataFrame."""
        self._check_data_loaded()
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df["y"] = self.y
        return df

    # ---------------------------------------------------------------------
    # Model fitting: single-variable quadratic
    # ---------------------------------------------------------------------
    def fit_single_variable_quadratics(self) -> Dict[int, Dict]:
        """
        Fit K separate OLS models:
            y ~ const + p_k + p_k^2

        For each model, compute the optimal p_k within its bounds and the
        corresponding predicted y.

        Returns
        -------
        dict keyed by dimension index k
        """
        self._check_data_loaded()
        self.single_models = {}

        for k in range(self.K):
            xk = self.X[:, k]
            X_design = np.column_stack([np.ones_like(xk), xk, xk**2])
            beta = self._ols_fit(X_design, self.y)
            y_hat = X_design @ beta
            r2 = self._r2(self.y, y_hat)

            lb, ub = self.bounds[k]
            p_star_k, y_star = self._solve_1d_quadratic(
                intercept=beta[0],
                b1=beta[1],
                b2=beta[2],
                lb=lb,
                ub=ub,
            )

            self.single_models[k] = {
                "dimension": k,
                "feature_name": self.feature_names[k],
                "coefficients": {
                    "intercept": beta[0],
                    "linear": beta[1],
                    "quadratic": beta[2],
                },
                "r2": r2,
                "optimal_point": p_star_k,
                "optimal_value": y_star,
            }

        return self.single_models

    # ---------------------------------------------------------------------
    # Model fitting: additive quadratic
    # ---------------------------------------------------------------------
    def fit_additive_quadratic(self) -> Dict:
        """
        Fit additive quadratic OLS:
            y ~ const + sum_k p_k + sum_k p_k^2

        Then solve for the jointly best p under box bounds.
        Since the model is separable across dimensions, the optimization is
        done dimension by dimension.

        Returns
        -------
        dict with fitted coefficients and optimal solution
        """
        self._check_data_loaded()

        linear_terms = self.X
        quad_terms = self.X**2
        X_design = np.column_stack([np.ones(self.X.shape[0]), linear_terms, quad_terms])

        beta = self._ols_fit(X_design, self.y)
        y_hat = X_design @ beta
        r2 = self._r2(self.y, y_hat)

        intercept = beta[0]
        a = beta[1 : 1 + self.K]
        d = beta[1 + self.K : 1 + 2 * self.K]  # diagonal quadratic terms

        p_star = np.zeros(self.K)
        for k in range(self.K):
            lb, ub = self.bounds[k]
            p_star[k], _ = self._solve_1d_quadratic(
                intercept=0.0,
                b1=a[k],
                b2=d[k],
                lb=lb,
                ub=ub,
            )

        y_star = intercept + np.dot(a, p_star) + np.dot(d, p_star**2)

        self.additive_model = {
            "model_name": "additive_quadratic",
            "intercept": intercept,
            "A": a,
            "diag_quadratic": d,
            "r2": r2,
            "optimal_p": p_star,
            "optimal_value": y_star,
        }
        return self.additive_model

    # ---------------------------------------------------------------------
    # Model fitting: full quadratic
    # ---------------------------------------------------------------------
    def fit_full_quadratic(
        self,
        prefer_gurobi: bool = True,
        n_starts: int = 20,
        scipy_method: str = "L-BFGS-B",
    ) -> Dict:
        """
        Fit full quadratic OLS:
            y ~ const + sum_k p_k + sum_k p_k^2 + sum_{i<j} p_i p_j

        Then solve the joint quadratic optimization problem.

        Returns
        -------
        dict with fitted coefficients and optimal solution
        """
        self._check_data_loaded()

        X_design, meta = self._build_full_quadratic_design(self.X)
        beta = self._ols_fit(X_design, self.y)
        y_hat = X_design @ beta
        r2 = self._r2(self.y, y_hat)

        intercept, A_hat, B_hat = self._parse_full_quadratic_coefficients(beta, meta)

        p_star, y_star, solver_used = self._solve_box_qp(
            intercept=intercept,
            A=A_hat,
            B=B_hat,
            bounds=self.bounds,
            prefer_gurobi=prefer_gurobi,
            n_starts=n_starts,
            scipy_method=scipy_method,
        )

        self.full_model = {
            "model_name": "full_quadratic",
            "intercept": intercept,
            "A": A_hat,
            "B": B_hat,
            "r2": r2,
            "optimal_p": p_star,
            "optimal_value": y_star,
            "solver_used": solver_used,
        }
        return self.full_model

    # ---------------------------------------------------------------------
    # Compare optimal gaps
    # ---------------------------------------------------------------------
    def compare_optimal_gaps(self) -> pd.DataFrame:
        """
        Compare the optimal values from:
            - best single-variable quadratic model
            - additive quadratic model
            - full quadratic model (benchmark)

        Also, if true A/B are known from simulation, report the true objective
        value at each model's estimated optimum and regret relative to the true optimum.

        Returns
        -------
        pd.DataFrame
        """
        if self.additive_model is None:
            raise ValueError("Run fit_additive_quadratic() first.")
        if self.full_model is None:
            raise ValueError("Run fit_full_quadratic() first.")
        if not self.single_models:
            raise ValueError("Run fit_single_variable_quadratics() first.")

        # Best single model by optimal predicted y
        best_single_idx = max(
            self.single_models.keys(),
            key=lambda k: self.single_models[k]["optimal_value"]
        )
        best_single = self.single_models[best_single_idx]

        # Build a K-dimensional p for the best single-variable model:
        # only one dimension moves to its own optimum, others set to their lower bounds.
        p_single = np.array([lb for lb, _ in self.bounds], dtype=float)
        p_single[best_single_idx] = best_single["optimal_point"]

        results = []

        # Full model as benchmark
        full_pred = self.full_model["optimal_value"]
        add_pred = self.additive_model["optimal_value"]

        results.append({
            "model": f"single_{best_single['feature_name']}",
            "predicted_optimal_value": np.nan,  # not comparable directly in K-d objective
            "benchmark_gap_vs_full_predicted": np.nan,
            "optimal_p": p_single,
        })
        results.append({
            "model": "additive_quadratic",
            "predicted_optimal_value": add_pred,
            "benchmark_gap_vs_full_predicted": full_pred - add_pred,
            "optimal_p": self.additive_model["optimal_p"],
        })
        results.append({
            "model": "full_quadratic",
            "predicted_optimal_value": full_pred,
            "benchmark_gap_vs_full_predicted": 0.0,
            "optimal_p": self.full_model["optimal_p"],
        })

        df = pd.DataFrame(results)

        # If true model known, evaluate true objective
        if self.true_A is not None and self.true_B is not None:
            p_true_opt, y_true_opt, solver_used = self.solve_true_optimum()

            true_vals = []
            true_regrets = []
            for _, row in df.iterrows():
                p = np.asarray(row["optimal_p"], dtype=float)
                val = self.evaluate_true_objective(p)
                true_vals.append(val)
                true_regrets.append(y_true_opt - val)

            df["true_objective_at_estimated_optimum"] = true_vals
            df["true_regret_vs_true_optimum"] = true_regrets
            df["true_solver_used"] = solver_used

        return df

    # ---------------------------------------------------------------------
    # Solve true optimum if simulation truth is known
    # ---------------------------------------------------------------------
    def solve_true_optimum(
        self,
        prefer_gurobi: bool = True,
        n_starts: int = 20,
        scipy_method: str = "L-BFGS-B",
    ) -> Tuple[np.ndarray, float, str]:
        """
        Solve the true optimization problem using the simulated true A/B.
        """
        if self.true_A is None or self.true_B is None:
            raise ValueError("True parameters are only available when data were simulated.")
        return self._solve_box_qp(
            intercept=self.true_intercept,
            A=self.true_A,
            B=self.true_B,
            bounds=self.bounds,
            prefer_gurobi=prefer_gurobi,
            n_starts=n_starts,
            scipy_method=scipy_method,
        )

    def evaluate_true_objective(self, p: np.ndarray) -> float:
        """
        Evaluate the true noiseless objective at p.
        """
        if self.true_A is None or self.true_B is None:
            raise ValueError("True parameters are only available when data were simulated.")
        p = np.asarray(p, dtype=float).reshape(-1)
        return float(self.true_intercept + p @ self.true_A + p @ self.true_B @ p)

    # ---------------------------------------------------------------------
    # Prediction helpers
    # ---------------------------------------------------------------------
    def predict_additive(self, P: np.ndarray) -> np.ndarray:
        if self.additive_model is None:
            raise ValueError("Run fit_additive_quadratic() first.")
        P = np.asarray(P, dtype=float)
        return (
            self.additive_model["intercept"]
            + P @ self.additive_model["A"]
            + (P**2) @ self.additive_model["diag_quadratic"]
        )

    def predict_full(self, P: np.ndarray) -> np.ndarray:
        if self.full_model is None:
            raise ValueError("Run fit_full_quadratic() first.")
        P = np.asarray(P, dtype=float)
        quad = np.einsum("ni,ij,nj->n", P, self.full_model["B"], P)
        return self.full_model["intercept"] + P @ self.full_model["A"] + quad

    # ---------------------------------------------------------------------
    # Internal utilities
    # ---------------------------------------------------------------------
    def _sample_p(self, n_samples: int, sampler: str = "uniform") -> np.ndarray:
        X = np.zeros((n_samples, self.K), dtype=float)

        for k, (lb, ub) in enumerate(self.bounds):
            if sampler == "uniform":
                X[:, k] = self.rng.uniform(lb, ub, size=n_samples)
            elif sampler == "truncated_normal":
                mu = 0.5 * (lb + ub)
                sigma = (ub - lb) / 6.0 if ub > lb else 1.0
                vals = self.rng.normal(mu, sigma, size=n_samples)
                X[:, k] = np.clip(vals, lb, ub)
            else:
                raise ValueError(f"Unknown sampler: {sampler}")

        return X

    def _check_data_loaded(self) -> None:
        if self.X is None or self.y is None:
            raise ValueError("No data loaded. Use simulate_data() or load_data().")

    @staticmethod
    def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta

    @staticmethod
    def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    @staticmethod
    def _solve_1d_quadratic(
        intercept: float,
        b1: float,
        b2: float,
        lb: float,
        ub: float,
    ) -> Tuple[float, float]:
        """
        Maximize intercept + b1*x + b2*x^2 over [lb, ub].
        """
        def obj(x):
            return intercept + b1 * x + b2 * x**2

        candidates = [lb, ub]

        if abs(b2) > 1e-12:
            x_stationary = -b1 / (2.0 * b2)
            if lb <= x_stationary <= ub:
                candidates.append(x_stationary)

        vals = [obj(x) for x in candidates]
        best_idx = int(np.argmax(vals))
        return float(candidates[best_idx]), float(vals[best_idx])

    def _build_full_quadratic_design(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Build design matrix for:
            const, p_i, p_i^2, p_i p_j (i<j)
        """
        n, K = X.shape
        cols = [np.ones(n)]
        linear_idx = []
        quad_diag_idx = []
        cross_idx = []

        # Linear
        for i in range(K):
            linear_idx.append(len(cols))
            cols.append(X[:, i])

        # Squared diagonal
        for i in range(K):
            quad_diag_idx.append(len(cols))
            cols.append(X[:, i] ** 2)

        # Cross terms
        cross_pairs = []
        for i in range(K):
            for j in range(i + 1, K):
                cross_idx.append(len(cols))
                cols.append(X[:, i] * X[:, j])
                cross_pairs.append((i, j))

        X_design = np.column_stack(cols)
        meta = {
            "linear_idx": linear_idx,
            "quad_diag_idx": quad_diag_idx,
            "cross_idx": cross_idx,
            "cross_pairs": cross_pairs,
        }
        return X_design, meta

    def _parse_full_quadratic_coefficients(self, beta: np.ndarray, meta: Dict) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Convert OLS coefficients into:
            intercept + A^T p + p^T B p
        where for cross term coefficient c_ij on p_i p_j, we store B_ij = c_ij / 2
        because p^T B p = sum_i B_ii p_i^2 + 2 sum_{i<j} B_ij p_i p_j
        """
        intercept = float(beta[0])
        A = np.zeros(self.K, dtype=float)
        B = np.zeros((self.K, self.K), dtype=float)

        for i, idx in enumerate(meta["linear_idx"]):
            A[i] = beta[idx]

        for i, idx in enumerate(meta["quad_diag_idx"]):
            B[i, i] = beta[idx]

        for idx, (i, j) in zip(meta["cross_idx"], meta["cross_pairs"]):
            B[i, j] = beta[idx] / 2.0
            B[j, i] = beta[idx] / 2.0

        return intercept, A, B

    def _solve_box_qp(
        self,
        intercept: float,
        A: np.ndarray,
        B: np.ndarray,
        bounds: List[Tuple[float, float]],
        prefer_gurobi: bool = True,
        n_starts: int = 20,
        scipy_method: str = "L-BFGS-B",
    ) -> Tuple[np.ndarray, float, str]:
        """
        Maximize:
            intercept + A^T p + p^T B p
        subject to box bounds.

        Try gurobipy first. If unavailable or it fails, use scipy multi-start.
        """
        A = np.asarray(A, dtype=float).reshape(-1)
        B = 0.5 * (np.asarray(B, dtype=float) + np.asarray(B, dtype=float).T)

        if prefer_gurobi:
            try:
                return self._solve_box_qp_gurobi(intercept, A, B, bounds)
            except Exception as e:
                warnings.warn(
                    f"Gurobi solver unavailable or failed ({e}). Falling back to scipy."
                )

        return self._solve_box_qp_scipy(
            intercept=intercept,
            A=A,
            B=B,
            bounds=bounds,
            n_starts=n_starts,
            method=scipy_method,
        )

    def _solve_box_qp_gurobi(
        self,
        intercept: float,
        A: np.ndarray,
        B: np.ndarray,
        bounds: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, float, str]:
        """
        Solve the nonconvex box-constrained QP with gurobipy.
        """
        import gurobipy as gp
        from gurobipy import GRB

        m = gp.Model("full_quadratic_opt")
        m.Params.OutputFlag = 0
        m.Params.NonConvex = 2

        x = []
        for k, (lb, ub) in enumerate(bounds):
            xk = m.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"p{k+1}")
            x.append(xk)

        obj = intercept
        obj += gp.quicksum(A[i] * x[i] for i in range(self.K))
        obj += gp.quicksum(B[i, i] * x[i] * x[i] for i in range(self.K))
        obj += gp.quicksum(
            2.0 * B[i, j] * x[i] * x[j]
            for i in range(self.K) for j in range(i + 1, self.K)
        )

        m.setObjective(obj, GRB.MAXIMIZE)
        m.optimize()

        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi did not return OPTIMAL status. Status={m.Status}")

        p_star = np.array([var.X for var in x], dtype=float)
        y_star = float(m.ObjVal)
        return p_star, y_star, "gurobi"

    def _solve_box_qp_scipy(
        self,
        intercept: float,
        A: np.ndarray,
        B: np.ndarray,
        bounds: List[Tuple[float, float]],
        n_starts: int = 20,
        method: str = "L-BFGS-B",
    ) -> Tuple[np.ndarray, float, str]:
        """
        Multi-start scipy local optimization for box-constrained QP.
        """
        def objective(p):
            return -(intercept + p @ A + p @ B @ p)

        best_x = None
        best_val = -np.inf

        # Start points: middle, random, corners
        starts = []

        midpoint = np.array([(lb + ub) / 2.0 for lb, ub in bounds], dtype=float)
        starts.append(midpoint)

        for _ in range(max(n_starts - 1, 0)):
            x0 = np.array([self.rng.uniform(lb, ub) for lb, ub in bounds], dtype=float)
            starts.append(x0)

        for x0 in starts:
            res = minimize(
                objective,
                x0=x0,
                method=method,
                bounds=bounds,
            )
            x_candidate = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
            val_candidate = intercept + x_candidate @ A + x_candidate @ B @ x_candidate

            if val_candidate > best_val:
                best_val = float(val_candidate)
                best_x = x_candidate.copy()

        return best_x, best_val, "scipy_multistart"

    # ---------------------------------------------------------------------
    # Convenience reporting
    # ---------------------------------------------------------------------
    def summary(self) -> None:
        """
        Print a brief summary of fitted models and optimal solutions.
        """
        print("=" * 80)
        print("QuadraticODAnalyzer Summary")
        print("=" * 80)

        if self.X is not None and self.y is not None:
            print(f"Data shape: X={self.X.shape}, y={self.y.shape}")
        else:
            print("No data loaded.")
            return

        print("\nBounds:")
        for k, (lb, ub) in enumerate(self.bounds):
            print(f"  {self.feature_names[k]} in [{lb}, {ub}]")

        if self.true_A is not None:
            print("\nTrue simulation parameters available.")

        if self.single_models:
            print("\nSingle-variable quadratic models:")
            for k, res in self.single_models.items():
                print(
                    f"  {res['feature_name']}: R2={res['r2']:.4f}, "
                    f"opt={res['optimal_point']:.4f}, "
                    f"pred_y*={res['optimal_value']:.4f}"
                )

        if self.additive_model is not None:
            print("\nAdditive quadratic model:")
            print(f"  R2={self.additive_model['r2']:.4f}")
            print(f"  optimal_p={np.round(self.additive_model['optimal_p'], 4)}")
            print(f"  pred_y*={self.additive_model['optimal_value']:.4f}")

        if self.full_model is not None:
            print("\nFull quadratic model:")
            print(f"  R2={self.full_model['r2']:.4f}")
            print(f"  optimal_p={np.round(self.full_model['optimal_p'], 4)}")
            print(f"  pred_y*={self.full_model['optimal_value']:.4f}")
            print(f"  solver={self.full_model['solver_used']}")


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    K = 4
    bounds = [(0.0, 1.0), (0.0, 1.5), (-0.5, 0.8), (0.0, 2.0)]

    analyzer = QuadraticODAnalyzer(K=K, bounds=bounds, random_state=42)

    # True data-generating parameters
    A_true = np.array([1.0, -0.5, 0.8, 0.3])
    B_true = np.array([
        [-1.2,  0.4,  0.2,  0.0],
        [ 0.4, -0.8, -0.3,  0.1],
        [ 0.2, -0.3, -1.0,  0.5],
        [ 0.0,  0.1,  0.5, -0.6],
    ])

    # 1) Simulate data
    X, y = analyzer.simulate_data(
        n_samples=2000,
        A=A_true,
        B=B_true,
        intercept=0.5,
        noise_std=0.2,
        sampler="uniform",
    )

    # 2) Alternatively, you could load your own data:
    # analyzer.load_data(X_external, y_external, feature_names=["p1", "p2", "p3", "p4"])

    # 3) Fit K separate 1D quadratics
    analyzer.fit_single_variable_quadratics()

    # 4) Fit additive quadratic and solve joint optimum
    analyzer.fit_additive_quadratic()

    # 5) Fit full quadratic with cross terms and solve joint optimum
    analyzer.fit_full_quadratic(prefer_gurobi=True, n_starts=30)

    # Summary
    analyzer.summary()

    # 6) Compare optimal gaps relative to full model
    gap_df = analyzer.compare_optimal_gaps()
    print("\nOptimal gap comparison:")
    print(gap_df.to_string(index=False))

    # If simulated, also solve the true optimum
    p_true_star, y_true_star, solver_used = analyzer.solve_true_optimum()
    print("\nTrue optimum (from true A, B):")
    print("  p* =", np.round(p_true_star, 4))
    print("  y* =", round(y_true_star, 4))
    print("  solver =", solver_used)