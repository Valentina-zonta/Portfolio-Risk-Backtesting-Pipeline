"""Portfolio optimizer – Global Minimum Variance (GMV)."""

import logging

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)


def solve_gmv(cov: np.ndarray, w_max: float = 0.15, long_only: bool = True) -> np.ndarray:
    """Solve the Global Minimum Variance portfolio.

    min  w' Σ w
    s.t. sum(w) = 1
         w >= 0        (if long_only)
         w <= w_max

    Parameters
    ----------
    cov : (n, n) covariance matrix
    w_max : maximum weight per asset
    long_only : enforce non-negativity

    Returns
    -------
    np.ndarray – optimal weight vector (n,)
    """
    n = cov.shape[0]
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cov))

    constraints = [
        cp.sum(w) == 1,
        w <= w_max,
    ]
    if long_only:
        constraints.append(w >= 0)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise cp.SolverError(f"Solver status: {prob.status}")
        weights = w.value
    except (cp.SolverError, Exception) as exc:
        logger.warning("OSQP failed (%s), trying SCS …", exc)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                raise cp.SolverError(f"Solver status: {prob.status}")
            weights = w.value
        except Exception as exc2:
            logger.error("All solvers failed (%s). Falling back to equal-weight.", exc2)
            weights = np.ones(n) / n

    # Clean up tiny negatives from numerical noise
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum()

    return weights
