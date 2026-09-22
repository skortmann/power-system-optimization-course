r"""Scenarios, chance constraints and the validation that keeps them honest.

The course's rule for this material: **a violation probability that was typed
into a model is not a result.** A chance constraint is a claim about the world,
and the only way to check it is to draw fresh samples the optimizer never saw
and count.

What a chance constraint is
---------------------------
.. math::

    \Pr[g(x, \xi) \le 0] \ge 1 - \epsilon

and *not*

.. code-block:: python

    constraint <= limit * 0.95          # this is not a chance constraint

For a linear constraint :math:`a^\top x + b^\top \xi \le c` with
:math:`\xi \sim \mathcal{N}(\mu, \Sigma)`, the reformulation is exact:

.. math::

    a^\top x + b^\top \mu
    + \Phi^{-1}(1-\epsilon)\,\sqrt{b^\top \Sigma b}
    \;\le\; c

The square root is a second-order cone constraint, which is why this module and
:mod:`psopt_course.relaxations` end up needing the same solvers.

Individual is not joint
-----------------------
Requiring each of :math:`m` constraints to hold with probability
:math:`1-\epsilon` says almost nothing about all of them holding together.
Boole's inequality gives the safe direction:

.. math::

    \Pr\Big[\bigcap_i A_i\Big] \ge 1 - \sum_i \epsilon_i

so allocating :math:`\epsilon_i = \epsilon/m` guarantees a joint level of
:math:`\epsilon` — conservatively, and more conservatively the more correlated
the constraints are. :func:`empirical_violation` measures what actually
happens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "GaussianUncertainty",
    "ViolationReport",
    "boole_joint_bound",
    "chance_constraint_margin",
    "empirical_violation",
    "equal_risk_allocation",
    "quantile",
    "sample_scenarios",
    "wilson_interval",
]


def quantile(epsilon: float) -> float:
    r"""$\Phi^{-1}(1-\epsilon)$, the safety factor a chance constraint buys.

    Worth printing next to any chance-constrained result: at
    :math:`\epsilon = 0.05` it is 1.645, and at :math:`\epsilon = 0.01` it is
    2.326. Tightening the risk level by four points costs 41% more margin.
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must lie strictly in (0, 1), got {epsilon}")
    return float(stats.norm.ppf(1.0 - epsilon))


@dataclass(frozen=True)
class GaussianUncertainty:
    r"""$\xi \sim \mathcal{N}(\mu, \Sigma)$, with the covariance kept explicit.

    Correlation is the whole reason individual and joint reliability differ, so
    the covariance is a first-class object here rather than a vector of
    standard deviations.
    """

    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        mean = np.atleast_1d(np.asarray(self.mean, dtype=float))
        cov = np.atleast_2d(np.asarray(self.covariance, dtype=float))
        if cov.shape != (mean.size, mean.size):
            raise ValueError(
                f"covariance must be {mean.size}x{mean.size} for a mean of "
                f"length {mean.size}, got {cov.shape}"
            )
        if not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError("covariance must be symmetric")
        eigenvalues = np.linalg.eigvalsh(cov)
        if eigenvalues.min() < -1e-10:
            raise ValueError(
                f"covariance must be positive semidefinite; smallest eigenvalue "
                f"is {eigenvalues.min():.3e}"
            )
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", cov)

    @property
    def dimension(self) -> int:
        return int(self.mean.size)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.covariance))

    @property
    def correlation(self) -> np.ndarray:
        s = self.std
        outer = np.outer(s, s)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(outer > 0, self.covariance / outer, 0.0)

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(self.mean, self.covariance, size=n)

    def spread(self, direction: np.ndarray) -> float:
        r"""$\sqrt{b^\top \Sigma b}$ — the std of a linear functional of $\xi$."""
        b = np.atleast_1d(np.asarray(direction, dtype=float))
        return float(np.sqrt(max(b @ self.covariance @ b, 0.0)))


def chance_constraint_margin(
    uncertainty: GaussianUncertainty, direction: np.ndarray, epsilon: float
) -> float:
    r"""The deterministic tightening $\Phi^{-1}(1-\epsilon)\sqrt{b^\top\Sigma b}$.

    Subtract it from the limit and the chance constraint becomes an ordinary
    linear constraint. That reformulation is **exact** for Gaussian
    :math:`\xi` and a linear :math:`g` — and it is exact for no other case you
    are likely to meet, which is why the assumption must be stated whenever the
    result is reported.
    """
    return quantile(epsilon) * uncertainty.spread(direction)


def equal_risk_allocation(epsilon: float, n_constraints: int) -> np.ndarray:
    r"""Split a joint budget evenly: $\epsilon_i = \epsilon / m$.

    The simplest allocation satisfying Boole's inequality, and usually a
    conservative one. Tutorial 07 compares it against an optimized allocation.
    """
    if n_constraints < 1:
        raise ValueError("need at least one constraint")
    return np.full(n_constraints, epsilon / n_constraints, dtype=float)


def boole_joint_bound(epsilons) -> float:
    r"""The guaranteed joint violation level, $\sum_i \epsilon_i$.

    An upper bound on the true joint violation probability, and only that. The
    true value can be far smaller when constraints are strongly correlated —
    which is exactly why the bound is safe and why it can be wasteful.
    """
    total = float(np.sum(np.asarray(list(epsilons), dtype=float)))
    return min(total, 1.0)


def sample_scenarios(
    uncertainty: GaussianUncertainty,
    n: int,
    seed: int,
    *,
    antithetic: bool = False,
) -> np.ndarray:
    """Draw scenarios, optionally antithetically to cut sampling noise.

    The seed is required rather than optional. An unseeded scenario set makes
    the whole experiment unreproducible, and a chance-constrained result that
    cannot be reproduced cannot be checked.
    """
    if antithetic:
        half = (n + 1) // 2
        base = uncertainty.sample(half, seed=seed)
        centred = base - uncertainty.mean
        mirrored = uncertainty.mean - centred
        return np.vstack([base, mirrored])[:n]
    return uncertainty.sample(n, seed=seed)


@dataclass
class ViolationReport:
    """What a fresh Monte Carlo sample said about a chance-constrained plan."""

    individual: np.ndarray
    joint: float
    n_samples: int
    requested_individual: np.ndarray | None = None
    requested_joint: float | None = None

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            {"empirical violation": self.individual},
            index=[f"constraint {i}" for i in range(self.individual.size)],
        )
        if self.requested_individual is not None:
            frame["requested"] = self.requested_individual
            frame["within budget"] = frame["empirical violation"] <= frame["requested"] + 1e-12
        return frame

    def summary(self) -> str:
        lines = [
            f"{self.n_samples:,} independent samples",
            f"worst individual violation: {self.individual.max():.4f}",
            f"JOINT violation:            {self.joint:.4f}",
        ]
        if self.requested_joint is not None:
            verdict = "within" if self.joint <= self.requested_joint else "ABOVE"
            lines.append(f"requested joint level:      {self.requested_joint:.4f} ({verdict})")
        low, high = wilson_interval(self.joint, self.n_samples)
        lines.append(f"95% CI on the joint rate:   [{low:.4f}, {high:.4f}]")
        return "\n".join(lines)


def empirical_violation(
    residuals: np.ndarray,
    *,
    requested_individual=None,
    requested_joint: float | None = None,
) -> ViolationReport:
    r"""Count violations in an out-of-sample set.

    Parameters
    ----------
    residuals:
        Shape ``(n_samples, n_constraints)``; the value of :math:`g(x, \xi)`,
        so a violation is ``> 0``. These must come from samples the optimizer
        never saw — reusing the in-sample scenarios measures how well the
        solver fitted them, not how reliable the plan is.

    Notes
    -----
    The joint rate is the fraction of samples in which **any** constraint is
    violated. It is not the product, and it is not the maximum, of the
    individual rates; that is the entire point of Tutorial 07.
    """
    values = np.atleast_2d(np.asarray(residuals, dtype=float))
    violated = values > 0.0
    individual = violated.mean(axis=0)
    joint = float(violated.any(axis=1).mean())
    requested = (
        None if requested_individual is None
        else np.atleast_1d(np.asarray(requested_individual, dtype=float))
    )
    return ViolationReport(
        individual=individual,
        joint=joint,
        n_samples=int(values.shape[0]),
        requested_individual=requested,
        requested_joint=requested_joint,
    )


def wilson_interval(rate: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for an estimated probability.

    Preferred over the normal approximation because violation rates sit near
    zero, where the normal interval happily returns a negative lower bound.
    Reporting an empirical violation of 0.048 from 200 samples without an
    interval overstates what was measured.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    denominator = 1.0 + z**2 / n
    centre = (rate + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denominator
    return (float(max(0.0, centre - margin)), float(min(1.0, centre + margin)))
