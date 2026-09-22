r"""Bookkeeping for Benders decomposition and column generation.

Both algorithms are the same story told twice: keep a lower bound and an upper
bound on one number, and add something each iteration that squeezes them. Benders
adds **constraints** (cuts) to a master; column generation adds **variables**
(columns) to a restricted master. This module holds the shared bookkeeping so
the notebooks can spend their code on the mathematics.

What is deliberately *not* here
-------------------------------
The cut itself. Constructing

.. math::

    \theta \ge \pi^\top (b - B y)

from a dual solution is the lesson of Tutorial 08, and the pricing problem is
the lesson of Tutorial 09. A student who imports ``make_benders_cut`` has been
robbed of the tutorial. What this module provides is iteration history, bound
tracking, convergence tests and the plots — the parts that are the same every
time and teach nothing.

The gaps are not interchangeable
--------------------------------
Tutorial 08 tracks two at once and they mean different things:

``Gap_BD``
    :math:`UB - LB` within the model being decomposed. Answers "has the
    algorithm converged?"
``Gap_AC``
    :math:`z^{feas}_{AC} - z^\star_{SOC}` between a relaxation and a feasible
    point of the original problem. Answers "is the relaxation tight?"

A run can close the first to zero while the second stays wide. Benders has then
solved the relaxation perfectly and the relaxation is simply not exact.
:class:`BoundHistory` keeps them in separate columns for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "BoundHistory",
    "ConvergenceTest",
    "Iteration",
    "relative_gap",
]


def relative_gap(lower: float, upper: float) -> float:
    r"""$(UB - LB) / \max(1, |UB|)$.

    The ``max(1, ...)`` keeps the measure finite when the optimum is near zero,
    which happens often enough in loss-minimising problems to matter.
    """
    if not np.isfinite(lower) or not np.isfinite(upper):
        return float("inf")
    return (upper - lower) / max(1.0, abs(upper))


@dataclass
class Iteration:
    """One pass of a decomposition algorithm."""

    index: int
    lower_bound: float
    upper_bound: float
    master_objective: float | None = None
    subproblem_objective: float | None = None
    n_cuts: int = 0
    master_seconds: float = 0.0
    subproblem_seconds: float = 0.0
    feasible: bool = True
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def gap(self) -> float:
        return relative_gap(self.lower_bound, self.upper_bound)


@dataclass
class BoundHistory:
    """Iteration log for Benders or column generation.

    Enforces the one invariant that catches most decomposition bugs: for a
    minimization the lower bound must never decrease and the upper bound must
    never increase. A master relaxation that *loses* ground means a cut was
    invalid — it cut off part of the true feasible region — and that is worth
    an exception rather than a wrong answer several iterations later.
    """

    name: str = "decomposition"
    iterations: list[Iteration] = field(default_factory=list)
    #: Set False for algorithms where the incumbent legitimately fluctuates.
    enforce_monotone: bool = True
    tolerance: float = 1e-7

    def add(self, iteration: Iteration) -> Iteration:
        if self.enforce_monotone and self.iterations:
            previous = self.iterations[-1]
            if iteration.lower_bound < previous.lower_bound - self.tolerance:
                raise ValueError(
                    f"{self.name}: the lower bound fell from "
                    f"{previous.lower_bound:.8g} to {iteration.lower_bound:.8g} at "
                    f"iteration {iteration.index}. A master relaxation cannot lose "
                    f"ground when cuts are only added, so a cut was invalid."
                )
            if iteration.upper_bound > previous.upper_bound + self.tolerance:
                raise ValueError(
                    f"{self.name}: the upper bound rose from "
                    f"{previous.upper_bound:.8g} to {iteration.upper_bound:.8g} at "
                    f"iteration {iteration.index}. The incumbent should only ever "
                    f"improve — keep the best solution found so far."
                )
        self.iterations.append(iteration)
        return iteration

    def record(
        self,
        lower_bound: float,
        upper_bound: float,
        **kwargs,
    ) -> Iteration:
        """Append an iteration, numbering it automatically."""
        return self.add(
            Iteration(
                index=len(self.iterations) + 1,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                **kwargs,
            )
        )

    @property
    def best_lower(self) -> float:
        return self.iterations[-1].lower_bound if self.iterations else -float("inf")

    @property
    def best_upper(self) -> float:
        return self.iterations[-1].upper_bound if self.iterations else float("inf")

    @property
    def gap(self) -> float:
        return relative_gap(self.best_lower, self.best_upper)

    def converged(self, tolerance: float = 1e-4) -> bool:
        return self.gap <= tolerance

    def to_frame(self) -> pd.DataFrame:
        if not self.iterations:
            return pd.DataFrame()
        frame = pd.DataFrame(
            [
                {
                    "iteration": it.index,
                    "lower_bound": it.lower_bound,
                    "upper_bound": it.upper_bound,
                    "relative_gap": it.gap,
                    "master_objective": it.master_objective,
                    "subproblem_objective": it.subproblem_objective,
                    "cuts": it.n_cuts,
                    "master_time_s": it.master_seconds,
                    "subproblem_time_s": it.subproblem_seconds,
                    "feasible": it.feasible,
                    "note": it.note,
                }
                for it in self.iterations
            ]
        ).set_index("iteration")
        return frame

    def summary(self) -> str:
        if not self.iterations:
            return f"{self.name}: no iterations recorded"
        total = sum(it.master_seconds + it.subproblem_seconds for it in self.iterations)
        feasibility_cuts = sum(1 for it in self.iterations if not it.feasible)
        return (
            f"{self.name}: {len(self.iterations)} iterations, "
            f"LB={self.best_lower:.6f}, UB={self.best_upper:.6f}, "
            f"gap={self.gap:.3%}, "
            f"{self.iterations[-1].n_cuts} cuts "
            f"({feasibility_cuts} from infeasible subproblems), "
            f"{total:.2f}s"
        )

    def plot(self, ax=None, *, log_gap: bool = True):
        """Bounds converging, with the gap beneath them.

        The picture the course reuses for branch-and-bound, Benders and column
        generation, so that they visibly do the same thing.
        """
        import matplotlib.pyplot as plt

        frame = self.to_frame()
        if frame.empty:
            raise ValueError("nothing to plot yet")

        if ax is None:
            _fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
        top, bottom = ax

        top.plot(frame.index, frame.lower_bound, "o-", ms=4, label="lower bound (master)")
        top.plot(frame.index, frame.upper_bound, "s-", ms=4, label="upper bound (incumbent)")
        top.fill_between(frame.index, frame.lower_bound, frame.upper_bound,
                         alpha=0.15, label="optimality gap")
        top.set_ylabel("objective")
        top.set_title(f"{self.name}: bounds converging")
        top.legend(loc="best", fontsize=8)

        bottom.plot(frame.index, frame.relative_gap, "o-", ms=4, color="0.3")
        if log_gap and (frame.relative_gap > 0).any():
            bottom.set_yscale("log")
        bottom.set_xlabel("iteration")
        bottom.set_ylabel("relative gap")
        bottom.grid(True, alpha=0.3)
        return ax


@dataclass
class ConvergenceTest:
    """When to stop, and why it stopped.

    Separating the reason from the fact matters: a run that hit the iteration
    cap has not converged, and reporting its bound as if it had is how a
    decomposition result becomes wrong.
    """

    tolerance: float = 1e-4
    max_iterations: int = 50
    max_seconds: float = float("inf")

    def should_stop(self, history: BoundHistory, elapsed: float = 0.0) -> tuple[bool, str]:
        if history.converged(self.tolerance):
            return True, f"converged: gap {history.gap:.3e} <= {self.tolerance:.0e}"
        if len(history.iterations) >= self.max_iterations:
            return True, (
                f"STOPPED at the iteration cap ({self.max_iterations}) with gap "
                f"{history.gap:.3e} — this is not convergence"
            )
        if elapsed >= self.max_seconds:
            return True, (
                f"STOPPED at the time limit ({self.max_seconds:.0f}s) with gap "
                f"{history.gap:.3e} — this is not convergence"
            )
        return False, ""
