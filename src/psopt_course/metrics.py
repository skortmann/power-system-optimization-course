r"""Gaps, bounds and the scoreboard — the course's shared vocabulary for "how good".

Four different things get called "the gap" in this subject and they are not
interchangeable. Keeping them apart is most of what this module is for:

============================  ==========================================
:func:`optimality_gap`        MILP: incumbent vs. best bound. "Has branch
                              and bound finished?"
:func:`relaxation_gap`        Convex relaxation vs. a feasible point of the
                              original. "Is the relaxation tight?"
:func:`integrality_gap`       MILP optimum vs. its LP relaxation. A property
                              of the *formulation*, not of a run.
:func:`certified_interval`    Lower bound from a relaxation, upper bound from
                              a feasible solution. What is actually known
                              about the global optimum.
============================  ==========================================

Tutorial 08 reports two of these simultaneously and the difference between them
is the point of the tutorial.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "CertifiedInterval",
    "Scoreboard",
    "certified_interval",
    "integrality_gap",
    "optimality_gap",
    "relaxation_gap",
]


def _relative(numerator: float, reference: float) -> float:
    return numerator / max(1.0, abs(reference))


def optimality_gap(incumbent: float, bound: float) -> float:
    r"""MILP gap: $(z^{inc} - z^{bound}) / \max(1, |z^{inc}|)$.

    Answers "has the search finished", and nothing else. A solver reporting a
    0% gap has proved optimality *for the model it was given*.
    """
    return _relative(incumbent - bound, incumbent)


def relaxation_gap(relaxed: float, feasible: float) -> float:
    r"""$(z^{feas} - z^{relax}) / \max(1, |z^{feas}|)$ for a minimization.

    Answers "is the relaxation tight". A small value is *suggestive* of
    exactness and does not establish it — a relaxed point can have almost the
    right objective and still be physically unrealisable. For the SOC branch
    flow model, ``SOCBFM.soc_residuals()`` is the test that speaks to
    exactness; this number does not.
    """
    if relaxed > feasible + 1e-6:
        raise ValueError(
            f"a relaxation cannot exceed a feasible objective for a minimization: "
            f"relaxed={relaxed:.6g} > feasible={feasible:.6g}. Either the two models "
            f"differ by more than the intended relaxation, or one of them is wrong."
        )
    return _relative(feasible - relaxed, feasible)


def integrality_gap(integer_optimum: float, lp_relaxation: float) -> float:
    r"""How much the LP relaxation understates the MILP optimum.

    A property of the formulation. Two formulations of the same problem can
    have very different integrality gaps, and the tighter one is usually the
    one worth solving even if it has more constraints.
    """
    return _relative(integer_optimum - lp_relaxation, integer_optimum)


@dataclass
class CertifiedInterval:
    r"""What is actually known about a global optimum.

    .. math::

        z^\star_{LB} \;\le\; z^\star \;\le\; z^{UB}

    Tutorial 08's central object: the lower bound comes from Benders on an SOC
    relaxation, the upper bound from an AC-feasible recovery. The *width* is
    the honest statement of remaining uncertainty.
    """

    lower: float
    upper: float
    lower_source: str = "convex relaxation"
    upper_source: str = "feasible solution"

    def __post_init__(self) -> None:
        if self.lower > self.upper + 1e-6:
            raise ValueError(
                f"lower bound {self.lower:.6g} exceeds upper bound {self.upper:.6g}; "
                f"the interval is empty, so one of the two models is misspecified"
            )

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def relative_width(self) -> float:
        return _relative(self.width, self.upper)

    def contains(self, value: float, tolerance: float = 1e-6) -> bool:
        return self.lower - tolerance <= value <= self.upper + tolerance

    def summary(self) -> str:
        return (
            f"z* in [{self.lower:.6f}, {self.upper:.6f}]  "
            f"(width {self.width:.6f}, {self.relative_width:.3%})\n"
            f"  lower from: {self.lower_source}\n"
            f"  upper from: {self.upper_source}"
        )


def certified_interval(
    lower: float, upper: float, *, lower_source: str = "convex relaxation",
    upper_source: str = "feasible solution",
) -> CertifiedInterval:
    return CertifiedInterval(lower, upper, lower_source, upper_source)


@dataclass
class Scoreboard:
    """The course-wide results table, accumulated from real runs.

    Never typed by hand. Every row comes from a solve, and
    :meth:`to_frame` is what ``docs/scoreboard.md`` is generated from.
    """

    rows: list[dict] = field(default_factory=list)

    def add(
        self,
        tutorial: str,
        formulation: str,
        problem_class: str,
        *,
        objective: float | None = None,
        solver: str = "",
        termination: str = "",
        seconds: float = float("nan"),
        n_variables: int = 0,
        n_binary: int = 0,
        n_constraints: int = 0,
        **extra,
    ) -> None:
        self.rows.append(
            {
                "tutorial": tutorial,
                "formulation": formulation,
                "class": problem_class,
                "variables": n_variables,
                "binary": n_binary,
                "constraints": n_constraints,
                "objective": objective,
                "solver": solver,
                "termination": termination,
                "seconds": seconds,
                **extra,
            }
        )

    def from_record(self, tutorial: str, formulation: str, record, **extra) -> None:
        """Add a row straight from a :class:`psopt_course.solvers.SolveRecord`."""
        self.add(
            tutorial,
            formulation,
            record.problem_class,
            objective=record.objective,
            solver=record.solver,
            termination=record.termination,
            seconds=record.seconds,
            n_variables=record.n_variables,
            n_binary=record.n_binary,
            n_constraints=record.n_constraints,
            **extra,
        )

    def to_frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame()
        return pd.DataFrame(self.rows)

    def to_markdown(self) -> str:
        frame = self.to_frame()
        if frame.empty:
            return "_no results recorded_"
        return frame.to_markdown(index=False, floatfmt=".4f")


def summarise_array(values, name: str = "value") -> pd.Series:
    """min / mean / max / worst-case, for residual and violation reporting."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return pd.Series(dtype=float, name=name)
    return pd.Series(
        {
            "min": float(array.min()),
            "mean": float(array.mean()),
            "max": float(array.max()),
            "max |.|": float(np.abs(array).max()),
        },
        name=name,
    )
