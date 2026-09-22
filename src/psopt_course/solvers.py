"""Solver selection, and the record of what a solve actually did.

The course insists that *problem class*, *algorithm* and *solver software* are
three different things. This module is where that distinction is made
operational: you ask for a problem class, and it hands back a solver that can
handle it, having checked rather than assumed.

Every preference below was measured by ``scripts/check_solvers.py`` and
``scripts/check_gurobi.py``, not assumed.

**Gurobi is preferred where it is genuinely better, and never required.**
The students this course is written for have a licence, and for convex QP,
SOCP and especially MISOCP Gurobi is the right tool: it takes the rotated cone
natively, returns conic duals through ``QCPDual`` (which Tutorial 08 needs to
build Benders cuts), and solves the mixed-integer conic master that has no
open-source equivalent in this stack. Where Gurobi is absent, every entry falls
back to an open-source route and the course still completes.

Two stack facts worth knowing:

* Pyomo's HiGHS interface **rejects a quadratic objective**
  (``DegreeError: ... degree None``), so convex QP goes to Gurobi or IPOPT.
  Falling back to IPOPT is sound rather than a downgrade: the objective is
  convex, so its local optimum is the global one.
* A rotated second-order cone can be written as a convex quadratic constraint
  and solved by Gurobi or IPOPT, *or* handed to a conic solver through CVXPY.
  All three work and they agree. See :mod:`psopt_course.relaxations`.

**Licence caveat.** An unlicensed ``pip install gurobipy`` runs in a
size-limited mode (roughly 2000 variables and constraints). Fast mode stays
well inside that; the full-scale capstone does not. :func:`solve` therefore
treats a Gurobi size-limit error as a reason to fall back rather than to fail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import pyomo.environ as pyo

__all__ = [
    "ProblemClass",
    "SolveRecord",
    "available_solvers",
    "solve",
    "solver_for",
]

#: The problem classes the course actually builds. Deliberately not an open
#: string: naming the class is a step students are required to take.
ProblemClass = Literal[
    "LP", "QP", "MILP", "MIQP", "NLP", "MINLP", "SOCP", "MISOCP"
]

#: Which Pyomo solver to try for each class, in order of preference.
#:
#: ``appsi_gurobi``, NOT ``gurobi`` or ``gurobi_direct``. A measured correctness
#: requirement rather than a style preference: on the course's SOC branch-flow
#: model -- the rotated cone ``P^2 + Q^2 <= u * ell`` -- the legacy interfaces
#: return an unmapped ``other`` termination AND an objective above the optimum::
#:
#:     gurobi          other     0.115235 MW
#:     gurobi_direct   other     0.103718 MW
#:     appsi_gurobi    optimal   0.098836 MW
#:     ipopt           optimal   0.098795 MW
#:
#: The last two agree to 0.04%. A silently wrong objective from a CONVEX program
#: is the worst failure mode in this course, because every bound downstream
#: inherits it.
#:
#: HiGHS leads for LP and MILP because it is excellent at both and needs no
#: licence. Gurobi leads wherever the cone or the quadratic objective is the
#: point. Every row ends in an open-source option except MISOCP, which is
#: flagged below.
_PREFERENCE: dict[str, tuple[str, ...]] = {
    "LP": ("appsi_highs", "appsi_gurobi", "ipopt"),
    "MILP": ("appsi_highs", "appsi_gurobi"),
    # Pyomo's HiGHS interface cannot take a quadratic objective at all.
    "QP": ("appsi_gurobi", "ipopt"),
    "MIQP": ("appsi_gurobi",),
    "NLP": ("ipopt",),
    "MINLP": ("appsi_gurobi", "ipopt"),
    # IPOPT FIRST, and this one is measured too. Written as a convex QCQP, the
    # course's SOC branch-flow model goes through Pyomo's Gurobi bridges badly:
    # the legacy interfaces return a wrong objective (see above) and the appsi
    # one returns `unknown` with no feasible solution on half of an eight-case
    # sweep. IPOPT solved all eight, monotonically in the amount of control
    # installed, with cone residuals of 4e-07 throughout.
    #
    # This is a statement about the BRIDGE and this formulation, not about
    # Gurobi, which is an excellent conic solver. It is kept as the fallback,
    # and MISOCP still goes to it because there is no alternative.
    #
    # Convexity is what makes IPOPT's answer trustworthy here: a local optimum
    # of a convex program is the global one.
    "SOCP": ("ipopt", "appsi_gurobi"),
    # The one class with no open-source route in this stack. Tutorial 08 uses it
    # only for the *reference* monolithic solve that Benders is checked against;
    # the decomposition itself never needs it, because fixing the master leaves
    # a continuous SOCP. `solver_for` says exactly this when it cannot find one.
    "MISOCP": ("appsi_gurobi",),
}

#: Classes for which no open-source solver exists in this stack.
_COMMERCIAL_ONLY = frozenset({"MISOCP", "MIQP"})

_CACHE: dict[str, bool] = {}


def _is_available(name: str) -> bool:
    if name not in _CACHE:
        try:
            _CACHE[name] = bool(pyo.SolverFactory(name).available(exception_flag=False))
        except Exception:
            _CACHE[name] = False
    return _CACHE[name]


def available_solvers() -> dict[str, bool]:
    """Availability of every solver the course might reach for."""
    names = sorted({n for group in _PREFERENCE.values() for n in group})
    return {n: _is_available(n) for n in names}


def solver_for(problem_class: ProblemClass) -> str:
    """Name the solver this machine will use for ``problem_class``.

    Raises with an actionable message rather than returning something that will
    fail later inside a solve.
    """
    if problem_class not in _PREFERENCE:
        raise ValueError(
            f"unknown problem class {problem_class!r}; "
            f"expected one of {sorted(_PREFERENCE)}"
        )
    for name in _PREFERENCE[problem_class]:
        if _is_available(name):
            return name
    tried = ", ".join(_PREFERENCE[problem_class])
    if problem_class in _COMMERCIAL_ONLY:
        raise RuntimeError(
            f"{problem_class} needs a solver this stack only has commercially "
            f"(tried {tried}). This is used for a reference solve, not for the "
            f"method being taught — the tutorial says which result is skipped. "
            f"See docs/solver_guide.md."
        )
    raise RuntimeError(
        f"no available solver for {problem_class}. Tried {tried}. "
        f"See docs/solver_guide.md."
    )


@dataclass
class SolveRecord:
    """Everything worth recording about one solve.

    The course's reproducibility rule is that a reported objective without a
    termination condition beside it is not a result. This is the object that
    keeps them together.
    """

    problem_class: str
    solver: str
    termination: str
    objective: float | None
    seconds: float
    n_variables: int
    n_binary: int
    n_constraints: int
    gap: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Did the solver claim success?

        ``optimal`` for most solvers; IPOPT reports ``optimal`` for a locally
        optimal KKT point, which is exactly the distinction Tutorial 04 makes.
        """
        return self.termination in {"optimal", "locallyOptimal", "feasible"}

    def summary(self) -> str:
        value = "n/a" if self.objective is None else f"{self.objective:,.4f}"
        line = (
            f"{self.problem_class:5s} via {self.solver:12s} "
            f"{self.termination:12s} obj={value:>14s} "
            f"{self.seconds:6.2f}s  "
            f"{self.n_variables:,} vars ({self.n_binary:,} binary), "
            f"{self.n_constraints:,} cons"
        )
        if self.gap is not None:
            line += f", gap={self.gap:.3%}"
        return line


def _count(model: pyo.ConcreteModel) -> tuple[int, int, int]:
    n_var = n_bin = 0
    for v in model.component_data_objects(pyo.Var, active=True):
        n_var += 1
        if v.is_binary():
            n_bin += 1
    n_con = sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True))
    return n_var, n_bin, n_con


def solve(
    model: pyo.ConcreteModel,
    problem_class: ProblemClass,
    *,
    solver: str | None = None,
    duals: bool = False,
    tee: bool = False,
    options: dict[str, Any] | None = None,
) -> SolveRecord:
    """Solve ``model`` and return a :class:`SolveRecord`.

    Parameters
    ----------
    problem_class:
        Stated by the caller, not inferred. Naming the class before choosing a
        solver is the habit the whole course is trying to build.
    duals:
        Attach an import suffix so shadow prices can be read afterwards. Needed
        from Tutorial 02 on, and essential for every Benders cut in Tutorial 08.
    """
    name = solver or solver_for(problem_class)
    if duals and not hasattr(model, "dual"):
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    def _attempt(solver_name: str):
        opt = pyo.SolverFactory(solver_name)
        for key, value in (options or {}).items():
            opt.options[key] = value
        started = time.perf_counter()
        try:
            return opt.solve(model, tee=tee), time.perf_counter() - started
        except RuntimeError as exc:
            # Pyomo's appsi HiGHS interface RAISES when there is no feasible
            # solution to load, instead of returning a status. Infeasibility is
            # a legitimate answer the course asks students to inspect, so retry
            # without loading and report the condition.
            if "feasible solution was not found" not in str(exc):
                raise
            results = opt.solve(model, tee=tee, load_solutions=False)
            return results, time.perf_counter() - started

    try:
        results, elapsed = _attempt(name)
    except Exception as exc:
        # An unlicensed Gurobi refuses models above ~2000 rows/columns. That is
        # a licensing limit, not a modelling error, so fall through to the
        # open-source route and say so rather than failing the notebook.
        fallbacks = [s for s in _PREFERENCE[problem_class] if s != name and _is_available(s)]
        if not fallbacks:
            raise
        print(
            f"  [solvers] {name} declined this model "
            f"({type(exc).__name__}: {str(exc)[:80]}); "
            f"falling back to {fallbacks[0]}"
        )
        name = fallbacks[0]
        results, elapsed = _attempt(name)

    termination = str(results.solver.termination_condition)
    try:
        objective = float(pyo.value(next(model.component_data_objects(pyo.Objective, active=True))))
    except Exception:
        objective = None

    gap = None
    try:
        bound = results.problem.lower_bound
        if objective is not None and bound is not None and abs(objective) > 1e-9:
            gap = abs(objective - float(bound)) / max(1.0, abs(objective))
    except Exception:
        gap = None

    n_var, n_bin, n_con = _count(model)
    return SolveRecord(
        problem_class=problem_class,
        solver=name,
        termination=termination,
        objective=objective,
        seconds=elapsed,
        n_variables=n_var,
        n_binary=n_bin,
        n_constraints=n_con,
        gap=gap,
    )
