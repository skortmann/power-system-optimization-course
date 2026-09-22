"""Course-wide numerical tolerances, in one place with reasons.

Scattering magic numbers like ``1e-6`` through ten notebooks invites two
failures: a check too tight for the solver that produced the number, and a
check so loose it passes on a wrong answer. Both happened during development.
Tutorial 05's bound check originally compared two IPOPT solutions to ``1e-9``
absolute — stricter than either solve — and printed ``False`` for two objectives
agreeing to 7.6e-07.

Each constant below says what it guards and why that magnitude.
"""

from __future__ import annotations

__all__ = [
    "BOUND_ORDERING",
    "CONIC_RESIDUAL",
    "DUAL_RELATIVE",
    "EQUALITY_FEASIBILITY",
    "FINITE_DIFFERENCE_STEP",
    "INEQUALITY_FEASIBILITY",
    "INTEGER_FEASIBILITY",
    "OBJECTIVE_ABSOLUTE",
    "OBJECTIVE_RELATIVE",
    "PROBABILITY_ABSOLUTE",
    "describe",
]

#: Equality constraint residual, e.g. nodal power balance.
#: IPOPT's default ``constr_viol_tol`` is 1e-4 and it typically converges far
#: below that; HiGHS's primal feasibility tolerance is 1e-7. 1e-6 sits above
#: both solvers' achievable accuracy and well below any physically meaningful
#: imbalance (1e-6 per unit on a 10 MVA base is 10 W).
EQUALITY_FEASIBILITY = 1e-6

#: Inequality violation, e.g. a voltage or thermal limit. Same reasoning; an
#: interior-point method approaches a bound from inside, so violations here are
#: smaller still.
INEQUALITY_FEASIBILITY = 1e-6

#: Distance from a binary variable to {0, 1}. MILP solvers report integrality
#: to about 1e-5 by default; anything tighter tests the solver, not the model.
INTEGER_FEASIBILITY = 1e-5

#: Second-order cone slack ``u*ell - P^2 - Q^2``. Looser than the equality
#: tolerance on purpose: this is a PRODUCT of solver-accurate quantities, so
#: its error is roughly the sum of their relative errors times their magnitude.
#: Measured residuals on the course's tight cases are ~4e-07.
CONIC_RESIDUAL = 1e-5

#: Relative agreement between two objectives that should be equal — a
#: decomposition against its monolith, or two formulations of one model.
#: Both sides carry solver error, so the comparison must be relative.
OBJECTIVE_RELATIVE = 1e-6

#: Absolute floor for the same comparison, so an optimum near zero does not
#: make the relative test meaningless.
OBJECTIVE_ABSOLUTE = 1e-8

#: Agreement between a dual variable and a finite-difference estimate of
#: d(objective)/d(right-hand side). Deliberately loose: the finite difference
#: is only valid while the ACTIVE SET is unchanged, and the estimate carries
#: both the step's truncation error and two solves' worth of solver error.
DUAL_RELATIVE = 1e-3

#: Slack allowed when checking a bound ordering such as z_relaxed <= z_feasible
#: or LB <= UB. Scaled by max(1, |reference|) at the call site.
BOUND_ORDERING = 1e-6

#: Absolute tolerance when comparing an empirical violation frequency against a
#: requested epsilon. A Monte Carlo estimate from N samples has standard error
#: ~sqrt(eps(1-eps)/N); at eps=0.05 and N=100,000 that is 0.0007, so 0.005 is
#: about seven standard errors — loose enough not to flag sampling noise.
PROBABILITY_ABSOLUTE = 5e-3

#: Step for finite-difference checks of dual variables. Small enough that the
#: active set is unlikely to change, large enough that the difference is well
#: above solver noise.
FINITE_DIFFERENCE_STEP = 1e-3


def describe() -> str:
    """A printable summary, for the audit report and for notebooks."""
    rows = [
        ("equality feasibility", EQUALITY_FEASIBILITY, "power balance, KVL"),
        ("inequality feasibility", INEQUALITY_FEASIBILITY, "voltage, thermal limits"),
        ("integer feasibility", INTEGER_FEASIBILITY, "distance to {0,1}"),
        ("conic residual", CONIC_RESIDUAL, "u*ell - P^2 - Q^2"),
        ("objective (relative)", OBJECTIVE_RELATIVE, "decomposition vs monolith"),
        ("objective (absolute)", OBJECTIVE_ABSOLUTE, "floor near zero"),
        ("dual (relative)", DUAL_RELATIVE, "vs finite difference"),
        ("bound ordering", BOUND_ORDERING, "LB <= UB"),
        ("probability (absolute)", PROBABILITY_ABSOLUTE, "empirical vs requested eps"),
    ]
    width = max(len(name) for name, _, _ in rows)
    lines = [f"{'quantity'.ljust(width)}  tolerance   guards"]
    lines += [f"{name.ljust(width)}  {value:<10.1e}  {why}" for name, value, why in rows]
    return "\n".join(lines)


def objectives_agree(a: float, b: float) -> bool:
    """Do two objectives agree to the course's tolerance?"""
    return abs(a - b) <= max(OBJECTIVE_ABSOLUTE, OBJECTIVE_RELATIVE * max(1.0, abs(b)))


def bound_holds(lower: float, upper: float) -> bool:
    """Is ``lower <= upper`` to the course's tolerance?"""
    return lower <= upper + BOUND_ORDERING * max(1.0, abs(upper))
