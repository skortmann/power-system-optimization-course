"""Re-test the independent reviewer's findings before acting on any of them.

A reviewer's finding is a hypothesis. Several of these contradict claims the
course makes and one contradicts a claim THIS AUDIT made, so each is re-run
here from the tutorial's own code rather than taken on report.

Run it as-is::

    uv run python scripts/audit_reviewer_claims.py
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Random price vectors per generator when testing the integrality property.
N_PRICE_PROBES = 300

#: Seed for those price vectors.
SEED = 20260923

# --------------------------------------------------------------------------

import contextlib
import warnings

warnings.filterwarnings("ignore")

import jupytext
import numpy as np
import pyomo.environ as pyo

from psopt_course.config import PROJECT_ROOT
from psopt_course.solvers import solve

SOURCES = PROJECT_ROOT / "tutorials" / "_sources"
VERDICTS: list[tuple[str, str, str]] = []


def verdict(claim: str, result: str, evidence: str) -> None:
    VERDICTS.append((claim, result, evidence))
    print(f"\n[{result}] {claim}\n    {evidence}")


def namespace_for(stem: str, stop_marker: str | None = None) -> dict:
    """Run a tutorial's non-exercise code cells and hand back the namespace."""
    path = next(SOURCES.glob(f"{stem}*.py"))
    ns: dict = {"__name__": "__psopt_audit__", "display": lambda *a, **k: None}
    for cell in jupytext.read(path, fmt="py:percent").cells:
        if cell["cell_type"] != "code":
            continue
        tags = set(cell.get("metadata", {}).get("tags", []) or [])
        if {"exercise", "advanced-exercise"} & tags:
            continue
        if stop_marker and stop_marker in cell["source"]:
            break
        # A later cell may still be usable even when an earlier one needs
        # something only the notebook provides, so a failure here is not fatal.
        with contextlib.suppress(Exception):
            exec(compile(cell["source"], f"<{stem}>", "exec"), ns)
    return ns


# ------------------------------------------------------- M2: integrality property


def check_integrality_property() -> None:
    """Does each single-generator subproblem have the integrality property?

    THE AUDIT'S OWN CLAIM IS AT STAKE. This audit verified that z_LP == z_DW
    and reported the tutorial's *attribution* of that coincidence to
    Geoffrion's integrality property as verified. Measuring that the two
    numbers agree does not establish WHY. The property itself says: for every
    objective direction, minimising over conv(X_g) equals minimising over the
    LP relaxation of X_g. That is a statement about all price vectors, so it
    is tested over many price vectors.
    """
    ns = namespace_for("09")
    GENS, T, DEMAND = ns["GENS"], ns["T"], ns["DEMAND"]
    rng = np.random.default_rng(SEED)

    def single_generator(spec, prices, relax):
        m = pyo.ConcreteModel()
        m.T = pyo.Set(initialize=range(T), ordered=True)
        domain = pyo.UnitInterval if relax else pyo.Binary
        m.u = pyo.Var(m.T, domain=domain)
        m.v = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.p = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.lo = pyo.Constraint(m.T, rule=lambda m, t: m.p[t] >= spec["pmin"] * m.u[t])
        m.hi = pyo.Constraint(m.T, rule=lambda m, t: m.p[t] <= spec["pmax"] * m.u[t])

        @m.Constraint(m.T)
        def startup(m, t):
            if t == m.T.first():
                return m.v[t] >= m.u[t]
            return m.v[t] >= m.u[t] - m.u[m.T.prev(t)]

        @m.Constraint(m.T)
        def minup(m, t):
            length = spec["minup"]
            if t + length > T:
                return pyo.Constraint.Skip
            return sum(m.u[tt] for tt in range(t, t + length)) >= length * m.v[t]

        m.obj = pyo.Objective(
            expr=sum(
                spec["c1"] * m.p[t] + spec["noload"] * m.u[t] + spec["start"] * m.v[t]
                - prices[t] * m.p[t]
                for t in m.T
            )
        )
        return solve(m, "LP" if relax else "MILP").objective

    rows = []
    any_gap = False
    for spec in GENS:
        worst = 0.0
        for _ in range(N_PRICE_PROBES):
            prices = rng.uniform(0.0, 2.0 * spec["c1"] + 40.0, size=T)
            z_lp = single_generator(spec, prices, relax=True)
            z_ip = single_generator(spec, prices, relax=False)
            worst = max(worst, z_ip - z_lp)
        rows.append(f"{spec['name']}: worst (MILP - LP) = {worst:.6f}")
        if worst > 1e-6:
            any_gap = True

    verdict(
        "M2: the per-generator subproblem has the integrality property",
        "REVIEWER CORRECT - the course's explanation is wrong" if any_gap
        else "REVIEWER WRONG - the property holds",
        f"{N_PRICE_PROBES} random price vectors per generator; "
        + "; ".join(rows)
        + (". A strictly positive gap means conv(X_g) is strictly inside the LP "
           "polytope, so the integrality property does NOT hold and cannot be "
           "the reason z_LP == z_DW here." if any_gap else ""),
    )
    # Demand is unused here but confirms we loaded the tutorial's real data.
    assert len(DEMAND) == T


# -------------------------------------------------------------- H1: T02 KKT case


def check_t02_interior_case() -> None:
    """Is the 'interior' KKT case actually interior?"""
    ns = namespace_for("02")
    system_factory = ns.get("two_generator_system")
    if system_factory is None:
        verdict("H1: T02 interior case", "COULD NOT TEST", "factory not in namespace")
        return

    rows = []
    contradiction = False
    for demand in (60.0, 70.0, 130.0):
        system = system_factory(demand=demand)
        gens = system.generators
        m = pyo.ConcreteModel()
        m.G = pyo.Set(initialize=[g.name for g in gens])
        by_name = {g.name: g for g in gens}
        m.p = pyo.Var(
            m.G,
            bounds=lambda m, g, _by=by_name: (_by[g].p_min, _by[g].p_max),
        )
        m.balance = pyo.Constraint(expr=sum(m.p[g] for g in m.G) == demand)
        m.cost = pyo.Objective(expr=sum(by_name[g].c1 * m.p[g] for g in m.G))
        rec = solve(m, "LP", duals=True)
        p = [pyo.value(m.p[g]) for g in m.G]
        at_ceiling = [
            abs(pyo.value(m.p[g]) - by_name[g].p_max) < 1e-9 for g in m.G
        ]
        rows.append(
            f"D={demand:5.0f}: p*={[round(x, 1) for x in p]}, "
            f"lambda={rec.objective and pyo.value(m.dual[m.balance]):.2f}, "
            f"cheap unit at ceiling: {at_ceiling[0]}"
        )
        if abs(demand - 70.0) < 1e-9 and at_ceiling[0]:
            contradiction = True

    verdict(
        "H1: T02's 'interior' case (D=70) has the cheap unit strictly inside",
        "REVIEWER CORRECT - D=70 is AT the ceiling" if contradiction
        else "REVIEWER WRONG - D=70 is interior",
        "; ".join(rows),
    )


# ------------------------------------------------------ H2: T02 integrality gap


def check_t02_gap_widens() -> None:
    """Does the integrality gap widen with start-up cost, as the text claims?"""
    ns = namespace_for("02")
    builder = ns.get("commitment_model") or ns.get("unit_commitment")
    if builder is None:
        names = [k for k in ns if "commit" in k.lower()]
        verdict("H2: T02 gap widens with start cost", "COULD NOT TEST",
                f"builder not found; candidates in namespace: {names}")
        return
    verdict("H2: T02 gap widens with start cost", "SEE MANUAL RUN",
            "builder located; measured separately")


# ----------------------------------------------------------- M7: IPOPT gap=inf%


def check_ipopt_gap() -> None:
    """Does every IPOPT solve report an infinite gap?"""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.c = pyo.Constraint(expr=m.x >= 2.0)
    m.obj = pyo.Objective(expr=(m.x - 3.0) ** 2)
    rec = solve(m, "NLP")
    infinite = not np.isfinite(rec.gap) if rec.gap is not None else False
    verdict(
        "M7: SolveRecord.gap is infinite for IPOPT solves",
        "REVIEWER CORRECT" if infinite else "REVIEWER WRONG",
        f"solver={rec.solver}, termination={rec.termination}, "
        f"objective={rec.objective:.6f}, gap={rec.gap}",
    )


def main() -> int:
    print("=" * 78)
    print("RE-TESTING THE REVIEWER'S FINDINGS")
    print("=" * 78)
    check_ipopt_gap()
    check_t02_interior_case()
    check_integrality_property()
    print("\n" + "=" * 78)
    for claim, result, _ in VERDICTS:
        print(f"  {result:<48} {claim}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
