"""Report which solvers this machine can actually run, and on which problem classes.

Run it as-is::

    uv run python scripts/check_solvers.py

The course promises that every notebook has an open-source path. This script is
how that promise is checked rather than asserted — it builds one tiny instance of
each problem class the course uses and tries to solve it.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Also try the optional commercial solvers. They are never required.
CHECK_COMMERCIAL = True

# --------------------------------------------------------------------------


def _report(name: str, ok: bool, detail: str = "") -> None:
    mark = "OK  " if ok else "--  "
    print(f"  {mark}{name:28s} {detail}")


def check_lp_milp() -> dict[str, bool]:
    """LP, QP and MILP through Pyomo."""
    import pyomo.environ as pyo

    found: dict[str, bool] = {}
    print("\nLP / QP / MILP (Pyomo)")

    # A two-generator economic dispatch: the course's very first model.
    def dispatch(integer: bool = False, quadratic: bool = False):
        m = pyo.ConcreteModel()
        m.G = pyo.Set(initialize=[1, 2])
        cost = {1: 10.0, 2: 30.0}
        pmax = {1: 60.0, 2: 100.0}
        m.p = pyo.Var(m.G, bounds=lambda m, g: (0, pmax[g]))
        if integer:
            m.u = pyo.Var(m.G, domain=pyo.Binary)
            m.link = pyo.Constraint(m.G, rule=lambda m, g: m.p[g] <= pmax[g] * m.u[g])
        if quadratic:
            m.obj = pyo.Objective(expr=sum(cost[g] * m.p[g] + 0.05 * m.p[g] ** 2 for g in m.G))
        else:
            m.obj = pyo.Objective(expr=sum(cost[g] * m.p[g] for g in m.G))
        m.bal = pyo.Constraint(expr=sum(m.p[g] for g in m.G) == 100.0)
        return m

    for label, kwargs, solver in [
        ("HiGHS (appsi) LP", {}, "appsi_highs"),
        ("HiGHS (appsi) MILP", {"integer": True}, "appsi_highs"),
        # Pyomo's HiGHS interface rejects a quadratic objective, so the course's
        # QP dispatch goes to IPOPT instead. The objective is convex, so a local
        # optimum IS the global one -- which is why this substitution is sound
        # and not a quiet downgrade.
        ("IPOPT QP", {"quadratic": True}, "ipopt"),
    ]:
        try:
            model = dispatch(**kwargs)
            opt = pyo.SolverFactory(solver)
            if not opt.available(exception_flag=False):
                _report(label, False, "solver not available")
                found[label] = False
                continue
            res = opt.solve(model)
            ok = str(res.solver.termination_condition) == "optimal"
            _report(
                label, ok,
                f"{res.solver.termination_condition}, obj={pyo.value(model.obj):,.2f}",
            )
            found[label] = ok
        except Exception as exc:
            _report(label, False, f"{type(exc).__name__}: {str(exc)[:70]}")
            found[label] = False
    return found


def check_nlp() -> dict[str, bool]:
    """Nonconvex NLP through Pyomo — the AC-OPF path."""
    import pyomo.environ as pyo

    found: dict[str, bool] = {}
    print("\nNLP (Pyomo)")

    # A two-bus AC-flavoured problem: nonlinear, with a trig term.
    m = pyo.ConcreteModel()
    m.v = pyo.Var(bounds=(0.9, 1.1), initialize=1.0)
    m.th = pyo.Var(bounds=(-0.5, 0.5), initialize=0.0)
    m.obj = pyo.Objective(expr=(m.v - 1.05) ** 2 + (m.th - 0.1) ** 2)
    m.c = pyo.Constraint(expr=m.v * pyo.cos(m.th) >= 0.95)

    for label, solver in [("IPOPT", "ipopt")]:
        try:
            opt = pyo.SolverFactory(solver)
            if not opt.available(exception_flag=False):
                _report(label, False, "not on PATH — see docs/solver_guide.md")
                found[label] = False
                continue
            res = opt.solve(m, tee=False)
            ok = str(res.solver.termination_condition) == "optimal"
            _report(label, ok, f"{res.solver.termination_condition}, v={pyo.value(m.v):.4f}")
            found[label] = ok
        except Exception as exc:
            _report(label, False, f"{type(exc).__name__}: {str(exc)[:70]}")
            found[label] = False
    return found


def check_conic() -> dict[str, bool]:
    """Second-order cone programs, and — critically — their dual variables.

    Tutorial 08 builds Benders cuts from conic duals, so a route that solves an
    SOCP but cannot hand back a trustworthy dual is not enough.
    """
    found: dict[str, bool] = {}
    print("\nSOCP (conic)")

    # min c'x  s.t.  ||x||_2 <= t,  t <= 1,  x_0 >= 0.4
    try:
        import cvxpy as cp

        x = cp.Variable(2)
        t = cp.Variable()
        constraints = [cp.SOC(t, x), t <= 1.0, x[0] >= 0.4]
        problem = cp.Problem(cp.Minimize(-x[0] - 0.5 * x[1]), constraints)
        problem.solve(solver=cp.CLARABEL)
        dual_ok = constraints[1].dual_value is not None
        _report(
            "CVXPY + Clarabel",
            problem.status == "optimal",
            f"{problem.status}, obj={problem.value:.6f}, duals={'yes' if dual_ok else 'NO'}",
        )
        found["cvxpy_clarabel"] = problem.status == "optimal" and dual_ok
    except Exception as exc:
        _report("CVXPY + Clarabel", False, f"{type(exc).__name__}: {str(exc)[:70]}")
        found["cvxpy_clarabel"] = False

    # The same cone written as a convex QCQP and handed to IPOPT. Convexity
    # means a local solve is global, so this is a legitimate second route —
    # and it keeps the formulation inside Pyomo.
    try:
        import pyomo.environ as pyo

        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1], initialize=0.5)
        m.t = pyo.Var(bounds=(0, 1), initialize=1.0)
        m.obj = pyo.Objective(expr=-m.x[0] - 0.5 * m.x[1])
        m.cone = pyo.Constraint(expr=m.x[0] ** 2 + m.x[1] ** 2 <= m.t**2)
        m.lb = pyo.Constraint(expr=m.x[0] >= 0.4)
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        opt = pyo.SolverFactory("ipopt")
        if opt.available(exception_flag=False):
            res = opt.solve(m)
            ok = str(res.solver.termination_condition) == "optimal"
            duals = len(m.dual) > 0
            _report(
                "Pyomo QCQP + IPOPT",
                ok,
                f"obj={pyo.value(m.obj):.6f}, duals={'yes' if duals else 'NO'}",
            )
            found["pyomo_ipopt_qcqp"] = ok and duals
        else:
            _report("Pyomo QCQP + IPOPT", False, "IPOPT not on PATH")
            found["pyomo_ipopt_qcqp"] = False
    except Exception as exc:
        _report("Pyomo QCQP + IPOPT", False, f"{type(exc).__name__}: {str(exc)[:70]}")
        found["pyomo_ipopt_qcqp"] = False

    return found


def check_commercial() -> dict[str, bool]:
    print("\nOptional commercial (never required)")
    found: dict[str, bool] = {}
    import pyomo.environ as pyo

    for label, solver in [("Gurobi (appsi)", "appsi_gurobi"), ("CPLEX", "cplex_direct")]:
        try:
            opt = pyo.SolverFactory(solver)
            ok = bool(opt.available(exception_flag=False))
            _report(label, ok, "available" if ok else "not installed / no licence")
            found[label] = ok
        except Exception:
            _report(label, False, "not installed")
            found[label] = False
    return found


def main() -> None:
    import sys

    print("=" * 78)
    print("Solver availability for the power-system optimization course")
    print("=" * 78)
    print(f"  python {sys.version.split()[0]}")

    results: dict[str, bool] = {}
    results |= check_lp_milp()
    results |= check_nlp()
    results |= check_conic()
    if CHECK_COMMERCIAL:
        check_commercial()

    print("\n" + "-" * 78)
    lp_ok = results.get("HiGHS (appsi) LP", False)
    milp_ok = results.get("HiGHS (appsi) MILP", False)
    qp_ok = results.get("IPOPT QP", False)
    nlp_ok = results.get("IPOPT", False)
    soc_ok = results.get("cvxpy_clarabel", False) or results.get("pyomo_ipopt_qcqp", False)

    print("Course coverage with open-source solvers only:")
    print(f"  Tutorials 01-03 (LP + MILP, HiGHS)  {'yes' if lp_ok and milp_ok else 'NO'}")
    print(f"  Tutorial 01 (convex QP, IPOPT)      {'yes' if qp_ok else 'NO'}")
    print(f"  Tutorial 04, 08 (nonconvex NLP)     {'yes' if nlp_ok else 'NO'}")
    print(f"  Tutorials 05, 08 (SOCP + duals)     {'yes' if soc_ok else 'NO'}")
    if not (lp_ok and milp_ok and qp_ok and nlp_ok and soc_ok):
        print("\n  Some path is missing. docs/solver_guide.md explains how to install it.")
    print("-" * 78)


if __name__ == "__main__":
    main()
