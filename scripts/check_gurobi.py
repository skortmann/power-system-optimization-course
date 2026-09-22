"""What can Gurobi do here, and where does the free licence stop?

Run it as-is::

    uv run python scripts/check_gurobi.py

`gurobipy` installs from PyPI with no licence file, in a size-limited mode. The
course needs to know three things before relying on it:

1. does it solve convex QP, SOCP and MISOCP?
2. does it return usable duals for the conic constraints?
3. where does the size limit actually bite?

Answers go into docs/solver_guide.md. Nothing in the course may *require* a
licence, so this establishes what the fallback has to cover.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Model sizes to probe when locating the free-licence ceiling.
PROBE_SIZES = (100, 500, 1_000, 2_000, 5_000)

# --------------------------------------------------------------------------


def _line(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else '--  '}{label:34s} {detail}")


def licence_info() -> bool:
    """Report the licence situation. Returns True if a licence FILE was found.

    This matters for what the course may claim. A machine with an academic
    licence solves models that an unlicensed `pip install gurobipy` refuses, so
    a size ceiling measured here says nothing about what an external student
    will see unless the licence state is reported alongside it.
    """
    import os
    from pathlib import Path

    import gurobipy as gp

    print("\nLicence")
    print(f"  gurobipy version: {'.'.join(str(v) for v in gp.gurobi.version())}")

    candidates = [Path.home() / "gurobi.lic", Path("/opt/gurobi/gurobi.lic")]
    if os.environ.get("GRB_LICENSE_FILE"):
        candidates.insert(0, Path(os.environ["GRB_LICENSE_FILE"]))
    found = next((p for p in candidates if p.exists()), None)

    if found is not None:
        _line("licence file", True, f"{found}")
        print("       -> sizes below reflect THIS licence, not the free pip limit")
    else:
        _line("licence file", False, "none found — size-limited pip mode")
        print("       -> Gurobi's unlicensed mode caps model size (~2000 vars/cons)")

    try:
        env = gp.Env(params={"OutputFlag": 0})
        env.dispose()
        _line("environment starts", True, "")
    except Exception as exc:
        _line("environment starts", False, f"{type(exc).__name__}: {str(exc)[:70]}")
    return found is not None


def size_ceiling() -> int | None:
    """Find the largest LP the current licence will actually solve."""
    import gurobipy as gp

    print("\nFree-licence size ceiling (plain LP)")
    last_ok: int | None = None
    for n in PROBE_SIZES:
        try:
            with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
                x = m.addVars(n, lb=0.0, ub=1.0)
                m.setObjective(sum(x[i] for i in range(n)), gp.GRB.MAXIMIZE)
                m.addConstr(sum(x[i] for i in range(n)) <= n / 2)
                m.optimize()
                ok = m.Status == gp.GRB.OPTIMAL
            _line(f"{n:,} variables", ok, "solved" if ok else "not optimal")
            if ok:
                last_ok = n
        except Exception as exc:
            _line(f"{n:,} variables", False, str(exc)[:70])
            break
    return last_ok


def conic() -> dict[str, bool]:
    """SOCP, MISOCP, and whether duals come back."""
    import gurobipy as gp

    found: dict[str, bool] = {}
    print("\nConic")

    # Rotated cone written the way the branch-flow relaxation writes it:
    #   P^2 + Q^2 <= u * ell,   u, ell >= 0
    try:
        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
            P = m.addVar(lb=-10, ub=10, name="P")
            Q = m.addVar(lb=-10, ub=10, name="Q")
            u = m.addVar(lb=0.81, ub=1.21, name="u")
            ell = m.addVar(lb=0.0, ub=5.0, name="ell")
            m.setObjective(ell, gp.GRB.MINIMIZE)
            m.addConstr(P >= 1.0)
            m.addConstr(Q >= 0.5)
            m.addQConstr(P * P + Q * Q <= u * ell, name="soc")
            m.optimize()
            ok = m.Status == gp.GRB.OPTIMAL
            _line("SOCP (rotated cone)", ok, f"obj={m.ObjVal:.6f}" if ok else str(m.Status))
            found["socp"] = ok
    except Exception as exc:
        _line("SOCP (rotated cone)", False, f"{type(exc).__name__}: {str(exc)[:70]}")
        found["socp"] = False

    # Duals on a continuous conic model. Gurobi exposes Pi for linear
    # constraints; QCPDual must be switched on for the quadratic ones.
    try:
        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
            m.Params.QCPDual = 1
            P = m.addVar(lb=-10, ub=10)
            Q = m.addVar(lb=-10, ub=10)
            u = m.addVar(lb=0.81, ub=1.21)
            ell = m.addVar(lb=0.0, ub=5.0)
            m.setObjective(ell, gp.GRB.MINIMIZE)
            lin = m.addConstr(P >= 1.0)
            qc = m.addQConstr(P * P + Q * Q <= u * ell)
            m.optimize()
            pi = lin.Pi
            qpi = qc.QCPi
            _line("conic duals", True, f"linear Pi={pi:.4f}, QCPi={qpi:.4f}")
            found["duals"] = True
    except Exception as exc:
        _line("conic duals", False, f"{type(exc).__name__}: {str(exc)[:70]}")
        found["duals"] = False

    # MISOCP: an integer master over a conic continuous part. This is the
    # structure Tutorial 08 decomposes.
    try:
        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
            y = m.addVar(vtype=gp.GRB.BINARY)
            P = m.addVar(lb=0, ub=10)
            Q = m.addVar(lb=0, ub=10)
            u = m.addVar(lb=0.81, ub=1.21)
            ell = m.addVar(lb=0.0, ub=5.0)
            m.setObjective(ell + 0.3 * y, gp.GRB.MINIMIZE)
            m.addConstr(P >= 1.0 + 0.5 * y)
            m.addConstr(Q >= 0.5)
            m.addQConstr(P * P + Q * Q <= u * ell)
            m.optimize()
            ok = m.Status == gp.GRB.OPTIMAL
            _line("MISOCP", ok, f"obj={m.ObjVal:.6f}, y={y.X:.0f}" if ok else str(m.Status))
            found["misocp"] = ok
    except Exception as exc:
        _line("MISOCP", False, f"{type(exc).__name__}: {str(exc)[:70]}")
        found["misocp"] = False

    return found


def through_pyomo() -> None:
    """The course models in Pyomo, so the Pyomo bridge has to work too."""
    import pyomo.environ as pyo

    print("\nVia Pyomo")
    for name in ("appsi_gurobi", "gurobi_direct", "gurobi"):
        try:
            opt = pyo.SolverFactory(name)
            ok = bool(opt.available(exception_flag=False))
            if not ok:
                _line(name, False, "not available")
                continue
            m = pyo.ConcreteModel()
            m.x = pyo.Var([0, 1], bounds=(0, 10), initialize=1.0)
            m.t = pyo.Var(bounds=(0, 5), initialize=1.0)
            m.obj = pyo.Objective(expr=sum(m.x[i] ** 2 for i in (0, 1)) + m.t)
            m.c = pyo.Constraint(expr=m.x[0] + m.x[1] >= 3.0)
            res = opt.solve(m)
            _line(
                f"{name} (convex QP)",
                str(res.solver.termination_condition) == "optimal",
                f"obj={pyo.value(m.obj):.6f}",
            )
        except Exception as exc:
            _line(name, False, f"{type(exc).__name__}: {str(exc)[:70]}")


def main() -> None:
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        print("gurobipy is not installed. The course runs without it; see")
        print("docs/solver_guide.md for the open-source route.")
        return

    print("=" * 78)
    print("Gurobi capability check")
    print("=" * 78)
    licensed = licence_info()
    ceiling = size_ceiling()
    conic()
    through_pyomo()

    print("\n" + "-" * 78)
    if ceiling is not None:
        where = "a licence file on this machine" if licensed else "the free pip mode"
        print(f"Largest LP solved here: {ceiling:,} variables, using {where}.")
        if licensed:
            print("An external student WITHOUT a licence is capped far lower")
            print("(~2000 variables / constraints), so the course must not assume this.")
    print("Everything above is OPTIONAL. psopt_course.solvers falls back to")
    print("HiGHS / IPOPT / Clarabel when Gurobi is missing or the model is too big.")
    print("-" * 78)


if __name__ == "__main__":
    main()
