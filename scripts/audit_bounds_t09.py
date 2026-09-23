"""Verify Tutorial 09's bound-ordering claim by measurement, not assertion.

Tutorial 09 states ``z_LP <= z_DW <= z_MILP`` and then claims that here the
first two COINCIDE, because the per-generator subproblem has the integrality
property (Geoffrion 1974). That is an empirical claim about this instance, so
this script measures all three numbers with the tutorial's own model code and
checks the ordering and the equality directly.

Run it as-is::

    uv run python scripts/audit_bounds_t09.py
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Cap on column-generation iterations. The instance converges in far fewer.
MAX_ITERATIONS = 40

# --------------------------------------------------------------------------

import warnings

warnings.filterwarnings("ignore")

import jupytext
import numpy as np

from psopt_course import tolerances as tol
from psopt_course.config import PROJECT_ROOT

SOURCE = PROJECT_ROOT / "tutorials" / "_sources" / "09_column_generation.py"


def load_tutorial_namespace() -> dict:
    """Execute the tutorial's provided cells so we test ITS code, not a copy.

    Re-typing the model here would test whether two transcriptions agree, which
    is not the question. The question is whether the tutorial's own monolithic
    model, its pricing problem and its master produce the ordering it claims.
    """
    cells = jupytext.read(SOURCE, fmt="py:percent").cells
    namespace: dict = {"__name__": "__psopt_audit__"}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        tags = set(cell.get("metadata", {}).get("tags", []) or [])
        if "exercise" in tags or "advanced-exercise" in tags:
            continue
        source = cell["source"]
        # Stop before the exercises; everything needed is defined by then.
        if "run_column_generation" in source and "def " not in source:
            continue
        try:
            exec(compile(source, "<tutorial-09>", "exec"), namespace)
        except Exception as error:
            print(f"  (skipped a cell: {type(error).__name__}: {error})")
    return namespace


def main() -> int:
    print("=" * 78)
    print("TUTORIAL 09 BOUND ORDERING")
    print("=" * 78)

    ns = load_tutorial_namespace()
    solve = ns["solve"]

    # --- z_MILP: the compact mixed-integer model, solved to optimality
    milp = ns["monolithic"](relax=False)
    z_milp = solve(milp, "MILP").objective

    # --- z_LP: the SAME model with the binaries relaxed
    lp = ns["monolithic"](relax=True)
    z_lp = solve(lp, "LP").objective

    # --- z_DW: column generation to convergence
    make_column = ns["make_column"]
    restricted_master = ns["restricted_master"]
    pricing = ns["pricing"]
    GENS, T = ns["GENS"], ns["T"]

    columns = {
        g: [make_column(np.ones(T), spec), make_column(np.zeros(T), spec)]
        for g, spec in enumerate(GENS)
    }
    iterations = 0
    while iterations < MAX_ITERATIONS:
        iterations += 1
        master = restricted_master(columns)
        solve(master, "LP", duals=True)
        prices = [master.dual[master.demand[t]] for t in range(T)]
        sigmas = {g: master.dual[master.convexity[g]] for g in range(len(GENS))}
        added = 0
        for g in range(len(GENS)):
            column, reduced = pricing(g, prices, sigmas[g])
            if reduced < -1e-6:
                columns[g].append(column)
                added += 1
        if added == 0:
            break
    z_dw = solve(restricted_master(columns), "LP").objective

    print(f"\n  z_LP   (compact LP relaxation) = {z_lp:.8f}")
    print(f"  z_DW   (Dantzig-Wolfe bound)   = {z_dw:.8f}")
    print(f"  z_MILP (compact MILP optimum)  = {z_milp:.8f}")
    print(f"  converged in {iterations} iterations, "
          f"{sum(len(v) for v in columns.values())} columns")

    ordering = tol.bound_holds(z_lp, z_dw) and tol.bound_holds(z_dw, z_milp)
    coincide = tol.objectives_agree(z_lp, z_dw)
    strict_gap = z_milp - z_dw

    print(f"\n  z_LP <= z_DW <= z_MILP            : {ordering}")
    print(f"  |z_DW - z_LP|                     : {abs(z_dw - z_lp):.3e}")
    print(f"  z_MILP - z_DW (the integrality gap): {strict_gap:.6f} "
          f"({100 * strict_gap / z_milp:.3f}%)")
    print(f"\n  tutorial's claim 'the two coincide': {coincide}")

    print("=" * 78)
    if not ordering:
        print("FAIL: the bound ordering the tutorial states does not hold.")
        return 1
    if not coincide:
        print("FAIL: the tutorial claims z_LP == z_DW; they differ.")
        return 1
    print("PASS: ordering holds and z_LP == z_DW, as Tutorial 09 claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
