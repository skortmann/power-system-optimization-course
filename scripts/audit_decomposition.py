"""Decomposition audit: Benders and column generation against ground truth.

The brief asks for "especially strong verification" here, because a
decomposition that converges smoothly to the wrong answer looks exactly like
one that works. Every check below compares against something independent:
a monolithic solve, a full enumeration, or a direct re-solve.

Run it as-is::

    uv run python scripts/audit_decomposition.py
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Random y vectors used to probe cut validity away from the incumbent.
N_RANDOM_PROBES = 24

# --------------------------------------------------------------------------

import copy
import itertools
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pyomo.environ as pyo

from psopt_course import tolerances as tol
from psopt_course.networks import radial_feeder
from psopt_course.relaxations import SOCBFM
from psopt_course.solvers import available_solvers, solve

RESULTS: list[dict] = []

VMIN, VMAX = 0.90, 1.05
Q_LIMIT = 0.03
INVEST_COST = 0.0008
N_CANDIDATES = 3
CLASS = {"linear": "LP", "soc": "SOCP", "exact": "NLP"}
NET = radial_feeder(n_pv=N_CANDIDATES, pv_mw=0.5)


def record(check: str, verdict: str, evidence: str) -> None:
    RESULTS.append({"check": check, "verdict": verdict, "evidence": evidence})
    print(f"  [{verdict}] {check}\n         {evidence}")


# ===================================================================== Benders


def subproblem(y, definition="soc"):
    """Exactly the tutorial's subproblem, rebuilt here independently."""
    model = SOCBFM(copy.deepcopy(NET), current_definition=definition)
    model.add_OPF(vmin=VMIN, vmax=VMAX)
    m = model.model
    gens = sorted(m.sG)
    m.Y = pyo.Param(range(len(gens)), initialize=dict(enumerate(y)), mutable=True)
    for g in gens:
        m.qsG[g].unfix()
        m.qsG[g].setlb(None)
        m.qsG[g].setub(None)

    @m.Constraint(range(len(gens)))
    def q_upper(m, k):
        return m.qsG[gens[k]] <= Q_LIMIT * m.Y[k]

    @m.Constraint(range(len(gens)))
    def q_lower(m, k):
        return -m.qsG[gens[k]] <= Q_LIMIT * m.Y[k]

    slack = sorted(m.eG)[0]
    m.obj = pyo.Objective(expr=m.pG[slack])
    record_ = solve(m, CLASS[definition], duals=True)
    return model, m, gens, record_


def value_function(definition="soc"):
    """Q(y) at every one of the eight patterns. Ground truth by enumeration."""
    table = {}
    for pattern in itertools.product([0, 1], repeat=N_CANDIDATES):
        _model, m, _gens, record_ = subproblem(list(pattern), definition)
        table[pattern] = pyo.value(m.obj) if record_.ok else float("nan")
    return table


def audit_cut_gradient_by_finite_difference(Q: dict) -> None:
    r"""Does the cut coefficient equal dQ/dy_k?

    y is binary, so a true derivative does not exist — but the cut must still
    under-estimate Q at the neighbouring vertices, and comparing the coefficient
    against the ACTUAL change Q(y + e_k) - Q(y) shows whether the sign and
    magnitude are right. A sign error here would show as a coefficient pointing
    the wrong way.
    """
    print("\n1. Benders cut coefficient vs the measured change in Q")

    y_hat = [0, 0, 0]
    _model, m, gens, _sub = subproblem(y_hat, "soc")
    pi = [
        (m.dual[m.q_upper[k]] + m.dual[m.q_lower[k]]) * Q_LIMIT
        for k in range(len(gens))
    ]
    q_hat = pyo.value(m.obj)

    rows = []
    signs_consistent = True
    for k in range(N_CANDIDATES):
        neighbour = list(y_hat)
        neighbour[k] = 1
        actual = Q[tuple(neighbour)] - q_hat
        # Installing control can only help (or do nothing), so Q must not rise.
        if actual > tol.EQUALITY_FEASIBILITY:
            signs_consistent = False
        # The cut coefficient must not over-predict the improvement, or the cut
        # would sit ABOVE Q at that vertex and be invalid.
        if pi[k] < actual - 1e-9:
            pass  # under-estimating is fine and expected
        rows.append(f"k={k}: pi={pi[k]:+.6e}, actual dQ={actual:+.6e}")

    record(
        "cut coefficients and measured dQ have consistent signs",
        "PASS" if signs_consistent else "FAIL",
        "; ".join(rows),
    )


def audit_cut_validity_everywhere(Q: dict) -> None:
    """A cut must lie below Q at EVERY y, not just where it was generated.

    This is the check that catches a sign error or a wrong constant term. An
    invalid cut can remove the true optimum from the master's feasible set, and
    the algorithm will then converge confidently to the wrong answer.
    """
    print("\n2. Cut validity at all 8 patterns, from several generating points")

    worst_overall = -np.inf
    detail = []
    for y_hat in itertools.product([0, 1], repeat=N_CANDIDATES):
        _model, m, gens, sub = subproblem(list(y_hat), "soc")
        if not sub.ok:
            continue
        q_hat = pyo.value(m.obj)
        pi = [
            (m.dual[m.q_upper[k]] + m.dual[m.q_lower[k]]) * Q_LIMIT
            for k in range(len(gens))
        ]
        worst_here = -np.inf
        for pattern in itertools.product([0, 1], repeat=N_CANDIDATES):
            cut = q_hat + sum(pi[k] * (pattern[k] - y_hat[k]) for k in range(N_CANDIDATES))
            violation = cut - Q[pattern]        # > 0 means the cut is INVALID
            worst_here = max(worst_here, violation)
        worst_overall = max(worst_overall, worst_here)
        detail.append(f"from {''.join(map(str, y_hat))}: worst={worst_here:+.2e}")

    record(
        "every cut under-estimates Q at every pattern",
        "PASS" if worst_overall <= tol.EQUALITY_FEASIBILITY else "FAIL",
        f"worst violation over 8x8 = {worst_overall:+.3e} "
        f"(must be <= {tol.EQUALITY_FEASIBILITY:.0e}); " + "; ".join(detail[:3]) + " ...",
    )


def audit_cut_tightness(Q: dict) -> None:
    """A cut must be TIGHT at the point it was generated from."""
    print("\n3. Cut tightness at its generating point")

    worst = 0.0
    for y_hat in itertools.product([0, 1], repeat=N_CANDIDATES):
        _model, m, _gens, sub = subproblem(list(y_hat), "soc")
        if not sub.ok:
            continue
        q_hat = pyo.value(m.obj)
        worst = max(worst, abs(q_hat - Q[y_hat]))
    record(
        "cut value equals Q at its generating point",
        "PASS" if worst <= tol.EQUALITY_FEASIBILITY else "FAIL",
        f"max |cut - Q| at the generating vertex = {worst:.2e}",
    )


def benders(definition="soc", max_iterations=25):
    """An independent re-implementation, not an import from the notebook."""
    master = pyo.ConcreteModel()
    master.K = pyo.Set(initialize=range(N_CANDIDATES))
    master.y = pyo.Var(master.K, domain=pyo.Binary)
    master.theta = pyo.Var(bounds=(0.0, None))
    master.obj = pyo.Objective(
        expr=INVEST_COST * sum(master.y[k] for k in master.K) + master.theta
    )
    master.cuts = pyo.ConstraintList()

    upper, best_y, lower = float("inf"), None, -float("inf")
    for _ in range(max_iterations):
        rec_master = solve(master, "MILP")
        y_hat = [round(pyo.value(master.y[k])) for k in master.K]
        lower = rec_master.objective

        _model, m, gens, rec_sub = subproblem(y_hat, definition)
        if not rec_sub.ok:
            raise RuntimeError(f"subproblem failed at {y_hat}: {rec_sub.termination}")
        q = pyo.value(m.obj)
        candidate = INVEST_COST * sum(y_hat) + q
        if candidate < upper:
            upper, best_y = candidate, list(y_hat)

        pi = [
            (m.dual[m.q_upper[k]] + m.dual[m.q_lower[k]]) * Q_LIMIT
            for k in range(len(gens))
        ]
        master.cuts.add(
            master.theta >= q + sum(pi[k] * (master.y[k] - y_hat[k]) for k in master.K)
        )
        if (upper - lower) / max(1.0, abs(upper)) <= 1e-9:
            break
    return lower, upper, best_y, len(master.cuts)


def audit_benders_against_enumeration(Q: dict, definition: str) -> None:
    """The decisive check: does Benders reproduce the monolithic optimum?"""
    print(f"\n4. Benders ({definition}) vs full enumeration")

    totals = {
        pattern: Q[pattern] + INVEST_COST * sum(pattern)
        for pattern in Q if np.isfinite(Q[pattern])
    }
    best_pattern = min(totals, key=totals.get)
    z_truth = totals[best_pattern]

    lower, upper, best_y, n_cuts = benders(definition)
    agrees = tol.objectives_agree(upper, z_truth)
    same_decision = tuple(best_y) == best_pattern
    record(
        f"Benders({definition}) objective == enumerated optimum",
        "PASS" if agrees else "FAIL",
        f"Benders={upper:.10f}, enumeration={z_truth:.10f}, "
        f"|difference|={abs(upper - z_truth):.2e}, cuts={n_cuts}",
    )
    record(
        f"Benders({definition}) finds the same y",
        "PASS" if same_decision else "FAIL",
        f"Benders y={''.join(map(str, best_y))}, "
        f"enumerated y={''.join(map(str, best_pattern))}",
    )
    record(
        f"Benders({definition}) bounds never invert",
        "PASS" if tol.bound_holds(lower, upper) else "FAIL",
        f"LB={lower:.10f} <= UB={upper:.10f}",
    )


# ========================================================== column generation

T = 6
DEMAND = np.array([60.0, 75.0, 110.0, 130.0, 95.0, 70.0])
GENS = [
    {"name": "coal", "pmin": 20.0, "pmax": 70.0, "c1": 25.0,
     "start": 400.0, "noload": 60.0, "minup": 3},
    {"name": "gas", "pmin": 10.0, "pmax": 60.0, "c1": 55.0,
     "start": 120.0, "noload": 20.0, "minup": 2},
    {"name": "peak", "pmin": 5.0, "pmax": 40.0, "c1": 95.0,
     "start": 40.0, "noload": 5.0, "minup": 1},
]
BIG_M = 1e5


def schedule_cost(u, p, spec):
    starts = sum(max(u[t] - (u[t - 1] if t else 0.0), 0.0) for t in range(T))
    return float(spec["c1"] * np.sum(p) + spec["noload"] * np.sum(u) + spec["start"] * starts)


def is_feasible_schedule(u, p, spec) -> tuple[bool, str]:
    """Is this column actually a runnable schedule?

    A negative reduced cost is irrelevant if the schedule it came from violates
    the generator's own constraints.
    """
    u = np.asarray(u)
    p = np.asarray(p)
    for t in range(T):
        if u[t] not in (0.0, 1.0):
            return False, f"u[{t}]={u[t]} is not binary"
        if u[t] == 0 and abs(p[t]) > tol.EQUALITY_FEASIBILITY:
            return False, f"off at t={t} but p={p[t]:.4f}"
        if u[t] == 1 and not (
            spec["pmin"] - tol.EQUALITY_FEASIBILITY
            <= p[t]
            <= spec["pmax"] + tol.EQUALITY_FEASIBILITY
        ):
            return False, f"p[{t}]={p[t]:.4f} outside [{spec['pmin']}, {spec['pmax']}]"
    # Minimum up-time: every start must be followed by `minup` on-periods.
    for t in range(T):
        started = u[t] - (u[t - 1] if t else 0.0)
        if started > 0.5:
            length = spec["minup"]
            window = u[t : min(t + length, T)]
            if len(window) == length and window.sum() < length - 1e-9:
                return False, f"min up-time violated at t={t}"
    return True, "ok"


def restricted_master(columns):
    m = pyo.ConcreteModel()
    m.G = pyo.Set(initialize=range(len(GENS)))
    m.T = pyo.Set(initialize=range(T))
    m.K = pyo.Set(initialize=[(g, k) for g in columns for k in range(len(columns[g]))],
                  dimen=2)
    m.lam = pyo.Var(m.K, domain=pyo.NonNegativeReals, bounds=(0, 1))
    m.up = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.dn = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    @m.Constraint(m.T)
    def demand(m, t):
        return (sum(columns[g][k]["p"][t] * m.lam[g, k] for (g, k) in m.K)
                + m.up[t] - m.dn[t] == DEMAND[t])

    @m.Constraint(m.G)
    def convexity(m, g):
        return sum(m.lam[g, k] for k in range(len(columns[g]))) == 1

    m.obj = pyo.Objective(
        expr=sum(columns[g][k]["cost"] * m.lam[g, k] for (g, k) in m.K)
        + BIG_M * sum(m.up[t] + m.dn[t] for t in m.T))
    return m


def pricing(g, prices, sigma):
    spec = GENS[g]
    m = pyo.ConcreteModel()
    m.T = pyo.Set(initialize=range(T), ordered=True)
    m.u = pyo.Var(m.T, domain=pyo.Binary)
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
        expr=sum(spec["c1"] * m.p[t] + spec["noload"] * m.u[t] + spec["start"] * m.v[t]
                 - prices[t] * m.p[t] for t in m.T) - sigma)
    rec = solve(m, "MILP")
    u = np.array([round(pyo.value(m.u[t])) for t in m.T], dtype=float)
    p = np.array([pyo.value(m.p[t]) for t in m.T])
    return {"u": u, "p": p, "cost": schedule_cost(u, p, spec)}, rec.objective


def audit_column_generation() -> None:
    print("\n5. Column generation vs an exhaustive column pool")

    def make(u, spec):
        u = np.asarray(u, dtype=float)
        p = u * spec["pmax"]
        return {"u": u, "p": p, "cost": schedule_cost(u, p, spec)}

    columns = {g: [make(np.ones(T), s), make(np.zeros(T), s)]
               for g, s in enumerate(GENS)}

    generated_feasible = True
    reason = "ok"
    for _ in range(40):
        master = restricted_master(columns)
        solve(master, "LP", duals=True)
        prices = np.array([master.dual[master.demand[t]] for t in range(T)])
        sigmas = {g: master.dual[master.convexity[g]] for g in range(len(GENS))}
        added = 0
        for g in range(len(GENS)):
            column, reduced = pricing(g, prices, sigmas[g])
            if reduced < -1e-6:
                ok, why = is_feasible_schedule(column["u"], column["p"], GENS[g])
                if not ok:
                    generated_feasible = False
                    reason = f"{GENS[g]['name']}: {why}"
                columns[g].append(column)
                added += 1
        if added == 0:
            break

    z_cg = solve(restricted_master(columns), "LP").objective
    n_generated = sum(len(v) for v in columns.values())

    record(
        "every generated column is a feasible schedule",
        "PASS" if generated_feasible else "FAIL",
        f"checked {n_generated} columns against pmin/pmax, on/off and min up-time; "
        f"{reason}",
    )

    # GROUND TRUTH: the LP over EVERY feasible commitment pattern, dispatched.
    exhaustive = {g: [] for g in range(len(GENS))}
    for g, spec in enumerate(GENS):
        for pattern in itertools.product([0, 1], repeat=T):
            u = np.array(pattern, dtype=float)
            ok, _ = is_feasible_schedule(u, u * spec["pmax"], spec)
            if ok:
                exhaustive[g].append(make(u, spec))
    z_full = solve(restricted_master(exhaustive), "LP").objective
    n_full = sum(len(v) for v in exhaustive.values())

    record(
        "column-generation LP == LP over the exhaustive pool",
        "PASS" if tol.objectives_agree(z_cg, z_full) else "FAIL",
        f"generated={z_cg:.8f} from {n_generated} columns, "
        f"exhaustive={z_full:.8f} from {n_full} columns, "
        f"|difference|={abs(z_cg - z_full):.2e}",
    )

    # REDUCED-COST SIGN: at the optimum, nothing improves; and a column with a
    # negative reduced cost must actually lower the master objective.
    master = restricted_master(columns)
    solve(master, "LP", duals=True)
    prices = np.array([master.dual[master.demand[t]] for t in range(T)])
    sigmas = {g: master.dual[master.convexity[g]] for g in range(len(GENS))}
    worst = min(pricing(g, prices, sigmas[g])[1] for g in range(len(GENS)))
    record(
        "no improving column remains at termination",
        "PASS" if worst >= -1e-6 else "FAIL",
        f"minimum reduced cost over all generators = {worst:+.3e} (must be >= 0)",
    )

    # Now the converse, which is the real sign test. Take a DELIBERATELY BAD
    # price vector, generate a column with negative reduced cost, and confirm
    # adding it to the master at those prices lowers the objective.
    sparse = {g: [make(np.ones(T), s), make(np.zeros(T), s)]
              for g, s in enumerate(GENS)}
    m0 = restricted_master(sparse)
    r0 = solve(m0, "LP", duals=True)
    p0 = np.array([m0.dual[m0.demand[t]] for t in range(T)])
    s0 = {g: m0.dual[m0.convexity[g]] for g in range(len(GENS))}
    improved = True
    notes = []
    for g in range(len(GENS)):
        column, reduced = pricing(g, p0, s0[g])
        if reduced >= -1e-6:
            continue
        trial = {k: list(v) for k, v in sparse.items()}
        trial[g].append(column)
        r1 = solve(restricted_master(trial), "LP")
        notes.append(f"{GENS[g]['name']}: rc={reduced:+.2f}, "
                     f"objective {r0.objective:.2f} -> {r1.objective:.2f}")
        if r1.objective > r0.objective + 1e-6:
            improved = False
    record(
        "a negative reduced cost really does improve the master",
        "PASS" if improved else "FAIL",
        "; ".join(notes) if notes else "no improving column at the initial basis",
    )


def main() -> int:
    print("=" * 78)
    print("DECOMPOSITION AUDIT")
    print("=" * 78)
    print(f"solvers: {available_solvers()}")

    if not available_solvers().get("ipopt", False):
        print("\nIPOPT unavailable; the Benders checks need it.")
        return 1

    Q_soc = value_function("soc")
    print(f"\nQ(y) enumerated (SOC): "
          f"{ {''.join(map(str, k)): round(v, 8) for k, v in Q_soc.items()} }")

    audit_cut_gradient_by_finite_difference(Q_soc)
    audit_cut_validity_everywhere(Q_soc)
    audit_cut_tightness(Q_soc)
    audit_benders_against_enumeration(Q_soc, "soc")

    Q_linear = value_function("linear")
    audit_benders_against_enumeration(Q_linear, "linear")

    audit_column_generation()

    print("\n" + "=" * 78)
    counts: dict[str, int] = {}
    for row in RESULTS:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print(f"{len(RESULTS)} checks: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    failures = [r for r in RESULTS if r["verdict"] == "FAIL"]
    for row in failures:
        print(f"  FAILED: {row['check']} — {row['evidence']}")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
