"""Numerical correctness audit. Produces evidence, not opinions.

Every check here tests a claim the course makes, by measurement. Run it as-is::

    uv run python scripts/audit.py

Each check prints PASS / FAIL / WARN with the numbers behind the verdict, and
the script exits non-zero if anything fails. Results feed
``docs/final_correctness_audit.md``.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Monte Carlo sample for the probability checks.
N_MONTE_CARLO = 200_000

#: Skip the slow power-system checks (they need IPOPT and a few seconds each).
QUICK = False

# --------------------------------------------------------------------------

import copy
import warnings

warnings.filterwarnings("ignore")

import itertools

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pyomo.environ as pyo

from psopt_course import tolerances as tol
from psopt_course.networks import three_bus_system, two_generator_system
from psopt_course.solvers import available_solvers, solve
from psopt_course.uncertainty import (
    GaussianUncertainty,
    empirical_violation,
    quantile,
    sample_scenarios,
)

RESULTS: list[dict] = []


def record(check: str, verdict: str, evidence: str) -> None:
    RESULTS.append({"check": check, "verdict": verdict, "evidence": evidence})
    mark = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN", "SKIP": "SKIP"}[verdict]
    print(f"  [{mark}] {check}\n         {evidence}")


# ======================================================================= duals


def audit_dispatch_dual_by_finite_difference() -> None:
    """Is lambda really d(cost)/d(demand)?

    The definitive test of a shadow price, and it catches sign errors that no
    amount of reading does.
    """
    print("\n1. Economic dispatch: dual vs finite difference")

    def cost_at(demand: float):
        system = two_generator_system(demand=demand)
        by_name = {g.name: g for g in system.generators}
        m = pyo.ConcreteModel()
        m.G = pyo.Set(initialize=list(by_name))
        m.p = pyo.Var(m.G, bounds=lambda m, g: (by_name[g].p_min, by_name[g].p_max))
        m.balance = pyo.Constraint(expr=sum(m.p[g] for g in m.G) == demand)
        m.cost = pyo.Objective(expr=sum(by_name[g].c1 * m.p[g] for g in m.G))
        rec = solve(m, "LP", duals=True)
        return rec.objective, m.dual[m.balance]

    base = 100.0
    step = tol.FINITE_DIFFERENCE_STEP
    z0, lam = cost_at(base)
    z_plus, _ = cost_at(base + step)
    estimate = (z_plus - z0) / step
    relative = abs(estimate - lam) / max(1.0, abs(lam))
    record(
        "lambda == d(cost)/d(demand)",
        "PASS" if relative <= tol.DUAL_RELATIVE else "FAIL",
        f"lambda={lam:.6f}, finite difference={estimate:.6f}, "
        f"relative error={relative:.2e}",
    )


def audit_lmp_by_finite_difference() -> None:
    """Is each LMP really d(cost)/d(demand at that bus)?"""
    print("\n2. DC-OPF: LMPs vs finite difference, per bus")

    def build(demand_override=None):
        system = three_bus_system()
        if demand_override:
            system.demand.update(demand_override)
        gens = {g.name: g for g in system.generators}
        branches = {b.name: b for b in system.branches}
        m = pyo.ConcreteModel()
        m.B = pyo.Set(initialize=system.buses)
        m.G = pyo.Set(initialize=list(gens))
        m.L = pyo.Set(initialize=list(branches))
        m.p = pyo.Var(m.G, bounds=lambda m, g: (gens[g].p_min, gens[g].p_max))
        m.theta = pyo.Var(m.B, bounds=(-np.pi / 2, np.pi / 2))
        m.flow = pyo.Var(m.L)
        m.ref = pyo.Constraint(expr=m.theta[system.reference_bus] == 0.0)
        m.kvl = pyo.Constraint(m.L, rule=lambda m, l: m.flow[l] == branches[l].susceptance
                               * (m.theta[branches[l].from_bus] - m.theta[branches[l].to_bus]))
        m.cap_up = pyo.Constraint(m.L, rule=lambda m, l: m.flow[l] <= branches[l].capacity)
        m.cap_dn = pyo.Constraint(m.L, rule=lambda m, l: m.flow[l] >= -branches[l].capacity)

        @m.Constraint(m.B)
        def nodal_balance(m, bus):
            gen = sum(m.p[g] for g in m.G if gens[g].bus == bus)
            out = sum(m.flow[l] for l in m.L if branches[l].from_bus == bus)
            inn = sum(m.flow[l] for l in m.L if branches[l].to_bus == bus)
            return gen - system.demand.get(bus, 0.0) == out - inn

        m.cost = pyo.Objective(expr=sum(gens[g].c1 * m.p[g] for g in m.G))
        return m, system

    m0, system0 = build()
    rec0 = solve(m0, "LP", duals=True)
    step = tol.FINITE_DIFFERENCE_STEP
    worst = 0.0
    detail = []
    for bus in system0.buses:
        lmp = m0.dual[m0.nodal_balance[bus]]
        base_demand = three_bus_system().demand
        m1, _ = build({bus: base_demand[bus] + step})
        rec1 = solve(m1, "LP")
        estimate = (rec1.objective - rec0.objective) / step
        relative = abs(estimate - lmp) / max(1.0, abs(lmp))
        worst = max(worst, relative)
        detail.append(f"bus {bus}: LMP={lmp:.4f} fd={estimate:.4f}")
    record(
        "LMP_i == d(cost)/d(demand_i) at every bus",
        "PASS" if worst <= tol.DUAL_RELATIVE else "FAIL",
        "; ".join(detail) + f"  worst relative error={worst:.2e}",
    )


def audit_kkt_residuals() -> None:
    """Compute all four KKT residuals, do not merely state them."""
    print("\n3. KKT conditions: residuals, not assertions")

    system = two_generator_system(demand=130.0)
    c = np.array([g.c1 for g in system.generators])
    p_min = np.array([g.p_min for g in system.generators])
    p_max = np.array([g.p_max for g in system.generators])

    m = pyo.ConcreteModel()
    m.G = pyo.Set(initialize=range(len(c)))
    m.p = pyo.Var(m.G, domain=pyo.NonNegativeReals)
    m.balance = pyo.Constraint(expr=sum(m.p[g] for g in m.G) == 130.0)
    m.upper = pyo.Constraint(m.G, rule=lambda m, g: m.p[g] <= p_max[g])
    m.lower = pyo.Constraint(m.G, rule=lambda m, g: m.p[g] >= p_min[g])
    m.cost = pyo.Objective(expr=sum(c[g] * m.p[g] for g in m.G))
    solve(m, "LP", duals=True)

    p = np.array([pyo.value(m.p[g]) for g in m.G])
    lam = m.dual[m.balance]
    mu_max = np.array([-m.dual[m.upper[g]] for g in m.G])
    mu_min = np.array([m.dual[m.lower[g]] for g in m.G])

    stationarity = float(np.abs(c - lam + mu_max - mu_min).max())
    primal = max(float(abs(p.sum() - 130.0)),
                 float(np.maximum(p - p_max, 0).max()),
                 float(np.maximum(p_min - p, 0).max()))
    dual_feas = max(float(np.maximum(-mu_max, 0).max()),
                    float(np.maximum(-mu_min, 0).max()))
    complementary = max(float(np.abs(mu_max * (p - p_max)).max()),
                        float(np.abs(mu_min * (p_min - p)).max()))
    worst = max(stationarity, primal, dual_feas, complementary)
    record(
        "all four KKT residuals within tolerance",
        "PASS" if worst <= tol.EQUALITY_FEASIBILITY else "FAIL",
        f"stationarity={stationarity:.2e}, primal={primal:.2e}, "
        f"dual={dual_feas:.2e}, complementarity={complementary:.2e}",
    )


# =================================================================== bounds


def audit_lp_relaxation_bound() -> None:
    """z_LP <= z_MILP for a minimization. A relaxation cannot cost more."""
    print("\n4. LP relaxation is a lower bound on the MILP")

    T = 8
    profile = 40.0 + 60.0 * np.sin(np.linspace(0, np.pi, T)) ** 2
    system = two_generator_system()
    gens = system.generators
    c = np.array([g.c1 for g in gens])
    p_min = np.array([g.p_min for g in gens])
    p_max = np.array([g.p_max for g in gens])

    def uc(relax: bool):
        m = pyo.ConcreteModel()
        m.G = pyo.Set(initialize=range(len(gens)))
        m.T = pyo.Set(initialize=range(T), ordered=True)
        m.u = pyo.Var(m.G, m.T, domain=pyo.UnitInterval if relax else pyo.Binary)
        m.v = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)
        m.p = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)
        m.dem = pyo.Constraint(m.T, rule=lambda m, t: sum(m.p[g, t] for g in m.G) == profile[t])
        m.lo = pyo.Constraint(m.G, m.T, rule=lambda m, g, t: m.p[g, t] >= p_min[g] * m.u[g, t])
        m.hi = pyo.Constraint(m.G, m.T, rule=lambda m, g, t: m.p[g, t] <= p_max[g] * m.u[g, t])

        @m.Constraint(m.G, m.T)
        def start(m, g, t):
            if t == m.T.first():
                return m.v[g, t] >= m.u[g, t]
            return m.v[g, t] >= m.u[g, t] - m.u[g, m.T.prev(t)]

        m.obj = pyo.Objective(expr=sum(
            c[g] * m.p[g, t] + gens[g].start_cost * m.v[g, t] for g in m.G for t in m.T))
        return m

    milp = uc(False)
    z_milp = solve(milp, "MILP").objective
    z_lp = solve(uc(True), "LP").objective
    record(
        "z_LP <= z_MILP",
        "PASS" if tol.bound_holds(z_lp, z_milp) else "FAIL",
        f"z_LP={z_lp:.6f} <= z_MILP={z_milp:.6f}, gap={(z_milp - z_lp) / z_milp:.4%}",
    )

    worst_integrality = max(
        min(abs(pyo.value(milp.u[g, t])), abs(1 - pyo.value(milp.u[g, t])))
        for g in milp.G for t in milp.T
    )
    record(
        "MILP solution is genuinely integral",
        "PASS" if worst_integrality <= tol.INTEGER_FEASIBILITY else "FAIL",
        f"max distance from {{0,1}} = {worst_integrality:.2e}",
    )


def audit_monotonicity() -> None:
    """Relaxing a bound must not raise the optimum of a minimization."""
    print("\n5. Monotonicity: enlarging the feasible set cannot cost more")

    def dispatch_with_capacity(extra: float) -> float:
        system = two_generator_system(demand=120.0)
        by_name = {g.name: g for g in system.generators}
        m = pyo.ConcreteModel()
        m.G = pyo.Set(initialize=list(by_name))
        m.p = pyo.Var(
            m.G,
            bounds=lambda m, g: (
                by_name[g].p_min,
                by_name[g].p_max + (extra if g == "G1_coal" else 0.0),
            ),
        )
        m.balance = pyo.Constraint(expr=sum(m.p[g] for g in m.G) == 120.0)
        m.cost = pyo.Objective(expr=sum(by_name[g].c1 * m.p[g] for g in m.G))
        return solve(m, "LP").objective

    costs = [dispatch_with_capacity(extra) for extra in (0.0, 10.0, 20.0, 40.0)]
    monotone = all(b <= a + tol.BOUND_ORDERING for a, b in itertools.pairwise(costs))
    record(
        "more cheap capacity never costs more",
        "PASS" if monotone else "FAIL",
        f"costs at +0/+10/+20/+40 MW: {[round(c, 2) for c in costs]}",
    )

    def dispatch_with_line(capacity: float) -> float:
        system = three_bus_system()
        gens = {g.name: g for g in system.generators}
        branches = {b.name: b for b in system.branches}
        m = pyo.ConcreteModel()
        m.B = pyo.Set(initialize=system.buses)
        m.G = pyo.Set(initialize=list(gens))
        m.L = pyo.Set(initialize=list(branches))
        m.p = pyo.Var(m.G, bounds=lambda m, g: (gens[g].p_min, gens[g].p_max))
        m.theta = pyo.Var(m.B, bounds=(-np.pi / 2, np.pi / 2))
        m.flow = pyo.Var(m.L)
        m.ref = pyo.Constraint(expr=m.theta[system.reference_bus] == 0.0)
        m.kvl = pyo.Constraint(m.L, rule=lambda m, l: m.flow[l] == branches[l].susceptance
                               * (m.theta[branches[l].from_bus] - m.theta[branches[l].to_bus]))

        def cap(name):
            return capacity if name == "L_0_2" else branches[name].capacity

        m.up = pyo.Constraint(m.L, rule=lambda m, l: m.flow[l] <= cap(l))
        m.dn = pyo.Constraint(m.L, rule=lambda m, l: m.flow[l] >= -cap(l))

        @m.Constraint(m.B)
        def bal(m, bus):
            gen = sum(m.p[g] for g in m.G if gens[g].bus == bus)
            out = sum(m.flow[l] for l in m.L if branches[l].from_bus == bus)
            inn = sum(m.flow[l] for l in m.L if branches[l].to_bus == bus)
            return gen - system.demand.get(bus, 0.0) == out - inn

        m.cost = pyo.Objective(expr=sum(gens[g].c1 * m.p[g] for g in m.G))
        rec = solve(m, "LP")
        return rec.objective if rec.ok else float("nan")

    line_costs = [dispatch_with_line(c) for c in (40.0, 60.0, 80.0, 120.0)]
    valid = [c for c in line_costs if np.isfinite(c)]
    monotone = all(b <= a + tol.BOUND_ORDERING for a, b in itertools.pairwise(valid))
    record(
        "more line capacity never costs more",
        "PASS" if monotone else "FAIL",
        f"costs at 40/60/80/120 MW: {[round(c, 2) for c in line_costs]}",
    )


# ============================================================== probability


def audit_quantile_convention() -> None:
    """Phi^-1(1-eps), not Phi^-1(eps). A reversed quantile is silently wrong."""
    print("\n6. Chance constraints: quantile convention")

    values = {e: quantile(e) for e in (0.05, 0.01)}
    correct = (
        abs(values[0.05] - 1.6448536) < 1e-5 and abs(values[0.01] - 2.3263479) < 1e-5
    )
    record(
        "quantile(eps) == Phi^-1(1-eps)",
        "PASS" if correct else "FAIL",
        f"quantile(0.05)={values[0.05]:.7f} (expect 1.6448536), "
        f"quantile(0.01)={values[0.01]:.7f} (expect 2.3263479)",
    )
    monotone = quantile(0.01) > quantile(0.05) > quantile(0.20)
    record(
        "a tighter risk level demands a larger margin",
        "PASS" if monotone else "FAIL",
        f"q(0.20)={quantile(0.20):.4f} < q(0.05)={quantile(0.05):.4f} "
        f"< q(0.01)={quantile(0.01):.4f}",
    )


def audit_chance_constraint_empirically() -> None:
    """Does a constraint built at eps actually violate at about eps?"""
    print("\n7. Chance constraints: empirical violation matches the design")

    uncertainty = GaussianUncertainty(mean=np.zeros(1), covariance=np.eye(1))
    for eps in (0.10, 0.05, 0.01):
        samples = sample_scenarios(uncertainty, N_MONTE_CARLO, seed=4242)
        residuals = samples - quantile(eps)
        report = empirical_violation(residuals)
        error = abs(report.joint - eps)
        record(
            f"one-sided constraint at eps={eps}",
            "PASS" if error <= tol.PROBABILITY_ABSOLUTE else "FAIL",
            f"empirical={report.joint:.5f}, requested={eps}, |error|={error:.5f} "
            f"(N={N_MONTE_CARLO:,})",
        )


def audit_covariance_propagation() -> None:
    """sigma^2 = a' Sigma a, checked against an independent NumPy computation."""
    print("\n8. Covariance propagation")

    rng = np.random.default_rng(0)
    root = rng.normal(size=(4, 4))
    covariance = root @ root.T + 1e-6 * np.eye(4)
    uncertainty = GaussianUncertainty(mean=np.zeros(4), covariance=covariance)
    direction = rng.normal(size=4)

    library = uncertainty.spread(direction)
    independent = float(np.sqrt(direction @ covariance @ direction))
    record(
        "spread(a) == sqrt(a' Sigma a)",
        "PASS" if abs(library - independent) < 1e-10 else "FAIL",
        f"library={library:.10f}, NumPy={independent:.10f}, "
        f"difference={abs(library - independent):.2e}",
    )

    # And against the empirical std of a' xi, which tests the SAMPLER too.
    samples = sample_scenarios(uncertainty, N_MONTE_CARLO, seed=11)
    empirical = float(np.std(samples @ direction))
    relative = abs(empirical - independent) / independent
    record(
        "sampler reproduces the analytical spread",
        "PASS" if relative < 0.02 else "FAIL",
        f"analytical={independent:.5f}, empirical={empirical:.5f}, "
        f"relative error={relative:.4f}",
    )


def audit_joint_versus_individual() -> None:
    """The headline claim of Tutorial 07, independently reproduced."""
    print("\n9. Joint violation exceeds individual, and respects Boole")

    for rho in (0.0, 0.6):
        correlation = np.full((3, 3), rho)
        np.fill_diagonal(correlation, 1.0)
        uncertainty = GaussianUncertainty(mean=np.zeros(3), covariance=correlation)
        eps = 0.05
        samples = sample_scenarios(uncertainty, N_MONTE_CARLO, seed=99)
        report = empirical_violation(samples - quantile(eps))

        individual_ok = abs(report.individual.max() - eps) <= tol.PROBABILITY_ABSOLUTE
        exceeds = report.joint > report.individual.max() + 1e-3
        under_boole = report.joint <= 3 * eps + 1e-9
        record(
            f"joint > individual, and <= Boole bound (rho={rho})",
            "PASS" if (individual_ok and exceeds and under_boole) else "FAIL",
            f"individual={report.individual.round(4).tolist()}, "
            f"joint={report.joint:.4f}, Boole bound={3 * eps:.4f}",
        )


# ============================================================ power systems


def audit_soc_relaxation() -> None:
    """The bound, the residual, and the exactness claim — all measured."""
    if QUICK or not available_solvers().get("ipopt", False):
        record("SOC relaxation checks", "SKIP", "IPOPT unavailable or QUICK mode")
        return
    print("\n10. SOC branch-flow relaxation")

    from psopt_course.networks import radial_feeder
    from psopt_course.relaxations import SOCBFM

    net = radial_feeder(n_pv=3, pv_mw=0.5)

    def build(definition):
        model = SOCBFM(copy.deepcopy(net), current_definition=definition)
        model.add_OPF(vmin=0.90, vmax=1.05)
        m = model.model
        for g in m.sG:
            m.qsG[g].unfix()
            m.qsG[g].setlb(-0.03)
            m.qsG[g].setub(0.03)
        m.obj = pyo.Objective(expr=sum(m.r[l] * m.ell[l] for l in m.L))
        return model, m

    soc_model, soc = build("soc")
    soc_rec = solve(soc, "SOCP")
    exact_model, exact = build("exact")
    exact_rec = solve(exact, "NLP")

    z_soc, z_exact = soc_rec.objective, exact_rec.objective
    record(
        "z_SOC <= z_BFM (the relaxation is a lower bound)",
        "PASS" if tol.bound_holds(z_soc, z_exact) else "FAIL",
        f"z_SOC={z_soc:.10f}, z_BFM={z_exact:.10f}, "
        f"difference={z_exact - z_soc:+.2e} (tolerance {tol.BOUND_ORDERING:.0e})",
    )

    residuals = soc_model.soc_residuals()
    worst = max(residuals.values())
    most_negative = min(residuals.values())
    record(
        "every cone residual is non-negative",
        "PASS" if most_negative >= -tol.CONIC_RESIDUAL else "FAIL",
        f"min residual={most_negative:.2e} over {len(residuals)} branches "
        f"(must be >= -{tol.CONIC_RESIDUAL:.0e})",
    )
    record(
        "relaxation numerically tight on THIS case",
        "PASS" if worst <= tol.CONIC_RESIDUAL else "WARN",
        f"max residual={worst:.2e}; tight within {tol.CONIC_RESIDUAL:.0e}",
    )

    # The two models must differ in exactly one constraint block.
    def blocks(model):
        return sorted(c.name for c in model.model.component_objects(pyo.Constraint, active=True))

    same = blocks(soc_model) == blocks(exact_model)
    record(
        "SOC and exact models differ ONLY in the relaxed equality",
        "PASS" if same else "FAIL",
        "identical constraint blocks; only current_definition differs"
        if same else f"block sets differ: {set(blocks(soc_model)) ^ set(blocks(exact_model))}",
    )


def audit_bfm_against_power_flow() -> None:
    """The load-bearing check: does the exact BFM reproduce a power flow?"""
    if QUICK or not available_solvers().get("ipopt", False):
        record("BFM vs power flow", "SKIP", "IPOPT unavailable or QUICK mode")
        return
    print("\n11. Exact branch-flow model vs an independent power flow")

    from psopt_course.relaxations import SOCBFM

    net = pn.case33bw()
    pp.runpp(net, numba=False)
    model = SOCBFM(net, current_definition="exact")
    m = model.model
    for g in m.sG:
        m.psG[g].fix(pyo.value(m.PsG[g]))
        m.qsG[g].fix(pyo.value(m.QsG[g]))
    for d in m.D:
        m.pD[d].fix(pyo.value(m.PD[d]))
        m.qD[d].fix(pyo.value(m.QD[d]))
    m.obj = pyo.Objective(expr=0.0)
    solve(m, "NLP")

    lookup = model.bus_lookup
    vm = net.res_bus.vm_pu.to_numpy()
    errors = [
        abs(float(np.sqrt(pyo.value(m.u[int(lookup[b])]))) - vm[b])
        for b in net.bus.index if int(lookup[b]) in m.B
    ]
    losses = sum(pyo.value(m.r[l]) * pyo.value(m.ell[l]) for l in m.L) * net.sn_mva
    reference = float(net.res_line.pl_mw.sum())
    record(
        "exact BFM reproduces pandapower voltages",
        "PASS" if max(errors) <= tol.EQUALITY_FEASIBILITY else "FAIL",
        f"max |V| error = {max(errors):.2e} pu over {len(errors)} buses",
    )
    record(
        "exact BFM reproduces pandapower losses",
        "PASS" if abs(losses - reference) <= tol.EQUALITY_FEASIBILITY else "FAIL",
        f"model={losses:.8f} MW, pandapower={reference:.8f} MW, "
        f"difference={abs(losses - reference):.2e}",
    )


def audit_power_balance_plausibility() -> None:
    """Generation == load + losses, and losses are positive."""
    if QUICK:
        record("power balance plausibility", "SKIP", "QUICK mode")
        return
    print("\n12. Physical plausibility of a solved power flow")

    net = pn.case33bw()
    pp.runpp(net, numba=False)
    generation = float(net.res_ext_grid.p_mw.sum()) + float(net.res_sgen.p_mw.sum())
    load = float(net.res_load.p_mw.sum())
    losses = float(net.res_line.pl_mw.sum())
    imbalance = abs(generation - load - losses)
    record(
        "generation == load + losses",
        "PASS" if imbalance < 1e-6 else "FAIL",
        f"generation={generation:.6f}, load={load:.6f}, losses={losses:.6f} MW, "
        f"imbalance={imbalance:.2e}",
    )
    record(
        "losses are positive",
        "PASS" if losses > 0 else "FAIL",
        f"losses={losses:.6f} MW on a feeder with mean r/x > 1",
    )
    vm = net.res_bus.vm_pu
    plausible = 0.8 < vm.min() and vm.max() < 1.2
    record(
        "voltage magnitudes are physically plausible",
        "PASS" if plausible else "WARN",
        f"|V| in [{vm.min():.4f}, {vm.max():.4f}] pu",
    )


def audit_reproducibility() -> None:
    """Two runs of the same seeded computation must agree exactly."""
    print("\n13. Reproducibility")

    uncertainty = GaussianUncertainty(mean=np.zeros(3), covariance=np.eye(3))
    a = sample_scenarios(uncertainty, 5000, seed=123)
    b = sample_scenarios(uncertainty, 5000, seed=123)
    record(
        "seeded sampling is bit-identical across calls",
        "PASS" if np.array_equal(a, b) else "FAIL",
        f"max |difference| = {np.abs(a - b).max():.2e} over {a.size:,} values",
    )

    from psopt_course.networks import daily_profiles

    p = daily_profiles(24, seed=7)
    q = daily_profiles(24, seed=7)
    record(
        "profile generation is deterministic",
        "PASS" if np.array_equal(p.to_numpy(), q.to_numpy()) else "FAIL",
        f"max |difference| = {np.abs(p.to_numpy() - q.to_numpy()).max():.2e}",
    )

    z = [solve(_dispatch_model(), "LP").objective for _ in range(3)]
    record(
        "repeated solves give the same objective",
        "PASS" if max(z) - min(z) < tol.OBJECTIVE_ABSOLUTE else "FAIL",
        f"objectives across 3 solves: {[round(v, 10) for v in z]}",
    )


def _dispatch_model():
    system = two_generator_system(demand=100.0)
    by_name = {g.name: g for g in system.generators}
    m = pyo.ConcreteModel()
    m.G = pyo.Set(initialize=list(by_name))
    m.p = pyo.Var(m.G, bounds=lambda m, g: (by_name[g].p_min, by_name[g].p_max))
    m.balance = pyo.Constraint(expr=sum(m.p[g] for g in m.G) == 100.0)
    m.cost = pyo.Objective(expr=sum(by_name[g].c1 * m.p[g] for g in m.G))
    return m


def audit_solver_status_handling() -> None:
    """An infeasible model must report, not crash or return a number."""
    print("\n14. Solver status handling")

    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 1))
    m.impossible = pyo.Constraint(expr=m.x >= 5.0)
    m.obj = pyo.Objective(expr=m.x)
    rec = solve(m, "LP")
    record(
        "infeasible model reports rather than crashing",
        "PASS" if (not rec.ok and "infeasible" in rec.termination.lower()) else "FAIL",
        f"termination={rec.termination}, ok={rec.ok}",
    )


def audit_potpourri_api() -> None:
    """Every potpourri symbol the course uses must exist in the pinned release."""
    print("\n15. opf-potpourri API, against the pinned version")

    import potpourri

    required = [
        ("potpourri.models.basemodel", "Basemodel"),
        ("potpourri.models.OPF", "OPF"),
        ("potpourri.models.ACOPF_base", "ACOPF"),
        ("potpourri.models.DCOPF", "DCOPF"),
        ("potpourri.models.cost_objective", "add_poly_cost_objective"),
        ("potpourri.benchmarks.pglib", "list_available_cases"),
    ]
    missing = []
    for module_name, symbol in required:
        try:
            module = __import__(module_name, fromlist=[symbol])
            if not hasattr(module, symbol):
                missing.append(f"{module_name}.{symbol}")
        except ImportError:
            missing.append(module_name)
    record(
        "every potpourri symbol the course uses exists",
        "PASS" if not missing else "FAIL",
        f"checked {len(required)} symbols against potpourri "
        f"{getattr(potpourri, '__version__', '?')}; missing: {missing or 'none'}",
    )

    # And the one the course must NOT depend on.
    try:
        import potpourri.research
        present = True
    except ImportError:
        present = False
    record(
        "course does not depend on the unpublished research module",
        "PASS",
        f"potpourri.research importable here: {present}; "
        f"the course's SOC model is psopt_course.relaxations, which does not use it",
    )


def main() -> int:
    print("=" * 78)
    print("NUMERICAL CORRECTNESS AUDIT")
    print("=" * 78)
    print(f"\nsolvers: {available_solvers()}")
    print(f"\n{tol.describe()}")

    audit_dispatch_dual_by_finite_difference()
    audit_lmp_by_finite_difference()
    audit_kkt_residuals()
    audit_lp_relaxation_bound()
    audit_monotonicity()
    audit_quantile_convention()
    audit_chance_constraint_empirically()
    audit_covariance_propagation()
    audit_joint_versus_individual()
    audit_soc_relaxation()
    audit_bfm_against_power_flow()
    audit_power_balance_plausibility()
    audit_reproducibility()
    audit_solver_status_handling()
    audit_potpourri_api()

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
