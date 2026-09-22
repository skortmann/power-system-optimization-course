"""Tests for the course infrastructure and the formulations it teaches.

The expensive check — that every solution notebook executes — belongs in CI and
lives in ``.github/workflows/ci.yml``. What is here runs in seconds.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import pytest

from psopt_course.config import PROJECT_ROOT, fast_mode, scaled, set_seed
from psopt_course.decomposition import BoundHistory, ConvergenceTest, relative_gap
from psopt_course.metrics import (
    certified_interval,
    integrality_gap,
    optimality_gap,
    relaxation_gap,
)
from psopt_course.networks import (
    daily_profiles,
    three_bus_system,
    two_generator_system,
)
from psopt_course.solvers import available_solvers, solve, solver_for
from psopt_course.uncertainty import (
    GaussianUncertainty,
    boole_joint_bound,
    empirical_violation,
    equal_risk_allocation,
    quantile,
    sample_scenarios,
    wilson_interval,
)

warnings.filterwarnings("ignore")

HAS_IPOPT = available_solvers().get("ipopt", False)
needs_ipopt = pytest.mark.skipif(not HAS_IPOPT, reason="IPOPT is not installed")


# ------------------------------------------------------------------ config


def test_scaled_changes_size_only():
    assert scaled(full=24, fast=6) in (24, 6)
    assert scaled(full=24, fast=6) == (6 if fast_mode() else 24)


def test_seed_is_reproducible():
    set_seed()
    first = np.random.default_rng(set_seed()).normal(size=5)
    second = np.random.default_rng(set_seed()).normal(size=5)
    assert np.allclose(first, second)


# ---------------------------------------------------------------- networks


def test_two_generator_system_is_servable():
    system = two_generator_system(demand=100.0)
    assert system.total_capacity >= system.total_demand
    assert [g.name for g in system.merit_order()] == ["G1_coal", "G2_gas"]


def test_impossible_demand_is_refused_by_the_data_check():
    with pytest.raises(ValueError, match="cannot serve"):
        two_generator_system(demand=1000.0)


def test_branch_susceptance_carries_its_base():
    """The units bug that made a DC-OPF silently infeasible."""
    system = three_bus_system()
    branch = system.branches[0]
    assert branch.susceptance_pu == pytest.approx(1.0 / branch.reactance)
    assert branch.susceptance == pytest.approx(branch.base_mva / branch.reactance)
    # The two differ by exactly the power base. Conflating them is what made a
    # DC-OPF need ten-radian angles and come back silently infeasible.
    assert branch.susceptance == pytest.approx(branch.base_mva * branch.susceptance_pu)
    assert branch.susceptance_pu < branch.susceptance


def test_daily_profiles_are_deterministic_and_bounded():
    a = daily_profiles(n_periods=24, seed=7)
    b = daily_profiles(n_periods=24, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert (a["pv"] >= 0).all() and (a["pv"] <= 1).all()
    assert (a["pv"] == 0).any(), "PV should be exactly zero at night"


# ----------------------------------------------------------------- solvers


def test_problem_class_must_be_named():
    with pytest.raises(ValueError, match="unknown problem class"):
        solver_for("definitely-not-a-class")


def test_lp_and_milp_have_open_source_routes():
    assert solver_for("LP")
    assert solver_for("MILP")


def test_economic_dispatch_matches_the_hand_calculation():
    """The T01 result, pinned: 60/40 MW at 3,700 EUR/h."""
    system = two_generator_system(demand=100.0)
    cheap, expensive = system.merit_order()

    m = pyo.ConcreteModel()
    m.G = pyo.Set(initialize=[g.name for g in system.generators])
    by_name = {g.name: g for g in system.generators}
    m.p = pyo.Var(m.G, bounds=lambda m, g: (by_name[g].p_min, by_name[g].p_max))
    m.balance = pyo.Constraint(expr=sum(m.p[g] for g in m.G) == system.total_demand)
    m.cost = pyo.Objective(expr=sum(by_name[g].c1 * m.p[g] for g in m.G))

    record = solve(m, "LP", duals=True)
    assert record.ok
    assert record.objective == pytest.approx(3700.0)
    assert pyo.value(m.p[cheap.name]) == pytest.approx(60.0)
    assert pyo.value(m.p[expensive.name]) == pytest.approx(40.0)
    # The marginal unit is the one NOT at a limit, so lambda is its cost.
    assert m.dual[m.balance] == pytest.approx(expensive.c1)


def test_infeasible_model_reports_rather_than_crashes():
    """Pyomo's appsi HiGHS raises instead of returning a status; solve() must not."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 1))
    m.impossible = pyo.Constraint(expr=m.x >= 5.0)
    m.obj = pyo.Objective(expr=m.x)
    record = solve(m, "LP")
    assert not record.ok
    assert "infeasible" in record.termination.lower()


# ----------------------------------------------------------------- metrics


def test_the_four_gaps_are_distinct_quantities():
    assert optimality_gap(110.0, 100.0) == pytest.approx(10.0 / 110.0)
    assert relaxation_gap(90.0, 100.0) == pytest.approx(10.0 / 100.0)
    assert integrality_gap(110.0, 100.0) == pytest.approx(10.0 / 110.0)


def test_a_relaxation_above_the_feasible_value_is_refused():
    """That ordering is impossible for a minimization, so say so loudly."""
    with pytest.raises(ValueError, match="cannot exceed"):
        relaxation_gap(120.0, 100.0)


def test_empty_certified_interval_is_refused():
    with pytest.raises(ValueError, match="exceeds upper bound"):
        certified_interval(100.0, 90.0)


def test_certified_interval_contains_the_truth():
    interval = certified_interval(18314.76, 19150.0)
    assert interval.contains(18510.0)
    assert not interval.contains(20000.0)


# ----------------------------------------------------------- decomposition


def test_bound_history_rejects_an_invalid_cut():
    """A master relaxation cannot lose ground when cuts are only added."""
    history = BoundHistory(name="test")
    history.record(lower_bound=10.0, upper_bound=100.0)
    with pytest.raises(ValueError, match="lower bound fell"):
        history.record(lower_bound=5.0, upper_bound=100.0)


def test_bound_history_rejects_a_worsening_incumbent():
    history = BoundHistory(name="test")
    history.record(lower_bound=10.0, upper_bound=100.0)
    with pytest.raises(ValueError, match="upper bound rose"):
        history.record(lower_bound=10.0, upper_bound=150.0)


def test_convergence_test_distinguishes_converged_from_capped():
    history = BoundHistory(name="test")
    history.record(lower_bound=99.999, upper_bound=100.0)
    stop, reason = ConvergenceTest(tolerance=1e-2).should_stop(history)
    assert stop and "converged" in reason

    capped = BoundHistory(name="test")
    capped.record(lower_bound=10.0, upper_bound=100.0)
    stop, reason = ConvergenceTest(tolerance=1e-9, max_iterations=1).should_stop(capped)
    assert stop and "not convergence" in reason


def test_relative_gap_handles_a_near_zero_optimum():
    assert np.isfinite(relative_gap(0.0, 1e-12))
    assert relative_gap(-np.inf, 1.0) == float("inf")


# ------------------------------------------------------------- uncertainty


def test_quantile_matches_the_standard_values():
    assert quantile(0.05) == pytest.approx(1.6449, abs=1e-4)
    assert quantile(0.01) == pytest.approx(2.3263, abs=1e-4)
    with pytest.raises(ValueError):
        quantile(0.0)


def test_covariance_must_be_valid():
    with pytest.raises(ValueError, match="symmetric"):
        GaussianUncertainty(mean=np.zeros(2), covariance=np.array([[1.0, 0.5], [0.2, 1.0]]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        GaussianUncertainty(mean=np.zeros(2), covariance=np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_joint_violation_exceeds_the_individual_ones():
    """The headline measurement of Tutorial 07, pinned."""
    covariance = np.eye(2)
    uncertainty = GaussianUncertainty(mean=np.zeros(2), covariance=covariance)
    epsilon = 0.05
    samples = sample_scenarios(uncertainty, 100_000, seed=7)
    residuals = samples - quantile(epsilon)
    report = empirical_violation(residuals)

    assert report.individual.max() == pytest.approx(epsilon, abs=0.005)
    assert report.joint > report.individual.max() * 1.5
    assert report.joint <= boole_joint_bound([epsilon, epsilon]) + 1e-9


def test_boole_allocation_sums_to_the_budget():
    allocation = equal_risk_allocation(0.05, 6)
    assert allocation.sum() == pytest.approx(0.05)
    assert boole_joint_bound(allocation) == pytest.approx(0.05)


def test_wilson_interval_stays_inside_zero_one():
    low, high = wilson_interval(0.0, 100)
    assert low >= 0.0 and high <= 1.0
    low, high = wilson_interval(0.05, 1000)
    assert low < 0.05 < high


# ------------------------------------------------------------- the project


def test_every_tutorial_source_has_both_notebooks():
    sources = sorted((PROJECT_ROOT / "tutorials" / "_sources").glob("[0-9][0-9]_*.py"))
    assert len(sources) == 10, f"expected 10 tutorials, found {len(sources)}"
    for source in sources:
        for suffix in ("_solution.ipynb", "_exercise.ipynb"):
            target = PROJECT_ROOT / "tutorials" / f"{source.stem}{suffix}"
            assert target.exists(), f"missing {target.name}"


def test_documentation_is_present():
    expected = {
        "course_overview.md", "mathematical_background.md",
        "optimization_problem_classes.md", "solver_guide.md",
        "decomposition_guide.md", "literature.md", "instructor_guide.md",
        "student_guide.md", "glossary.md",
    }
    present = {p.name for p in (PROJECT_ROOT / "docs").glob("*.md")}
    assert expected <= present, f"missing: {sorted(expected - present)}"


@pytest.mark.parametrize("notebook", sorted(
    (PROJECT_ROOT / "tutorials").glob("*.ipynb")
))
def test_no_local_paths_or_errors_in_committed_notebooks(notebook: Path):
    import nbformat

    text = notebook.read_text(encoding="utf-8")
    assert "/home/" not in text, f"{notebook.name} embeds an absolute home path"

    parsed = nbformat.read(notebook, as_version=4)
    for cell in parsed.cells:
        for output in cell.get("outputs", []):
            assert output.output_type != "error", (
                f"{notebook.name}: {output.get('ename')}: {output.get('evalue')}"
            )


def test_exercise_notebooks_carry_no_outputs_and_no_solutions():
    import nbformat

    for notebook in sorted((PROJECT_ROOT / "tutorials").glob("*_exercise.ipynb")):
        parsed = nbformat.read(notebook, as_version=4)
        for index, cell in enumerate(parsed.cells):
            if cell.cell_type != "code":
                continue
            assert not cell.get("outputs"), f"{notebook.name} cell {index} has outputs"
            assert "ANSWER." not in cell.source, (
                f"{notebook.name} cell {index} leaks a worked answer"
            )


def test_exercise_notebooks_actually_scaffold():
    """A student notebook with nothing to fill in is not an exercise."""
    import nbformat

    for notebook in sorted((PROJECT_ROOT / "tutorials").glob("*_exercise.ipynb")):
        parsed = nbformat.read(notebook, as_version=4)
        todos = sum(
            1 for cell in parsed.cells
            if cell.cell_type == "code" and "TODO" in cell.source
        )
        assert todos >= 3, f"{notebook.name} has only {todos} TODO cells"
