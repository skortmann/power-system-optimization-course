"""The branch-flow model and its second-order cone relaxation.

The load-bearing test is :func:`test_exact_bfm_reproduces_the_power_flow`. With
every injection fixed the exact BFM has a unique solution and it must be the
power flow; if that fails, every bound in Tutorials 05, 08 and 10 is built on
sand. It currently agrees to about 3e-09 per unit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pyomo.environ as pyo
import pytest

from psopt_course.relaxations import BFM, SOCBFM

warnings.filterwarnings("ignore")

VMIN, VMAX = 0.90, 1.05


@pytest.fixture(scope="module")
def solved_net():
    net = pn.case33bw()
    pp.runpp(net, numba=False)
    return net


def _feeder_with_pv():
    """case33bw plus three controllable PV inverters.

    The bare feeder has no control at all, so an OPF over it is either
    infeasible or trivial. The inverters give the reactive freedom that makes
    the comparison meaningful.
    """
    net = pn.case33bw()
    for bus, p in [(17, 0.5), (24, 0.5), (32, 0.5)]:
        pp.create_sgen(
            net, bus=bus, p_mw=p, q_mvar=0.0, controllable=True,
            max_p_mw=p, min_p_mw=0.0, max_q_mvar=0.3, min_q_mvar=-0.3,
        )
    return net


def _loss_minimising(definition):
    model = SOCBFM(_feeder_with_pv(), current_definition=definition)
    model.add_OPF(vmin=VMIN, vmax=VMAX)
    mod = model.model
    for g in mod.sG:
        mod.qsG[g].unfix()
        mod.qsG[g].setlb(-0.03)
        mod.qsG[g].setub(0.03)
    mod.obj = pyo.Objective(expr=sum(mod.r[l] * mod.ell[l] for l in mod.L))
    result = pyo.SolverFactory("ipopt").solve(mod)
    return model, str(result.solver.termination_condition)


# ---------------------------------------------------------------- physics


def test_exact_bfm_reproduces_the_power_flow(solved_net):
    """Fixed injections, no limits: the BFM must BE the power flow."""
    model = SOCBFM(solved_net, current_definition="exact")
    mod = model.model
    for g in mod.sG:
        mod.psG[g].fix(pyo.value(mod.PsG[g]))
        mod.qsG[g].fix(pyo.value(mod.QsG[g]))
    for d in mod.D:
        mod.pD[d].fix(pyo.value(mod.PD[d]))
        mod.qD[d].fix(pyo.value(mod.QD[d]))
    mod.obj = pyo.Objective(expr=0.0)

    result = pyo.SolverFactory("ipopt").solve(mod)
    assert str(result.solver.termination_condition) == "optimal"

    lookup = model.bus_lookup
    vm_pp = solved_net.res_bus.vm_pu.to_numpy()
    errors = [
        abs(float(np.sqrt(pyo.value(mod.u[int(lookup[b])]))) - vm_pp[b])
        for b in solved_net.bus.index
        if int(lookup[b]) in mod.B
    ]
    assert max(errors) < 1e-6, f"max voltage error {max(errors):.3e} pu"

    losses = sum(pyo.value(mod.r[l]) * pyo.value(mod.ell[l]) for l in mod.L) * solved_net.sn_mva
    assert losses == pytest.approx(float(solved_net.res_line.pl_mw.sum()), abs=1e-6)


def test_impedances_come_from_the_same_ppc_rows_potpourri_uses():
    """A data mismatch here would masquerade as a relaxation gap."""
    from pandapower.pypower.idx_brch import BR_R, BR_X

    model = SOCBFM(pn.case33bw())
    branch = model.net._ppc["branch"]
    for position, line_index in enumerate(model.line_data.index):
        if not bool(model.line_data["in_service"].iloc[position]):
            continue
        row = model._line_rows_ppc[position]
        assert pyo.value(model.model.r[line_index]) == pytest.approx(
            float(np.real(branch[row, BR_R]))
        )
        assert pyo.value(model.model.x[line_index]) == pytest.approx(
            float(np.real(branch[row, BR_X]))
        )


# ------------------------------------------------------------- relaxation


def test_soc_relaxation_is_a_lower_bound():
    """z_SOC <= z_BFM, because the feasible set only grew."""
    soc, soc_term = _loss_minimising("soc")
    exact, exact_term = _loss_minimising("exact")
    assert soc_term == "optimal" and exact_term == "optimal"

    z_soc = pyo.value(soc.model.obj)
    z_exact = pyo.value(exact.model.obj)
    assert z_soc <= z_exact + 1e-6, f"relaxation is not a lower bound: {z_soc} > {z_exact}"


def test_relaxation_is_exact_under_the_farivar_low_conditions():
    """Radial, loss objective (monotone in ell), upper voltage bound slack."""
    soc, term = _loss_minimising("soc")
    assert term == "optimal"
    residuals = soc.soc_residuals()
    assert max(residuals.values()) < 1e-5, (
        f"expected a tight relaxation here, max residual {max(residuals.values()):.3e}"
    )
    assert soc.is_exact(tolerance=1e-5)


def test_a_constant_objective_breaks_exactness():
    """Exactness needs the objective to PULL the cone tight.

    With a constant objective nothing is monotone in ``ell``, the Farivar-Low
    precondition fails, and the relaxed solution drifts into the interior of
    the enlarged set. This is the clearest demonstration in the course that a
    relaxation is not an approximation: the model is still solved exactly, and
    the answer is still physically meaningless.
    """
    model = SOCBFM(pn.case33bw(), current_definition="soc")
    mod = model.model
    for g in mod.sG:
        mod.psG[g].fix(pyo.value(mod.PsG[g]))
        mod.qsG[g].fix(pyo.value(mod.QsG[g]))
    for d in mod.D:
        mod.pD[d].fix(pyo.value(mod.PD[d]))
        mod.qD[d].fix(pyo.value(mod.QD[d]))
    mod.obj = pyo.Objective(expr=0.0)
    pyo.SolverFactory("ipopt").solve(mod)

    assert not model.is_exact(tolerance=1e-5), (
        "without a monotone objective the relaxation should NOT be tight"
    )


def test_soc_and_exact_differ_in_exactly_one_constraint():
    """The two models must be identical apart from the relaxed equality.

    If anything else differed, a measured 'relaxation gap' would be a modelling
    discrepancy instead.
    """
    soc = SOCBFM(pn.case33bw(), current_definition="soc")
    exact = SOCBFM(pn.case33bw(), current_definition="exact")

    def signature(model):
        return sorted(
            c.name for c in model.model.component_objects(pyo.Constraint, active=True)
        )

    assert signature(soc) == signature(exact)
    assert soc.problem_class == "SOCP"
    assert exact.problem_class == "nonconvex QCQP"


# ------------------------------------------------------------ guard rails


def test_meshed_network_is_refused_by_default():
    """The exactness theory is radial; say so rather than returning a number."""
    net = pn.case9()  # meshed transmission case
    with pytest.raises((ValueError, NotImplementedError)):
        BFM(net)


def test_transformers_are_refused_with_an_explanation():
    net = pn.case4gs()
    with pytest.raises((NotImplementedError, ValueError)):
        BFM(net)


def test_voltage_limits_are_squared_not_linearised():
    """u bounds are v^2 exactly: squaring a positive bound is a bijection."""
    model = SOCBFM(pn.case33bw())
    model.add_OPF(vmin=0.93, vmax=1.04)
    assert pyo.value(model.model.u_min) == pytest.approx(0.93**2)
    assert pyo.value(model.model.u_max) == pytest.approx(1.04**2)
