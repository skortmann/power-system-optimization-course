"""Branch-flow model and its second-order cone relaxation.

This module **extends** ``opf-potpourri`` rather than standing beside it. It
follows the same mixin pattern the library uses for its own models::

    potpourri:   ACOPF(AC, OPF)        DCOPF(DC, OPF)
    here:        SOCBFM(BFM, OPF)

so it inherits the whole pandapower pipeline for free — bus numbering, the
``_pd2ppc`` lookups, generator and load sets, per-unit conversion, apparent
power limits, ``solve()``, ``diagnose()`` and the result mappers. What is added
here is only the physics that potpourri does not ship: the branch-flow
formulation and its conic relaxation.

Why a new model at all
----------------------
``potpourri.research.lin_opf.socbfm`` implements the same relaxation, but it is
absent from both the PyPI wheel and the public ``v0.7.0`` tag, so nothing a
student can install contains it. It also links squared voltage to the AC
voltage variable through a Taylor expansion about a base solution, which is
fine for its purpose and fatal for ours: the course teaches
:math:`z_{SOC} \\le z_{AC}` as a *certified* bound, and a bound cannot rest on a
linearisation about a point.

The formulation
---------------
Work in **lifted** variables, which is what makes the relaxation possible:

.. math::

    u_i = |V_i|^2, \\qquad \\ell_l = |I_l|^2.

For a branch :math:`l = (i, j)` with series impedance :math:`r_l + \\jmath x_l`,
carrying :math:`P_{ij}, Q_{ij}` **into** the branch at bus :math:`i`:

.. math::

    u_j &= u_i - 2\\,(r_l P_{ij} + x_l Q_{ij}) + (r_l^2 + x_l^2)\\,\\ell_l
        &&\\text{(KVL)} \\\\
    P_{ij} + P_{ji} &= r_l \\ell_l &&\\text{(active loss)} \\\\
    Q_{ij} + Q_{ji} &= x_l \\ell_l &&\\text{(reactive loss)} \\\\
    P_{ij}^2 + Q_{ij}^2 &= u_i \\ell_l &&\\text{(current definition)}

The last equation is a **quadratic equality**, and that single line is the
entire source of nonconvexity. Replacing it by

.. math::

    P_{ij}^2 + Q_{ij}^2 \\le u_i \\ell_l

enlarges the feasible set into a convex (rotated second-order) cone. Because
the set only grew, for a minimization the relaxed optimum can only be lower:

.. math::

    z_{SOC}^\\star \\le z_{BFM}^\\star.

That inequality is the whole point, and it is why a relaxation gives a bound
while an approximation such as DC-OPF does not.

Exactness
---------
The relaxation is *exact* when the inequality binds at the optimum. Sufficient
conditions are known for **radial** networks with a monotone objective and
non-binding upper voltage limits (Farivar & Low 2013; Gan, Li, Topcu & Low
2015). They are sufficient, not necessary, and they are easy to violate — which
is why :meth:`SOCBFM.soc_residuals` exists and why the course checks the
residual instead of trusting the objective gap.

References
----------
R. A. Jabr, "Radial distribution load flow using conic programming",
IEEE Trans. Power Syst., 21(3), 2006.

M. Farivar and S. H. Low, "Branch flow model: relaxations and
convexification", IEEE Trans. Power Syst., 28(3), 2013.

L. Gan, N. Li, U. Topcu and S. H. Low, "Exact convex relaxation of optimal
power flow in radial networks", IEEE Trans. Autom. Control, 60(1), 2015.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pyomo.environ as pyo
from pandapower.pypower.idx_brch import BR_R, BR_X
from potpourri.models.basemodel import Basemodel
from potpourri.models.OPF import OPF

__all__ = ["BFM", "SOCBFM", "CurrentDefinition"]

#: How to write the current-definition equation.
#:
#: ``"exact"``   keeps :math:`P^2 + Q^2 = u\\ell`; the model is a nonconvex QCQP.
#: ``"soc"``     relaxes it to :math:`\\le`; the model is a convex SOCP.
CurrentDefinition = Literal["exact", "soc"]


class BFM(Basemodel):
    """Branch-flow (DistFlow) physics, as a potpourri physics mixin.

    Plays the role that :class:`potpourri.models.AC.AC` and
    :class:`potpourri.models.DC.DC` play for the library's own models: it adds
    the network physics to whatever :class:`~potpourri.models.basemodel.Basemodel`
    has already built, and nothing else.

    Unlike the AC model, there is no voltage-angle variable and no ``v``. The
    state is :math:`(u, \\ell)` — squared voltage magnitude and squared branch
    current — because the relaxation is only convex in those coordinates.

    Parameters
    ----------
    net:
        A pandapower network. **Radial**: the branch-flow model assumes a tree,
        and the exactness results quoted above do not apply otherwise.
    current_definition:
        ``"soc"`` for the convex relaxation, ``"exact"`` for the nonconvex
        original. Everything else about the two models is identical, which is
        what makes them comparable.
    """

    def __init__(
        self,
        net,
        current_definition: CurrentDefinition = "soc",
        *,
        allow_meshed: bool = False,
    ) -> None:
        if current_definition not in ("exact", "soc"):
            raise ValueError(
                f"current_definition must be 'exact' or 'soc', got {current_definition!r}"
            )
        self.current_definition: CurrentDefinition = current_definition
        self._allow_meshed = allow_meshed
        super().__init__(net)
        # potpourri's physics mixins (AC, DC) each call create_model() at the
        # end of their own __init__ rather than letting Basemodel do it. Follow
        # that convention, or self.model never exists.
        self.create_model()

    # -- data ------------------------------------------------------------

    def _branch_impedances(self) -> tuple[dict[int, float], dict[int, float]]:
        """Per-unit series impedance of every in-service line, by line index.

        Read out of the ppc branch table through potpourri's own
        ``_line_rows_ppc`` mapping, so the impedances are exactly the ones its
        AC model uses. Deriving them from ``net.line`` instead would risk a
        different answer whenever pandapower's bus lookup is not the identity,
        and an apparent relaxation gap caused by a data mismatch is the single
        most misleading failure this course can produce.
        """
        branch = self.net._ppc["branch"]
        rows = self._line_rows_ppc
        r: dict[int, float] = {}
        x: dict[int, float] = {}
        for position, line_index in enumerate(self.line_data.index):
            if not bool(self.line_data["in_service"].iloc[position]):
                continue
            row = rows[position]
            r[int(line_index)] = float(np.real(branch[row, BR_R]))
            x[int(line_index)] = float(np.real(branch[row, BR_X]))
        return r, x

    def _reactive_data(self) -> tuple[dict, dict]:
        """Reactive demand and static-generation set points, in per unit.

        Mirrors the per-unit convention `Basemodel` uses for the active
        quantities: divide by ``net.sn_mva`` and apply the element scaling.
        """
        qd = (self.net.load.q_mvar * self.net.load.scaling / self.baseMVA).to_dict()
        qsg = (self.net.sgen.q_mvar * self.net.sgen.scaling / self.baseMVA).to_dict()
        return qd, qsg

    def _check_radial(self) -> None:
        n_branch = len(self.model.L) + len(self.model.TRANSF)
        if n_branch != len(self.model.B) - 1 and not self._allow_meshed:
            raise ValueError(
                f"the branch-flow model assumes a radial network, but this one has "
                f"{len(self.model.B)} buses and {n_branch} branches "
                f"(a tree would have {len(self.model.B) - 1}). "
                f"Pass allow_meshed=True to build it anyway — the formulation still "
                f"has a meaning, but the exactness results do not apply."
            )

    # -- model -----------------------------------------------------------

    def create_model(self) -> None:
        """Build the model, then add branch-flow physics on top of it."""
        super().create_model()
        self.model.name = f"BFM-{self.current_definition}"

        if len(self.model.TRANSF) > 0:
            raise NotImplementedError(
                "this branch-flow model covers lines only; the network has "
                f"{len(self.model.TRANSF)} transformer(s). Tap-changing transformers "
                "need extra ratio variables in the KVL equation, which is deliberately "
                "out of scope here — use potpourri's ACOPF for those networks."
            )
        self._check_radial()

        r_data, x_data = self._branch_impedances()
        self.model.r = pyo.Param(self.model.L, initialize=r_data, within=pyo.Reals)
        self.model.x = pyo.Param(self.model.L, initialize=x_data, within=pyo.Reals)

        qd_data, qsg_data = self._reactive_data()
        self.model.QD = pyo.Param(
            self.model.D,
            initialize={d: qd_data.get(d, 0.0) for d in self.model.D},
            within=pyo.Reals,
        )
        self.model.QsG = pyo.Param(
            self.model.sG,
            initialize={g: qsg_data.get(g, 0.0) for g in self.model.sG},
            within=pyo.Reals,
        )

        # --- lifted state ---------------------------------------------
        # u and ell ARE the state. There is no |V| and no angle: the
        # relaxation is convex only in these coordinates.
        # Generous physical bounds so the solver never wanders into |V| = 0,
        # where the cone degenerates. The OPF layer tightens these to the real
        # operating band for the buses that have one.
        self.model.u = pyo.Var(
            self.model.B, domain=pyo.NonNegativeReals, bounds=(0.25, 2.25), initialize=1.0
        )
        self.model.ell = pyo.Var(self.model.L, domain=pyo.NonNegativeReals, initialize=0.0)

        # --- reactive counterparts of Basemodel's active variables -----
        self.model.qG = pyo.Var(self.model.G, domain=pyo.Reals, initialize=0.0)
        self.model.qsG = pyo.Var(self.model.sG, domain=pyo.Reals, initialize=0.0)
        self.model.qD = pyo.Var(self.model.D, domain=pyo.Reals, initialize=0.0)
        self.model.qLfrom = pyo.Var(self.model.L, domain=pyo.Reals, initialize=0.0)
        self.model.qLto = pyo.Var(self.model.L, domain=pyo.Reals, initialize=0.0)

        # Loads are not controllable here unless an OPF layer frees them.
        for d in self.model.D:
            self.model.qD[d].fix(pyo.value(self.model.QD[d]))
        for g in self.model.sG:
            self.model.qsG[g].fix(pyo.value(self.model.QsG[g]))

        # Reference bus: the AC set point, squared. `v_b0` is an AC-mixin
        # parameter and does not exist here, so read the magnitude from
        # `bus_data`, which is the same ppc column the AC model builds it from.
        for b in self.model.b0:
            self.model.u[b].fix(float(self.bus_data.v_m[b]) ** 2)

        self._add_kcl()
        self._add_kvl_and_losses()
        self._add_current_definition()

    def _add_kcl(self) -> None:
        """Nodal balance, active and reactive.

        Same shape as potpourri's DC ``KCL_const``, with the reactive twin
        beside it. ``pLfrom``/``pLto`` are both powers *entering* the branch, so
        each appears on the consumption side of its own bus.
        """

        @self.model.Constraint(self.model.B)
        def KCL_p(model, b):
            balance = sum(
                model.psG[g] for g in model.sG if (g, b) in model.sGbs
            ) + sum(model.pG[g] for g in model.G if (g, b) in model.Gbs) == sum(
                model.pD[d] for d in model.D if (b, d) in model.Dbs
            ) + sum(model.pLfrom[l] for l in model.L if model.A[l, 1] == b) + sum(
                model.pLto[l] for l in model.L if model.A[l, 2] == b
            ) + sum(model.GB[s] for s in model.SHUNT if (b, s) in model.SHUNTbs)
            if isinstance(balance, bool | np.bool_):
                return pyo.Constraint.Skip
            return balance

        @self.model.Constraint(self.model.B)
        def KCL_q(model, b):
            balance = sum(
                model.qsG[g] for g in model.sG if (g, b) in model.sGbs
            ) + sum(model.qG[g] for g in model.G if (g, b) in model.Gbs) == sum(
                model.qD[d] for d in model.D if (b, d) in model.Dbs
            ) + sum(model.qLfrom[l] for l in model.L if model.A[l, 1] == b) + sum(
                model.qLto[l] for l in model.L if model.A[l, 2] == b
            )
            if isinstance(balance, bool | np.bool_):
                return pyo.Constraint.Skip
            return balance

    def _add_kvl_and_losses(self) -> None:
        """Voltage drop along a branch, and the losses it dissipates."""

        @self.model.Constraint(self.model.L)
        def KVL(model, l):
            i, j = model.A[l, 1], model.A[l, 2]
            return model.u[j] == (
                model.u[i]
                - 2.0 * (model.r[l] * model.pLfrom[l] + model.x[l] * model.qLfrom[l])
                + (model.r[l] ** 2 + model.x[l] ** 2) * model.ell[l]
            )

        @self.model.Constraint(self.model.L)
        def loss_p(model, l):
            return model.pLfrom[l] + model.pLto[l] == model.r[l] * model.ell[l]

        @self.model.Constraint(self.model.L)
        def loss_q(model, l):
            return model.qLfrom[l] + model.qLto[l] == model.x[l] * model.ell[l]

    def _add_current_definition(self) -> None:
        """The one equation that decides the problem class.

        ``==`` gives a nonconvex QCQP whose feasible set is the true branch-flow
        manifold. ``<=`` gives a convex rotated second-order cone containing it.
        Nothing else about the two models differs — which is exactly what makes
        the objective values comparable, and the difference attributable to the
        relaxation rather than to a modelling discrepancy.
        """
        if self.current_definition == "exact":

            @self.model.Constraint(self.model.L)
            def current_definition(model, l):
                i = model.A[l, 1]
                return model.pLfrom[l] ** 2 + model.qLfrom[l] ** 2 == model.u[i] * model.ell[l]

        else:

            @self.model.Constraint(self.model.L)
            def current_definition(model, l):
                i = model.A[l, 1]
                return model.pLfrom[l] ** 2 + model.qLfrom[l] ** 2 <= model.u[i] * model.ell[l]


class SOCBFM(BFM, OPF):
    """Branch-flow OPF, relaxed to a second-order cone program by default.

    Mirrors ``ACOPF(AC, OPF)``: the physics comes from :class:`BFM`, the
    operational limits and cost data come from potpourri's
    :class:`~potpourri.models.OPF.OPF`.

    Examples
    --------
    >>> import pandapower.networks as pn
    >>> from psopt_course.relaxations import SOCBFM
    >>> model = SOCBFM(pn.case33bw())            # doctest: +SKIP
    >>> model.add_OPF(vmin=0.95, vmax=1.05)      # doctest: +SKIP
    """

    def __init__(
        self,
        net,
        current_definition: CurrentDefinition = "soc",
        *,
        allow_meshed: bool = False,
    ) -> None:
        super().__init__(net, current_definition, allow_meshed=allow_meshed)

    @property
    def problem_class(self) -> str:
        """The class a solver actually sees. Name it before choosing a solver."""
        return "SOCP" if self.current_definition == "soc" else "nonconvex QCQP"

    def add_OPF(
        self,
        vmin: float = 0.95,
        vmax: float = 1.05,
        max_loading_percent: float = 100.0,
        **kwargs,
    ) -> None:
        """Add operational limits, expressed in the lifted variables.

        Parameters
        ----------
        vmin, vmax:
            Voltage magnitude limits in per unit. They enter as
            :math:`v_{min}^2 \\le u_i \\le v_{max}^2`, which is **exact** —
            squaring a positive bound is a bijection, not an approximation.
            This is the step the research implementation linearises and this one
            does not.
        max_loading_percent:
            Thermal limit, forwarded to potpourri's ``SLmax`` computation.
        """
        OPF.add_OPF(self, max_loading_percent=max_loading_percent, **kwargs)

        self.model.u_min = pyo.Param(initialize=vmin**2, within=pyo.NonNegativeReals)
        self.model.u_max = pyo.Param(initialize=vmax**2, within=pyo.NonNegativeReals)

        @self.model.Constraint(self.model.Bpd)
        def voltage_limits_lower(model, b):
            if b in model.b0:
                return pyo.Constraint.Skip
            return model.u[b] >= model.u_min

        @self.model.Constraint(self.model.Bpd)
        def voltage_limits_upper(model, b):
            if b in model.b0:
                return pyo.Constraint.Skip
            return model.u[b] <= model.u_max

        # Thermal limit, also in lifted form: S^2 <= Smax^2 is a convex
        # quadratic constraint and needs no cone of its own.
        @self.model.Constraint(self.model.L)
        def thermal_limit_from(model, l):
            return model.pLfrom[l] ** 2 + model.qLfrom[l] ** 2 <= model.SLmax[l] ** 2

        @self.model.Constraint(self.model.L)
        def thermal_limit_to(model, l):
            return model.pLto[l] ** 2 + model.qLto[l] ** 2 <= model.SLmax[l] ** 2

    # -- diagnostics -----------------------------------------------------

    def soc_residuals(self) -> dict[int, float]:
        """Per-branch slack in the cone, after a solve.

        .. math::

            r_l = u_i \\ell_l - P_{ij}^2 - Q_{ij}^2 \\;\\ge\\; 0

        A residual at zero means the relaxed constraint binds and the point
        satisfies the *original* equality; the relaxation was exact on that
        branch. A positive residual means it did not, and the relaxed point is
        not a physical operating state however good its objective looks.

        Checking this is not optional. A small objective gap and an exact
        relaxation are different claims, and only this residual speaks to the
        second one.
        """
        residuals: dict[int, float] = {}
        for l in self.model.L:
            i = self.model.A[l, 1]
            residuals[int(l)] = float(
                pyo.value(self.model.u[i]) * pyo.value(self.model.ell[l])
                - pyo.value(self.model.pLfrom[l]) ** 2
                - pyo.value(self.model.qLfrom[l]) ** 2
            )
        return residuals

    def is_exact(self, tolerance: float = 1e-6) -> bool:
        """Did the relaxation come out tight on every branch?"""
        return max(self.soc_residuals().values(), default=0.0) <= tolerance

    def voltages_pu(self) -> dict[int, float]:
        """Recover :math:`|V_i| = \\sqrt{u_i}` for reporting and validation."""
        return {int(b): float(np.sqrt(max(pyo.value(self.model.u[b]), 0.0))) for b in self.model.B}
