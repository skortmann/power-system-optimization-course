"""The systems the course optimizes, from two generators to a real feeder.

One example evolves through the whole course so the mathematics changes while
the physical problem stays recognisable::

    two_generator_system()   T01  economic dispatch          LP / QP
        + on/off             T02  unit commitment            MILP
    three_bus_system()       T03  DC-OPF                     LP
        + AC physics         T04  AC-OPF                     nonconvex NLP
    radial_feeder()          T05  SOC-BFM relaxation         SOCP
        + time               T06  storage                    LP / MILP
        + uncertainty        T07  chance constraints         LP / SOCP
        + structure          T08/09 decomposition
    simbench_feeder()        T10  capstone
    pglib_case()             T10  benchmark discipline

The small systems are plain dataclasses rather than pandapower networks, on
purpose: Tutorials 01-03 are about writing the equations down, and a dataclass
puts the data where a student can read every number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "Branch",
    "DispatchSystem",
    "Generator",
    "daily_profiles",
    "pglib_case",
    "radial_feeder",
    "simbench_feeder",
    "three_bus_system",
    "two_generator_system",
]


@dataclass(frozen=True)
class Generator:
    """One dispatchable unit.

    Costs are affine-plus-quadratic:
    :math:`C(p) = c_0 + c_1 p + c_2 p^2`. Tutorial 01 uses ``c2 = 0`` to keep
    the first model an LP, then switches it on to show that changing the cost
    curve changes the *problem class* — which is what forces a different solver,
    not any change in the physics.
    """

    name: str
    bus: int
    p_min: float
    p_max: float
    c1: float
    c0: float = 0.0
    c2: float = 0.0
    #: Unit-commitment data. Unused until Tutorial 02.
    start_cost: float = 0.0
    min_up: int = 1
    min_down: int = 1
    ramp: float | None = None

    def cost(self, p: float) -> float:
        return self.c0 + self.c1 * p + self.c2 * p**2


@dataclass(frozen=True)
class Branch:
    """A transmission line in the DC sense: reactance and a thermal rating.

    ``reactance`` is **per unit** on ``base_mva``; ``capacity`` is in **MW**.
    Mixing those two up is the classic units bug, and it does not announce
    itself — it produces a model that is quietly infeasible because the angles
    required to move the power exceed any physical bound.
    """

    name: str
    from_bus: int
    to_bus: int
    reactance: float
    capacity: float
    #: Power base for the per-unit reactance. 100 MVA is the usual convention.
    base_mva: float = 100.0

    @property
    def susceptance_pu(self) -> float:
        r"""$B_{ij} = 1 / x_{ij}$, in per unit."""
        return 1.0 / self.reactance

    @property
    def susceptance(self) -> float:
        r"""$B_{ij}$ in **MW per radian**: $S_{base} / x_{pu}$.

        This is what multiplies an angle difference to give a flow in MW. With
        $x = 0.2$ p.u. on a 100 MVA base it is 500 MW/rad, so moving 50 MW takes
        0.1 rad — a sensible angle. Using the per-unit value directly would ask
        for 10 rad and make the model infeasible.
        """
        return self.base_mva / self.reactance


@dataclass
class DispatchSystem:
    """A tiny power system, written out so every number is visible."""

    name: str
    generators: list[Generator]
    demand: dict[int, float]
    branches: list[Branch] = field(default_factory=list)
    reference_bus: int = 0

    @property
    def buses(self) -> list[int]:
        found = {g.bus for g in self.generators} | set(self.demand)
        for b in self.branches:
            found |= {b.from_bus, b.to_bus}
        return sorted(found)

    @property
    def total_demand(self) -> float:
        return float(sum(self.demand.values()))

    @property
    def total_capacity(self) -> float:
        return float(sum(g.p_max for g in self.generators))

    def generator_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "name": g.name, "bus": g.bus, "p_min": g.p_min, "p_max": g.p_max,
                    "c1 [EUR/MWh]": g.c1, "c2 [EUR/MWh^2]": g.c2,
                    "start [EUR]": g.start_cost,
                }
                for g in self.generators
            ]
        ).set_index("name")

    def branch_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "name": b.name, "from": b.from_bus, "to": b.to_bus,
                    "x [pu]": b.reactance, "B [MW/rad]": round(b.susceptance, 1),
                    "capacity [MW]": b.capacity,
                }
                for b in self.branches
            ]
        ).set_index("name")

    def merit_order(self) -> list[Generator]:
        """Cheapest first, by marginal cost at minimum output.

        For a purely linear cost this IS the dispatch order, which is why the
        LP solution in Tutorial 01 can be predicted before the solver runs.
        """
        return sorted(self.generators, key=lambda g: g.c1)

    def check(self) -> None:
        """Fail loudly on a system that cannot possibly be served."""
        if self.total_capacity < self.total_demand:
            raise ValueError(
                f"{self.name}: capacity {self.total_capacity} MW cannot serve "
                f"demand {self.total_demand} MW — the model will be infeasible, "
                f"and that is the data's fault rather than the solver's."
            )
        floor = sum(g.p_min for g in self.generators)
        if floor > self.total_demand:
            raise ValueError(
                f"{self.name}: minimum outputs sum to {floor} MW but demand is "
                f"{self.total_demand} MW. Without on/off decisions this is "
                f"infeasible — which is exactly why Tutorial 02 adds them."
            )


def two_generator_system(demand: float = 100.0, quadratic: bool = False) -> DispatchSystem:
    """The course's first system: one bus, two units, one balance constraint.

    Deliberately small enough to solve graphically and by hand before any
    solver is involved. With ``quadratic=True`` the cost curves gain a
    :math:`c_2 p^2` term, which turns the LP into a QP — the first time in the
    course that the problem class changes without the physics changing.
    """
    # c2 = 0.40, not 0.08. The point of the quadratic variant is that a
    # strictly convex cost moves the optimum OFF the vertex and equalises
    # marginal costs. With c2 = 0.08 the equal-marginal-cost point sits at
    # p1 = 163.6 MW, far outside this unit's 60 MW ceiling, so the QP optimum
    # was [60, 40] -- the same vertex as the LP -- with marginal costs 34.6 and
    # 57.4, a 22.8 EUR/MWh spread. The equal-marginal-cost reading needs the
    # optimum to be interior, which needs 2*c2 + 2*0.03 > 0.6, i.e. c2 > 0.27.
    # At c2 = 0.40 the optimum is p = [41.86, 58.14] with both marginal costs
    # 58.49 EUR/MWh.
    cheap = Generator("G1_coal", bus=0, p_min=20.0, p_max=60.0, c1=25.0,
                      c2=0.40 if quadratic else 0.0, start_cost=500.0,
                      min_up=3, min_down=2, ramp=30.0)
    expensive = Generator("G2_gas", bus=0, p_min=10.0, p_max=80.0, c1=55.0,
                          c2=0.03 if quadratic else 0.0, start_cost=150.0,
                          min_up=1, min_down=1, ramp=60.0)
    system = DispatchSystem("two-generator", [cheap, expensive], {0: demand})
    system.check()
    return system


def three_bus_system(scale: float = 1.0) -> DispatchSystem:
    """Three buses, three lines, two generators, one load.

    The smallest system in which a network constraint can bind without making
    the problem trivial, which is what Tutorial 03 needs to produce distinct
    nodal prices. Line 0-2 is deliberately weak: raise its capacity and the
    congestion — and the price separation — disappears.
    """
    generators = [
        Generator("G1_cheap", bus=0, p_min=0.0, p_max=150.0, c1=20.0, start_cost=800.0),
        Generator("G2_mid", bus=1, p_min=0.0, p_max=100.0, c1=45.0, start_cost=300.0),
        Generator("G3_peaker", bus=2, p_min=0.0, p_max=80.0, c1=90.0, start_cost=100.0),
    ]
    branches = [
        Branch("L_0_1", 0, 1, reactance=0.20, capacity=100.0),
        Branch("L_1_2", 1, 2, reactance=0.25, capacity=100.0),
        Branch("L_0_2", 0, 2, reactance=0.10, capacity=40.0),  # the binding one
    ]
    # The cheap generator alone cannot serve everything, so the network must
    # carry power and the weak line must matter. Verified by `check()` below.
    system = DispatchSystem(
        "three-bus",
        generators,
        demand={0: 20.0 * scale, 1: 60.0 * scale, 2: 130.0 * scale},
        branches=branches,
        reference_bus=0,
    )
    system.check()
    return system


def radial_feeder(n_pv: int = 3, pv_mw: float = 0.5):
    """``case33bw`` with controllable PV inverters — the SOC-BFM test bed.

    The bare feeder has no control at all, so an OPF over it is either
    infeasible or trivial. Reactive-capable inverters give it something to
    decide, which is what makes the AC and relaxed models worth comparing.

    Its minimum voltage without control is 0.913 pu, so a 0.95 lower limit is
    genuinely unreachable rather than numerically awkward. Tutorial 05 uses
    that as its infeasibility example.
    """
    import pandapower as pp
    import pandapower.networks as pn

    net = pn.case33bw()
    buses = [17, 24, 32][:n_pv]
    for bus in buses:
        pp.create_sgen(
            net, bus=bus, p_mw=pv_mw, q_mvar=0.0, controllable=True,
            max_p_mw=pv_mw, min_p_mw=0.0,
            max_q_mvar=0.6 * pv_mw, min_q_mvar=-0.6 * pv_mw,
        )
    return net


def simbench_feeder(code: str = "1-LV-rural1--0-sw"):
    """A SimBench network, for the capstone's realistic case."""
    import simbench as sb

    return sb.get_simbench_net(code)


def pglib_case(name: str = "pglib_opf_case14_ieee"):
    """A PGLib-OPF benchmark through potpourri's own loader.

    Benchmark discipline matters: a new formulation is tested against trusted
    reference cases before it is used for a novel claim.
    """
    from potpourri.benchmarks import pglib

    return pglib, name


def daily_profiles(n_periods: int = 24, seed: int = 20260101) -> pd.DataFrame:
    """Normalised demand, PV and wind shapes for the multi-period tutorials.

    Deterministic given ``seed``. Shapes are stylised rather than measured —
    the point of Tutorial 06 is intertemporal *coupling*, and a transparent
    profile keeps the storage behaviour interpretable.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(n_periods) * (24.0 / n_periods)

    demand = 0.62 + 0.18 * np.sin((hours - 7.0) / 24.0 * 2 * np.pi) \
        + 0.16 * np.exp(-0.5 * ((hours - 19.0) / 2.2) ** 2)
    demand += rng.normal(0.0, 0.012, n_periods)

    solar_elevation = np.sin(np.pi * (hours - 6.0) / 12.0)
    pv = np.clip(solar_elevation, 0.0, None) ** 1.3
    pv *= 1.0 - 0.18 * rng.random(n_periods)
    pv[pv < 1e-3] = 0.0   # snap twilight to exact zero so "is it night" is crisp

    wind = 0.45 + 0.30 * np.sin((hours - 2.0) / 24.0 * 2 * np.pi)
    wind += rng.normal(0.0, 0.07, n_periods)

    return pd.DataFrame(
        {
            "demand": np.clip(demand, 0.05, 1.0),
            "pv": np.clip(pv, 0.0, 1.0),
            "wind": np.clip(wind, 0.0, 1.0),
        },
        index=pd.Index(hours, name="hour"),
    )
