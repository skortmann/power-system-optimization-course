"""Is the answer physically real, or merely mathematically optimal?

A solver returning ``optimal`` is a statement about the *model*. Whether the
model represents the physical system is a separate question, and this module is
where the course keeps asking it.

Three levels, deliberately distinct:

1. :func:`constraint_residuals` — does the solution satisfy the model it was
   solved from? (Catches tolerance and scaling problems.)
2. :func:`power_flow_check` — does an independent AC power flow, run on the
   dispatch the optimizer chose, reproduce it? (Catches formulation errors.)
3. :func:`limit_violations` — does the *validated* state respect the
   operational limits? (Catches models that optimized the wrong thing.)

A result that passes 1 and fails 2 is the interesting case, and the course
constructs one on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyomo.environ as pyo

__all__ = [
    "PowerFlowCheck",
    "ResidualReport",
    "bounds_table",
    "constraint_residuals",
    "limit_violations",
    "power_flow_check",
]


@dataclass
class ResidualReport:
    """Worst-case violation of each constraint block in a solved model."""

    worst: dict[str, float] = field(default_factory=dict)
    tolerance: float = 1e-6

    @property
    def feasible(self) -> bool:
        return all(v <= self.tolerance for v in self.worst.values())

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            {"max |violation|": pd.Series(self.worst)}
        ).sort_values("max |violation|", ascending=False)
        frame["within tolerance"] = frame["max |violation|"] <= self.tolerance
        return frame

    def summary(self) -> str:
        if self.feasible:
            return f"all constraint blocks satisfied to {self.tolerance:.0e}"
        offenders = [k for k, v in self.worst.items() if v > self.tolerance]
        return f"{len(offenders)} block(s) violated: {', '.join(sorted(offenders)[:5])}"


def constraint_residuals(model, tolerance: float = 1e-6) -> ResidualReport:
    """Largest violation within each named constraint block.

    Reported per *block* rather than per constraint index, because a name like
    ``KCL_p`` tells a student where to look and ``c[48193]`` does not. This is
    the same motivation as potpourri's diagnostics layer, which
    :func:`describe_violations_on_network` hands off to for pandapower models.
    """
    worst: dict[str, float] = {}
    for block in model.component_objects(pyo.Constraint, active=True):
        largest = 0.0
        for index in block:
            constraint = block[index]
            try:
                body = pyo.value(constraint.body)
            except (ValueError, TypeError):
                continue
            lower = pyo.value(constraint.lower) if constraint.lower is not None else None
            upper = pyo.value(constraint.upper) if constraint.upper is not None else None
            if lower is not None:
                largest = max(largest, lower - body)
            if upper is not None:
                largest = max(largest, body - upper)
        worst[block.name] = float(max(largest, 0.0))
    return ResidualReport(worst=worst, tolerance=tolerance)


@dataclass
class PowerFlowCheck:
    """What an independent AC power flow said about an optimizer's dispatch."""

    converged: bool
    max_vm_error: float
    vm_min: float
    vm_max: float
    losses_mw: float
    max_line_loading: float
    detail: pd.DataFrame | None = None

    def summary(self) -> str:
        if not self.converged:
            return "power flow did NOT converge on the optimized dispatch"
        return (
            f"power flow converged | |V| in [{self.vm_min:.4f}, {self.vm_max:.4f}] pu "
            f"| losses {self.losses_mw:.4f} MW "
            f"| max loading {self.max_line_loading:.1f}%"
        )


def power_flow_check(
    net,
    *,
    vm_expected: dict[int, float] | None = None,
    numba: bool = False,
) -> PowerFlowCheck:
    """Run a real power flow on ``net`` and compare against the model's voltages.

    Pass ``vm_expected`` keyed by **pandapower** bus index to get the agreement
    error. The distinction that matters: the optimizer's voltages come from its
    own equations, and this runs pandapower's. When they disagree, the
    formulation is wrong — not the solver.
    """
    import pandapower as pp

    try:
        pp.runpp(net, numba=numba)
    except Exception:
        return PowerFlowCheck(False, float("nan"), float("nan"), float("nan"),
                              float("nan"), float("nan"))

    vm = net.res_bus.vm_pu
    error = float("nan")
    detail = None
    if vm_expected:
        rows = []
        for bus, expected in vm_expected.items():
            if bus in vm.index:
                rows.append({"bus": bus, "model": expected, "power_flow": float(vm[bus]),
                             "error": abs(expected - float(vm[bus]))})
        if rows:
            detail = pd.DataFrame(rows).set_index("bus")
            error = float(detail["error"].max())

    loading = net.res_line.loading_percent if len(net.res_line) else pd.Series([0.0])
    return PowerFlowCheck(
        converged=True,
        max_vm_error=error,
        vm_min=float(vm.min()),
        vm_max=float(vm.max()),
        losses_mw=float(net.res_line.pl_mw.sum()) if len(net.res_line) else 0.0,
        max_line_loading=float(loading.max()),
        detail=detail,
    )


def limit_violations(
    net,
    *,
    vm_min: float = 0.95,
    vm_max: float = 1.05,
    max_loading_percent: float = 100.0,
) -> pd.DataFrame:
    """Operational limits broken by the *validated* state, named by element.

    Returns one row per violated element, with the pandapower index, so a
    student can look it up rather than decoding a constraint number.
    """
    rows: list[dict] = []
    if hasattr(net, "res_bus") and len(net.res_bus):
        for bus, vm_pu in net.res_bus.vm_pu.items():
            if vm_pu < vm_min:
                rows.append({"element": "bus", "index": bus, "quantity": "vm_pu",
                             "value": float(vm_pu), "limit": vm_min,
                             "violation": float(vm_min - vm_pu)})
            elif vm_pu > vm_max:
                rows.append({"element": "bus", "index": bus, "quantity": "vm_pu",
                             "value": float(vm_pu), "limit": vm_max,
                             "violation": float(vm_pu - vm_max)})
    if hasattr(net, "res_line") and len(net.res_line):
        for line, loading in net.res_line.loading_percent.items():
            if loading > max_loading_percent:
                rows.append({"element": "line", "index": line, "quantity": "loading_percent",
                             "value": float(loading), "limit": max_loading_percent,
                             "violation": float(loading - max_loading_percent)})
    if hasattr(net, "res_trafo") and len(net.res_trafo):
        for trafo, loading in net.res_trafo.loading_percent.items():
            if loading > max_loading_percent:
                rows.append({"element": "trafo", "index": trafo, "quantity": "loading_percent",
                             "value": float(loading), "limit": max_loading_percent,
                             "violation": float(loading - max_loading_percent)})

    if not rows:
        return pd.DataFrame(
            columns=["element", "index", "quantity", "value", "limit", "violation"]
        )
    return pd.DataFrame(rows).sort_values("violation", ascending=False).reset_index(drop=True)


def bounds_table(
    lower: float,
    upper: float,
    *,
    lower_label: str = "lower bound",
    upper_label: str = "upper bound",
) -> pd.DataFrame:
    """The course's recurring bound picture, as a small table.

    Used by Tutorials 02, 05, 08, 09 and 10 so that branch-and-bound, convex
    relaxation, Benders and column generation all report their progress in the
    same shape. The unifying claim is that they are all squeezing the same
    interval.
    """
    if lower > upper + 1e-9:
        raise ValueError(
            f"{lower_label} ({lower:.6g}) exceeds {upper_label} ({upper:.6g}); "
            "for a minimization that is impossible, so one of them is wrong"
        )
    gap = upper - lower
    relative = gap / max(1.0, abs(upper))
    return pd.DataFrame(
        {
            "value": [lower, upper, gap, relative],
        },
        index=[lower_label, upper_label, "absolute gap", "relative gap"],
    )


def describe_violations_on_network(model, level: str = "standard"):
    """Hand off to potpourri's diagnostics, which name pandapower elements.

    Available on any model built from :class:`potpourri.models.basemodel.Basemodel`,
    including the course's :class:`~psopt_course.relaxations.SOCBFM`.
    """
    if not hasattr(model, "diagnose"):
        raise TypeError(
            "this model has no diagnose(); it is not built on potpourri's Basemodel"
        )
    return model.diagnose(level=level)


def max_abs(values) -> float:
    """Largest absolute value, tolerant of empty input."""
    array = np.asarray(list(values), dtype=float)
    return float(np.abs(array).max()) if array.size else 0.0
