# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial 05 — Convex Relaxations and Second-Order Cone OPF
#
# > **Can we solve an easier problem that still tells us something rigorous
# > about the difficult AC-OPF?**
#
# ## Where we are
#
# ```text
# ED → UC → DC-OPF → AC-OPF → >> SOCP << → multi-period → uncertainty
#                                         → Benders → column generation → capstone
#                              ^
#                              YOU ARE HERE
# ```
#
# ## Learning objectives
#
# 1. State precisely what a **relaxation** is, and why it differs from an
#    **approximation**;
# 2. build a McCormick envelope for a bilinear term and see that its quality is
#    governed by the variable *bounds*;
# 3. derive the **branch-flow model** and find the single nonconvex equation;
# 4. relax that equation into a **second-order cone** and prove the bound
#    $z_{SOC} \le z_{AC}$;
# 5. test **exactness** with conic residuals rather than the objective gap;
# 6. meet **conic duality** and **Slater's condition**, which Tutorial 08 needs.
#
# ## Two claims this notebook tests
#
# > "A relaxation approximates the original problem."
#
# > "A small objective gap proves the relaxed solution is AC-feasible."
#
# Both false, and both are shown to be false by measurement.

# %% tags=["provided"]
import copy
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandapower as pp
import pandas as pd
import pyomo.environ as pyo

from psopt_course.config import fast_mode, set_seed
from psopt_course.metrics import relaxation_gap
from psopt_course.networks import radial_feeder
from psopt_course.plotting import (
    COLORS,
    plot_mccormick_envelope,
    plot_relaxation_vs_approximation,
    plot_second_order_cone,
    use_course_style,
)
from psopt_course.relaxations import SOCBFM
from psopt_course.solvers import solve

use_course_style()
set_seed()
print(f"reduced (fast) configuration: {fast_mode()}")

# %% [markdown]
# ## 1. Relaxation is a statement about sets
#
# Let the original problem be $\min\{f(x) : x \in \mathcal{F}\}$. A
# **relaxation** replaces $\mathcal{F}$ by a larger set:
#
# $$\mathcal{F} \subseteq \mathcal{F}_{relax}
#   \quad\Longrightarrow\quad
#   z_{relax} = \min_{\mathcal{F}_{relax}} f \;\le\; \min_{\mathcal{F}} f = z^\star .$$
#
# The inequality is immediate — minimising over a bigger set cannot give a
# larger answer — and it is the *only* thing a relaxation promises.
#
# An **approximation** is a different animal. It changes the equations, so the
# feasible set moves rather than grows, and there is no inequality in either
# direction.
#
# | | relaxation | approximation |
# |---|---|---|
# | changes | the feasible **set** | the **equations** |
# | relation | $\mathcal{F} \subseteq \mathcal{F}_{relax}$ | none |
# | gives a bound? | **yes** | **no** |
# | example here | SOC branch flow | DC power flow |
# | solution usable? | maybe not physical | usually approximately physical |
#
# Students routinely have this backwards: they expect DC-OPF to bound the AC
# optimum, and expect a convex relaxation to be an approximate solution.

# %% tags=["provided"]
plot_relaxation_vs_approximation()
plt.show()

# %% [markdown]
# ## 2. McCormick: the simplest useful relaxation
#
# The bilinear term $w = xy$ with $x \in [\underline{x}, \overline{x}]$,
# $y \in [\underline{y}, \overline{y}]$ is nonconvex. Its convex hull over the
# box is given by four linear inequalities:
#
# $$
# \begin{aligned}
# w &\ge \underline{x}y + x\underline{y} - \underline{x}\,\underline{y} \\
# w &\ge \overline{x}y + x\overline{y} - \overline{x}\,\overline{y} \\
# w &\le \overline{x}y + x\underline{y} - \overline{x}\,\underline{y} \\
# w &\le \underline{x}y + x\overline{y} - \underline{x}\,\overline{y}
# \end{aligned}
# $$
#
# Every one of them mentions a **bound**. That is the lesson: the strength of
# this relaxation is entirely inherited from how tightly $x$ and $y$ are boxed.
# Bound tightening is not housekeeping; it is where the relaxation gets its
# power.

# %% tags=["provided"]
plot_mccormick_envelope(x_bounds=(1.0, 4.0), y_bounds=(1.0, 3.0))
plt.show()


def mccormick_width(x_bounds, y_bounds, n=40):
    """Mean gap between the McCormick envelope and the true surface w = xy."""
    xl, xu = x_bounds
    yl, yu = y_bounds
    X, Y = np.meshgrid(np.linspace(xl, xu, n), np.linspace(yl, yu, n))
    lower = np.maximum(xl * Y + X * yl - xl * yl, xu * Y + X * yu - xu * yu)
    upper = np.minimum(xu * Y + X * yl - xu * yl, xl * Y + X * yu - xl * yu)
    return float(np.mean(upper - lower))


widths = pd.DataFrame(
    [
        {"x range": f"[{a:.1f}, {b:.1f}]", "width": b - a,
         "mean envelope gap": mccormick_width((a, b), (1.0, 3.0))}
        for a, b in [(2.0, 2.5), (1.5, 3.0), (1.0, 4.0), (0.5, 6.0)]
    ]
).set_index("x range")
display(widths.round(4))
print("Halve the box, and the relaxation tightens roughly in proportion.")

# %% [markdown]
# ## 3. The branch-flow model
#
# Tutorial 04's polar form had trigonometry. The **branch-flow (DistFlow)**
# formulation avoids it by working in *lifted* variables:
#
# $$u_i = |V_i|^2, \qquad \ell_{ij} = |I_{ij}|^2 .$$
#
# For a branch $(i,j)$ with impedance $r + jx$ carrying $P_{ij}, Q_{ij}$ into
# the branch at bus $i$:
#
# $$
# \begin{aligned}
# u_j &= u_i - 2(r P_{ij} + x Q_{ij}) + (r^2 + x^2)\,\ell_{ij}
#    && \text{(KVL)} \\
# P_{ij} + P_{ji} &= r\,\ell_{ij} && \text{(active loss)} \\
# Q_{ij} + Q_{ji} &= x\,\ell_{ij} && \text{(reactive loss)} \\
# P_{ij}^2 + Q_{ij}^2 &= u_i\,\ell_{ij} && \text{(current definition)}
# \end{aligned}
# $$
#
# Three of those are **linear**. The fourth is a quadratic **equality**, and it
# is the only nonconvexity in the whole model. Relax it:
#
# $$\boxed{P_{ij}^2 + Q_{ij}^2 \;\le\; u_i\,\ell_{ij}}$$
#
# ### Why that is a second-order cone
#
# The set $\{(P,Q,u,\ell) : P^2+Q^2 \le u\ell,\ u,\ell \ge 0\}$ is a **rotated**
# second-order cone, and it maps to the standard cone
# $\mathcal{Q}^{n+1} = \{(t,z) : \lVert z\rVert_2 \le t\}$ by
#
# $$P^2 + Q^2 \le u\ell
#   \iff
#   \left\lVert \begin{pmatrix} 2P \\ 2Q \\ u - \ell \end{pmatrix} \right\rVert_2
#   \;\le\; u + \ell .$$
#
# Both sides are convex, and conic solvers exploit exactly this structure.

# %% tags=["provided"]
plot_second_order_cone()
plt.show()

# Verify the rotated-to-standard identity numerically rather than trusting it.
rng = np.random.default_rng(0)
samples = rng.uniform(0.1, 2.0, size=(2000, 4))
P, Q, u, ell = samples.T
rotated = P**2 + Q**2 <= u * ell + 1e-12
standard = np.sqrt((2 * P) ** 2 + (2 * Q) ** 2 + (u - ell) ** 2) <= u + ell + 1e-9
print(f"the two forms agree on {np.mean(rotated == standard):.2%} of 2000 random points")
assert np.mean(rotated == standard) > 0.999

# %% [markdown]
# ## 4. Build the relaxation on a real feeder
#
# `psopt_course.relaxations.SOCBFM` extends `opf-potpourri` using its own mixin
# pattern — `SOCBFM(BFM, OPF)` mirrors the library's `ACOPF(AC, OPF)` — so it
# inherits bus numbering, per-unit conversion, generation limits and the result
# mappers, and adds only the branch-flow physics.
#
# The `current_definition` switch is the entire experiment: `"exact"` keeps the
# equality and gives a nonconvex QCQP, `"soc"` relaxes it and gives an SOCP.
# **Nothing else differs**, which is what makes the two objectives comparable.

# %% tags=["provided"]
net = radial_feeder(n_pv=3, pv_mw=0.5)
print(f"feeder: {len(net.bus)} buses, {int(net.line.in_service.sum())} lines in service, "
      f"{len(net.sgen)} PV inverters, base {net.sn_mva} MVA")

pp.runpp(net, numba=False)
print(f"base power flow: |V| in [{net.res_bus.vm_pu.min():.4f}, "
      f"{net.res_bus.vm_pu.max():.4f}] pu, losses {net.res_line.pl_mw.sum():.4f} MW")
print("\nNote the minimum voltage: a 0.95 pu limit is NOT reachable on this")
print("feeder without control. That is a data fact, not a solver problem, and")
print("Exercise 5.3 uses it.")

VMIN, VMAX = 0.90, 1.05


def build(definition):
    """Loss-minimising OPF on the feeder, with the inverters free in Q."""
    model = SOCBFM(copy.deepcopy(net), current_definition=definition)
    model.add_OPF(vmin=VMIN, vmax=VMAX)
    mod = model.model
    for g in mod.sG:
        mod.qsG[g].unfix()
        mod.qsG[g].setlb(-0.03)   # per unit on the 10 MVA base
        mod.qsG[g].setub(0.03)
    # Minimise active losses: monotone increasing in ell, which is what the
    # Farivar-Low exactness conditions require.
    mod.obj = pyo.Objective(expr=sum(mod.r[l] * mod.ell[l] for l in mod.L))
    return model


soc = build("soc")
print(f"\nSOC model problem class : {soc.problem_class}")
soc_record = solve(soc.model, "SOCP")
print(soc_record.summary())

exact = build("exact")
print(f"\nexact model problem class: {exact.problem_class}")
exact_record = solve(exact.model, "NLP")
print(exact_record.summary())

# %% [markdown]
# ## 5. The bound, and how to test exactness properly

# %% tags=["provided"]
base = net.sn_mva
z_soc = pyo.value(soc.model.obj) * base
z_exact = pyo.value(exact.model.obj) * base

print(f"z_SOC   = {z_soc:.8f} MW  (relaxation -> LOWER bound)")
print(f"z_BFM   = {z_exact:.8f} MW  (original nonconvex problem)")

# Both were solved to a finite tolerance, so the comparison must be made at a
# tolerance. Demanding z_SOC <= z_BFM to 1e-9 ABSOLUTE is stricter than either
# solve, and when the relaxation is exact the two coincide -- at which point
# their ORDERING within tolerance carries no information.
difference = z_exact - z_soc
tolerance = 1e-6 * max(1.0, abs(z_exact))
print(f"\nz_BFM - z_SOC = {difference:+.3e} MW   (solver tolerance ~{tolerance:.1e})")
if abs(difference) <= tolerance:
    print("The two agree to solver tolerance: the relaxation is TIGHT here, so the")
    print("bound is attained and the sign of a 1e-12 difference means nothing.")
else:
    print(f"bound holds: z_SOC <= z_BFM  ->  {z_soc <= z_exact + tolerance}")
print(f"objective gap: {relaxation_gap(min(z_soc, z_exact), max(z_exact, z_soc)):.6%}")

residuals = soc.soc_residuals()
worst = max(residuals.values())
print(f"\nmax SOC residual  r_l = u_i*ell_l - P^2 - Q^2  :  {worst:.3e}")
print(f"relaxation exact to 1e-6: {soc.is_exact(1e-6)}")

voltages = soc.voltages_pu()
print(f"|V| from the relaxation: [{min(voltages.values()):.4f}, "
      f"{max(voltages.values()):.4f}] pu")

# %% [markdown]
# ### Why the residual and not the gap
#
# The objective gap answers "do the two problems have similar optimal values?".
# **Exactness** asks something different: "does the relaxed solution satisfy the
# original equality?" A relaxed point can have almost the right objective and
# still sit strictly inside the cone, where
# $P^2 + Q^2 < u\ell$ — which corresponds to no physical current at all.
#
# So the test is the residual, per branch, and Exercise 5.2 constructs a case
# where the two answers disagree.

# %% tags=["provided"]
fig, ax = plt.subplots(figsize=(8.5, 3.2))
branch_ids = sorted(residuals)
ax.bar(range(len(branch_ids)), [residuals[b] for b in branch_ids],
       color=COLORS["relaxed"])
ax.set_xlabel("branch")
ax.set_ylabel("SOC residual [pu]")
ax.set_title("Cone slack per branch — zero means the relaxation is tight there")
ax.axhline(1e-6, ls="--", color=COLORS["infeasible"], label="1e-6 tolerance")
ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# ## 6. Conic duality, and why Tutorial 08 needs it
#
# LP duality (Tutorial 02) generalises. A conic program
#
# $$\min_x\ c^\top x \quad \text{s.t.} \quad Ax + b \in \mathcal{K}$$
#
# over a convex cone $\mathcal{K}$ has a dual
#
# $$\max_y\ -b^\top y \quad \text{s.t.} \quad c + A^\top y = 0,\ \ y \in \mathcal{K}^*$$
#
# with $\mathcal{K}^*$ the dual cone. Weak duality always holds. **Strong**
# duality needs a regularity condition, and the usual one is
#
# > **Slater's condition:** there exists a **strictly** feasible point, i.e. one
# > lying in the *interior* of the cone.
#
# Convexity alone is not enough. This matters concretely in Tutorial 08: a
# Benders cut built from a dual solution is only valid if strong duality holds
# for the subproblem. For an LP that is automatic; for a conic program it is a
# condition you should check rather than assume.

# %% tags=["provided"]
# Check strict feasibility on our solved relaxation: is any branch strictly
# inside the cone? (Slater asks for one strictly feasible point, not optimality.)
strict = sum(1 for r in residuals.values() if r > 1e-9)
print(f"branches strictly inside the cone at the optimum: {strict}/{len(residuals)}")
print()
print("At the OPTIMUM a tight relaxation sits on the cone boundary, which is")
print("exactly what exactness means. Slater's condition is about the FEASIBLE")
print("SET having a strict interior, not about where the optimum lands — and the")
print("bounded, strictly positive voltage box here provides that.")

# %% [markdown]
# ## 7. The landscape of relaxations
#
# | | class | convex | voltage | reactive | tightness | scales to |
# |---|---|---|---|---|---|---|
# | DC-OPF | LP | yes | no (assumed 1.0) | no | *approximation*, no bound | very large |
# | LinDistFlow | LP | yes | linearised | yes | approximation | large |
# | **SOC branch flow** | SOCP | yes | yes | yes | exact on radial nets under conditions | large |
# | QC relaxation | SOCP+ | yes | yes | yes | often tighter than SOC | medium |
# | SDP (Lavaei–Low) | SDP | yes | yes | yes | tightest of these | small–medium |
# | AC-OPF | nonconvex NLP | **no** | yes | yes | exact by definition | large, locally |
#
# Two rows are approximations and give no bound. The rest are relaxations and do.
# Reading down the "tightness" column against "scales to" is the whole
# engineering trade-off.

# %% [markdown]
# ## 8. Exercises
#
# ---

# %% [markdown]
# ### Exercise 5.1 — Implement the SOC constraint yourself
#
# **Difficulty:** Intermediate
#
# The library does this for you. Do it once by hand so you know what it does.
#
# #### Your task
#
# Write `soc_slack(P, Q, u, ell)` returning $u\ell - P^2 - Q^2$, and
# `is_in_cone(...)` testing membership. Then verify on the solved model that
#
# 1. every branch satisfies the relaxed inequality;
# 2. the *exact* model satisfies the equality;
# 3. the relaxed model's residuals match `soc.soc_residuals()`.
#
# #### Expected result
#
# All three checks pass. Residuals from your function and the library's should
# agree to machine precision.
#
# #### Hint
#
# `soc.model.A[l, 1]` gives the from-bus of branch `l`.

# %% tags=["exercise"]
# TODO: implement soc_slack and is_in_cone, then verify against the solved models.
def soc_slack(P, Q, u, ell):
    """Slack in the rotated cone: u*ell - P^2 - Q^2, non-negative if feasible."""
    # TODO
    pass


my_residuals = None

# %% tags=["solution"]
def soc_slack(P, Q, u, ell):
    """Slack in the rotated cone: u*ell - P^2 - Q^2, non-negative if feasible."""
    return u * ell - P**2 - Q**2


def is_in_cone(P, Q, u, ell, tolerance=1e-9):
    return u >= -tolerance and ell >= -tolerance and soc_slack(P, Q, u, ell) >= -tolerance


def residuals_of(model):
    mod = model.model
    out = {}
    for l in mod.L:
        i = mod.A[l, 1]
        out[int(l)] = soc_slack(
            pyo.value(mod.pLfrom[l]), pyo.value(mod.qLfrom[l]),
            pyo.value(mod.u[i]), pyo.value(mod.ell[l]),
        )
    return out


my_residuals = residuals_of(soc)
library = soc.soc_residuals()
difference = max(abs(my_residuals[k] - library[k]) for k in my_residuals)
print(f"my residuals vs the library's: max difference {difference:.3e}")

exact_residuals = residuals_of(exact)
print(f"\nrelaxed model: max slack {max(my_residuals.values()):.3e}  (>= 0 required)")
print(f"exact model  : max |slack| {max(abs(v) for v in exact_residuals.values()):.3e}"
      f"  (== 0 required)")

in_cone = all(
    is_in_cone(pyo.value(soc.model.pLfrom[l]), pyo.value(soc.model.qLfrom[l]),
               pyo.value(soc.model.u[soc.model.A[l, 1]]), pyo.value(soc.model.ell[l]))
    for l in soc.model.L
)
print(f"\nevery branch inside the cone: {in_cone}")

# %% tags=["validation"]
assert my_residuals is not None, "implement soc_slack and build `my_residuals`"
assert min(my_residuals.values()) > -1e-8, "a branch violates the relaxed inequality"
assert max(abs(my_residuals[k] - soc.soc_residuals()[k]) for k in my_residuals) < 1e-12, (
    "your residuals disagree with the library's"
)
assert max(abs(v) for v in exact_residuals.values()) < 1e-5, (
    "the exact model should satisfy the EQUALITY, not just the inequality"
)
print("Checks passed: inequality in the relaxation, equality in the original.")

# %% [markdown]
# #### Interpretation
#
# **Does the objective gap imply the relaxed solution is AC feasible?**

# %% tags=["solution"]
print("ANSWER. No, and the two questions are not even about the same thing.")
print()
print(f"Here the objective gap is {relaxation_gap(z_soc, max(z_exact, z_soc)):.4%} and the")
print(f"maximum cone residual is {max(my_residuals.values()):.2e}, so on THIS instance")
print("both say the same thing: the relaxation is tight.")
print()
print("But they can disagree, and the logic only runs one way:")
print()
print("  relaxation exact   =>  objective gap is zero")
print("  objective gap zero  =/=>  relaxation exact")
print()
print("A relaxed point can sit strictly inside the cone -- P^2 + Q^2 < u*ell --")
print("while still achieving nearly the right objective, because the objective")
print("may be insensitive to slack on branches that carry little power. Such a")
print("point corresponds to no physical current, so it is not an operating state")
print("at all, however good its cost looks.")
print()
print("That is why the residual is the test and the gap is not. Exercise 5.2")
print("builds a case where the relaxation is visibly NOT tight.")

# %% [markdown]
# ---
#
# ### Exercise 5.2 — Break the exactness conditions on purpose
#
# **Difficulty:** Advanced
#
# The Farivar–Low / Gan–Li–Topcu–Low conditions are *sufficient*: radial
# topology, an objective monotone increasing in $\ell$, and non-binding upper
# voltage limits. Violate one and see what happens.
#
# #### Your task
#
# 1. Solve the relaxation with a **constant** objective (nothing monotone in
#    $\ell$).
# 2. Measure the residuals.
# 3. Compare the resulting voltages against a real power flow.
# 4. Explain what the solver did, and whether it did anything wrong.
#
# #### Think before coding
#
# If the objective does not reward small $\ell$, what stops the solver from
# choosing an $\ell$ far larger than the physics allows?
#
# #### Expected result
#
# Residuals far from zero, and voltages that do not correspond to any power
# flow. The solver will still report `optimal`.

# %% tags=["exercise"]
# TODO: solve the relaxation with a constant objective and measure exactness.
broken = None

# %% tags=["solution"]
broken = SOCBFM(copy.deepcopy(net), current_definition="soc")
broken.add_OPF(vmin=VMIN, vmax=VMAX)
mod = broken.model
for g in mod.sG:
    mod.qsG[g].unfix()
    mod.qsG[g].setlb(-0.03)
    mod.qsG[g].setub(0.03)
# A CONSTANT objective: nothing is monotone in ell any more.
mod.obj = pyo.Objective(expr=0.0)

broken_record = solve(mod, "SOCP")
print(f"solver says: {broken_record.termination}")

broken_residuals = broken.soc_residuals()
print(f"max SOC residual: {max(broken_residuals.values()):.3e}")
print(f"exact to 1e-6   : {broken.is_exact(1e-6)}")

implied_losses = sum(
    pyo.value(mod.r[l]) * pyo.value(mod.ell[l]) for l in mod.L
) * base
pp.runpp(net, numba=False)
print(f"\nlosses implied by the relaxed solution: {implied_losses:,.3f} MW")
print(f"losses from an actual power flow      : {net.res_line.pl_mw.sum():.3f} MW")
print(f"ratio: {implied_losses / net.res_line.pl_mw.sum():,.0f}x")

broken_v = broken.voltages_pu()
print(f"\n|V| from the relaxation: [{min(broken_v.values()):.4f}, "
      f"{max(broken_v.values()):.4f}] pu")

# %% tags=["validation"]
assert broken is not None, "build the broken case first"
assert broken_record.ok, "the solver should still succeed — the MODEL is fine"
assert not broken.is_exact(1e-5), (
    "with a constant objective the relaxation should NOT be tight"
)
assert max(broken_residuals.values()) > max(soc.soc_residuals().values()), (
    "the broken case should be less tight than the loss-minimising one"
)
print("Checks passed: solver succeeded, relaxation is not exact.")

# %% [markdown]
# #### Interpretation
#
# **Did the solver do anything wrong?**

# %% tags=["solution"]
print("ANSWER. No. The solver solved the problem it was given, exactly and")
print("correctly, and returned a globally optimal point of a convex program.")
print()
print("What went wrong is the MODEL's relationship to physics. With a constant")
print("objective nothing penalises large ell, so the solver is free to pick any")
print(f"point in the enlarged set. It chose one implying {implied_losses:,.0f} MW of")
print(f"losses on a feeder whose real losses are {net.res_line.pl_mw.sum():.2f} MW --")
print("absurd physically, perfectly feasible for the relaxation.")
print()
print("This is the cleanest demonstration in the course that a RELAXATION IS NOT")
print("AN APPROXIMATION. The relaxed feasible set contains points that are not")
print("operating states, and only the objective pulling against the cone keeps")
print("the solution on the boundary where the original equality holds.")
print()
print("Practically: never report a relaxed solution as an operating point without")
print("checking the residuals. The bound is always valid; the POINT is not always")
print("meaningful.")

# %% [markdown]
# ---
#
# ### Exercise 5.3 — Prepare the model for decomposition
#
# **Difficulty:** Advanced
#
# Tutorial 08 fixes some decisions and solves the rest. For that to work, the
# problem must become **convex once those decisions are fixed**.
#
# #### Your task
#
# 1. Treat the PV inverters' *reactive capability* as a discrete investment: for
#    each inverter, a binary $y_b$ saying whether Q control is installed.
# 2. For a few fixed $y$, solve the SOC subproblem and record the loss.
# 3. Confirm the subproblem is a convex SOCP for every fixed $y$.
# 4. Explain why the same construction with the **AC** model would not give
#    Tutorial 08 what it needs.
#
# #### Expected result
#
# Installing more Q capability should reduce losses. Each subproblem should be a
# convex SOCP, solved to global optimality.

# %% tags=["exercise"]
# TODO: fix a binary investment pattern, solve the SOC subproblem for each,
#       and tabulate the resulting losses in `investment`.
investment = None

# %% tags=["solution"]
import itertools

def subproblem_loss(pattern):
    """Loss-minimising SOC subproblem with Q control installed where pattern is 1."""
    model = SOCBFM(copy.deepcopy(net), current_definition="soc")
    model.add_OPF(vmin=VMIN, vmax=VMAX)
    mod = model.model
    for position, g in enumerate(sorted(mod.sG)):
        if position < len(pattern) and pattern[position]:
            mod.qsG[g].unfix()
            mod.qsG[g].setlb(-0.03)
            mod.qsG[g].setub(0.03)
        else:
            mod.qsG[g].fix(0.0)          # no Q control installed here
    mod.obj = pyo.Objective(expr=sum(mod.r[l] * mod.ell[l] for l in mod.L))
    record = solve(mod, "SOCP")
    return model, record


n_inverters = len(soc.model.sG)
patterns = list(itertools.product([0, 1], repeat=n_inverters))
if fast_mode():
    patterns = [p for p in patterns if sum(p) in (0, n_inverters)]

rows = []
for pattern in patterns:
    model, record = subproblem_loss(pattern)
    rows.append({
        "pattern": "".join(str(b) for b in pattern),
        "installed": sum(pattern),
        "loss [MW]": pyo.value(model.model.obj) * base if record.ok else np.nan,
        "class": model.problem_class,
        "termination": record.termination,
        "max residual": max(model.soc_residuals().values()) if record.ok else np.nan,
    })

investment = pd.DataFrame(rows).set_index("pattern")
display(investment.round(6))

print(f"\nevery subproblem is a {investment['class'].unique()[0]} — convex, so each")
print("was solved to GLOBAL optimality, and its dual carries globally valid")
print("information about the value function Q(y).")
best = investment["loss [MW]"].idxmin()
print(f"\nlowest loss at pattern {best} "
      f"({investment.loc[best, 'loss [MW]']:.6f} MW)")

# %% tags=["validation"]
assert investment is not None, "build the `investment` table first"
_done = investment.dropna(subset=["loss [MW]"])
assert len(_done) >= 2, "at least two investment patterns should solve"
assert (_done["class"] == "SOCP").all(), (
    "every fixed-y subproblem must be a convex SOCP — that is the whole point"
)
_none = _done[_done["installed"] == 0]["loss [MW]"]
_all = _done[_done["installed"] == _done["installed"].max()]["loss [MW]"]
if len(_none) and len(_all):
    assert _all.iloc[0] <= _none.iloc[0] + 1e-9, (
        "more reactive capability cannot increase losses"
    )
print("Checks passed: every subproblem is a convex SOCP, and more control helps.")

# %% [markdown]
# #### Interpretation
#
# **Why would the AC model not give Tutorial 08 what it needs?**

# %% tags=["solution"]
print("ANSWER. Because of what the dual variables would mean.")
print()
print("Benders builds a cut on the value function Q(y) out of DUAL information")
print("from the subproblem. For the cut to be valid everywhere -- not just near")
print("the current y -- that dual must describe the GLOBAL optimum of the")
print("subproblem. Two ingredients are needed:")
print()
print("  1. the subproblem is CONVEX once y is fixed, so a local optimum is")
print("     global;")
print("  2. STRONG DUALITY holds, so the dual value equals the primal value.")
print()
print("Fixing y in the SOC model leaves a convex SOCP: both hold, under Slater.")
print("Fixing y in the AC model leaves a nonconvex NLP: IPOPT returns a KKT")
print("point of whichever basin it fell into, and its multipliers describe that")
print("basin only. A cut built from them can slice off part of the true feasible")
print("region, and then the 'lower bound' is not a bound at all.")
print()
print("So the SOC relaxation is not merely a nicer model here. It is what makes")
print("rigorous decomposition POSSIBLE -- and Tutorial 08 measures the difference")
print("by running Benders three ways: DC, AC and SOC.")
print()
print("The caveat that comes with it, and it is essential: solving the MISOCP")
print("rigorously solves the RELAXATION. Whether that certifies the original AC")
print("problem depends on exactness, or on the gap to an independently obtained")
print("AC-feasible upper bound.")

# %% [markdown]
# ## 9. Key takeaways
#
# 1. **A relaxation enlarges the feasible set**, so it gives a bound. An
#    **approximation** changes the equations and gives none.
# 2. **McCormick envelopes are only as tight as the variable bounds.**
# 3. **The branch-flow model isolates the nonconvexity into one equation**,
#    $P^2+Q^2 = u\ell$, and relaxing it to $\le$ gives a rotated second-order
#    cone.
# 4. **$z_{SOC} \le z_{AC}$ always; exactness sometimes.** The residual
#    $u_i\ell_l - P^2 - Q^2$ tests exactness; the objective gap does not.
# 5. **A relaxed solution need not be a physical operating state.** With a
#    constant objective this notebook produced one implying absurd losses — from
#    a correctly solved convex program.
# 6. **Conic duality needs Slater's condition**, and that is what makes
#    Tutorial 08's cuts valid.
#
# ## Further reading
#
# - Jabr, "Radial distribution load flow using conic programming", *IEEE Trans.
#   Power Syst.* 21(3), 2006.
# - Farivar & Low, "Branch flow model: relaxations and convexification (parts I
#   and II)", *IEEE Trans. Power Syst.* 28(3), 2013.
# - Gan, Li, Topcu & Low, "Exact convex relaxation of optimal power flow in
#   radial networks", *IEEE Trans. Autom. Control* 60(1), 2015.
# - Low, "Convex relaxation of optimal power flow (parts I and II)", *IEEE
#   Trans. Control Netw. Syst.* 1(1) and 1(2), 2014.
# - Molzahn & Hiskens, *A Survey of Relaxations and Approximations of the Power
#   Flow Equations*, NOW Publishers, 2019.
#
# ## Next
#
# Tutorial 06 adds time. Storage couples one period to the next, and a decision
# now changes what is possible later.
