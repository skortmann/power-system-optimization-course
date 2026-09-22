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
# # Tutorial 04 — AC Optimal Power Flow and the Origin of Nonconvexity
#
# > **Why is real optimal power flow substantially harder than DC-OPF?**
#
# ## Where we are
#
# ```text
# ED → UC → DC-OPF → >> AC-OPF << → SOCP → multi-period → uncertainty
#                                  → Benders → column generation → capstone
#                     ^
#                     YOU ARE HERE
# ```
#
# ## Learning objectives
#
# 1. Write the AC nodal power equations and point at the terms that make them
#    nonconvex;
# 2. build an AC-OPF **by hand** in Pyomo and solve it with IPOPT;
# 3. compare AC against DC on the same network and say what DC discarded;
# 4. explain why `optimal` from a local NLP solver is not a claim about the
#    global optimum;
# 5. read `opf-potpourri`'s architecture, and reproduce the hand-built answer
#    with it.
#
# ## The claim this notebook tests
#
# > "IPOPT returned optimal, therefore we have the global AC-OPF optimum."
#
# False in general, and the notebook shows why rather than asserting it.

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
from psopt_course.plotting import COLORS, use_course_style
from psopt_course.solvers import solve
from psopt_course.validation import power_flow_check

use_course_style()
set_seed()
print(f"reduced (fast) configuration: {fast_mode()}")

# %% [markdown]
# ## 1. A three-bus AC network
#
# Small enough to write every equation out. Cheap generation at bus 0,
# expensive at bus 1, all the load at bus 2.

# %% tags=["provided"]
def three_bus_ac():
    """Three buses, three lines, in pandapower so potpourri can read it later."""
    net = pp.create_empty_network(sn_mva=100.0)
    buses = [pp.create_bus(net, vn_kv=110.0, name=f"B{i}") for i in range(3)]
    pp.create_ext_grid(net, buses[0], vm_pu=1.0, max_p_mw=200, min_p_mw=0,
                       max_q_mvar=100, min_q_mvar=-100)
    pp.create_gen(net, buses[1], p_mw=40, vm_pu=1.0, controllable=True,
                  max_p_mw=100, min_p_mw=0, max_q_mvar=60, min_q_mvar=-60)
    pp.create_load(net, buses[2], p_mw=120, q_mvar=40)
    # r/x = 0.15: a transmission-like ratio, so the DC comparison is fair.
    for (f, t, x) in [(0, 1, 0.06), (1, 2, 0.08), (0, 2, 0.10)]:
        pp.create_line_from_parameters(
            net, buses[f], buses[t], length_km=1.0,
            r_ohm_per_km=x * 0.15 * 121, x_ohm_per_km=x * 121,
            c_nf_per_km=0.0, max_i_ka=1.0,
        )
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=20.0)
    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=60.0)
    return net


net = three_bus_ac()
pp.runpp(net, numba=False)
print(f"base power flow: |V| = {net.res_bus.vm_pu.round(4).tolist()} pu")
print(f"                 losses = {net.res_line.pl_mw.sum():.4f} MW")
print(f"                 ext grid supplies {net.res_ext_grid.p_mw.iloc[0]:.2f} MW")

# %% [markdown]
# ## 2. The AC equations, and where convexity dies
#
# With complex voltage $V_i = |V_i| e^{j\theta_i}$ and admittance
# $Y_{ij} = G_{ij} + jB_{ij}$, the power injected at bus $i$ is
#
# $$
# \begin{aligned}
# P_i &= |V_i| \sum_j |V_j| \big(G_{ij}\cos\theta_{ij} + B_{ij}\sin\theta_{ij}\big) \\
# Q_i &= |V_i| \sum_j |V_j| \big(G_{ij}\sin\theta_{ij} - B_{ij}\cos\theta_{ij}\big)
# \end{aligned}
# \qquad \theta_{ij} = \theta_i - \theta_j
# $$
#
# Two things went wrong at once, and they are different kinds of wrong:
#
# | term | why it is a problem |
# |---|---|
# | $\lvert V_i\rvert\,\lvert V_j\rvert$ | a **product of variables** — bilinear, so not convex |
# | $\cos\theta_{ij},\ \sin\theta_{ij}$ | **trigonometric**, so not convex either |
#
# And critically these appear in **equality** constraints. An equality
# $h(x) = 0$ with nonlinear $h$ defines a curved surface, and a curved surface
# is not a convex set — even when $h$ itself is a perfectly nice function. That
# is the whole story: AC-OPF is nonconvex because the physics is an equality
# between products of unknowns.
#
# Compare with what Tutorial 03 had: $P_{ij} = B_{ij}(\theta_i - \theta_j)$,
# linear, and a linear equality defines a *flat* set, which is convex.
#
# ### Polar or rectangular?
#
# | | variables | pros | cons |
# |---|---|---|---|
# | polar | $\lvert V\rvert, \theta$ | limits are simple bounds; matches intuition | trigonometric terms |
# | rectangular | $e_i = \Re V_i,\ f_i = \Im V_i$ | **quadratic**, no trig; suits QCQP and relaxations | voltage limits become $v^2 \le e^2+f^2 \le \bar v^2$, and the lower one is nonconvex |
#
# Neither removes the nonconvexity — it is a property of the physics, not of
# the coordinates. But rectangular is where Tutorial 05 starts, because a
# *quadratic* nonconvexity is one we know how to relax.

# %% [markdown]
# ## 3. AC-OPF by hand
#
# Polar form, built straight from the equations above, with $Y$ taken from
# pandapower so there is no chance of a different network being modelled.

# %% tags=["provided"]
def ac_opf(net, *, vmin=0.94, vmax=1.10, start_vm=1.0, start_va=0.0):
    """Polar AC-OPF for a network with one ext_grid and one generator."""
    pp.runpp(net, numba=False)
    ybus = np.array(net._ppc["internal"]["Ybus"].todense())
    G, B = ybus.real, ybus.imag
    n = ybus.shape[0]
    base = net.sn_mva

    # Where things are, in ppc numbering.
    lookup = net._pd2ppc_lookups["bus"]
    slack_bus = int(lookup[net.ext_grid.bus.iloc[0]])
    gen_bus = int(lookup[net.gen.bus.iloc[0]])
    load = {int(lookup[r.bus]): (r.p_mw / base, r.q_mvar / base)
            for r in net.load.itertuples()}

    m = pyo.ConcreteModel(name="AC-OPF (polar)")
    m.N = pyo.Set(initialize=range(n))

    m.vm = pyo.Var(m.N, bounds=(vmin, vmax), initialize=start_vm)
    m.va = pyo.Var(m.N, bounds=(-np.pi / 3, np.pi / 3), initialize=start_va)
    m.p_slack = pyo.Var(bounds=(0.0, 200.0 / base), initialize=1.0)
    m.q_slack = pyo.Var(bounds=(-100.0 / base, 100.0 / base), initialize=0.0)
    m.p_gen = pyo.Var(bounds=(0.0, 100.0 / base), initialize=0.4)
    m.q_gen = pyo.Var(bounds=(-60.0 / base, 60.0 / base), initialize=0.0)

    m.reference = pyo.Constraint(expr=m.va[slack_bus] == 0.0)

    def injection_p(m, i):
        return m.vm[i] * sum(
            m.vm[j] * (G[i, j] * pyo.cos(m.va[i] - m.va[j])
                       + B[i, j] * pyo.sin(m.va[i] - m.va[j]))
            for j in m.N
        )

    def injection_q(m, i):
        return m.vm[i] * sum(
            m.vm[j] * (G[i, j] * pyo.sin(m.va[i] - m.va[j])
                       - B[i, j] * pyo.cos(m.va[i] - m.va[j]))
            for j in m.N
        )

    @m.Constraint(m.N)
    def balance_p(m, i):
        generation = (m.p_slack if i == slack_bus else 0.0) + (m.p_gen if i == gen_bus else 0.0)
        return generation - load.get(i, (0.0, 0.0))[0] == injection_p(m, i)

    @m.Constraint(m.N)
    def balance_q(m, i):
        generation = (m.q_slack if i == slack_bus else 0.0) + (m.q_gen if i == gen_bus else 0.0)
        return generation - load.get(i, (0.0, 0.0))[1] == injection_q(m, i)

    # Costs are per MW, so convert the per-unit variables back.
    m.cost = pyo.Objective(
        expr=20.0 * m.p_slack * base + 60.0 * m.p_gen * base, sense=pyo.minimize
    )
    return m


model = ac_opf(net)
record = solve(model, "NLP", duals=True)
print(record.summary())
print()
print(f"|V|      = {[round(pyo.value(model.vm[i]), 4) for i in model.N]} pu")
print(f"angle    = {[round(np.degrees(pyo.value(model.va[i])), 3) for i in model.N]} deg")
print(f"P slack  = {pyo.value(model.p_slack) * net.sn_mva:.3f} MW  @ 20 EUR/MW")
print(f"P gen    = {pyo.value(model.p_gen) * net.sn_mva:.3f} MW  @ 60 EUR/MW")
print(f"Q slack  = {pyo.value(model.q_slack) * net.sn_mva:.3f} Mvar")
print(f"Q gen    = {pyo.value(model.q_gen) * net.sn_mva:.3f} Mvar")

# %% [markdown]
# ### Validate it against a power flow
#
# The optimizer solved *its own* equations. Whether those equations describe the
# network is a separate question, and the way to answer it is to hand the
# dispatch to an independent power flow.

# %% tags=["provided"]
validation = copy.deepcopy(net)
validation.gen.loc[0, "p_mw"] = pyo.value(model.p_gen) * net.sn_mva
validation.gen.loc[0, "vm_pu"] = pyo.value(model.vm[
    int(net._pd2ppc_lookups["bus"][net.gen.bus.iloc[0]])])
validation.ext_grid.loc[0, "vm_pu"] = pyo.value(model.vm[
    int(net._pd2ppc_lookups["bus"][net.ext_grid.bus.iloc[0]])])

check = power_flow_check(validation)
print(check.summary())
model_losses = (
    pyo.value(model.p_slack) * net.sn_mva
    + pyo.value(model.p_gen) * net.sn_mva
    - net.load.p_mw.sum()
)
print(f"\nlosses: model {model_losses:.4f} MW  |  power flow {check.losses_mw:.4f} MW")
print(f"agreement: {abs(model_losses - check.losses_mw):.2e} MW")

# %% [markdown]
# ## 4. AC against DC on the same network
#
# DC discarded losses, reactive power and voltage. What did that cost?

# %% tags=["provided"]
def dc_opf_on(net):
    """The DC approximation of the same network, for comparison."""
    pp.rundcpp(net, numba=False)
    ybus = np.array(net._ppc["internal"]["Bbus"].todense())
    n = ybus.shape[0]
    base = net.sn_mva
    lookup = net._pd2ppc_lookups["bus"]
    slack_bus = int(lookup[net.ext_grid.bus.iloc[0]])
    gen_bus = int(lookup[net.gen.bus.iloc[0]])
    load = {int(lookup[r.bus]): r.p_mw / base for r in net.load.itertuples()}

    m = pyo.ConcreteModel(name="DC-OPF")
    m.N = pyo.Set(initialize=range(n))
    m.va = pyo.Var(m.N, bounds=(-np.pi / 3, np.pi / 3), initialize=0.0)
    m.p_slack = pyo.Var(bounds=(0.0, 200.0 / base))
    m.p_gen = pyo.Var(bounds=(0.0, 100.0 / base))
    m.reference = pyo.Constraint(expr=m.va[slack_bus] == 0.0)

    @m.Constraint(m.N)
    def balance(m, i):
        generation = (m.p_slack if i == slack_bus else 0.0) + (m.p_gen if i == gen_bus else 0.0)
        return generation - load.get(i, 0.0) == sum(ybus[i, j] * m.va[j] for j in m.N)

    m.cost = pyo.Objective(
        expr=20.0 * m.p_slack * base + 60.0 * m.p_gen * base, sense=pyo.minimize
    )
    return m


dc_model = dc_opf_on(copy.deepcopy(net))
dc_record = solve(dc_model, "LP")

comparison = pd.DataFrame(
    {
        "DC-OPF": {
            "class": "LP",
            "cost [EUR/h]": dc_record.objective,
            "P slack [MW]": pyo.value(dc_model.p_slack) * net.sn_mva,
            "P gen [MW]": pyo.value(dc_model.p_gen) * net.sn_mva,
            "losses [MW]": 0.0,
            "reactive power": "not modelled",
            "voltage": "assumed 1.0 pu",
        },
        "AC-OPF": {
            "class": "nonconvex NLP",
            "cost [EUR/h]": record.objective,
            "P slack [MW]": pyo.value(model.p_slack) * net.sn_mva,
            "P gen [MW]": pyo.value(model.p_gen) * net.sn_mva,
            "losses [MW]": model_losses,
            "reactive power": "modelled",
            "voltage": f"{min(pyo.value(model.vm[i]) for i in model.N):.3f}"
                       f"–{max(pyo.value(model.vm[i]) for i in model.N):.3f} pu",
        },
    }
)
display(comparison)

print(f"\nDC underestimates cost by "
      f"{(record.objective - dc_record.objective) / record.objective:.2%},")
print(f"almost exactly the {model_losses:.2f} MW of losses it does not model,")
print(f"priced at the cheap unit's {20.0:.0f} EUR/MW.")
print()
print("Note what this is NOT: the DC number is not a lower bound on the AC")
print("optimum. It happens to be below it here because losses are the dominant")
print("missing term, but DC solves DIFFERENT equations, so nothing guarantees a")
print("direction. Tutorial 05 introduces a model that DOES give a bound.")

# %% [markdown]
# ## 5. Local versus global
#
# IPOPT is an interior-point method for nonlinear programs. On a **convex**
# problem a KKT point is the global optimum. On a nonconvex one it is a *local*
# optimum, and IPOPT's `optimal` means exactly "I found a point satisfying the
# KKT conditions to tolerance" — not "no better point exists".
#
# The standard probe is multi-start: launch from many initial points and see
# whether they land in the same place.

# %% tags=["provided"]
rng = np.random.default_rng(0)
starts = [(1.0, 0.0), (0.95, -0.1), (1.08, 0.15), (0.94, 0.25), (1.10, -0.25)]
if not fast_mode():
    starts += [(float(rng.uniform(0.94, 1.10)), float(rng.uniform(-0.3, 0.3)))
               for _ in range(5)]

multistart = []
for vm0, va0 in starts:
    trial = ac_opf(copy.deepcopy(net), start_vm=vm0, start_va=va0)
    trial_record = solve(trial, "NLP")
    multistart.append({
        "start |V|": vm0, "start angle": va0,
        "termination": trial_record.termination,
        "cost": trial_record.objective if trial_record.ok else np.nan,
    })

multistart = pd.DataFrame(multistart)
display(multistart.round(5))

converged = multistart.dropna(subset=["cost"])
spread = converged["cost"].max() - converged["cost"].min()
print(f"\n{len(converged)}/{len(multistart)} starts converged")
print(f"objective spread across starts: {spread:.6f} EUR/h")
if spread < 1e-4:
    print("\nEvery start found the SAME point. That is reassuring and it is NOT a")
    print("proof: multi-start explores a finite sample of a continuous space, so")
    print("it can only ever fail to find a better optimum, never establish that")
    print("none exists. A certificate needs a LOWER BOUND, which is Tutorial 05.")
else:
    print("\nDifferent starts found different optima — the feasible set is")
    print("nonconvex and the solver reports whichever basin it fell into.")

# %% [markdown]
# ## 6. Now introduce `opf-potpourri`
#
# Everything above was written by hand, which is why it is worth reading. For a
# real network you do not want to re-derive $Y_{bus}$ indexing, per-unit
# conversion, generator capability curves and thermal limits every time.
#
# `opf-potpourri` builds Pyomo models over pandapower networks. Its pipeline:
#
# ```text
# pandapower network
#       ↓   Basemodel: deep-copy, fuse bus-bus switches, run a base power flow
# ppc tables + _pd2ppc lookups
#       ↓   create_model(): sets B, L, G, sG, D ... and active-power variables
# Pyomo ConcreteModel
#       ↓   AC mixin: admittances, |V| and angle, AC balance
# AC physics
#       ↓   OPF mixin: generation and demand limits, thermal limits
# AC-OPF
#       ↓   cost_objective: poly cost from net.poly_cost
# solve()  →  pyo_to_net  →  pandapower result tables
# ```
#
# The mixin structure is worth noticing, because Tutorial 05 extends it:
#
# ```text
# ACOPF(AC, OPF)      DCOPF(DC, OPF)      SOCBFM(BFM, OPF)   <- ours
# ```

# %% tags=["provided"]
from potpourri.models.ACOPF_base import ACOPF
from potpourri.models.cost_objective import add_poly_cost_objective

potpourri_model = ACOPF(copy.deepcopy(net))
potpourri_model.add_OPF()
add_poly_cost_objective(potpourri_model)

print("sets   :", sorted(c.name for c in potpourri_model.model.component_objects(
    pyo.Set, active=True))[:12])
print("vars   :", sorted(c.name for c in potpourri_model.model.component_objects(
    pyo.Var, active=True)))
print(f"\nthe network was deep-copied: "
      f"{potpourri_model.net is not net}")

potpourri_model.solve(solver="ipopt")
objective = pyo.value(next(potpourri_model.model.component_data_objects(
    pyo.Objective, active=True)))
print(f"\npotpourri AC-OPF cost: {objective:,.4f} EUR/h")
print(f"hand-built AC-OPF cost: {record.objective:,.4f} EUR/h")
print(f"difference: {abs(objective - record.objective):,.4f} EUR/h")

# %% [markdown]
# ## 7. Exercises
#
# ---

# %% [markdown]
# ### Exercise 4.1 — Tighten the voltage band and watch the cost
#
# **Difficulty:** Basic
#
# The AC-OPF pushed voltages to their upper limit. That is not an accident:
# higher voltage means lower current for the same power, and lower current means
# lower $I^2R$ losses.
#
# #### Your task
#
# 1. Solve the AC-OPF for several upper voltage limits.
# 2. Record cost, losses and the highest bus voltage.
# 3. Explain the trend.
#
# #### Expected result
#
# Cost should fall as the ceiling rises, and the highest voltage should sit
# *at* the ceiling every time — the constraint is binding.
#
# #### Hint
#
# `ac_opf(net, vmax=...)`.

# %% tags=["exercise"]
vmax_sweep = [1.00, 1.02, 1.05, 1.08, 1.10]

# TODO: sweep the upper voltage limit; record cost, losses and max |V|
#       in a DataFrame called `voltage_band`.
voltage_band = None

# %% tags=["solution"]
vmax_sweep = [1.00, 1.02, 1.05, 1.08, 1.10]

rows = []
for vmax in vmax_sweep:
    m = ac_opf(copy.deepcopy(net), vmax=vmax)
    rec = solve(m, "NLP")
    if not rec.ok:
        rows.append({"vmax": vmax, "cost": np.nan, "status": rec.termination})
        continue
    total_gen = (pyo.value(m.p_slack) + pyo.value(m.p_gen)) * net.sn_mva
    rows.append({
        "vmax": vmax, "cost": rec.objective,
        "losses [MW]": total_gen - net.load.p_mw.sum(),
        "max |V| [pu]": max(pyo.value(m.vm[i]) for i in m.N),
        "status": rec.termination,
    })

voltage_band = pd.DataFrame(rows).set_index("vmax")
display(voltage_band.round(5))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].plot(voltage_band.index, voltage_band["cost"], "o-", color=COLORS["feasible"])
axes[0].set_xlabel("upper voltage limit [pu]")
axes[0].set_ylabel("cost [EUR/h]")
axes[0].set_title("A wider voltage band is cheaper")
axes[1].plot(voltage_band.index, voltage_band["losses [MW]"], "o-", color=COLORS["accent"])
axes[1].set_xlabel("upper voltage limit [pu]")
axes[1].set_ylabel("losses [MW]")
axes[1].set_title("...because losses fall")
fig.tight_layout()
plt.show()

# %% tags=["validation"]
assert voltage_band is not None, "build the `voltage_band` DataFrame first"
_v = voltage_band.dropna(subset=["cost"])
assert (_v["cost"].diff().dropna() <= 1e-6).all(), (
    "relaxing a constraint cannot increase the optimal cost"
)
assert (_v["max |V| [pu]"] <= _v.index.to_numpy() + 1e-6).all(), (
    "a bus voltage exceeded its own limit"
)
print("Checks passed: cost is monotone in the limit, and no limit is violated.")

# %% [markdown]
# #### Interpretation
#
# **Why does the optimizer push voltage up, and what did DC-OPF have to say
# about any of this?**

# %% tags=["solution"]
print("ANSWER.")
print()
print("For a given power transfer P = |V| |I| cos(phi), raising |V| lowers |I|.")
print("Losses go as I^2 R, so they fall roughly as 1/|V|^2. The optimizer is not")
print("doing anything subtle -- it is buying cheaper losses with voltage, right up")
print("to the constraint.")
print()
print(f"Over this sweep, raising the ceiling from {vmax_sweep[0]:.2f} to "
      f"{vmax_sweep[-1]:.2f} pu cut losses from")
print(f"{_v['losses [MW]'].iloc[0]:.3f} to {_v['losses [MW]'].iloc[-1]:.3f} MW and cost by "
      f"{(_v['cost'].iloc[0] - _v['cost'].iloc[-1]):.2f} EUR/h.")
print()
print("DC-OPF had NOTHING to say about it. Assumption 1 fixed every voltage at")
print("1.0 pu, so voltage is not a variable, there are no voltage limits to")
print("bind, and losses are identically zero. An entire dimension of the")
print("operating decision is invisible to it.")
print()
print("This is the real cost of the DC approximation, and it is not 'a few")
print("percent of accuracy'. It is that some questions cannot be ASKED.")

# %% [markdown]
# ---
#
# ### Exercise 4.2 — Compare your model against potpourri's, honestly
#
# **Difficulty:** Intermediate
#
# Two AC-OPF models of the same network should agree. When they do not, the
# difference is almost always a modelling discrepancy rather than a solver
# problem — and finding it is the skill.
#
# #### Your task
#
# 1. Read off the voltage limits potpourri actually applied.
# 2. Rebuild the hand-written model with **those** limits.
# 3. Compare cost and dispatch.
# 4. If they still differ, list what else could differ and check one.
#
# #### Think before coding
#
# Before comparing any two formulations, what must be identical? Make the list
# first — objective, generator limits, voltage bounds, thermal limits, shunts,
# per-unit base, sign conventions.
#
# #### Hint
#
# `potpourri_model.model.Vmax` and `.Vmin` hold the limits it used.

# %% tags=["exercise"]
# TODO: align the two models and compare them.
alignment = None

# %% tags=["solution"]
# What limits did potpourri actually use?
pot = potpourri_model.model
pot_vmax = {b: pyo.value(pot.Vmax[b]) for b in pot.Bpd} if hasattr(pot, "Vmax") else {}
pot_vmin = {b: pyo.value(pot.Vmin[b]) for b in pot.Bpd} if hasattr(pot, "Vmin") else {}
print(f"potpourri Vmax: {sorted(set(round(v, 4) for v in pot_vmax.values()))}")
print(f"potpourri Vmin: {sorted(set(round(v, 4) for v in pot_vmin.values()))}")

aligned_vmax = max(pot_vmax.values()) if pot_vmax else 1.10
aligned_vmin = min(pot_vmin.values()) if pot_vmin else 0.94
aligned = ac_opf(copy.deepcopy(net), vmin=aligned_vmin, vmax=aligned_vmax)
aligned_record = solve(aligned, "NLP")

pot_objective = pyo.value(
    next(potpourri_model.model.component_data_objects(pyo.Objective, active=True))
)
alignment = pd.DataFrame(
    {
        "hand-built (aligned)": {
            "cost [EUR/h]": aligned_record.objective,
            "P slack [MW]": pyo.value(aligned.p_slack) * net.sn_mva,
            "P gen [MW]": pyo.value(aligned.p_gen) * net.sn_mva,
            "max |V| [pu]": max(pyo.value(aligned.vm[i]) for i in aligned.N),
        },
        "potpourri": {
            "cost [EUR/h]": pot_objective,
            "P slack [MW]": pyo.value(pot.pG[0]) * net.sn_mva,
            "P gen [MW]": pyo.value(pot.pG[1]) * net.sn_mva,
            "max |V| [pu]": max(pyo.value(pot.v[b]) for b in pot.B),
        },
    }
)
alignment["difference"] = (alignment.iloc[:, 0] - alignment.iloc[:, 1]).abs()
display(alignment.round(5))
print(f"\nrelative cost difference: "
      f"{alignment.loc['cost [EUR/h]', 'difference'] / pot_objective:.3%}")

# %% tags=["validation"]
assert alignment is not None, "build the `alignment` table first"
_rel = alignment.loc["cost [EUR/h]", "difference"] / abs(pot_objective)
assert _rel < 0.02, (
    f"the two AC-OPF models differ by {_rel:.2%}; that is a modelling "
    f"discrepancy worth finding, not solver noise"
)
print(f"Check passed: the two independent AC-OPF models agree to {_rel:.3%}.")

# %% [markdown]
# #### Interpretation
#
# **What has to match before two formulations can be compared at all?**

# %% tags=["solution"]
print("ANSWER. The checklist, and it is not optional:")
print()
print("  objective          same cost coefficients, same units (EUR/MW vs EUR/pu)")
print("  generator limits   P and Q, min and max, on the same elements")
print("  voltage bounds     per bus, including whether the slack is free")
print("  thermal limits     current or apparent power, and at which end")
print("  shunts and taps    present in both, or absent from both")
print("  slack treatment    which bus, and is its magnitude fixed or free")
print("  per-unit base      sn_mva, and whether costs are per MW or per pu")
print("  sign conventions   injection positive, or consumption positive")
print()
print("Skip any one of these and you will measure a difference that looks like a")
print("modelling insight and is actually a bug. This matters enormously in")
print("Tutorial 05: an apparent 'relaxation gap' between an AC model and an SOC")
print("model is only meaningful if the two are identical APART from the intended")
print("relaxation. That is why the course's SOCBFM reads its impedances through")
print("potpourri's own ppc row mapping rather than re-deriving them.")

# %% [markdown]
# ---
#
# ### Exercise 4.3 — Does `optimal` mean global? (research-style)
#
# **Difficulty:** Advanced
#
# #### Your task
#
# 1. Run the multi-start study with a wider spread of starting points.
# 2. Record how many converged, and to what.
# 3. Then answer: **what would it take to prove** that the best point found is
#    the global optimum?
#
# #### Expected result
#
# On this small, well-conditioned network you will probably find one optimum
# every time. State precisely what that does and does not establish.

# %% tags=["exercise"]
# TODO: run a wider multi-start and summarise what it can and cannot establish.
wide_multistart = None

# %% tags=["solution"]
rng = np.random.default_rng(42)
n_starts = 6 if fast_mode() else 20
rows = []
for k in range(n_starts):
    vm0 = float(rng.uniform(0.94, 1.10))
    va0 = float(rng.uniform(-0.4, 0.4))
    trial = ac_opf(copy.deepcopy(net), start_vm=vm0, start_va=va0)
    rec = solve(trial, "NLP")
    rows.append({
        "start |V|": vm0, "start angle [rad]": va0,
        "termination": rec.termination,
        "cost": rec.objective if rec.ok else np.nan,
    })

wide_multistart = pd.DataFrame(rows)
converged = wide_multistart.dropna(subset=["cost"])
distinct = converged["cost"].round(4).nunique()
print(f"{len(converged)}/{n_starts} starts converged")
print(f"distinct optima found (4 dp): {distinct}")
print(f"best found: {converged['cost'].min():,.4f} EUR/h")
print(f"spread    : {converged['cost'].max() - converged['cost'].min():.2e} EUR/h")

fig, ax = plt.subplots(figsize=(6.5, 3.4))
spread = float(converged["cost"].max() - converged["cost"].min())
if spread < 1e-9:
    # Every start landed on the same point, so there is no range to bin. Show
    # the starting points instead, which is the more informative picture: a
    # scatter of launch conditions that all funnelled to one optimum.
    ax.scatter(converged["start |V|"], converged["start angle [rad]"],
               c=COLORS["feasible"], s=40, zorder=3)
    ax.set_xlabel("starting |V| [pu]")
    ax.set_ylabel("starting angle [rad]")
    ax.set_title(f"{n_starts} scattered starts, all converging to one optimum\n"
                 f"({converged['cost'].iloc[0]:,.4f} EUR/h)")
else:
    ax.hist(converged["cost"], bins=min(20, len(converged)),
            color=COLORS["feasible"], alpha=0.8)
    ax.set_xlabel("objective found [EUR/h]")
    ax.set_ylabel("number of starts")
    ax.set_title(f"{n_starts} random starts, {distinct} distinct optima")
plt.show()

# %% tags=["validation"]
assert wide_multistart is not None, "run the wide multi-start first"
_c = wide_multistart.dropna(subset=["cost"])
assert len(_c) >= 2, "at least two starts should converge"
assert (_c["cost"] >= record.objective - 1e-4).all(), (
    "a start found a better point than the original solve — worth investigating"
)
print("Check passed: no start beat the original solution.")

# %% [markdown]
# #### Interpretation
#
# **What would it take to prove global optimality?**

# %% tags=["solution"]
print("ANSWER.")
print()
print(f"All {len(_c)} converged starts found the same objective to 4 decimal")
print("places. That is evidence and it is not proof, and the distinction is not")
print("pedantry:")
print()
print("  - Multi-start samples a FINITE set of points in a continuous space. It")
print("    can only ever FAIL to find a better optimum. It cannot establish that")
print("    none exists.")
print("  - IPOPT's 'optimal' means it found a point satisfying the KKT conditions")
print("    to tolerance. On a CONVEX problem that is global. On a nonconvex one it")
print("    is a statement about a neighbourhood.")
print("  - This network is small, lightly meshed and well conditioned. Nonconvex")
print("    AC-OPF instances with multiple local optima are documented in the")
print("    literature (Bukhsh et al. 2013 collects them), and some are tiny.")
print()
print("To PROVE it you need a LOWER BOUND on the global optimum, and then to show")
print("your feasible point attains it:")
print()
print("      z_lower_bound  <=  z_global  <=  z_your_feasible_point")
print()
print("If the two ends meet, the point is global. If they do not, the width of")
print("that interval is exactly what you do not know.")
print()
print("A convex RELAXATION produces such a lower bound, and that is the entire")
print("subject of Tutorial 05. It is also why the course spent Tutorial 02 on")
print("relaxations before ever reaching AC physics: this is the same idea, applied")
print("to a cone instead of a box.")

# %% [markdown]
# ## 8. Key takeaways
#
# 1. **AC-OPF is nonconvex because the physics is an equality between products
#    of unknowns** — bilinear voltage products and trigonometric angle terms, in
#    equality constraints.
# 2. **Polar and rectangular are coordinates, not cures.** Rectangular makes the
#    nonconvexity *quadratic*, which is what makes Tutorial 05 possible.
# 3. **DC-OPF is not a bound.** It underestimated cost here by about the size of
#    the losses it does not model, but that direction is a property of this
#    instance, not a guarantee.
# 4. **AC-OPF answers questions DC cannot ask** — voltage limits, reactive
#    dispatch, losses.
# 5. **`optimal` from a local NLP solver is a statement about a KKT point.**
#    Multi-start gives evidence; only a bound gives proof.
# 6. **Before comparing two formulations, align them.** An apparent gap is
#    usually a modelling discrepancy.
#
# ## Further reading
#
# - Carpentier, "Contribution à l'étude du dispatching économique", *Bulletin de
#   la Société Française des Électriciens*, 1962 — the original OPF.
# - Cain, O'Neill & Castillo, "History of optimal power flow and formulations",
#   FERC staff paper, 2012.
# - Bukhsh, Grothey, McKinnon & Trodden, "Local solutions of the optimal power
#   flow problem", *IEEE Trans. Power Syst.* 28(4), 2013 — small instances with
#   multiple local optima.
#
# ## Next
#
# Tutorial 05 builds a convex relaxation of these equations, proves it gives a
# lower bound, and asks when that bound is tight.
