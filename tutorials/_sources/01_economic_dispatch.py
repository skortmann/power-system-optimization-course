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
# # Tutorial 01 — From Equations to Optimization: Economic Dispatch
#
# > **What is an optimization problem, and how does a power-system engineering
# > problem become one?**
#
# ## Where we are
#
# ```text
# >> ED << → UC → DC-OPF → AC-OPF → SOCP → multi-period → uncertainty
#           → Benders → column generation → capstone
#    ^
#    YOU ARE HERE
# ```
#
# ## Learning objectives
#
# By the end of this notebook you should be able to:
#
# 1. separate **parameters** from **decision variables**, and say why the
#    distinction is the first modelling decision you make;
# 2. write an objective, equality constraints and inequality constraints, and
#    describe the feasible set they define;
# 3. name the **problem class** of a formulation — LP, QP, MILP, NLP, SOCP —
#    *before* choosing a solver;
# 4. solve the same economic dispatch four ways: graphically, by hand, with
#    SciPy, and in Pyomo;
# 5. explain why changing a cost curve from linear to quadratic changes the
#    solver you need, even though the physical problem is untouched.
#
# ## The one message
#
# > Optimization is not "call a solver". The hard part is building the right
# > mathematical representation — and knowing what kind of object you built.

# %% tags=["provided"]
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from scipy.optimize import linprog

from psopt_course.config import fast_mode, set_seed
from psopt_course.networks import two_generator_system
from psopt_course.plotting import COLORS, plot_feasible_region, use_course_style
from psopt_course.solvers import solve, solver_for

use_course_style()
set_seed()
print(f"reduced (fast) configuration: {fast_mode()}")

# %% [markdown]
# ## 1. The physical problem
#
# Two generating units supply one demand. Nothing moves through a network yet —
# that arrives in Tutorial 03. The question is only: **how much should each unit
# produce?**

# %% tags=["provided"]
system = two_generator_system(demand=100.0)
print(f"system: {system.name},  demand = {system.total_demand:.0f} MW")
display(system.generator_table())
print(f"\nmerit order (cheapest first): {[g.name for g in system.merit_order()]}")

# %% [markdown]
# ## 2. From engineering to mathematics
#
# ### Sets
#
# $$g \in \mathcal{G} = \{\text{G1\_coal},\ \text{G2\_gas}\}$$
#
# ### Parameters — known numbers, fixed before we optimize
#
# | symbol | meaning | units |
# |---|---|---|
# | $c_g$ | marginal cost of unit $g$ | EUR/MWh |
# | $P^{min}_g,\ P^{max}_g$ | output limits | MW |
# | $D$ | demand | MW |
#
# ### Decision variables — what the optimizer chooses
#
# $$p_g \ge 0 \quad \text{active power output of unit } g \ \text{[MW]}$$
#
# **This split is the first modelling decision, and it is a choice.** Demand is
# a parameter here. In Tutorial 06 part of it becomes a variable (flexible load)
# and the model changes character completely. Nothing about the physics decided
# that — we did.
#
# ### The problem
#
# $$
# \begin{aligned}
# \min_{p} \quad & \sum_{g \in \mathcal{G}} c_g\, p_g
#   && \text{(total cost)} \\
# \text{s.t.} \quad & \sum_{g \in \mathcal{G}} p_g = D
#   && \text{(power balance)} \quad [\lambda] \\
# & P^{min}_g \le p_g \le P^{max}_g \quad \forall g
#   && \text{(unit limits)}
# \end{aligned}
# $$
#
# ### Problem class
#
# The objective is **linear** in $p$. Every constraint is **linear** in $p$.
# Therefore this is a **linear program (LP)**.
#
# That sentence is the habit this whole course is trying to build. Classify
# first; choose a solver second.

# %% [markdown]
# ## 3. Solve it four ways
#
# Four routes to the same three numbers. If they disagree, one of them is wrong,
# and finding out which is more instructive than any of them being right.
#
# ### 3.1 By hand
#
# With a linear cost the answer is pure merit order: run the cheapest unit as
# hard as you can, then the next. The only subtlety is the **minimum** output —
# G2 must produce at least 10 MW whether we want it or not.

# %% tags=["provided"]
cheap, expensive = system.merit_order()

# Fill in merit order, but respect BOTH limits: every unit must run at least at
# its floor, and whatever the cheap unit cannot take must go somewhere.
p_cheap = min(cheap.p_max, system.total_demand - expensive.p_min)
p_expensive = system.total_demand - p_cheap
assert expensive.p_min - 1e-9 <= p_expensive <= expensive.p_max + 1e-9, (
    "the remainder does not fit inside the second unit's limits"
)

hand_cost = cheap.cost(p_cheap) + expensive.cost(p_expensive)
print(f"by hand:  {cheap.name} = {p_cheap:.1f} MW, "
      f"{expensive.name} = {p_expensive:.1f} MW")
print(f"          total    = {p_cheap + p_expensive:.1f} MW "
      f"(demand {system.total_demand:.0f} MW)")
print(f"          cost     = {hand_cost:,.2f} EUR/h")

# The marginal unit is the one that is NOT pinned at a limit -- it is the only
# one free to respond to one more MW of demand.
at_limit = {
    cheap.name: not (cheap.p_min + 1e-6 < p_cheap < cheap.p_max - 1e-6),
    expensive.name: not (expensive.p_min + 1e-6 < p_expensive < expensive.p_max - 1e-6),
}
marginal = next((name for name, pinned in at_limit.items() if not pinned), None)
print(f"\n{cheap.name} sits at its {cheap.p_max:.0f} MW ceiling, so it cannot")
print(f"respond. The marginal generator is {marginal}: it is the one still free")
print("to move, so ITS cost sets the system price -- not the cheapest unit's.")

# %% [markdown]
# ### 3.2 Graphically
#
# Two variables means we can draw the whole problem. The feasible set is the
# intersection of the box $[P^{min}, P^{max}]^2$ with the balance line; the
# dashed lines are contours of constant cost. The optimum sits where the lowest
# contour still touches the feasible set — at a **vertex**.
#
# That is not a coincidence. An LP optimum can always be found at a vertex,
# which is exactly what the simplex method exploits.

# %% tags=["provided"]
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.2, 5.0))
D = system.total_demand
# The balance p1 + p2 = D as two inequalities, so the shading works.
constraints = [
    (1.0, 1.0, D, "$p_1 + p_2 \\leq D$"),
    (-1.0, -1.0, -D, "$p_1 + p_2 \\geq D$"),
    (1.0, 0.0, cheap.p_max, f"$p_1 \\leq {cheap.p_max:.0f}$"),
    (0.0, 1.0, expensive.p_max, f"$p_2 \\leq {expensive.p_max:.0f}$"),
    (-1.0, 0.0, -cheap.p_min, f"$p_1 \\geq {cheap.p_min:.0f}$"),
    (0.0, -1.0, -expensive.p_min, f"$p_2 \\geq {expensive.p_min:.0f}$"),
]
plot_feasible_region(
    constraints, (cheap.c1, expensive.c1), bounds=((0, 90), (0, 90)), ax=ax,
    title="Economic dispatch: the feasible set is a line segment",
)
ax.plot([p_cheap], [p_expensive], "*", ms=18, color=COLORS["optimum"], zorder=5,
        label="optimum")
ax.set_xlabel(f"$p_1$ — {cheap.name} [MW]")
ax.set_ylabel(f"$p_2$ — {expensive.name} [MW]")
ax.legend(fontsize=8, loc="upper right")
plt.show()

print("The equality constraint collapses the feasible REGION to a feasible")
print("SEGMENT. Every point on it serves the load; only one is cheapest.")

# %% [markdown]
# ### 3.3 With SciPy
#
# `linprog` takes the problem in matrix form. Writing it out by hand once is
# worth doing — it shows what a modelling language is saving you from.

# %% tags=["provided"]
c = np.array([cheap.c1, expensive.c1])
A_eq = np.array([[1.0, 1.0]])
b_eq = np.array([D])
bounds = [(cheap.p_min, cheap.p_max), (expensive.p_min, expensive.p_max)]

scipy_result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
print(f"scipy:    {scipy_result.x.round(2)} MW,  cost = {scipy_result.fun:,.2f} EUR/h")
print(f"          status: {scipy_result.message}")

# %% [markdown]
# ### 3.4 With Pyomo
#
# The matrix form does not survive contact with a real system — try writing
# `A_eq` for a 2000-bus network. An algebraic modelling language lets you write
# the *equations*, indexed by sets, and keeps the matrix assembly out of sight.
#
# The Pyomo vocabulary, in the order you meet it:
#
# | component | what it is |
# |---|---|
# | `ConcreteModel` | the container |
# | `Set` | an index set, e.g. the generators |
# | `Param` | known data |
# | `Var` | a decision variable |
# | `Constraint` | an equation or inequality |
# | `Objective` | what to minimise |
# | `Suffix` | a channel for solver output such as duals |

# %% tags=["provided"]
def economic_dispatch(system, quadratic: bool = False) -> pyo.ConcreteModel:
    """Build the economic-dispatch model for `system`.

    With ``quadratic=True`` the cost gains a `c2 * p**2` term. That single
    change turns an LP into a QP — same physics, different problem class.
    """
    m = pyo.ConcreteModel(name="economic dispatch")

    m.G = pyo.Set(initialize=[g.name for g in system.generators])
    by_name = {g.name: g for g in system.generators}

    m.c1 = pyo.Param(m.G, initialize={g: by_name[g].c1 for g in m.G})
    m.c2 = pyo.Param(m.G, initialize={g: by_name[g].c2 for g in m.G})
    m.p_min = pyo.Param(m.G, initialize={g: by_name[g].p_min for g in m.G})
    m.p_max = pyo.Param(m.G, initialize={g: by_name[g].p_max for g in m.G})
    m.demand = pyo.Param(initialize=system.total_demand)

    m.p = pyo.Var(m.G, domain=pyo.NonNegativeReals,
                  bounds=lambda m, g: (m.p_min[g], m.p_max[g]))

    @m.Constraint()
    def power_balance(m):
        """Generation must equal demand. Its dual is the system price."""
        return sum(m.p[g] for g in m.G) == m.demand

    if quadratic:
        m.cost = pyo.Objective(
            expr=sum(m.c1[g] * m.p[g] + m.c2[g] * m.p[g] ** 2 for g in m.G),
            sense=pyo.minimize,
        )
    else:
        m.cost = pyo.Objective(
            expr=sum(m.c1[g] * m.p[g] for g in m.G), sense=pyo.minimize
        )
    return m


model = economic_dispatch(system)
record = solve(model, "LP", duals=True)
print(record.summary())
print()
for g in model.G:
    print(f"  {g:10s} {pyo.value(model.p[g]):7.2f} MW")
# Assert rather than print: a disagreement should stop the notebook, not scroll
# past as a False that the prose below then contradicts.
assert np.isclose(record.objective, hand_cost), (
    f"hand calculation {hand_cost:,.2f} != solver {record.objective:,.2f}"
)
assert np.isclose(record.objective, scipy_result.fun), (
    f"scipy {scipy_result.fun:,.2f} != Pyomo {record.objective:,.2f}"
)
print(f"\nall four methods agree on {record.objective:,.2f} EUR/h.")

# %% [markdown]
# ## 4. The dual variable is already telling us something
#
# We asked for duals, so the balance constraint has a shadow price attached.
# Tutorial 02 derives what it means; for now, notice the number.

# %% tags=["provided"]
lam = model.dual[model.power_balance]
print(f"dual of the power balance:      lambda = {lam:,.2f} EUR/MWh")
print(f"marginal cost of {cheap.name} (at its ceiling): {cheap.c1:,.2f} EUR/MWh")
print(f"marginal cost of {expensive.name} (still free) : {expensive.c1:,.2f} EUR/MWh")

assert np.isclose(lam, expensive.c1), (
    "lambda should equal the marginal unit's cost"
)
print()
print(f"lambda equals {expensive.name}'s cost, NOT the cheapest unit's. One more MW")
print(f"of demand cannot come from {cheap.name} -- it is already at its limit -- so")
print("it must come from the next unit in merit order, at that unit's cost.")
print()
print("This is worth pausing on. The system price is set by the marginal unit,")
print("which is whichever unit is free to move. It is a property of the ACTIVE")
print("CONSTRAINT SET, not of the generation mix. Tutorial 02 derives exactly why.")

# %% [markdown]
# ## 5. Exercises
#
# ---

# %% [markdown]
# ### Exercise 1.1 — Sweep the demand and find the marginal generator
#
# **Difficulty:** Basic
#
# As demand rises, the dispatch changes and so does the price. Somewhere the
# cheap unit hits its ceiling and the expensive one takes over at the margin.
#
# #### Your task
#
# 1. Solve the dispatch for every demand in `demand_sweep`.
# 2. Record each generator's output, the total cost, and the dual $\lambda$.
# 3. Collect the results into a `DataFrame` indexed by demand.
#
# #### Expected result
#
# The cheap unit should fill up first. Once it saturates, $\lambda$ should jump
# to the expensive unit's marginal cost — a *step*, not a ramp.
#
# #### Hint
#
# `economic_dispatch(two_generator_system(demand=D))` builds the model for a
# given demand, and `solve(model, "LP", duals=True)` returns a record whose
# `.objective` you can use.

# %% tags=["exercise"]
demand_sweep = np.arange(35.0, 141.0, 5.0)

# TODO: solve the dispatch for each demand and collect the results.
#       Build a DataFrame `sweep` indexed by demand with one column per
#       generator, plus "total cost" and "lambda".
sweep = None

# %% tags=["solution"]
demand_sweep = np.arange(35.0, 141.0, 5.0)

rows = []
for demand in demand_sweep:
    case = two_generator_system(demand=float(demand))
    m = economic_dispatch(case)
    rec = solve(m, "LP", duals=True)
    row = {"demand": demand, "total cost": rec.objective,
           "lambda": m.dual[m.power_balance]}
    for g in m.G:
        row[g] = pyo.value(m.p[g])
    rows.append(row)

sweep = pd.DataFrame(rows).set_index("demand")
display(sweep.head(8).round(2))

# %% tags=["validation"]
assert sweep is not None, "build the `sweep` DataFrame first"
assert len(sweep) == len(demand_sweep)
_balance_error = (sweep[[g.name for g in system.generators]].sum(axis=1)
                  - sweep.index.to_numpy())
assert np.abs(_balance_error).max() < 1e-6, (
    f"power balance is violated by {np.abs(_balance_error).max():.3e} MW"
)
assert sweep["lambda"].nunique() >= 2, (
    "lambda never changed — the cheap unit should saturate somewhere in this sweep"
)
print("Checks passed: balance holds at every demand, and the price moves.")

# %% [markdown]
# #### Interpretation
#
# **Which constraint determines the price?**

# %% tags=["solution"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for g in system.generators:
    axes[0].plot(sweep.index, sweep[g.name], "o-", ms=3, label=g.name)
axes[0].set_xlabel("demand [MW]")
axes[0].set_ylabel("output [MW]")
axes[0].set_title("Dispatch follows merit order")
axes[0].legend(fontsize=8)

axes[1].step(sweep.index, sweep["lambda"], where="mid", color=COLORS["accent"])
axes[1].set_xlabel("demand [MW]")
axes[1].set_ylabel("$\\lambda$ [EUR/MWh]")
axes[1].set_title("The price is a step function")
fig.tight_layout()
plt.show()

switch = sweep.index[sweep["lambda"].diff().fillna(0) > 0]
print(f"lambda jumps at demand = {switch[0]:.0f} MW, which is exactly where")
print(f"{cheap.name} reaches its {cheap.p_max:.0f} MW ceiling.")
print()
print("ANSWER. The price is set by whichever constraint is ACTIVE at the margin.")
print("Below the switch it is the cheap unit's cost, because that unit can still")
print("move. Above it, the cheap unit is pinned at its upper bound and can no")
print("longer respond, so the next megawatt must come from the expensive unit.")
print("The price is a property of the BINDING CONSTRAINT SET, not of the mix.")

# %% [markdown]
# ---
#
# ### Exercise 1.2 — Quadratic costs, and what they do to the problem class
#
# **Difficulty:** Intermediate
#
# Real thermal units have convex, roughly quadratic cost curves:
#
# $$C_g(p_g) = c_{0,g} + c_{1,g}\,p_g + c_{2,g}\,p_g^2 .$$
#
# #### Your task
#
# 1. Build the dispatch with `quadratic=True`.
# 2. **Before solving**, write down what problem class it now is.
# 3. Solve it and compare the dispatch against the linear case.
# 4. Ask `solver_for` which solver the course picks for each class, and explain
#    why they differ.
#
# #### Think before coding
#
# With a *linear* cost the optimum was at a vertex and one unit was marginal.
# With a *strictly convex* cost, is the optimum still at a vertex? Should both
# units now be producing?
#
# #### Hint
#
# `two_generator_system(quadratic=True)` gives the system with `c2` set.

# %% tags=["exercise"]
# TODO: build and solve the quadratic version, then compare with the linear one.
#       Report the problem class and the solver used for each.
quadratic_system = None
quadratic_model = None

# %% tags=["solution"]
quadratic_system = two_generator_system(demand=100.0, quadratic=True)
quadratic_model = economic_dispatch(quadratic_system, quadratic=True)

print(f"LP  is solved with: {solver_for('LP')}")
print(f"QP  is solved with: {solver_for('QP')}")
print()

record_qp = solve(quadratic_model, "QP", duals=True)
print(record_qp.summary())

comparison = pd.DataFrame(
    {
        "linear cost (LP)": {g: pyo.value(model.p[g]) for g in model.G},
        "quadratic cost (QP)": {g: pyo.value(quadratic_model.p[g]) for g in quadratic_model.G},
    }
)
display(comparison.round(2))
print(f"lambda: LP {model.dual[model.power_balance]:.2f} -> "
      f"QP {quadratic_model.dual[quadratic_model.power_balance]:.2f} EUR/MWh")

# %% tags=["validation"]
assert quadratic_model is not None, "build the quadratic model first"
_total = sum(pyo.value(quadratic_model.p[g]) for g in quadratic_model.G)
assert abs(_total - 100.0) < 1e-6, f"balance violated by {abs(_total - 100.0):.3e} MW"
for _g in quadratic_model.G:
    _v = pyo.value(quadratic_model.p[_g])
    assert pyo.value(quadratic_model.p_min[_g]) - 1e-6 <= _v <= pyo.value(quadratic_model.p_max[_g]) + 1e-6
print("Checks passed: balance holds and both units are within limits.")

# %% [markdown]
# #### Interpretation
#
# **Why did the solver change, and what exactly changed about the problem?**

# %% tags=["solution"]
print("ANSWER.")
print()
print("The PHYSICS did not change at all. Two units, one balance, the same")
print("limits. What changed is the mathematical class: a linear objective became")
print("a convex quadratic one, so the LP became a QP.")
print()
print("That is what forced a different solver — and concretely so. Pyomo's HiGHS")
print("interface REFUSES a quadratic objective outright:")
print("    DegreeError: Highs interface does not support expressions of degree None")
print("so the course sends QP to Gurobi, or to IPOPT where there is no licence.")
print("Falling back to IPOPT is sound rather than a downgrade: the objective is")
print("convex, so a local optimum IS the global one.")
print()
print("The SOLUTION changed shape too. With a strictly convex cost the optimum is")
print("no longer at a vertex: both units share the load so that their MARGINAL")
print("costs are equal. That is the classical equal-incremental-cost rule, and it")
print("falls straight out of the KKT conditions in Tutorial 02.")

marginal = {
    g: pyo.value(quadratic_model.c1[g]) + 2 * pyo.value(quadratic_model.c2[g])
    * pyo.value(quadratic_model.p[g])
    for g in quadratic_model.G
}
print()
print("marginal cost at the QP optimum:")
for g, mc in marginal.items():
    print(f"  {g:10s} {mc:7.3f} EUR/MWh")
print(f"  spread: {max(marginal.values()) - min(marginal.values()):.3e} EUR/MWh")

# %% [markdown]
# ---
#
# ### Exercise 1.3 — Make it fail, then diagnose it
#
# **Difficulty:** Intermediate
#
# A model that cannot fail teaches nothing about failure. Here you will build an
# infeasible dispatch deliberately and read what the solver says.
#
# #### Your task
#
# 1. Ask for a demand larger than the total installed capacity.
# 2. Solve it and inspect the **termination condition** rather than the objective.
# 3. Explain in one sentence whose fault the failure is.
#
# #### Expected result
#
# The solver should report infeasibility. It has not malfunctioned — it has
# correctly proved that no dispatch exists.
#
# #### Hint
#
# `system.check()` raises before you ever reach the solver. Build the model by
# hand to get past it, or catch the exception.

# %% tags=["exercise"]
# TODO: construct an infeasible instance and report the termination condition.
infeasible_record = None

# %% tags=["solution"]
total_capacity = system.total_capacity
print(f"installed capacity: {total_capacity:.0f} MW")

try:
    two_generator_system(demand=total_capacity + 50.0)
except ValueError as exc:
    print(f"\nthe DATA check caught it first:\n  {exc}")

# Build it anyway, bypassing the guard, to see what the solver does.
impossible = two_generator_system(demand=100.0)
impossible.demand[0] = total_capacity + 50.0
bad_model = economic_dispatch(impossible)
infeasible_record = solve(bad_model, "LP")
print(f"\nsolver termination: {infeasible_record.termination}")
print(f"record.ok         : {infeasible_record.ok}")

# %% tags=["validation"]
assert infeasible_record is not None, "solve the infeasible instance first"
assert not infeasible_record.ok, (
    "this instance should NOT be solvable — check that demand really exceeds capacity"
)
print("Check passed: the solver correctly refused an impossible problem.")

# %% [markdown]
# #### Interpretation
#
# **Whose fault is an infeasible model?**

# %% tags=["solution"]
print("ANSWER. Almost never the solver's.")
print()
print("'Infeasible' is a PROOF, not a failure: the solver has established that no")
print("point satisfies all the constraints simultaneously. Here that is obviously")
print("true, because we asked 140 MW of plant to serve 190 MW of load.")
print()
print("The useful habit is to read the termination condition BEFORE the objective.")
print("An objective value from a run that terminated 'infeasible' is meaningless,")
print("and it is still a float that will happily propagate into a plot.")
print()
print("Tutorial 03 meets the harder version of this, where the network rather than")
print("the capacity makes a problem infeasible, and the cause is much less obvious.")

# %% [markdown]
# ## 6. Key takeaways
#
# 1. **Modelling is a sequence of choices**, starting with what is a parameter
#    and what is a variable. Nothing in the physics makes that choice for you.
# 2. **Classify before you solve.** Linear objective + linear constraints = LP.
#    Convex quadratic objective = QP. The class determines which algorithms
#    apply, and only then which software.
# 3. **The class can change without the physics changing.** Adding $c_2 p^2$
#    kept the same two generators and the same balance, and still forced a
#    different solver.
# 4. **An LP optimum sits at a vertex; a strictly convex QP optimum generally
#    does not.** That is why the LP dispatched one unit at its limit and the QP
#    shared the load at equal marginal cost.
# 5. **The dual of the power balance is the price, set by the marginal unit** —
#    the one still free to move, which need not be the cheapest one. Tutorial 02
#    proves it.
# 6. **Read the termination condition first.** An objective from an infeasible
#    run is a number without a meaning.
#
# ## Further reading
#
# - Wood, Wollenberg & Sheblé, *Power Generation, Operation and Control*, 3rd
#   ed., Wiley 2013 — chapters 3–4 are the classical treatment of economic
#   dispatch and the equal-incremental-cost rule.
# - Boyd & Vandenberghe, *Convex Optimization*, CUP 2004 — chapters 1 and 4 for
#   problem classes.
# - Bynum et al., *Pyomo — Optimization Modeling in Python*, 3rd ed., Springer
#   2021.
#
# ## Next
#
# Tutorial 02 asks what the dual variable actually *is*, derives the KKT
# conditions, and then adds the on/off decisions that turn this LP into a MILP.
