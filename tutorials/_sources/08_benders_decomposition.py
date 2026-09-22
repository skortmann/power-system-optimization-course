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
# # Tutorial 08 — Benders Decomposition, from LP to Convexified AC
#
# > **Can we exploit the structure of a hard mixed-integer power-system problem
# > by separating strategic decisions from operational optimization?**
#
# ## Where we are
#
# ```text
# ED → UC → DC-OPF → AC-OPF → SOCP → multi-period → uncertainty
#                                    → >> Benders << → column generation → capstone
#                                        ^
#                                        YOU ARE HERE
# ```
#
# This is the synthesis point of the course. It needs **all** of:
#
# ```text
# duality (T02) + MILP (T02) + AC-OPF (T04) + convex relaxation (T05) + SOCP (T05)
#                                    ↓
#                          Benders decomposition
# ```
#
# ## Learning objectives
#
# 1. Derive the Benders **optimality cut** from LP duality — not from a library;
# 2. implement the loop by hand and verify it against a monolithic solve;
# 3. track lower and upper bounds and understand what each one *is*;
# 4. run the same problem with **three** subproblem classes — LP, convex SOCP,
#    nonconvex AC — and compare what each can **certify**;
# 5. distinguish the **Benders convergence gap** from the **relaxation gap**.
#
# ## The claim this notebook tests
#
# > "Any model can be decomposed with Benders."
#
# False, and the reason is precise: Benders needs the subproblem's dual to
# describe its *global* optimum.

# %% tags=["provided"]
import copy
import itertools
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo

from psopt_course.config import fast_mode, set_seed
from psopt_course.decomposition import BoundHistory, ConvergenceTest, relative_gap
from psopt_course.networks import radial_feeder
from psopt_course.plotting import COLORS, use_course_style
from psopt_course.relaxations import SOCBFM
from psopt_course.solvers import solve

use_course_style()
set_seed()
print(f"reduced (fast) configuration: {fast_mode()}")

# %% [markdown]
# ## 1. The structure Benders looks for
#
# $$
# \min_{y \in Y,\ x}\ c^\top y + f(x, y)
# \quad\text{s.t.}\quad g(x, y) \le 0
# $$
#
# with $y$ the **complicating** decisions — few, usually discrete — and $x$ the
# operational ones. Define the **value function**
#
# $$Q(y) = \min_x \{\,f(x,y) : g(x,y) \le 0\,\}$$
#
# so the problem becomes
#
# $$\min_{y \in Y}\ c^\top y + Q(y).$$
#
# If we knew $Q$, we would be done. We do not. Benders builds an increasingly
# accurate **lower** approximation of it out of dual information:
#
# ```text
#    MASTER: choose y, using the cuts built so far
#         │
#         ▼
#    SUBPROBLEM: fix y, solve for x, read the duals
#         │
#         ▼
#    CUT: theta >= Q(y_hat) + pi^T (y - y_hat)
#         │
#         └──────────────► back to the MASTER
# ```
#
# ## 2. Deriving the cut
#
# Write the subproblem so that $y$ enters **only** the right-hand side:
#
# $$Q(y) = \min_x\ c_x^\top x \quad\text{s.t.}\quad Ax \ge b - By \quad [\pi]$$
#
# Its dual is
#
# $$Q(y) = \max_{\pi \ge 0}\ \pi^\top (b - By)
#   \quad\text{s.t.}\quad A^\top \pi \le c_x .$$
#
# Now the key step. The dual **feasible set does not depend on $y$** — $y$ only
# moves the dual objective. So any dual-feasible $\pi^k$, for *any* $y$, gives
#
# $$Q(y) \;\ge\; {\pi^k}^\top (b - By) \qquad \text{for all } y,$$
#
# by weak duality. That inequality is the **Benders optimality cut**:
#
# $$\boxed{\ \theta \;\ge\; {\pi^k}^\top\,(b - By)\ }$$
#
# and it is valid *globally*, not just near $y^k$. That is the whole trick, and
# it is why Tutorial 02's weak duality mattered.
#
# ### Where it can fail
#
# The derivation used two things: the dual feasible set is independent of $y$,
# and strong duality holds so $\pi^k$ actually describes $Q(y^k)$. For an LP
# both are automatic. For a **convex** subproblem they hold under a regularity
# condition (Slater). For a **nonconvex** subproblem the multipliers describe a
# local KKT point, and the cut can slice off feasible territory. Part 6 measures
# that.

# %% [markdown]
# ## 3. The application
#
# **Where should we install reactive-capable inverters on a distribution
# feeder?**
#
# * **Master** ($y$): binary, one per candidate bus — install Q control or not.
#   Each costs a fixed amount.
# * **Subproblem** ($x$): operational OPF. With $y$ fixed, what does the feeder
#   cost to run?
#
# The objective is **energy imported at the slack bus**, which equals load plus
# losses. Reactive control reduces losses, so it pays for itself or it does not.
#
# Crucially, $y$ enters the subproblem only through the bounds
#
# $$-Q^{max} y_k \;\le\; q_k \;\le\; Q^{max} y_k$$
#
# which is exactly the right-hand-side structure the cut derivation needs.

# %% tags=["provided"]
VMIN, VMAX = 0.90, 1.05
Q_LIMIT = 0.03           # per unit on the 10 MVA base
INVEST_COST = 0.0008     # per inverter, in the same per-unit currency
N_CANDIDATES = 3

net = radial_feeder(n_pv=N_CANDIDATES, pv_mw=0.5)
CLASS = {"linear": "LP", "soc": "SOCP", "exact": "NLP"}

print(f"feeder: {len(net.bus)} buses, {N_CANDIDATES} candidate inverter sites")
print(f"investment cost: {INVEST_COST} per site, Q capability +/-{Q_LIMIT} pu")


def build_subproblem(y, definition):
    """Operational problem with inverter k's Q capability gated by y[k].

    Returns (wrapper, pyomo model, generator keys). `y` enters ONLY through the
    right-hand side of q_upper / q_lower, which is what makes the Benders cut a
    linear function of y.
    """
    model = SOCBFM(copy.deepcopy(net), current_definition=definition)
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
    m.obj = pyo.Objective(expr=m.pG[slack])      # import at the slack bus
    return model, m, gens


def solve_subproblem(y, definition):
    model, m, gens = build_subproblem(y, definition)
    record = solve(m, CLASS[definition], duals=True)
    return model, m, gens, record


# %% [markdown]
# ## 4. Ground truth: enumerate everything
#
# With three binaries there are eight patterns. Solving all of them gives the
# exact answer, which is what Benders must reproduce. Do this *first* — an
# algorithm you cannot check is an algorithm you cannot trust.

# %% tags=["provided"]
def enumerate_all(definition):
    rows = []
    for pattern in itertools.product([0, 1], repeat=N_CANDIDATES):
        _model, m, _gens, record = solve_subproblem(list(pattern), definition)
        if not record.ok:
            rows.append({"y": "".join(map(str, pattern)), "Q(y)": np.nan,
                         "total": np.nan, "status": record.termination})
            continue
        q = pyo.value(m.obj)
        rows.append({
            "y": "".join(map(str, pattern)), "Q(y)": q,
            "total": q + INVEST_COST * sum(pattern),
            "status": record.termination,
        })
    return pd.DataFrame(rows).set_index("y").sort_values("total")


truth = {}
for definition in ("linear", "soc", "exact"):
    truth[definition] = enumerate_all(definition)
    best = truth[definition]["total"].idxmin()
    print(f"{definition:7s} ({CLASS[definition]:4s})  best y = {best}  "
          f"total = {truth[definition].loc[best, 'total']:.8f}")

display(truth["soc"].round(8))

# %% [markdown]
# ### Look at the LP level before going further

# %% tags=["provided"]
lp = truth["linear"]
print(f"LinDistFlow Q(y) values: {sorted(lp['Q(y)'].round(8).unique())}")
print(f"distinct values: {lp['Q(y)'].round(10).nunique()}")
print()
print("Every investment pattern gives the SAME operational cost. LinDistFlow")
print("sets the branch currents to zero, so it has no losses; the import at the")
print("slack is then exactly the load, whatever the inverters do.")
print()
print("So the LP subproblem's value function is CONSTANT, the cuts are flat, and")
print("the master correctly concludes that investment is pure cost. It decides")
print(f"y = {lp['total'].idxmin()} — install nothing.")
print()
print("Nothing went wrong numerically. The model simply cannot represent the")
print("quantity the investment improves. Section 7 compares the decisions.")

# %% [markdown]
# ## 5. Benders by hand
#
# The loop, written out. Note what is *not* imported: the cut.

# %% tags=["provided"]
def benders(definition, *, max_iterations=20, tolerance=1e-7, verbose=True):
    """Benders decomposition with the operational subproblem of `definition`."""
    master = pyo.ConcreteModel(name=f"master ({definition})")
    master.K = pyo.Set(initialize=range(N_CANDIDATES))
    master.y = pyo.Var(master.K, domain=pyo.Binary)
    master.theta = pyo.Var(bounds=(0.0, None))
    master.obj = pyo.Objective(
        expr=INVEST_COST * sum(master.y[k] for k in master.K) + master.theta,
        sense=pyo.minimize,
    )
    master.cuts = pyo.ConstraintList()

    history = BoundHistory(name=f"Benders ({CLASS[definition]})")
    stopper = ConvergenceTest(tolerance=tolerance, max_iterations=max_iterations)
    upper, best_y = float("inf"), None

    for _ in range(max_iterations):
        master_record = solve(master, "MILP")
        y_hat = [int(round(pyo.value(master.y[k]))) for k in master.K]
        lower = master_record.objective

        _model, m, gens, sub_record = solve_subproblem(y_hat, definition)
        if not sub_record.ok:
            # A feasibility cut would go here. This formulation cannot produce
            # an infeasible subproblem: y = 0 simply means no Q control, which
            # is still operable. Exercise 8.3 builds one that can.
            raise RuntimeError(f"subproblem failed: {sub_record.termination}")

        q_value = pyo.value(m.obj)
        candidate = INVEST_COST * sum(y_hat) + q_value
        if candidate < upper:
            upper, best_y = candidate, list(y_hat)

        # THE CUT. y enters only the RHS of q_upper/q_lower, so
        #     dQ/dy_k = (pi_upper_k + pi_lower_k) * Q_LIMIT
        # with Pyomo's sign convention for <= constraints.
        pi = [
            (m.dual[m.q_upper[k]] + m.dual[m.q_lower[k]]) * Q_LIMIT
            for k in range(len(gens))
        ]
        master.cuts.add(
            master.theta
            >= q_value + sum(pi[k] * (master.y[k] - y_hat[k]) for k in master.K)
        )

        history.record(
            lower_bound=lower, upper_bound=upper,
            master_objective=master_record.objective,
            subproblem_objective=q_value,
            n_cuts=len(master.cuts),
            master_seconds=master_record.seconds,
            subproblem_seconds=sub_record.seconds,
            note="".join(map(str, y_hat)),
        )
        if verbose:
            print(f"  it {len(history.iterations):2d}  y={''.join(map(str, y_hat))}  "
                  f"Q={q_value:.8f}  LB={lower:.8f}  UB={upper:.8f}  "
                  f"gap={history.gap:.2e}")

        stop, reason = stopper.should_stop(history)
        if stop:
            if verbose:
                print(f"  {reason}")
            break

    return history, best_y, upper


print("=== Benders with the SOC subproblem ===")
soc_history, soc_y, soc_value = benders("soc")

reference = truth["soc"]["total"].min()
reference_y = truth["soc"]["total"].idxmin()
print(f"\nBenders : y = {''.join(map(str, soc_y))}  total = {soc_value:.8f}")
print(f"truth   : y = {reference_y}  total = {reference:.8f}")
print(f"match   : {abs(soc_value - reference) < 1e-7 and ''.join(map(str, soc_y)) == reference_y}")

# %% tags=["provided"]
soc_history.plot()
plt.tight_layout()
plt.show()
display(soc_history.to_frame().round(8))
print(soc_history.summary())

# %% [markdown]
# ### What the two bounds actually are
#
# | bound | comes from | why it is a bound |
# |---|---|---|
# | **lower** | the master problem | the cuts under-estimate $Q$, so the master under-estimates the true cost |
# | **upper** | any $y$ actually evaluated | $c^\top y^k + Q(y^k)$ is achievable, so the optimum is no worse |
#
# The lower bound rises as cuts accumulate; the upper bound falls as better $y$
# are found. When they meet, the master's approximation of $Q$ is exact *where
# it matters* and the answer is proved optimal.
#
# This is the same picture as branch-and-bound in Tutorial 02 and as column
# generation in Tutorial 09. One idea, three costumes.

# %% [markdown]
# ## 6. The same problem, three subproblem classes

# %% tags=["provided"]
runs = {}
for definition in ("linear", "soc", "exact"):
    print(f"\n=== Benders with the {definition} subproblem ({CLASS[definition]}) ===")
    try:
        history, y_star, value = benders(definition, verbose=False)
        runs[definition] = {"history": history, "y": y_star, "value": value}
        print(f"  converged to y = {''.join(map(str, y_star))}, total = {value:.8f}, "
              f"{len(history.iterations)} iterations")
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {str(exc)[:160]}")
        runs[definition] = None

summary = []
for definition in ("linear", "soc", "exact"):
    run = runs[definition]
    exact_best = truth[definition]["total"].idxmin()
    summary.append({
        "subproblem": f"{definition} ({CLASS[definition]})",
        "convex": {"linear": "yes", "soc": "yes", "exact": "no"}[definition],
        "Benders y": "".join(map(str, run["y"])) if run else "—",
        "Benders total": run["value"] if run else np.nan,
        "enumerated y": exact_best,
        "enumerated total": truth[definition].loc[exact_best, "total"],
        "iterations": len(run["history"].iterations) if run else np.nan,
    })
summary = pd.DataFrame(summary).set_index("subproblem")
summary["reproduces truth"] = np.isclose(
    summary["Benders total"], summary["enumerated total"], atol=1e-7
)
display(summary.round(8))

# %% [markdown]
# ## 7. The decisions differ — and that is the point
#
# All three algorithms converged. They did not agree.

# %% tags=["provided"]
decisions = pd.DataFrame(
    {
        definition: {
            "decision y": "".join(map(str, runs[definition]["y"])) if runs[definition] else "—",
            "sites installed": sum(runs[definition]["y"]) if runs[definition] else np.nan,
            "operational cost Q(y)": (
                truth[definition].loc["".join(map(str, runs[definition]["y"])), "Q(y)"]
                if runs[definition] else np.nan
            ),
        }
        for definition in ("linear", "soc", "exact")
    }
)
display(decisions)

lp_decision = "".join(map(str, runs["linear"]["y"]))
soc_decision = "".join(map(str, runs["soc"]["y"]))
print(f"\nThe LP subproblem chose {lp_decision}; the SOC and AC subproblems chose "
      f"{soc_decision}.")
print()
print("Benders did not fail in any of the three runs — each solved the problem")
print("it was given, exactly. The LP run solved a model in which the investment")
print("has no benefit, and got the right answer to that question.")
print()
print("This is the practical lesson of the whole tutorial: the decomposition is")
print("only ever as good as the subproblem's physics.")

# %% [markdown]
# ## 8. Two gaps that are not the same gap
#
# | gap | definition | answers |
# |---|---|---|
# | **Benders convergence** | $UB_{SOC} - LB_{SOC}$ | has the algorithm converged? |
# | **relaxation** | $z^{feas}_{AC} - z^\star_{SOC}$ | is the relaxation tight? |
#
# A run can close the first to zero while the second stays wide. Benders has
# then solved the *relaxation* perfectly, and the relaxation is simply not
# exact. Reporting them interchangeably is a category error.

# %% tags=["provided"]
z_soc = runs["soc"]["value"]
z_ac = runs["exact"]["value"]

gap_bd = runs["soc"]["history"].gap
gap_relaxation = relative_gap(z_soc, z_ac)

print(f"Benders convergence gap (within the SOC model): {gap_bd:.3e}")
print(f"relaxation gap  (SOC lower vs AC feasible)    : {gap_relaxation:.3e}")
print()
print(f"    LB_SOC = {z_soc:.8f}")
print(f"    UB_AC  = {z_ac:.8f}")
print(f"    certified interval width = {abs(z_ac - z_soc):.3e}")
print()
if abs(z_ac - z_soc) < 1e-6:
    print("On this instance the SOC relaxation is tight, so the two bounds")
    print("coincide and the AC optimum is certified. That is a property of this")
    print("feeder and this objective, not a general guarantee — Tutorial 05's")
    print("Exercise 5.2 shows the relaxation failing badly when the objective")
    print("stops being monotone in the current.")

fig, ax = plt.subplots(figsize=(7.5, 3.0))
frame = runs["soc"]["history"].to_frame()
ax.plot(frame.index, frame.lower_bound, "o-", ms=4, label="$LB_{SOC}$ (Benders master)")
ax.plot(frame.index, frame.upper_bound, "s-", ms=4, label="$UB_{SOC}$ (incumbent)")
ax.axhline(z_ac, ls="--", color=COLORS["infeasible"], label="$UB_{AC}$ (AC-feasible)")
ax.set_xlabel("iteration")
ax.set_ylabel("objective")
ax.set_title("Benders gap closes inside the relaxation; the AC bound sits outside it")
ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# ## 9. What each subproblem class lets you certify
#
# | | LP / LinDistFlow | nonconvex AC | convex SOCP |
# |---|---|---|---|
# | subproblem class | LP | nonconvex QCQP | SOCP |
# | convex | yes | **no** | yes |
# | voltage magnitude | yes (linearised) | yes | yes |
# | reactive power | yes | yes | yes |
# | losses | **no** | yes | yes (relaxed) |
# | dual information | LP duals | local KKT multipliers | conic duals |
# | strong duality | automatic | **not in general** | under Slater |
# | cut valid globally | yes | **not automatically** | yes, for the relaxation |
# | certifies | the LP model | nothing global | the MISOCP relaxation |
# | AC-feasible point | no | yes, if the NLP point is feasible | not necessarily |
#
# Read the last two rows together. The SOC run certifies the **relaxation**.
# Whether that certifies the AC problem depends on exactness — which is why the
# course measures conic residuals rather than trusting objective gaps.
#
# The AC run produced the right answer here. It is not *certified*: IPOPT
# returned a KKT point of each subproblem, and multipliers from a local solution
# support the value function only locally. With a nastier feeder the cuts could
# be invalid and the master could be steered away from the true optimum, with
# nothing in the output to say so.

# %% [markdown]
# ## 10. Exercises
#
# ---

# %% [markdown]
# ### Exercise 8.1 — Derive and verify a cut by hand
#
# **Difficulty:** Intermediate
#
# A cut is only useful if it is **valid**: it must lie below $Q(y)$ everywhere,
# not just at the point it was built from.
#
# #### Your task
#
# 1. Build the cut at one $\hat y$.
# 2. Evaluate the cut and the true $Q(y)$ at **all eight** patterns.
# 3. Confirm the cut never exceeds $Q$.
# 4. Confirm it is **tight** at $\hat y$.
#
# #### Expected result
#
# Cut $\le Q$ at every pattern, with equality at $\hat y$. That is exactly what
# "supporting hyperplane of the value function" means.

# %% tags=["exercise"]
y_hat = [0, 0, 0]

# TODO: build the cut at y_hat, then check it against Q(y) at all 8 patterns.
#       Build a DataFrame `cut_check` with columns "Q(y)", "cut", "valid".
cut_check = None

# %% tags=["solution"]
y_hat = [0, 0, 0]

_model, m_hat, gens_hat, record_hat = solve_subproblem(y_hat, "soc")
q_hat = pyo.value(m_hat.obj)
pi = [
    (m_hat.dual[m_hat.q_upper[k]] + m_hat.dual[m_hat.q_lower[k]]) * Q_LIMIT
    for k in range(len(gens_hat))
]
print(f"Q(y_hat) = {q_hat:.8f} at y_hat = {''.join(map(str, y_hat))}")
print(f"cut gradient pi = {np.round(pi, 8)}")
print(f"\ncut:  theta >= {q_hat:.8f} + sum_k pi_k (y_k - {y_hat})")

rows = []
for pattern in itertools.product([0, 1], repeat=N_CANDIDATES):
    _mm, m, _g, rec = solve_subproblem(list(pattern), "soc")
    q_true = pyo.value(m.obj)
    cut_value = q_hat + sum(pi[k] * (pattern[k] - y_hat[k]) for k in range(N_CANDIDATES))
    rows.append({
        "y": "".join(map(str, pattern)),
        "Q(y)": q_true, "cut": cut_value,
        "slack Q - cut": q_true - cut_value,
        "valid": cut_value <= q_true + 1e-9,
    })

cut_check = pd.DataFrame(rows).set_index("y")
display(cut_check.round(9))

tight = cut_check.loc["".join(map(str, y_hat))]
print(f"\nat y_hat the slack is {tight['slack Q - cut']:.2e} — the cut is TIGHT there")
print(f"valid everywhere: {bool(cut_check['valid'].all())}")

# %% tags=["validation"]
assert cut_check is not None, "build the `cut_check` DataFrame first"
assert cut_check["valid"].all(), (
    "the cut must lie below Q(y) at EVERY y — an invalid cut can remove the "
    "true optimum from the master's feasible set"
)
assert abs(cut_check.loc["".join(map(str, y_hat)), "slack Q - cut"]) < 1e-8, (
    "the cut should be tight at the point it was generated from"
)
print("Checks passed: the cut supports Q(y) and is tight at its generating point.")

# %% [markdown]
# #### Interpretation
#
# **Which cut caused the largest improvement in the lower bound?**

# %% tags=["solution"]
frame = runs["soc"]["history"].to_frame()
improvement = frame["lower_bound"].diff().fillna(frame["lower_bound"])
best_iteration = improvement.idxmax()
print("ANSWER.")
print()
display(pd.DataFrame({
    "y evaluated": frame["note"],
    "lower bound": frame["lower_bound"],
    "improvement": improvement,
}).round(8))
print(f"The biggest jump came at iteration {best_iteration} "
      f"(+{improvement.max():.2e}), from evaluating "
      f"y = {frame.loc[best_iteration, 'note']}.")
print()
print("Early cuts move the bound most because the master starts knowing nothing")
print("about Q: theta is only bounded below by zero. The first cut replaces that")
print("with a real supporting hyperplane, and later cuts refine a region the")
print("master has already narrowed.")
print()
print("This is why cut SELECTION matters in large problems. A cut generated at a")
print("y the master would never choose again contributes almost nothing, and")
print("every cut is a constraint that the master must carry for the rest of the")
print("run. Pareto-optimal cut selection and cut pruning both exist for this.")

# %% [markdown]
# ---
#
# ### Exercise 8.2 — Single-cut versus multi-cut
#
# **Difficulty:** Advanced
#
# With several scenarios, $Q(y) = \sum_s p_s Q_s(y)$. You can add one aggregated
# cut per iteration, or one cut per scenario.
#
# #### Your task
#
# 1. Build three demand scenarios by scaling the feeder load.
# 2. Implement **single-cut** Benders: one cut on the weighted average.
# 3. Implement **multi-cut**: one $\theta_s$ and one cut per scenario.
# 4. Compare iterations, cuts added and total time.
#
# #### Expected result
#
# Multi-cut should need fewer iterations — it gives the master more information
# per pass — but each master solve is bigger. Which wins depends on the cost
# balance between master and subproblems.

# %% tags=["exercise"]
SCENARIOS = {"low": 0.85, "base": 1.0, "high": 1.15}
PROBABILITY = {"low": 0.3, "base": 0.4, "high": 0.3}

# TODO: implement single-cut and multi-cut Benders over the scenarios and
#       compare them in `cut_strategy`.
cut_strategy = None

# %% tags=["solution"]
SCENARIOS = {"low": 0.85, "base": 1.0, "high": 1.15}
PROBABILITY = {"low": 0.3, "base": 0.4, "high": 0.3}


def scenario_net(scale):
    scaled_net = copy.deepcopy(net)
    scaled_net.load.loc[:, "p_mw"] *= scale
    scaled_net.load.loc[:, "q_mvar"] *= scale
    return scaled_net


SCENARIO_NETS = {name: scenario_net(scale) for name, scale in SCENARIOS.items()}


def scenario_subproblem(y, scenario):
    model = SOCBFM(copy.deepcopy(SCENARIO_NETS[scenario]), current_definition="soc")
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
    record = solve(m, "SOCP", duals=True)
    pi = [
        (m.dual[m.q_upper[k]] + m.dual[m.q_lower[k]]) * Q_LIMIT
        for k in range(len(gens))
    ]
    return pyo.value(m.obj), pi, record


def stochastic_benders(multi_cut, max_iterations=20, tolerance=1e-7):
    master = pyo.ConcreteModel()
    master.K = pyo.Set(initialize=range(N_CANDIDATES))
    master.y = pyo.Var(master.K, domain=pyo.Binary)
    if multi_cut:
        master.S = pyo.Set(initialize=list(SCENARIOS))
        master.theta = pyo.Var(master.S, bounds=(0.0, None))
        expected = sum(PROBABILITY[s] * master.theta[s] for s in master.S)
    else:
        master.theta = pyo.Var(bounds=(0.0, None))
        expected = master.theta
    master.obj = pyo.Objective(
        expr=INVEST_COST * sum(master.y[k] for k in master.K) + expected,
        sense=pyo.minimize,
    )
    master.cuts = pyo.ConstraintList()

    history = BoundHistory(name="multi-cut" if multi_cut else "single-cut")
    upper, best_y = float("inf"), None
    total_seconds = 0.0

    for _ in range(max_iterations):
        rec_master = solve(master, "MILP")
        y_hat = [int(round(pyo.value(master.y[k]))) for k in master.K]
        total_seconds += rec_master.seconds

        q_values, pis = {}, {}
        for name in SCENARIOS:
            q, pi, rec = scenario_subproblem(y_hat, name)
            q_values[name], pis[name] = q, pi
            total_seconds += rec.seconds

        expected_q = sum(PROBABILITY[s] * q_values[s] for s in SCENARIOS)
        candidate = INVEST_COST * sum(y_hat) + expected_q
        if candidate < upper:
            upper, best_y = candidate, list(y_hat)

        if multi_cut:
            for name in SCENARIOS:
                master.cuts.add(
                    master.theta[name]
                    >= q_values[name]
                    + sum(pis[name][k] * (master.y[k] - y_hat[k]) for k in master.K)
                )
        else:
            averaged = [
                sum(PROBABILITY[s] * pis[s][k] for s in SCENARIOS)
                for k in range(N_CANDIDATES)
            ]
            master.cuts.add(
                master.theta
                >= expected_q
                + sum(averaged[k] * (master.y[k] - y_hat[k]) for k in master.K)
            )

        history.record(lower_bound=rec_master.objective, upper_bound=upper,
                       n_cuts=len(master.cuts))
        if history.gap <= tolerance:
            break

    return history, best_y, upper, total_seconds


rows = []
for multi in (False, True):
    history, y_star, value, seconds = stochastic_benders(multi_cut=multi)
    rows.append({
        "strategy": "multi-cut" if multi else "single-cut",
        "iterations": len(history.iterations),
        "cuts added": history.iterations[-1].n_cuts,
        "expected total": value,
        "y": "".join(map(str, y_star)),
        "solver seconds": seconds,
    })

cut_strategy = pd.DataFrame(rows).set_index("strategy")
display(cut_strategy.round(8))

# %% tags=["validation"]
assert cut_strategy is not None, "build the `cut_strategy` table first"
assert cut_strategy["y"].nunique() == 1, (
    "both cut strategies decompose the SAME problem and must reach the same y"
)
_vals = cut_strategy["expected total"]
assert abs(_vals.iloc[0] - _vals.iloc[1]) < 1e-6, (
    "both strategies must converge to the same objective"
)
print("Checks passed: same optimum from both strategies.")

# %% [markdown]
# #### Interpretation
#
# **Which scenarios generated useful cuts?**

# %% tags=["solution"]
single = cut_strategy.loc["single-cut"]
multi = cut_strategy.loc["multi-cut"]
print("ANSWER.")
print()
print(f"single-cut: {single['iterations']:.0f} iterations, "
      f"{single['cuts added']:.0f} cuts, {single['solver seconds']:.3f}s")
print(f"multi-cut : {multi['iterations']:.0f} iterations, "
      f"{multi['cuts added']:.0f} cuts, {multi['solver seconds']:.3f}s")
print()
print("Multi-cut passes the master one hyperplane PER SCENARIO instead of one")
print("averaged hyperplane, so the master learns the shape of each Q_s rather")
print("than only their weighted sum. That is strictly more information per")
print("iteration, which is why it usually needs fewer of them.")
print()
print("The cost is that the master grows three times faster in constraints and")
print("carries one theta per scenario. When the master is the expensive part --")
print("a large MILP -- single-cut can win despite needing more passes.")
print()
print("As for WHICH scenarios matter: the useful cuts come from the scenarios")
print("that BIND. Here the high-demand scenario drives the losses and therefore")
print("the value of reactive control; the low scenario's cut is nearly flat and")
print("adds little. In a large study that observation is actionable -- scenario")
print("reduction keeps the ones that shape the value function and drops the")
print("rest.")
print()
print("Note also the structure Benders exploited: with y fixed, the three")
print("scenarios are completely INDEPENDENT. They could be solved in parallel,")
print("which is the main practical reason stochastic programs get decomposed at")
print("all.")

# %% [markdown]
# ---
#
# ### Exercise 8.3 — Where does Benders stop being rigorous? (research-style)
#
# **Difficulty:** Advanced
#
# The AC run converged to the right answer. Investigate whether it was entitled
# to.
#
# #### Your task
#
# 1. For the **nonconvex** AC subproblem, generate a cut at some $\hat y$.
# 2. Test its validity against $Q_{AC}(y)$ at all eight patterns, exactly as in
#    Exercise 8.1.
# 3. Do the same for the SOC subproblem.
# 4. State the conditions under which each cut is guaranteed valid.
#
# #### Expected result
#
# Both may pass on this well-behaved feeder. The point is the *argument*: for
# the SOC subproblem validity follows from convexity and strong duality; for the
# AC one, passing a test at eight points establishes nothing.

# %% tags=["exercise"]
# TODO: test cut validity for both the SOC and the exact AC subproblems.
validity = None

# %% tags=["solution"]
def cut_validity(definition, y_hat=(0, 1, 0)):
    y_hat = list(y_hat)
    _mm, m_hat, gens_hat, rec_hat = solve_subproblem(y_hat, definition)
    q_hat = pyo.value(m_hat.obj)
    pi = [
        (m_hat.dual[m_hat.q_upper[k]] + m_hat.dual[m_hat.q_lower[k]]) * Q_LIMIT
        for k in range(len(gens_hat))
    ]
    worst = -float("inf")
    for pattern in itertools.product([0, 1], repeat=N_CANDIDATES):
        _m2, m, _g, rec = solve_subproblem(list(pattern), definition)
        q_true = pyo.value(m.obj)
        cut_value = q_hat + sum(pi[k] * (pattern[k] - y_hat[k])
                                for k in range(N_CANDIDATES))
        worst = max(worst, cut_value - q_true)      # > 0 means the cut is INVALID
    return {
        "subproblem": f"{definition} ({CLASS[definition]})",
        "convex": {"linear": "yes", "soc": "yes", "exact": "no"}[definition],
        "worst cut violation": worst,
        "valid at all 8 points": worst <= 1e-8,
    }


validity = pd.DataFrame([cut_validity(d) for d in ("soc", "exact")]).set_index("subproblem")
display(validity.round(10))

# %% tags=["validation"]
assert validity is not None, "build the `validity` table first"
assert validity[validity.index.str.startswith("soc")]["valid at all 8 points"].all(), (
    "the SOC cut must be valid — convexity plus strong duality guarantee it"
)
print("Check passed: the convex subproblem's cut is valid, as the theory says.")

# %% [markdown]
# #### Interpretation
#
# **Is the AC-Benders result certified?**

# %% tags=["solution"]
# Look the row up by prefix: the label carries the SOLVER class ("NLP"),
# not the mathematical one ("nonconvex QCQP"), and hard-coding either is
# how a rename turns into a KeyError three sections later.
ac_row = validity[validity.index.str.startswith("exact")].iloc[0]
print("ANSWER. No — and note that it PASSED the test.")
print()
print(f"The AC cut's worst violation over all eight patterns was "
      f"{ac_row['worst cut violation']:.2e},")
print("so on this feeder it behaved perfectly well. That is evidence about eight")
print("points and nothing more.")
print()
print("Why the two cases are not comparable:")
print()
print("  SOC subproblem. Convex, and Slater's condition holds because the")
print("  voltage box has a strict interior. Strong duality therefore holds, the")
print("  conic dual describes the GLOBAL optimum of Q_SOC, and the cut is a")
print("  genuine supporting hyperplane of the value function. Valid for every y,")
print("  provable in advance.")
print()
print("  AC subproblem. Nonconvex. IPOPT returns a KKT point of whichever basin")
print("  it landed in, and its multipliers describe THAT basin. If a different")
print("  y_hat leads to a different local solution, the resulting cut can lie")
print("  ABOVE the true Q_AC somewhere and remove the real optimum from the")
print("  master. Nothing in the solver output would say so.")
print()
print("So the honest description of the three runs:")
print("  linear : rigorous, for a model that cannot see the benefit")
print("  SOC    : rigorous, for the RELAXATION")
print("  AC     : a heuristic that happened to work here")
print()
print("And the certificate chain that matters:")
print()
print("      LB_SOC   <=   z*_AC   <=   UB_AC-feasible")
print(f"    {runs['soc']['value']:.8f}  <=  z*  <=  {runs['exact']['value']:.8f}")
print()
print("The left bound comes from Benders on a convex relaxation; the right from")
print("an AC-feasible point. Their width is what remains unknown. Here it is")
print(f"{abs(runs['exact']['value'] - runs['soc']['value']):.2e}, because the relaxation")
print("is tight on this feeder -- which Tutorial 05 verified with conic")
print("residuals rather than by looking at the objective.")

# %% [markdown]
# ## 11. Key takeaways
#
# 1. **The Benders cut comes from duality**, not from fitting past objective
#    values. Weak duality makes it valid for *all* $y$.
# 2. **Two bounds, squeezed together** — the same picture as branch-and-bound
#    and column generation.
# 3. **Verify against a monolithic solve** whenever you can. Benders must
#    reproduce the optimum of the model it decomposes.
# 4. **The decomposition is only as good as the subproblem's physics.** The LP
#    run was rigorous and chose differently, because LinDistFlow cannot
#    represent losses.
# 5. **Convexity is what makes the cut globally valid.** For a nonconvex AC
#    subproblem, Benders is a heuristic — however well it behaves on a given
#    network.
# 6. **The convergence gap and the relaxation gap are different numbers.**
#
# ## Further reading
#
# - Benders, "Partitioning procedures for solving mixed-variables programming
#   problems", *Numerische Mathematik* 4, 1962.
# - Geoffrion, "Generalized Benders decomposition", *J. Optimization Theory and
#   Applications* 10(4), 1972.
# - Rahmaniani, Crainic, Gendreau & Rei, "The Benders decomposition algorithm: a
#   literature review", *European J. Operational Research* 259(3), 2017.
# - Hijazi, Coffrin & Van Hentenryck, "Convex quadratic relaxations for
#   mixed-integer nonlinear programs in power systems", *Mathematical
#   Programming Computation* 9, 2017.
#
# ## Next
#
# Tutorial 09 decomposes the other way. Benders adds constraints to a master
# with too few; column generation adds variables to a master with too many.
