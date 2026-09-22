# Solver guide

Every claim here was measured by `scripts/check_solvers.py` and
`scripts/check_gurobi.py` on the machine the course was built on. Run them
yourself — the numbers below are a record, not a promise about your setup.

## What the course uses

| problem class | preferred | fallback | why |
|---|---|---|---|
| LP | HiGHS (`appsi_highs`) | Gurobi, IPOPT | excellent and licence-free |
| MILP | HiGHS | Gurobi | same |
| convex QP | Gurobi (`appsi_gurobi`) | IPOPT | Pyomo's HiGHS interface rejects a quadratic objective |
| nonconvex NLP | IPOPT | — | the AC-OPF route |
| SOCP | **IPOPT** | Gurobi | see the measurement below |
| MISOCP | Gurobi | **none** | no open-source route in this stack |

`psopt_course.solvers.solver_for(problem_class)` returns the choice, and
`solve(model, problem_class)` records what actually happened.

## Three measured findings

### 1. Pyomo's HiGHS interface cannot take a quadratic objective

```
DegreeError: Highs interface does not support expressions of degree None
```

So Tutorial 01's quadratic dispatch goes elsewhere. Falling back to IPOPT is
sound rather than a downgrade: the objective is convex, so a local optimum is
the global one. The tutorial uses this as its first concrete example of
"changing the problem class forced a different solver, while the physics stayed
the same".

### 2. The legacy Gurobi interfaces return a WRONG objective on a rotated cone

On the course's SOC branch-flow model, `P² + Q² ≤ u·ℓ`:

```
gurobi          other     0.115235 MW
gurobi_direct   other     0.103718 MW
appsi_gurobi    optimal   0.098836 MW
ipopt           optimal   0.098795 MW
```

The last two agree to 0.04%; the first two report an unmapped status *and* an
objective above the optimum. A silently wrong objective from a convex program is
the worst failure mode in this course, because every downstream bound inherits
it. **Use `appsi_gurobi`, never `gurobi` or `gurobi_direct`.**

### 3. SOCP goes to IPOPT, and that is a statement about the bridge

`appsi_gurobi` reports correctly but returned `unknown`, with no feasible
solution, on half of an eight-case investment sweep of the same model. IPOPT
solved all eight, monotonically in the amount of control installed, with cone
residuals of 4e-07 throughout.

This is about Pyomo's bridge and this formulation, not about Gurobi, which is an
excellent conic solver. It remains the fallback and still owns MISOCP.

## Installing

```bash
uv sync
uv run python scripts/check_solvers.py
```

`highspy`, `clarabel`, `cvxpy` and `gurobipy` all arrive as wheels. **IPOPT does
not** — it is a system binary. Options:

```bash
conda install -c conda-forge ipopt          # simplest
# or: idaes get-extensions                   # ships a prebuilt binary
# or: build from source (COIN-OR)
```

`scripts/check_solvers.py` tells you which paths are live and which parts of the
course each one covers.

## Gurobi and licences

Gurobi is used where it is genuinely better and is **never required**.
`gurobipy` pip-installs without a licence in a size-limited mode (roughly 2000
variables and constraints), which covers every fast-mode model here but not the
full-scale capstone.

`scripts/check_gurobi.py` detects whether a licence file is present and says so,
because a size ceiling measured on a licensed machine tells an unlicensed reader
nothing. Confirmed working with a licence: rotated-cone SOCP, conic duals via
`QCPDual`, and MISOCP.

MISOCP is the single gap without a licence. It is used only for the *reference*
solve that Tutorial 08 verifies Benders against — the decomposition itself never
needs it, because fixing the master leaves a continuous SOCP. Without a licence
that one verification is skipped with an explicit message.

## Reading a termination condition

| condition | meaning |
|---|---|
| `optimal` | the solver's optimality criteria are met — **for the model as given** |
| `optimal` from IPOPT on a nonconvex problem | a locally optimal KKT point. Not a global claim |
| `infeasible` | a *proof* that no feasible point exists. Usually the data's fault |
| `unbounded` | almost always a missing constraint |
| `maxIterations`, `maxTimeLimit` | **not** convergence. Do not report the objective as an optimum |
| `other`, `unknown` | the interface could not map the solver's status. Investigate before trusting anything |

Read the termination condition before the objective. A number from a run that
terminated `infeasible` is still a float, and it will propagate into a plot
without complaint.
