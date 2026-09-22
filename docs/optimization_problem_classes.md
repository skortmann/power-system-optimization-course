# Optimization problem classes

Naming the class is the step this course keeps insisting on, because the class
— not the application — determines which algorithms apply.

```text
Optimization
│
├── Continuous
│   ├── LP     linear objective, linear constraints          convex
│   ├── QP     quadratic objective, linear constraints       convex IFF the form is PSD
│   ├── QCQP   quadratic constraints                          convex only in special cases
│   ├── SOCP   second-order cone constraints                  convex by construction
│   ├── SDP    positive-semidefinite matrix variables         convex by construction
│   └── NLP    general smooth nonlinear                        convex or not
│
└── Discrete
    ├── MILP    LP   + integer variables
    ├── MIQP    QP   + integer variables
    ├── MISOCP  SOCP + integer variables
    └── MINLP   NLP  + integer variables
```

## An acronym is not a convexity claim

This is the trap. Two of these classes are convex *by construction*; the rest
depend on the data.

| class | convex? |
|---|---|
| LP | always |
| SOCP, SDP | always — the cone is convex by definition |
| QP | **only if** the quadratic form is positive semidefinite |
| QCQP | only if every constraint's form is PSD *and* signed correctly |
| NLP | only if the objective and the feasible set both are |
| anything MI- | **never**, as a whole — the integrality itself is nonconvex |

`min x'Qx` is a convex QP when `Q ⪰ 0` and a nonconvex one otherwise, and the
acronym is the same either way.

## Where the course's models land

| model | class | convex | where |
|---|---|---|---|
| economic dispatch, linear cost | LP | yes | T01 |
| economic dispatch, quadratic cost | QP | yes | T01 |
| unit commitment | MILP | no | T02, T09 |
| DC-OPF | LP | yes | T03 |
| AC-OPF (polar) | NLP | **no** | T04 |
| branch-flow model, exact | QCQP | **no** | T05 |
| SOC branch-flow relaxation | SOCP | yes | T05, T08, T10 |
| LinDistFlow | LP | yes | T08, T10 |
| chance-constrained dispatch | LP or SOCP | yes | T07 |
| inverter siting + SOC operation | MISOCP | no | T08 |

## Why the class decides the algorithm

| class | typical algorithm | what "optimal" means |
|---|---|---|
| LP | simplex, interior point | global |
| convex QP / SOCP / SDP | interior point | global |
| MILP / MISOCP | branch-and-bound / cut | global, once the gap closes |
| convex NLP | interior point, SQP | global |
| **nonconvex NLP** | interior point, SQP | **local** — a KKT point |
| MINLP | spatial branch-and-bound, outer approximation | global only with a global solver |

The row that causes the most trouble is the second-to-last. IPOPT reports
`optimal` for a nonconvex AC-OPF exactly as it does for a convex problem, and
the word means something weaker. Tutorial 04 measures this.

## Relaxation versus approximation

| | relaxation | approximation |
|---|---|---|
| changes | the feasible **set** | the **equations** |
| relation | `F ⊆ F_relax` | none |
| bound | yes: `z_relax ≤ z*` | **no** |
| example | SOC branch flow | DC power flow, LinDistFlow |

Both can look like "a simpler model". Only one gives a bound. Tutorial 05 makes
this the centre of the notebook because it is the most common confusion in the
subject.
