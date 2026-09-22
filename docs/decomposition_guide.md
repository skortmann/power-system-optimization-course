# Decomposition guide

Which method suits which structure — and what each can actually certify.

## The comparison

| method | primarily adds | main idea | structural signal | *T* |
|---|---|---|---|---|
| branch-and-bound | nodes | split the discrete space | integer variables | 02 |
| cutting planes | constraints | cut off fractional points | MILP, convexification | 02 |
| **Benders** | cuts | project out the subproblem | few complicating *variables* | 08 |
| **Dantzig-Wolfe / column generation** | columns | represent a block by its patterns | block-angular; huge variable sets | 09 |
| Lagrangian relaxation | dual penalties | relax the coupling constraints | separable blocks | 09 |
| ADMM | coordination terms | distributed convex optimization | decentralised blocks | — |

They are not interchangeable. Each reads a *different* structural signal.

## Choosing

```text
SMALL PROBLEM
     └── solve it directly

LARGE / STRUCTURED
     ├── a few complicating variables, and fixing them makes the rest easy?
     │        └── BENDERS
     ├── an enormous set of possible patterns per block?
     │        └── COLUMN GENERATION
     ├── separable blocks joined by a few coupling constraints?
     │        └── LAGRANGIAN / ADMM
     └── nonconvex physics?
              ├── local NLP         (fast, no guarantee)
              ├── convex relaxation (a bound — and rigorous decomposition)
              └── global solver     (small instances only)
```

Guidance, not a decision tree. Most real problems show more than one signal.

## Benders and column generation are mirror images

| | Benders | column generation |
|---|---|---|
| master starts with | too **few** constraints | too **few** variables |
| adds | cuts | columns |
| subproblem returns | a dual solution | a primal column |
| subproblem asks | "what does this decision cost?" | "is there a better pattern?" |
| converges when | no cut improves the bound | no column has negative reduced cost |

Both maintain a lower and an upper bound and squeeze them. So does
branch-and-bound. That is the thread running through the whole course.

## What each subproblem class lets you certify

This is the heart of Tutorial 08, and the row that matters is the last.

| subproblem | dual information | strong duality | cut valid globally | certifies |
|---|---|---|---|---|
| LP | LP duals | automatic | yes | the LP model |
| convex NLP | Lagrange multipliers | under Slater | yes | that convex model |
| **SOCP** | conic duals | under Slater | yes | the **relaxation** |
| **nonconvex NLP** | local KKT multipliers | **not in general** | **no** | nothing global |
| logic-based | inference / logical cuts | n/a | depends on the derivation | depends |

"Global guarantee" always refers to **the formulation actually being solved**.
Benders on an SOC relaxation rigorously solves the MISOCP. Whether that
certifies the original AC problem depends on exactness, or on the gap to an
independently obtained AC-feasible point.

## Nested gaps

Four different numbers get called "the gap". Reporting them interchangeably is a
category error.

```text
1. decomposition convergence gap    UB − LB inside the decomposed model
2. convex relaxation gap            z_AC_feasible − z_SOC
3. MIP optimality gap               from the branch-and-bound solver
4. numerical feasibility tolerance   per solver
```

A Benders run can close (1) to zero while (2) stays wide. Benders has then
solved the relaxation perfectly, and the relaxation is simply not exact.

## The bound hierarchy

```text
                     AC MINLP
                       z*
                       │
          ┌────────────┴────────────┐
          │                         │
   AC-feasible candidate      SOCP relaxation
          │                         │
          ▼                         ▼
     UPPER BOUND               LOWER BOUND
          │                         │
          └────────────┬────────────┘
                       ▼
              certified interval
```

and inside the relaxation:

```text
Benders master lower bound
             │  Benders gap
Benders feasible relaxed solution
```

## Practical notes

**Feasibility cuts.** When a master decision makes the subproblem infeasible,
the dual ray gives a cut removing that decision. Tutorial 08's formulation
cannot produce one — no reactive control is still operable — which the notebook
states rather than glosses over.

**Single- vs multi-cut.** Multi-cut passes one hyperplane per scenario instead
of one aggregate, so the master learns more per iteration and usually needs
fewer. The master grows faster. Which wins depends on where the time goes.

**Cut management.** Early cuts move the bound most — Tutorial 08 measures this.
Every cut is a constraint the master carries for the rest of the run, so large
implementations prune or select.

**Warm starts.** Both methods reach the same optimum from any start; only the
path changes (Tutorial 09, Exercise 9.2). Reusing yesterday's columns or cuts is
free and saves work.

**When Benders does not help.** If the subproblem is as hard as the original, or
if there are no complicating variables, decomposition adds overhead and buys
nothing. "Benders always makes a model faster" is false, and so is "any model
can be decomposed with Benders".
