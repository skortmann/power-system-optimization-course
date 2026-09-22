# From Economic Dispatch to Large-Scale Grid Optimization

### Hands-On Mathematical Optimization for Power Systems with Python, Pyomo and opf-potpourri

A ten-tutorial course taking you from a two-generator dispatch to certified
bounds on a nonconvex AC optimal power flow — building every formulation from
the equations before reaching for a library.

```text
economic dispatch → duality & MILP → DC-OPF → AC-OPF → convex relaxation
   → multi-period → uncertainty → Benders → column generation → capstone
```

Each tutorial comes in two versions: an **exercise** notebook with scaffolded
tasks, and a fully worked **solution** notebook. Both are generated from one
tagged source, so they cannot drift apart.

---

## The course

| # | Tutorial | Optimization concept | Problem class | Power-system problem |
|---|---|---|---|---|
| 01 | [Economic dispatch](tutorials/01_economic_dispatch_exercise.ipynb) | variables, objectives, feasible sets | LP, QP | merit-order dispatch |
| 02 | [Duality, MILP, relaxation](tutorials/02_duality_milp_relaxations_exercise.ipynb) | duality, KKT, branch-and-bound | LP, MILP | unit commitment |
| 03 | [DC optimal power flow](tutorials/03_dc_optimal_power_flow_exercise.ipynb) | network constraints, PTDF, LMP | LP | congestion and nodal prices |
| 04 | [AC optimal power flow](tutorials/04_ac_optimal_power_flow_exercise.ipynb) | nonconvexity, local optima | nonconvex NLP | AC network operation |
| 05 | [Convex and SOC relaxations](tutorials/05_convex_and_soc_relaxations_exercise.ipynb) | relaxation vs approximation, conic duality | SOCP | relaxed AC-OPF, bounds |
| 06 | [Multi-period and storage](tutorials/06_multiperiod_storage_flexibility_exercise.ipynb) | intertemporal coupling | LP, MILP | BESS, flexibility |
| 07 | [Chance constraints](tutorials/07_chance_constrained_optimization_exercise.ipynb) | individual vs **joint** reliability | LP, SOCP | uncertain PV and demand |
| 08 | [Benders decomposition](tutorials/08_benders_decomposition_exercise.ipynb) | LP / conic duality, cuts, bounds | MILP + LP/NLP/SOCP | inverter siting with AC operation |
| 09 | [Column generation](tutorials/09_column_generation_exercise.ipynb) | Dantzig-Wolfe, pricing, reduced cost | LP + MILP pricing | unit commitment by schedules |
| 10 | [Scalable grid optimization](tutorials/10_scalable_grid_optimization_exercise.ipynb) | method selection, certified gaps | all of the above | distribution-grid capstone |

---

## What makes this course different

**It measures instead of asserting.** Several results contradict the folklore,
and a few contradicted what I expected while writing them:

- DC power-flow angle differences **shrink** as load grows on the three-bus
  system (5.35° → 3.82°), because extra demand is served locally rather than
  transported. The exercise asks you to predict the trend first.
- A correctly solved convex relaxation produces an operating point implying
  **6.79 MW of losses on a feeder whose real losses are 0.099 MW** — a factor of
  69 — because the objective stopped being monotone in the current.
- Three chance constraints at ε = 0.05 each yield a **joint** violation of
  0.1012, twice the worst individual rate.
- In Benders, the LP subproblem decides to install **nothing** while the SOC and
  AC subproblems install two inverters. The LP run is rigorous and wrong.
- Dantzig-Wolfe's bound comes out **equal** to the LP relaxation — the
  integrality property, reported rather than hidden.

**Two threads run through everything.** *Bounds*: branch-and-bound, convex
relaxation, Benders and column generation are one idea in four costumes.
*Relaxation is not approximation*: one gives a bound, the other gives nothing,
and students arrive with this backwards.

**Nothing is a black box.** The Benders cut is derived from weak duality, not
imported. The pricing problem is built. `opf-potpourri` appears in Tutorial 04 —
*after* you have written an AC-OPF by hand, and its answer agrees with yours to
0.000%.

---

## Install

```bash
git clone https://github.com/skortmann/power-system-optimization-course
cd power-system-optimization-course
uv sync
uv run python scripts/check_solvers.py     # what can this machine run?
uv run jupyter lab
```

Python 3.12 (pinned: `opf-potpourri` requires `<3.13`).

### Solvers

Most arrive as wheels. **IPOPT does not** — it is a system binary and roughly
half the course needs it:

```bash
conda install -c conda-forge ipopt     # or: idaes get-extensions
```

| class | preferred | fallback |
|---|---|---|
| LP, MILP | HiGHS | Gurobi |
| convex QP | Gurobi | IPOPT |
| nonconvex NLP | IPOPT | — |
| SOCP | IPOPT | Gurobi |
| MISOCP | Gurobi | none |

Gurobi is used where it is genuinely better and is **never required**; every row
but MISOCP has an open-source path, and MISOCP is only used for a verification
solve that is skipped with an explicit message. `docs/solver_guide.md` records
three measured solver findings, including one where a legacy Gurobi interface
returns a *silently wrong objective* on a rotated cone.

### Fast mode

```bash
PSOPT_FAST=1 uv run jupyter lab
```

Shrinks horizons, scenarios and iteration caps. **Never** changes a formulation.

---

## Repository

```text
tutorials/
├── _sources/                  tagged jupytext sources — EDIT THESE
├── NN_*_exercise.ipynb        generated: scaffolded tasks
└── NN_*_solution.ipynb        generated: worked, with outputs

src/psopt_course/
├── config.py                  fast mode, seeds, paths
├── solvers.py                 problem class -> solver, with a solve record
├── relaxations.py             SOC branch-flow model, extending opf-potpourri
├── networks.py                the recurring test systems
├── uncertainty.py             scenarios, chance constraints, validation
├── decomposition.py           bound bookkeeping for Benders and column generation
├── validation.py              residuals, power-flow checks, limit violations
├── metrics.py                 the four different "gaps", kept apart
└── plotting.py                figures that clarify rather than decorate

docs/       course_overview, mathematical_background, optimization_problem_classes,
            solver_guide, decomposition_guide, literature, instructor_guide,
            student_guide, glossary
scripts/    check_solvers, check_gurobi, build_exercise_notebooks
tests/      including a check that the branch-flow model reproduces a power flow
```

**Edit `tutorials/_sources/*.py`, never the `.ipynb`.** Both notebooks are
generated:

```bash
uv run python scripts/build_exercise_notebooks.py
```

---

## The SOC branch-flow extension

`psopt_course.relaxations` extends `opf-potpourri` using the library's own mixin
pattern — `SOCBFM(BFM, OPF)` mirrors its `ACOPF(AC, OPF)` — so it inherits the
whole pandapower pipeline and adds only the physics the library does not ship.

It is validated rather than asserted. With every injection fixed, the exact
branch-flow model **reproduces a pandapower power flow to 3e-09 per unit** and
matches its losses to six decimals. On a radial feeder with a loss objective the
relaxation comes out tight (max cone residual 4e-07) and the bound
`z_SOC ≤ z_BFM` holds. A constant objective breaks exactness — the Farivar–Low
monotonicity condition failing visibly — and there is a test for that too.

Why a new model when `potpourri.research.lin_opf.socbfm` exists: it is in
**neither** the PyPI wheel **nor** the public `v0.7.0` tag, checked against both.
See `docs/course_overview.md` §2.3.

---

## Documentation

| file | what it is for |
|---|---|
| [`course_overview.md`](docs/course_overview.md) | the design record, including what reconnaissance changed |
| [`mathematical_background.md`](docs/mathematical_background.md) | enough to start T01 without a separate textbook |
| [`optimization_problem_classes.md`](docs/optimization_problem_classes.md) | LP/QP/SOCP/MILP…, and why an acronym is not a convexity claim |
| [`solver_guide.md`](docs/solver_guide.md) | what to use, what breaks, three measured findings |
| [`decomposition_guide.md`](docs/decomposition_guide.md) | which method for which structure, and what each certifies |
| [`literature.md`](docs/literature.md) | foundational papers, grouped by where the course uses them |
| [`instructor_guide.md`](docs/instructor_guide.md) | timings, sticking points, discussion questions |
| [`student_guide.md`](docs/student_guide.md) | how to work through an exercise |
| [`glossary.md`](docs/glossary.md) | precise definitions, pointing at where each idea is built |

---

## The one thing to carry away

Faced with a new problem, the first question is not

> "which solver should I use?"

but

> **"what mathematical structure does this problem have, and which parts of it
> can I exploit?"**

---

## Licence

MIT, for the code and the notebooks. Third-party libraries and data carry their
own terms; `opf-potpourri` is MIT, and PGLib-OPF is not redistributed here.
