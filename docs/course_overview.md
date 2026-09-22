# Course overview

*From Economic Dispatch to Large-Scale Grid Optimization —
Hands-On Mathematical Optimization for Power Systems with Python, Pyomo and opf-potpourri*

This document is the design record. It is written before the notebooks and
updated when an experiment contradicts the plan, so it should always describe
what the course actually does rather than what it was hoped to do.

---

## 1. The argument the course makes

Most optimization teaching in power systems collapses four distinct layers into
one. Students learn "AC-OPF is solved with IPOPT" and carry that sentence
around as if it were a definition. It is four statements glued together, and
each of them can be true or false independently:

```
Physical system            an AC network with generators, loads and limits
      ↓
Mathematical formulation   polar AC-OPF, rectangular AC-OPF, branch-flow model
      ↓
Problem class              nonconvex NLP  (or MINLP, or — after relaxation — SOCP)
      ↓
Numerical algorithm        interior point  →  the software called IPOPT
```

The course's whole structure is an attempt to keep these apart. Every tutorial
states the problem class explicitly before it selects a solver, and the
misconception list in §9 is mostly a list of layer confusions.

The second argument is about **bounds**. Branch-and-bound, LP relaxation,
convex relaxation, Benders decomposition and column generation are usually
taught as five unrelated techniques. They are all doing the same thing: keeping
a lower bound and an upper bound on the same number and squeezing them
together. Making that one thread visible is worth more than any individual
algorithm.

---

## 2. What the reconnaissance established

Phase 1 of the brief asked for a software review before implementation. It
changed four planning assumptions, so they are recorded here rather than buried.

### 2.1 `opf-potpourri` comes from the public GitHub tag

Development happens on an IAEW GitLab instance that external readers cannot
reach, so the course depends on the **public mirror** at a pinned tag:

```toml
"opf-potpourri @ git+https://github.com/RWTH-IAEW/opf-potpourri@v0.7.0"
```

A tag, not a branch. Tutorial 04 reads this library's internals and later
tutorials compare against its models; tracking `main` would silently change
what students see between semesters. The identically-numbered PyPI release
carries the same package contents if a wheel is preferred, but the public
repository is the citable source and is what the course installs.

Import name is `potpourri`, not `opf_potpourri`.

### 2.2 Python is pinned to 3.12 by the dependency, not by preference

`opf-potpourri` 0.7.0 declares `requires-python = ">=3.10,<3.13"`. The brief
preferred 3.12 anyway, but it is worth knowing this is now forced: 3.13 does not
resolve.

### 2.3 An SOC branch-flow model already exists — and the course still needs its own

`potpourri.research.lin_opf.socbfm` implements the Farivar–Low / Jabr
second-order cone relaxation of the branch-flow model, in Pyomo, with the
exactness conditions of Gan, Li, Topcu & Low (2015) named in its docstring.
That is a genuine asset and Tutorial 05 will read it.

It is **not** usable as the course's relaxation, for three reasons, the last of
which is decisive:

1. **Its voltage link is an approximation.** The module lifts to `u = |V|²` and
   `ℓ = |I|²` correctly, but links `u` back to the shared `v` variable through a
   first-order Taylor expansion `u_i ≈ V̂_i(2v_i − V̂_i)` around a base AC
   solution. The SOC relaxation is rigorous *in the lifted space*, but a course
   that teaches `z_SOC ≤ z_AC` as a certified bound cannot rest that claim on a
   linearisation around a point.
2. **It solves through Gurobi only.** `LinOPFBase.solve` calls Gurobi directly,
   with no fallback.
3. **It is not in any published release.** `potpourri.research` is absent from
   both the PyPI wheel and the public GitHub tag `v0.7.0` — checked directly
   against both. It exists only in a local working tree. A course that anyone
   outside one machine is expected to run cannot depend on it at all.

So the course builds its **own** clean, self-contained SOC-BFM — documented in
Tutorial 05, code in `psopt_course.relaxations`. Where a local checkout of the
research module happens to be present, Tutorial 05 reads it as a comparison;
nothing depends on it. This is the route the brief's amendment anticipated:
"Add a small educational SOCP extension … Keep the new formulation isolated and
documented."

### 2.4 Solver reality on an open-source-only machine

`scripts/check_solvers.py` builds one tiny instance of each class the course
needs and solves it. Measured, not assumed:

| Problem class | Preferred | Open-source fallback | Status |
|---|---|---|---|
| LP | HiGHS | — (HiGHS *is* the open one) | works |
| MILP | HiGHS | — | works |
| convex QP | Gurobi | IPOPT | works |
| nonconvex NLP | IPOPT | — | works |
| SOCP (+ duals) | Gurobi (`QCPDual`) | IPOPT as QCQP, or CVXPY→Clarabel | works |
| **MISOCP** | Gurobi | **none in this stack** | works, licence-dependent |

**Gurobi is preferred where it is genuinely better, and never required.** The
students this course is written for have a licence. For convex QP, SOCP and
especially MISOCP Gurobi is the right tool: it takes the rotated cone natively,
returns conic duals through `QCPDual` — which is precisely what Tutorial 08
needs to build Benders cuts — and solves the mixed-integer conic problem that
has no open-source equivalent here. `psopt_course.solvers` falls back
automatically when Gurobi is absent, and `solve()` treats a Gurobi size-limit
refusal as a reason to fall back rather than to fail.

MISOCP is the one gap. It is used only for the **reference monolithic solve**
that Tutorial 08 checks Benders against — the decomposition itself never needs
it, because fixing the master leaves a *continuous* SOCP. Without a licence that
one verification result is skipped and the tutorial says so explicitly.

**Licence caveat, measured.** `scripts/check_gurobi.py` found a licence file on
the development machine, so its 5,000-variable probe says nothing about an
external reader. An unlicensed `pip install gurobipy` is capped near 2,000
variables and constraints. Fast mode stays well inside that; the full-scale
capstone does not.

Two further findings worth carrying into the notebooks:

**Pyomo's HiGHS interface rejects a quadratic objective.** It raises
`DegreeError: Highs interface does not support expressions of degree None`. So
the quadratic-cost dispatch in Tutorial 01 cannot use the same solver as the
linear one: it goes to Gurobi, or to IPOPT without a licence. Falling back to
IPOPT is sound rather than a quiet downgrade — the objective is convex, so a
local optimum *is* the global one.

Tutorial 01 says all of this out loud, because it is the first concrete example
of the course's central distinction: changing the cost curve from linear to
quadratic changed the **problem class**, and that is what forced a different
**solver**. The physical problem did not change at all.

**There are three independent conic routes and they agree.** On the check
problem, CVXPY+Clarabel and Pyomo-QCQP+IPOPT both return `-1.118034`, and Gurobi
solves the rotated-cone form with duals available. All three are used
deliberately:

- **Gurobi** recognises the rotated cone natively and exposes `QCPi`, the dual
  of the quadratic constraint. Tutorial 08's conic Benders cuts are built from
  these.
- **Pyomo + IPOPT** keeps the SOC formulation in the same modelling language as
  the rest of the course, so students read it beside the AC-OPF. Convexity makes
  its local optimum global.
- **CVXPY + Clarabel** is a true conic interior-point method, kept as the
  licence-free route to conic duals.

Tutorial 05 solves the same relaxation more than one way and checks the answers
agree. That agreement is itself a lesson: a rotated cone written as a convex
quadratic constraint, and the same cone handed to a conic solver, are the same
mathematics expressed in two notations.

---

## 3. The ten tutorials

| # | Tutorial | Optimization concept | Problem class | Power-system problem |
|---|---|---|---|---|
| 01 | Economic dispatch | variables, objectives, feasible sets | LP, QP | merit-order dispatch |
| 02 | Duality, MILP, relaxation | duality, KKT, branch-and-bound, LP relaxation | LP, MILP | unit commitment |
| 03 | DC optimal power flow | network constraints, PTDF, LMP | LP | congestion and nodal prices |
| 04 | AC optimal power flow | nonlinearity, nonconvexity, local optima | nonconvex NLP | AC network operation |
| 05 | Convex and SOC relaxations | relaxation vs approximation, conic duality | SOCP | relaxed AC-OPF, bounds |
| 06 | Multi-period and storage | intertemporal coupling | LP / MILP / NLP | BESS, flexibility, curtailment |
| 07 | Uncertainty and chance constraints | CC, joint CC, risk allocation | LP / SOCP | uncertain PV and demand |
| 08 | Benders decomposition | LP / conic duality, cuts, bounds | MILP + LP/NLP/SOCP | planning with AC operation |
| 09 | Column generation | Dantzig-Wolfe, pricing, reduced cost | LP + MILP pricing | unit commitment by schedules |
| 10 | Scalable grid optimization | method selection, certified gaps | all of the above | distribution-grid capstone |

### The recurring system

One example evolves through the whole course, so the mathematics changes while
the physical problem stays recognisable:

```
2 generators, single bus      T01  economic dispatch
      ↓ discrete on/off       T02  unit commitment
      ↓ a network             T03  3-bus DC-OPF
      ↓ AC physics            T04  3-bus AC-OPF
      ↓ relax the nonconvexity T05 SOC-BFM on a radial feeder
      ↓ time                  T06  24 h with storage
      ↓ uncertainty           T07  scenarios and chance constraints
      ↓ structure             T08/09 decomposition
      ↓ realism               T10  SimBench feeder, PGLib benchmark
```

Bigger systems (IEEE cases via PGLib, SimBench distribution feeders) enter only
once the formulation being tested is already understood on three buses.

---

## 4. When `opf-potpourri` appears, and why not sooner

```
T01 – T03   manual Pyomo only. Students build economic dispatch, unit
            commitment and DC-OPF from the equations.
T04         manual AC-OPF on 3 buses FIRST. Only then is potpourri introduced,
            by reading its architecture — network → model → Pyomo components →
            solver → result tables — and comparing its answer with the
            hand-built one.
T05 – T10   both, chosen per objective. potpourri where realism matters
            (multi-period, SimBench, PGLib, diagnostics); explicit Pyomo where
            the formulation IS the lesson.
```

The rule: a library is introduced after the thing it abstracts is understood,
never before.

`potpourri.diagnostics` is used throughout from T04 on. It maps constraint
violations back to `pandapower` elements — bus, line, trafo, sgen — which is
what §10 of the brief asks for when it says students should not have to
interpret `c[48193]`.

---

## 5. The bound thread

Stated once here, then reused in five tutorials:

```
                       LOWER BOUND                  UPPER BOUND
T02  unit commitment   LP relaxation                integer-feasible solution
T05  AC-OPF            SOC relaxation               AC-feasible point
T08  Benders           master problem               feasible master + subproblem
T09  column generation restricted master + pricing  (LP relaxation only)
T10  capstone          SOC-Benders                  AC-feasible recovery
```

And the distinction that the brief's amendment makes central — there are
**nested** gaps, which must never be reported interchangeably:

```
1. Benders convergence gap     UB_SOC − LB_SOC     "did the algorithm converge?"
2. Relaxation gap              z_AC_feas − z_SOC   "is the relaxation tight?"
3. MIP optimality gap          from the MILP solver
4. Feasibility tolerance       numerical, per solver
```

A Benders run can close gap 1 to zero while gap 2 stays wide. That is not a
failure of Benders; it is a property of the relaxation. Tutorial 08 constructs
a case where this happens.

---

## 6. Approximation is not relaxation

The single most confused pair in the subject, so it gets a figure and a table:

| | Approximation | Relaxation |
|---|---|---|
| What changes | the equations | the feasible set |
| Direction | different physics | enlarged set, `F ⊆ F_relax` |
| Example | DC power flow | SOC relaxation of the branch-flow model |
| Gives a bound? | **no** | yes: `z_relax ≤ z*` for minimization |
| Solution usable? | usually, approximately | not necessarily physically realisable |

DC-OPF is an *approximation* and gives no bound on the AC optimum. The SOC-BFM
is a *relaxation* and gives a genuine lower bound. Students routinely believe
the first gives a bound and the second gives an approximate solution; both are
backwards.

---

## 7. The Benders thread (the amendment's centre of gravity)

Tutorial 08 runs the same planning problem three ways and compares what each
can *certify*:

| | DC-Benders | AC-Benders | SOCP-Benders |
|---|---|---|---|
| Subproblem class | LP | nonconvex NLP | convex SOCP |
| Dual information | LP duals | local KKT multipliers | conic duals |
| Strong duality | yes | not in general | yes under Slater |
| Cuts globally valid | yes | **not automatically** | yes, for the relaxation |
| Certifies | the DC model | nothing global | the MISOCP relaxation |

The lesson is the last row. Replacing a nonconvex AC subproblem with a convex
SOC relaxation is what makes the dual information globally meaningful — and the
certificate then applies **to the relaxation first**. Whether it also certifies
the AC problem depends on exactness, or on the gap to an independently obtained
AC-feasible upper bound:

```
z_SOC  ≤  z_AC*  ≤  z_AC_feasible
└──────── certified interval ────────┘
```

---

## 8. Fast mode

`FAST_MODE` shrinks **scale only** — fewer buses, periods, scenarios, Benders
iterations. It never changes a formulation. A tutorial that tests a
mathematical claim must test the same claim in both modes; only the instance
gets smaller. CI runs fast mode; the classroom runs full mode.

---

## 9. Misconceptions the course argues against

Each is addressed where it naturally arises, with a measurement rather than an
assertion.

| Misconception | Where it is dismantled |
|---|---|
| "Linear means inaccurate" | T03 — DC-OPF matches AC dispatch closely on a well-conditioned case |
| "Nonlinear means more accurate" | T04 — a converged local solution can be worse than a good LP |
| "IPOPT said optimal, so we have the global AC optimum" | T04 — multi-start finds different local optima |
| "A relaxation approximates the problem" | T05 — set inclusion, with a figure |
| "SOCP is just another nonlinear solver" | T05 — conic structure, and two solvers agreeing |
| "A small objective gap proves AC feasibility" | T05 — small gap, non-zero SOC residual |
| "MILP solvers try all binary combinations" | T02 — node counts vs `2^n` |
| "Chance-constrained means conservative" | T07 — it can be cheaper than the deterministic robust choice |
| "95% individual implies 95% joint" | T07 — Monte Carlo, the headline experiment |
| "More scenarios is always better" | T07 — in-sample vs out-of-sample |
| "Benders always speeds things up" | T08 — a case where the monolith wins |
| "Anything can be Benders-decomposed" | T08 — the nonconvex AC subproblem |
| "Column generation gives an integer solution" | T09 — it solves the LP relaxation |
| "Optimization failed, the solver is bad" | every tutorial's failure section |

---

## 10. Reproducibility

Fixed seeds; scenario generation documented and re-derivable; independent
validation samples never used for tuning; per-unit conventions stated per model.
Every recorded experiment logs solver, version, termination condition,
objective, runtime, gap and the residuals that matter for that model class.

The course-wide scoreboard in `docs/scoreboard.md` is **generated** from
executed notebooks, never typed by hand.

---

## 11. Status

Phase 1 (research) and the solver gate are complete. This document is the
Phase 2 deliverable. Implementation status is tracked in `docs/instructor_guide.md`
as tutorials land.
