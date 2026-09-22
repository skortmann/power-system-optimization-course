# Instructor guide

Timings, where students get stuck, and the discussion that each notebook is
built around. Assume **90–150 minutes of contact time** per tutorial, with the
notebooks executed in advance so nobody watches a solver.

## Before the first session

- Have everyone run `uv sync` the week before, and then
  `uv run python scripts/check_solvers.py`. **IPOPT is the one that bites** — it
  is a system binary, not a wheel, and half of the course needs it.
- Read `docs/course_overview.md` §2. Several of the course's measured results
  contradict folklore on purpose, and one contradicts a claim I made while
  writing it (Tutorial 03, Exercise 3.3). If a student says "but the small-angle
  assumption degrades with load", the notebook has a measurement.
- Decide whether to hand out `*_exercise.ipynb` only, or both. See
  `docs/student_guide.md`.

## The two threads

Point at them repeatedly; they are what makes the course one story rather than
ten topics.

**Bounds.** Branch-and-bound (T02), convex relaxation (T05), Benders (T08) and
column generation (T09) are all the same picture: a lower bound, an upper bound,
and something added each iteration to squeeze them.

**Relaxation is not approximation.** A relaxation enlarges the feasible set and
gives a bound. An approximation changes the equations and gives none. Students
arrive with this backwards.

---

## Tutorial 01 — Economic dispatch

**Duration** 90 min. **Runtime** ~5 s. **Prerequisites** none.

**Emphasis** modelling, not solving. The four-ways-to-solve section is the point.

**Where students get stuck**
- *Parameters versus variables.* That demand is a parameter here and a variable
  in T06 is a *choice*, not a fact about physics.
- *Merit order with a minimum.* The by-hand calculation must give the remainder
  to the next unit; getting this wrong was a real bug during development, and
  the notebook now asserts agreement across all four methods.
- *Which unit is marginal.* At D = 100 MW the cheap unit is at its ceiling, so
  λ = 55 (the gas unit's cost), not 25. Students reliably say 25.

**Discussion** Adding `c2·p²` changed the solver without changing the physics.
What exactly changed?

---

## Tutorial 02 — Duality, MILP, relaxation

**Duration** 120 min. **Runtime** ~5 s. **Prerequisites** T01.

**Emphasis** the theoretical core. Everything after T05 depends on it.

**Where students get stuck**
- *Dual sign conventions.* Pyomo's `model.dual` on a `≤` constraint is the
  negative of the textbook μ ≥ 0. Section 3 shows the fix; expect confusion.
- *"The LP relaxation is an approximation."* No — it is a relaxation, and the
  distinction is the whole course.
- *Complementary slackness* is easy to state and hard to feel. The two-case
  comparison in Exercise 2.2 is where it lands.

**Discussion** Exercise 2.4 shows solve time growing by 1.1× while brute force
would grow by 1.7e7×. What is branch-and-bound actually doing?

---

## Tutorial 03 — DC-OPF

**Duration** 120 min. **Runtime** ~5 s. **Prerequisites** T01, basic circuits.

**Emphasis** the four DC assumptions, stated explicitly, and what each discards.

**Where students get stuck**
- *Per unit versus MW.* Susceptance in per unit against powers in MW makes the
  model silently infeasible. This happened during development and is now
  documented in `Branch.susceptance`.
- *Why three prices?* Because there are three balance constraints.

**Discussion** Exercise 3.3 is the notebook's best moment and it is a **negative
result**: angle differences *shrink* as load grows (5.35° → 3.82°), because
extra demand is met locally rather than transported. Ask students to predict the
trend first — most say "grow" — then measure it. The model also becomes
infeasible at 1.2× load long before accuracy becomes a problem.

---

## Tutorial 04 — AC-OPF

**Duration** 120 min. **Runtime** ~10 s. **Prerequisites** T03.

**Emphasis** where nonconvexity comes from, and what `optimal` means.

**Where students get stuck**
- *"Nonconvex" as a vague complaint.* Be specific: bilinear voltage products and
  trigonometric terms, appearing in **equality** constraints.
- *Polar versus rectangular as a cure.* It is not. It is a change of coordinates
  that makes the nonconvexity quadratic, which is what T05 exploits.

**Discussion** 20 multi-start runs all found the same optimum (spread 5e-12).
Does that prove global optimality? No — and the answer names what would: a lower
bound, which is T05.

This is where `opf-potpourri` enters, *after* the hand-built model. Its answer
agrees to 0.000%.

---

## Tutorial 05 — Convex and SOC relaxations

**Duration** 150 min. **Runtime** ~10 s. **Prerequisites** T02, T04.

**Emphasis** the mathematical centre of the course. Do not rush it.

**Where students get stuck**
- *Relaxation versus approximation.* Use the figure, then the table, then the
  measurement.
- *Exactness versus objective gap.* These are different claims. A relaxation can
  match the objective and still be physically meaningless.

**Discussion** Exercise 5.2 is the one to spend time on. With a constant
objective the solver returns `optimal` on a convex program and produces a point
implying **6.79 MW of losses on a feeder whose real losses are 0.099 MW** — a
factor of 69. Nothing went wrong numerically. Ask what did.

---

## Tutorial 06 — Multi-period and storage

**Duration** 90 min. **Runtime** ~5 s. **Prerequisites** T01.

**Emphasis** one recursion couples the whole horizon.

**Where students get stuck**
- *Efficiency asymmetry.* Charging puts in η_c·P; discharging takes out P/η_d.
- *Why no binary is needed* — until the objective rewards waste.

**Discussion** Exercise 6.2: removing the terminal condition makes the model
*cheapest* and drains the store to 0.0%. Which number would you put in a report?

---

## Tutorial 07 — Chance constraints

**Duration** 150 min. **Runtime** ~5 s. **Prerequisites** T03, basic probability.

**Emphasis** individual reliability is not joint reliability.

**Where students get stuck**
- *"Chance-constrained means conservative."* At ε = 0.5 it is reckless.
- *In-sample validation.* Measuring violation on the scenarios you optimised
  over is a training-set score.

**Discussion** The headline measurement: three constraints at ε = 0.05 each gave
individual rates of 0.051, 0.052, 0.000 — and a **joint violation of 0.1012**,
twice the worst individual. Then Boole's allocation meets the joint target but
delivers 0.0167 against 0.05, three times more conservative than required, for
5.05% extra cost. Both halves matter.

---

## Tutorial 08 — Benders

**Duration** 150 min+. **Runtime** ~30 s. **Prerequisites** T02, T04, T05.

**Emphasis** the synthesis notebook. It needs everything before it.

**Where students get stuck**
- *Why the cut is valid for all y.* Because the dual feasible set does not
  depend on y — this is the single most important step and worth doing slowly.
- *Which gap is which.* Convergence gap ≠ relaxation gap.

**Discussion** The three-level comparison produces a genuine disagreement: the
LP subproblem installs **nothing**, while SOC and AC both install two inverters.
The LP run was rigorous and wrong, because a lossless model cannot see the
quantity the investment improves. Then Exercise 8.3: both the convex and the
nonconvex cut pass a validity test at all eight points — and only one of them
was *entitled* to.

---

## Tutorial 09 — Column generation

**Duration** 150 min. **Runtime** ~10 s. **Prerequisites** T02, T08.

**Emphasis** the mirror image of Benders.

**Where students get stuck**
- *Why artificial variables are needed.* An infeasible master has no duals, so
  there is nothing to price with.
- *Reduced cost as economics.* It is a profit calculation at the master's own
  prices.

**Discussion** Two honest findings. The Dantzig-Wolfe bound came out **equal** to
the LP relaxation — the integrality property, not a failure. And the integer
recovery lands 3.46% above the true optimum, because column generation solves an
LP. Two earlier attempts at the recovery are kept in the notebook because both
fail instructively.

---

## Tutorial 10 — Capstone

**Duration** 150 min. **Runtime** ~50 s. **Prerequisites** all.

**Emphasis** method selection, and the validation pipeline.

**Discussion** Exercise 10.1 asks which formulation each of five questions
needs. One of them — where to install the next inverter — cannot be answered by
*any* of the five, because they are all operational. That is the moment the
course's structure clicks.

Close on Exercise 10.3: how would you *prove* a new method is better? The design
matters more than the numbers, and "what this cannot establish" is the part
students omit.

---

## Assessment

**Project (recommended).** Apply one tutorial's method to a network the student
brings. Deliverables: a notebook, a baseline comparison, a physical validation,
and a limitations section. Mark the limitations section hardest.

**Written.** Give them a recent AC-OPF paper and ask: is the comparison fair?
Are the formulations aligned? Is a relaxation reported as an approximation?

**Oral.** "You have a nonconvex AC subproblem under integer decisions. What can
Benders certify, and what can it not?"

## Misconceptions to pre-empt

Each is dismantled by a measurement, not an assertion.

| misconception | where |
|---|---|
| "Linear means inaccurate" | T03 Ex 3.3 |
| "IPOPT said optimal, so it is global" | T04 Ex 4.3 |
| "A relaxation approximates the problem" | T05 Ex 5.2 |
| "A small objective gap proves feasibility" | T05 §5 |
| "MILP solvers enumerate" | T02 Ex 2.4 |
| "95% individual implies 95% joint" | T07 §5 |
| "More scenarios is always better" | T07 Ex 7.3 |
| "Any model can be Benders-decomposed" | T08 Ex 8.3 |
| "Column generation gives an integer solution" | T09 Ex 9.3 |
| "Optimization failed, so the solver is bad" | T01 Ex 1.3 |

## If something breaks

| symptom | cause | fix |
|---|---|---|
| `ApplicationError: No executable found for solver 'ipopt'` | IPOPT is not installed | `conda install -c conda-forge ipopt`; see `solver_guide.md` |
| `DegreeError: ... degree None` | HiGHS was handed a quadratic objective | use `solver_for("QP")` |
| termination `other` on an SOCP | a legacy Gurobi interface | use `appsi_gurobi`, or IPOPT |
| MISOCP unavailable | no Gurobi licence | the verification solve is skipped; the tutorial says so |
| numbers differ slightly from the committed outputs | different BLAS or solver version | expected; the *conclusions* should hold |
| a notebook is much slower than the table says | another is running | they are CPU-bound |

Rebuild everything with:

```bash
uv run python scripts/build_exercise_notebooks.py
```
