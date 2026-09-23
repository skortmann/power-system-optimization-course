# Final correctness audit

This document records an audit of the course's mathematics, implementations and
results. It is written to be checked, not believed: every claim below names the
script that produced it and the number it produced.

**The audit found real defects, including one Critical.** A commissioned
independent reviewer found that several tutorials' *narratives contradict their
own printed output*, and that the capstone's scenario study was inert — it
re-solved one identical model twelve times while the text discussed the spread.
Those are exactly the failures that survive a green test suite, and the audit's
own harnesses did not catch them, because the harnesses test the library and the
algorithms rather than whether each tutorial's prose matches its own numbers.

Totals: **one Critical, seven High, and nine Medium defects found**; all
Critical and High resolved. The audit also found **four defects in its own
instruments**, three of which produced false failures against the course, and it
**corrects one overclaim this document made in an earlier revision** (§10).

Nothing here rests on "the notebooks all execute". They do — all twenty, from
cleared sources, at full scale — and that fact is reported in §12 as one check
among many, not as a conclusion. Two of the defects below were present in
notebooks that executed cleanly and passed their own validation cells.

## Contents

1. [Scope and method](#1-scope-and-method)
2. [Independent review](#2-independent-review)
3. [Concepts, definitions and literature](#3-concepts-definitions-and-literature)
4. [Units, signs and per-unit consistency](#4-units-signs-and-per-unit-consistency)
5. [Convexity claims and problem classification](#5-convexity-claims-and-problem-classification)
6. [Relaxation versus approximation](#6-relaxation-versus-approximation)
7. [Duality, KKT conditions and prices](#7-duality-kkt-conditions-and-prices)
8. [Bounds and bound ordering](#8-bounds-and-bound-ordering)
9. [Benders decomposition](#9-benders-decomposition)
10. [Column generation and Dantzig-Wolfe](#10-column-generation-and-dantzig-wolfe)
11. [Uncertainty and chance constraints](#11-uncertainty-and-chance-constraints)
12. [Physics, reproducibility and execution](#12-physics-reproducibility-and-execution)
13. [Exercises, solutions and validation cells](#13-exercises-solutions-and-validation-cells)
14. [Issues found, severity and resolution](#14-issues-found-severity-and-resolution)
15. [Verification matrix](#15-verification-matrix)

---

## 1. Scope and method

The audit was run as four executable harnesses plus a targeted manual review.
Each harness is a plain script with a configuration block at the top and no
command-line interface, so re-running it is one command and changing what it
measures is one edit.

| Harness | Checks | What it establishes |
|---|---|---|
| `scripts/audit.py` | 31 | Duals, KKT, bounds, monotonicity, chance constraints, covariance, the SOC relaxation, branch-flow physics, reproducibility, solver status, the potpourri API |
| `scripts/audit_decomposition.py` | 13 | Benders cut validity and tightness, Benders against full enumeration at two subproblem classes, column generation against an exhaustive column pool, reduced-cost signs |
| `scripts/audit_exercises.py` | 9 | Pairing, scaffold leakage, validation-cell reachability, difficulty labels, deliverables |
| `scripts/audit_bounds_t09.py` | 2 | Tutorial 09's `z_LP <= z_DW <= z_MILP` claim, measured on the tutorial's own model code |
| `scripts/audit_reviewer_claims.py` | 3 | Re-tests the independent reviewer's findings before any of them is acted on |
| one-off (§9) | 2 | The nonconvex AC Benders level against enumeration |

Total: **60 evidence-backed checks**, alongside **56 pytest tests** and **20
executed notebooks**.

A fifth input was not a harness: the independent reviewer read each tutorial's
prose against its own printed output. That is where six of the seven High-or-
above defects came from, and none of them was reachable by a harness — see the
closing section.

Three principles governed the work.

**Measure, do not assert.** Where the course states a mathematical relationship,
the audit computes both sides. Duals are checked against finite differences,
relaxations against the models they relax, decompositions against monolithic
solves, chance constraints against 200,000-sample Monte Carlo.

**Compare against something independent.** A decomposition that converges
smoothly to the wrong answer is indistinguishable from one that works, so
Benders is checked against full enumeration of all eight investment patterns
and column generation against an LP over every feasible schedule. The
branch-flow model is checked against pandapower's power flow, which shares no
code with it.

**Re-derive the audit's own tools.** `scripts/audit_decomposition.py`
re-implements the Benders loop, the restricted master and the pricing problem
rather than importing the tutorial's. Two independent implementations reaching
18314.76190476 is evidence; one implementation agreeing with itself is not.

### Tolerances

Tolerances live in `src/psopt_course/tolerances.py`, one constant per concept,
each documented with the solver accuracy that justifies its magnitude. This
module was created during the audit, after the discovery that scattered
literals had produced both failure modes: a check too tight for the solvers
that produced the numbers (issue **A-03**), and a check loose enough to hide a
sign (issue **A-02**).

```
quantity                tolerance   guards
equality feasibility    1e-06       power balance, KVL
inequality feasibility  1e-06       voltage, thermal limits
integer feasibility     1e-05       distance to {0,1}
conic residual          1e-05       u*ell - P^2 - Q^2
objective (relative)    1e-06       decomposition vs monolith
objective (absolute)    1e-08       floor near zero
dual (relative)         1e-03       vs finite difference
bound ordering          1e-06       LB <= UB
probability (absolute)  5e-03       empirical vs requested eps
```

The dual tolerance is deliberately three orders looser than the others: a
finite-difference estimate of `d(objective)/d(rhs)` is only valid while the
active set is unchanged, and carries the step's truncation error on top of two
solves' worth of solver error. The measured agreement was far better than the
tolerance demands (§7), but the tolerance must be set by what the method can
guarantee, not by what it happened to deliver.

---

## 2. Independent review

An independent reviewer agent was commissioned with a concrete assignment
rather than a general request to "look for problems". It was directed to run
experiments rather than read code, and given five priority areas chosen because
each is a place where a sign or factor error produces plausible-looking output:

1. the SOC branch-flow formulation's signs and factors, against a power flow;
2. the Benders cut sign convention, specifically
   `pi_k = (dual[q_upper[k]] + dual[q_lower[k]]) * Q_LIMIT`;
3. the chance-constraint reformulation and its two-sided/Boole consistency;
4. column-generation reduced-cost signs and column feasibility;
5. PTDF and LMP sign conventions, by finite difference.

It returned findings on all five, plus tutorials 01, 02, 04, 06, 09 and 10 and
the library.

### How its findings were treated

Every finding was re-tested before anything was changed, in
`scripts/audit_reviewer_claims.py` and in targeted experiments. A reviewer's
finding is a hypothesis; several contradicted claims the course makes, and one
contradicted a claim *this document* made.

The outcome of that re-testing:

| Reviewer finding | Re-test result |
|---|---|
| T10's PV scenarios never perturb the model | **Confirmed.** Objective spread across 12 scenarios `3.87e-13`; solved `psG` pinned at `max_p_mw` in every one |
| T01 Ex 1.2's equal-marginal-cost conclusion is false | **Confirmed.** QP optimum `[60, 40]` — the LP vertex — marginal costs 34.6 vs 57.4 |
| T02 Ex 2.2's "interior" case is at a ceiling | **Confirmed.** At D=70 the optimum is `[60, 10]`, identical duals to the "binding" case |
| T02 Ex 2.3's gap does not widen | **Confirmed.** Absolute gap exactly invariant at 77.2785 across a 0→10,000 sweep |
| T07's wind error has the wrong sign | **Confirmed.** Mismatch is exactly `2*xi_wind` in every realisation |
| T07's Boole bound counts N, not 2N constraints | **Confirmed.** The allocation uses `2 * N_BUS`; the reported guarantee used `N_BUS` |
| T09's integrality-property explanation is wrong | **Confirmed, and it corrects this document** — see §10 |
| Three `relaxation_gap` call sites sort their arguments | **Confirmed** by inspection; all three fixed |
| T08 prints an inverted bound chain via `abs()` | **Confirmed**; routed through `certified_interval` |
| Every IPOPT solve reports `gap=inf%` | **Not reproduced.** `SolveRecord.gap` is `None`, not infinite, on a fresh NLP solve. Recorded as unconfirmed |

One further defect was found *while* fixing another: T07's scenario sweep drew
an **independent** sample per size (`seed=1000 + n`), so the scenario sets were
not nested and the monotonicity the validation cell asserts did not follow from
anything. It passed because the particular draws happened to cooperate; the
corrected wind sign changed the draws and it failed immediately. Recorded as
**R-10**.

## 3. Concepts, definitions and literature

### Terminology

The course's core distinctions were checked against their sources:

- **Problem class is not a convexity claim.** `docs/optimization_problem_classes.md`
  and Tutorial 10 both state that QCQP covers convex and nonconvex problems and
  that the acronym alone settles nothing. `SOCBFM.problem_class` returns
  `"SOCP"` for the relaxed model and `"nonconvex QCQP"` for the exact one, and
  a test asserts both (`tests/test_relaxations.py`).
- **Relaxation versus approximation** is treated in §6.
- **Certification** is treated in §9.

### Literature

Every citation in `docs/literature.md` carrying a volume, issue or year was
checked against the bibliographic record. All were correct as printed. The
highest-risk entries — those where a plausible wrong year or volume is easy to
introduce — are:

| Citation | As printed | Verified |
|---|---|---|
| Benders, "Partitioning procedures…" | *Numerische Mathematik* 4, 1962 | correct (pp. 238–252) |
| Geoffrion, "Generalized Benders decomposition" | *JOTA* 10(4), 1972 | correct (pp. 237–260) |
| Dantzig & Wolfe, "Decomposition principle…" | *Operations Research* 8(1), 1960 | correct (pp. 101–111) |
| Jabr, "Radial distribution load flow using conic programming" | *IEEE Trans. Power Syst.* 21(3), 2006 | correct |
| Farivar & Low, "Branch flow model… (parts I and II)" | *IEEE Trans. Power Syst.* 28(3), 2013 | correct — **not** the 27(3)/2012 that is often miscited |
| Gan, Li, Topcu & Low, "Exact convex relaxation…" | *IEEE Trans. Autom. Control* 60(1), 2015 | correct |
| Low, "Convex relaxation of OPF (I and II)" | *IEEE Trans. Control Netw. Syst.* 1(1), 1(2), 2014 | correct |
| Lavaei & Low, "Zero duality gap…" | *IEEE Trans. Power Syst.* 27(1), 2012 | correct |
| Lübbecke & Desrosiers, "Selected topics in column generation" | *Operations Research* 53(6), 2005 | correct |
| Geoffrion, "Lagrangean relaxation for integer programming" | *Math. Programming Study* 2, 1974 | correct |

One citation makes a claim about the course's own results rather than about the
literature, and so was measured rather than checked: Tutorial 09 attributes the
coincidence of its Dantzig-Wolfe and LP bounds to Geoffrion's integrality
property. That attribution is verified numerically in §10.

---

## 4. Units, signs and per-unit consistency

Unit errors are the course's most dangerous class of defect because they
produce output that is wrong by a clean factor and therefore looks like a
modelling choice. One such defect was found during development and is recorded
as **D-03**: `branch.susceptance` returned the per-unit `1/x` while powers were
carried in MW, so DC-OPF required roughly ten-radian angle differences and was
silently infeasible at realistic loadings. The fix separated `susceptance`
(MW/rad) from `susceptance_pu` and added `base_mva`; a test now asserts the
exact relationship between them rather than an inequality, after an earlier
version of that test asserted `susceptance > 100 * susceptance_pu` when the
ratio is exactly 100.

The strongest current evidence that units and signs are consistent is that the
branch-flow model reproduces an independent power flow:

```
max |V| error = 3.20e-09 pu over 33 buses
losses: model 0.20267713 MW, pandapower 0.20267711 MW, difference 1.38e-08
```

This is decisive for the distribution half of the course. The exact BFM is
written in lifted variables `u = |V|^2` and `ell = |I|^2`, converts losses back
through `sum(r * ell) * net.sn_mva`, and reads its impedances from the same
`_ppc["branch"]` rows potpourri uses (asserted in
`tests/test_relaxations.py::test_impedances_come_from_the_same_ppc_rows_potpourri_uses`).
A per-unit or base-power error anywhere in that chain would not reproduce
pandapower to nine decimal places.

Power balance closes on the solved network:

```
generation=3.917677, load=3.715000, losses=0.202677 MW, imbalance=7.69e-08
losses positive; |V| in [0.9131, 1.0000] pu
```

Sign conventions on prices are verified by finite difference in §7, which is
the test that would catch an inverted LMP.

---

## 5. Convexity claims and problem classification

Every statement of the form "this is convex", "this is globally optimal" or
"this certifies X" was located by grep across the ten tutorial sources and read
in context. The claims are correctly hedged. The load-bearing ones:

- Tutorial 04: *"On a convex problem a KKT point is the global optimum. On a
  nonconvex one it is a local…"* — correct; KKT conditions are sufficient for
  global optimality on a convex program.
- Tutorial 05: *"…returned a globally optimal point of a convex program."* —
  correct for IPOPT on an SOCP, and the justification (local implies global on
  a convex problem) is stated rather than assumed.
- Tutorial 05: *"…rigorously solves the RELAXATION. Whether that certifies the
  original AC…"* — correct, and the distinction is the point of the section.
- Tutorial 08's comparison table is the course's sharpest statement and is
  reproduced here because it is what the amended brief asked for:

  | | LP / LinDistFlow | nonconvex AC | convex SOCP |
  |---|---|---|---|
  | convex | yes | **no** | yes |
  | strong duality | automatic | **not in general** | under Slater |
  | cut valid globally | yes | **not automatically** | yes, for the relaxation |
  | certifies | the LP model | nothing global | the MISOCP relaxation |

  Every entry is correct. The row that matters most is the last: the LP level
  certifies *the LP model*, not the AC problem.

This last point deserves emphasis because the numbers invite the opposite
conclusion. The LinDistFlow level returns **0.2215**, which is *below* the SOC
lower bound of **0.22935**. A reader who assumed LinDistFlow were a relaxation
would see a smaller number and take it for a valid bound. It is not: setting
`ell = 0` changes the equations rather than enlarging the feasible set, so it
bounds nothing. The course says so explicitly — *"The LP run was rigorous and
chose differently, because LinDistFlow cannot represent losses"* — and never
places the LP value in a bound ordering.

---

## 6. Relaxation versus approximation

The SOC branch-flow relaxation was verified on four counts.

**It is a lower bound.** `z_SOC = 0.0057753374`, `z_BFM = 0.0057752618`. The
difference is `-7.56e-08`, within the objective tolerance of `1e-06` and far
below either solve's own accuracy.

**Its residuals have the right sign.** Over 32 branches the minimum cone
residual `u*ell - P^2 - Q^2` is `+1.39e-08` — non-negative everywhere, as the
relaxed inequality requires.

**It is tight here, and tightness is measured rather than inferred.** Maximum
residual `4.24e-07`, inside the conic tolerance of `1e-05`. This is checked
independently of the objective gap, because a relaxed point can carry almost
the right objective and still be physically unrealisable.

**The two models differ in exactly one constraint.** A test compares the active
constraint blocks of the SOC and exact models and asserts they are identical,
so a measured "relaxation gap" cannot be a modelling discrepancy in disguise.

The distinction is then demonstrated destructively. Tutorial 05's Exercise 5.2
replaces the loss objective with a constant, breaking the Farivar–Low
monotonicity precondition. The solver still reports `optimal`, the relaxation
is no longer tight, and the implied losses exceed the true ones by a factor of
about 69. This is the course's clearest evidence that solving a relaxation
exactly and obtaining a physically meaningful answer are different events.

---

## 7. Duality, KKT conditions and prices

Duals were checked against finite differences of the objective with respect to
the right-hand side — the definition, not a restatement of it.

**Economic dispatch.** The balance constraint's multiplier against a perturbed
demand:

```
lambda = 55.000000, finite difference = 55.000000, relative error 5.29e-12
```

**DC-OPF locational marginal prices.** Each bus perturbed separately:

```
bus 0: LMP = 20.0000  fd = 20.0000
bus 1: LMP = 51.1111  fd = 51.1111
bus 2: LMP = 90.0000  fd = 90.0000
worst relative error 2.18e-11
```

The spread across buses is the congestion signal, and the finite-difference
agreement establishes both magnitude and **sign** at every bus. An inverted
price convention — the failure this check exists to catch — would show as a
sign disagreement at the congested buses, where the LMPs are furthest apart.

**KKT conditions**, computed as residuals rather than asserted:

```
stationarity 0.00e+00   primal 0.00e+00   dual 0.00e+00   complementarity 0.00e+00
```

A related pedagogical defect was found during development and is recorded as
**D-02**: Tutorial 01 originally identified the cheapest unit as marginal while
the measured shadow price was 55, the gas unit's cost. The text now identifies
the marginal unit as the one not at a limit, which is the definition that
matches the multiplier.

---

## 8. Bounds and bound ordering

Every bound relationship the course states was measured.

```
z_LP = 17041.711142  <=  z_MILP = 17122.281643     gap 0.4706%
MILP solution integrality: max distance from {0,1} = 0.00e+00
z_SOC = 0.0057753374 <=  z_BFM  = 0.0057752618     (within 1e-06)
z_LP  = 18314.76190476 = z_DW   = 18314.76190476 <= z_MILP = 18510.00000000
```

Monotonicity — enlarging a feasible set cannot increase a minimum — was checked
in two directions:

```
generator capacity +0/+10/+20/+40 MW : [4800.0, 4500.0, 4200.0, 3600.0]
line capacity 40/60/80/120 MW        : [11133.33, 9422.22, 7711.11, 5700.0]
```

Both sequences are non-increasing. This is a weak check individually and a
useful one in aggregate: a sign error in a capacity constraint typically breaks
monotonicity before it breaks feasibility.

Two defects in the bound machinery itself were found during this audit and are
recorded as **A-02** and **A-03**. Both concern the guard `lower <= upper`:
`validation.bounds_table` used `1e-9`, tighter than either solve, and raised a
`ValueError` on the genuine SOC/BFM pair above; `metrics.CertifiedInterval`
used `1e-6` and therefore accepted the pair, but then reported it as an
interval of **negative width**, with a negative relative gap that would satisfy
any "is the gap small enough" test. Both now draw on
`tolerances.bound_holds`, and the reported width is clamped at zero, with the
reasoning recorded in the code. A genuinely empty interval is still rejected:

```
certified_interval(1.0, 0.5) -> ValueError: lower bound 1 exceeds upper bound 0.5
```

This was not a hypothetical. The same crossing appears a second time in
Tutorial 08's capstone comparison, where `z_SOC - z_AC = +7.40e-08`, and would
have produced a second negative width.

---

## 9. Benders decomposition

Benders is the course's highest-risk algorithm to verify, because an invalid
cut removes the true optimum from the master's feasible set and the algorithm
then converges cleanly to the wrong answer with nothing in the output to say
so. It was checked against full enumeration of the value function `Q(y)` at all
eight investment patterns, at all three subproblem classes.

```
Q(y), SOC subproblem:
  000 0.23137945   001 0.22905383   010 0.23075513   011 0.22850372
  100 0.22979254   101 0.22775436   110 0.22924021   111 0.22727531
```

**Cut validity.** Each of the eight cuts was evaluated at each of the eight
patterns — 64 (cut, vertex) pairs:

```
worst violation over 8x8 = +0.000e+00   (must be <= 1e-06)
```

The worst value being exactly zero is not a coincidence and is the expected
signature of a correct cut: it is attained only at each cut's own generating
vertex, where the cut must be tight, and is strictly negative everywhere else.

**Cut tightness.** `max |cut - Q|` at the generating vertex: `0.00e+00`.

**Cut coefficient signs.** The coefficient
`pi_k = (dual[q_upper[k]] + dual[q_lower[k]]) * Q_LIMIT` was compared against
the measured change in `Q`:

```
k=0: pi = -1.538786e-02   actual dQ = -1.586910e-03
k=1: pi = -1.507729e-02   actual dQ = -6.243249e-04
k=2: pi = -1.553812e-02   actual dQ = -2.325627e-03
```

The coefficients are an order of magnitude larger than the measured changes.
This was examined rather than accepted, because a wrong factor would look the
same. It is convexity: `Q` is a convex function of the right-hand side, `pi` is
its slope at `y = 0`, and the marginal benefit of reactive capability falls off
as more is installed — so the local slope necessarily exceeds the average slope
over the unit step. The cut-validity check above confirms the consequence: the
cut under-estimates `Q` everywhere despite the steeper coefficient.

**Against enumeration**, at all three levels:

| Subproblem | Benders | Enumeration | Difference | Decision | Cuts |
|---|---|---|---|---|---|
| LP / LinDistFlow | 0.2215000000 | 0.2215000000 | 0.00e+00 | y = 000 | 2 |
| convex SOCP | 0.2293543574 | 0.2293543574 | 0.00e+00 | y = 101 | 5 |
| nonconvex AC | 0.2293542834 | 0.2293542834 | 0.00e+00 | y = 101 | 5 |

Bounds never inverted in any run.

Three things in this table are worth stating plainly.

**The LP level converges correctly to a different decision.** Its value
function is constant across all eight patterns — LinDistFlow sets branch
currents to zero, so it has no losses, and the slack import equals the load
whatever the inverters do. The master correctly concludes that investment is
pure cost and installs nothing. Nothing went wrong numerically; the model
cannot represent the quantity the investment improves.

**The AC level reached the right answer here and is still not certified.**
IPOPT returns a KKT point of each subproblem, and multipliers from a local
solution support the value function only locally, so the cuts are not
guaranteed valid. The course says exactly this, and the audit's agreement with
enumeration on *this* feeder does not upgrade a heuristic to a guarantee. It
confirms the course's empirical claim and nothing more.

**The SOC bound sits 7.4e-08 above the AC value.** That is solver noise between
two independent IPOPT solves, not a violated bound — `tolerances.bound_holds`
accepts it, and a raw `1e-9` comparison rejects it. The audit itself made that
mistake once and is recorded as **A-09**.

---

## 10. Column generation and Dantzig-Wolfe

Column generation was checked against an LP over an exhaustive pool containing
**every** feasible single-generator schedule, enumerated by brute force over all
`2^6` on/off patterns per generator and filtered for `pmin`/`pmax`, on/off
consistency and minimum up-time.

```
generated  : 18314.76190476 from  29 columns
exhaustive : 18314.76190476 from 111 columns
difference : 7.28e-12
```

This is the check that matters: column generation claims to solve the LP over
*all* columns while visiting only a few, and the only way to establish that is
to build all of them.

**Column feasibility.** All 29 generated columns were re-checked against the
generator constraints independently of the pricing problem that produced them —
binary commitment, zero output when off, output within limits when on, and
minimum up-time honoured after every start. All 29 passed. A negative reduced
cost on an infeasible schedule would otherwise corrupt the master silently.

**Reduced-cost signs**, in both directions:

- At termination the minimum reduced cost over all generators is `+0.000e+00`,
  so no improving column remains.
- The converse is the real sign test. Starting from a deliberately impoverished
  two-column basis, each generator's pricing problem returns a column with
  negative reduced cost, and adding it lowers the master objective:

  ```
  coal: rc = -20994170.00   objective 13012930.00 -> 5021517.14
  gas : rc = -17989920.00   objective 13012930.00 -> 5017980.00
  peak: rc = -11988545.00   objective 13012930.00 -> 5022715.00
  ```

  A reversed sign convention would pass the first test and fail this one.

**The integrality-property claim.** Tutorial 09 states `z_LP <= z_DW <= z_MILP`
and then claims the first two coincide here, attributing it to Geoffrion's
integrality property. Measured with the tutorial's own model code
(`scripts/audit_bounds_t09.py`):

```
z_LP   = 18314.76190476
z_DW   = 18314.76190476      |z_DW - z_LP| = 3.638e-12
z_MILP = 18510.00000000      integrality gap 195.238095 (1.055%)
converged in 10 iterations, 29 columns
```

The ordering holds and the two bounds coincide. Two independent
implementations — the tutorial's and the audit's own reconstruction in
`audit_decomposition.py` — reach `18314.76190476`.

### Correction: this document previously overclaimed here

An earlier revision of this report said the tutorial's *attribution* of that
coincidence to Geoffrion's integrality property was "verified rather than
assumed". That was wrong, and the independent reviewer caught it.

What had been measured was that the two numbers agree. The attribution is a
different claim: that they agree **because** minimising over the LP relaxation
of each single-generator subproblem always lands on an integral point. That is
a statement about every price vector, so it has to be tested over price
vectors. Doing so (`scripts/audit_reviewer_claims.py`, 300 random price vectors
per generator, each subproblem solved as an LP and as a MILP):

| generator | worst (MILP − LP) |
|---|---|
| coal | 204.797463 |
| gas | 218.450151 |
| peak | 0.000000 |

Two of the three subproblems have fractional extreme points **already**, with
the minimum up-time constraints as written. The integrality property does not
hold here and therefore cannot be the reason the bounds agree. The real reason
is instance-specific: the compact LP's optimum happens to be representable as a
convex combination of integral schedules, so tightening to `conv(X_g)` removes
nothing the LP optimum was using.

The tutorial's text and its takeaway have been corrected to say this, and to
show the counterexample. Recorded as **R-07**.

The general lesson is the one this correction illustrates: *an agreement
between two numbers is not an explanation of itself*, and an audit that checks
the numbers has not thereby checked the reason given for them.

Tutorial 09's integer-recovery section deliberately retains two failed
approaches before the working one (**D-10**): a λ-binary restriction that pins
dispatch to a single column's profile and lets artificial variables absorb the
mismatch, and a commit-then-redispatch split that is infeasible at `t = 4`.
The working approach selects commitment and dispatches in one MILP, reaching
`19150.00` against a true optimum of `18510.00` — 3.46% high, and reported as a
heuristic rather than as an answer.

---

## 11. Uncertainty and chance constraints

**Quantile convention.** The reformulation uses `Phi^-1(1 - eps)`:

```
quantile(0.05) = 1.6448536  (expected 1.6448536)
quantile(0.01) = 2.3263479  (expected 2.3263479)
q(0.20) = 0.8416 < q(0.05) = 1.6449 < q(0.01) = 2.3263
```

A tighter risk level demands a larger margin, as it must. Using `Phi^-1(eps)`
instead — a common sign error — would reverse this ordering.

**Out-of-sample validation**, 200,000 fresh samples per risk level:

| Requested `eps` | Empirical violation rate | Error |
|---|---|---|
| 0.10 | 0.10054 | 0.00054 |
| 0.05 | 0.05040 | 0.00039 |
| 0.01 | 0.01009 | 0.00009 |

Each is within one to two standard errors of the design value
(`sqrt(eps(1-eps)/N)` is 0.00067 at `eps = 0.05`), which is the correct
outcome for an exactly-reformulated individual chance constraint under the
Gaussian assumption the model makes.

**Covariance propagation.** The analytical spread `sqrt(a' Sigma a)` against
NumPy, and against sampling:

```
library 3.7462600635, NumPy 3.7462600635, difference 0.00e+00
analytical 3.74626, empirical 3.74482, relative error 0.0004
```

**Joint versus individual.** This is the distinction most often elided, so it
was measured at two correlations:

| Correlation | Individual rates | Joint rate | Boole bound |
|---|---|---|---|
| ρ = 0.0 | 0.0501, 0.0503, 0.0492 | **0.1425** | 0.1500 |
| ρ = 0.6 | 0.0510, 0.0493, 0.0504 | **0.1111** | 0.1500 |

Three constraints each satisfied individually at 5% are jointly violated 14.25%
of the time when independent — nearly three times the individual rate, and just
under the Boole bound of 15%, which is tight in exactly this independent,
rarely-co-occurring regime. Correlation *reduces* the joint rate to 11.11%,
because violations coincide rather than accumulate. Both are correct and the
second is the counter-intuitive one worth teaching.

Two development-phase defects in this tutorial are recorded as **D-08** and
**D-09**: a monotonicity assertion that assumed the sweep arrived in order, and
an `in-sample <= out-of-sample` assertion that fails at `N = 1000` where both
quantities are around `1e-3`. The latter now tests that the *gap shrinks* with
sample size, which is the claim that is actually true.

---

## 12. Physics, reproducibility and execution

**Physical plausibility** on the solved feeder:

```
generation 3.917677 = load 3.715000 + losses 0.202677 MW   imbalance 7.69e-08
losses positive
|V| in [0.9131, 1.0000] pu
```

**Reproducibility:**

```
seeded sampling identical across calls : max |difference| = 0.00e+00 over 15,000 values
profile generation deterministic       : max |difference| = 0.00e+00
repeated solves                        : [3700.0, 3700.0, 3700.0]
```

**Solver status handling.** An infeasible model reports `termination=infeasible,
ok=False` rather than raising or returning a number. Two development-phase
solver defects are recorded: **D-05**, in which Pyomo's legacy `gurobi` and
`gurobi_direct` interfaces returned wrong objectives on a rotated cone
(0.115235 and 0.103718 against a true 0.098836) while reporting an unmapped
`other` status — resolved by moving to `appsi_gurobi`; and **D-06**, in which
`appsi_gurobi` returns `unknown` on four of eight SOCP investment patterns
while IPOPT solves all eight monotonically with `4.24e-07` residuals —
resolved by preferring IPOPT for SOCP, which is sound because a local solution
of a convex program is global.

**Execution from scratch.** All twenty notebooks were regenerated from their
jupytext sources — which carry no outputs, so this is a genuine cold run — and
every solution notebook executed end to end:

```
10 tutorial(s), 0 failure(s)          build(execute=True) returned 0 after 121.3s
fast_mode() = False
```

The last line matters. `PSOPT_FAST` was unset, so this was the full-scale
configuration, not the reduced one CI uses. The run was repeated after the
fixes in §8 and §13, again with 0 failures.

Supporting checks: `ruff check .` passes; `pytest` reports 56 passed; the
staleness check reports all 20 notebooks match their sources; and the pinned
potpourri 0.7.0 provides all 6 symbols the course imports, while
`potpourri.research` is confirmed **not** importable — establishing that the
course's SOC branch-flow model stands on its own rather than on an unpublished
module.

---

## 13. Exercises, solutions and validation cells

Both notebooks are generated from one tagged source, so they cannot drift the
way hand-maintained pairs do. The audit therefore targeted the failures the
generator cannot catch.

```
32 tasks, 32 validation cells
per tutorial: 01=3 02=4 03=3 04=3 05=3 06=4 07=3 08=3 09=3 10=3
difficulty labels: 14 Advanced, 12 Intermediate, 6 Basic  (32 of 32)
```

All nine checks pass: every scaffold has a solution behind it, every
implementation solution has a scaffold in front of it, every interpretation
answer follows markdown that actually asks a question, no scaffold leaks a
solution marker, every validation cell contains an assertion, every task
carries a difficulty label and states a deliverable.

One substantive defect was found here and is recorded as **A-01**. Three
validation cells asserted on names that the student's notebook never
introduces:

| Tutorial | Validation cell asserts | Scaffold provided |
|---|---|---|
| 03, Exercise 3.2 | `ptdf_model`, `ptdf_record` | `ptdf_model` only |
| 05, Exercise 5.1 | `my_residuals`, `exact_residuals` | `my_residuals` only |
| 05, Exercise 5.2 | `broken`, `broken_record`, `broken_residuals` | `broken` only |

Confirmed against the generated exercise notebooks: each missing name occurs
*only* inside the validation cell that asserts on it. A student who did the
task correctly but chose a different variable name would hit a `NameError` from
the cell that is supposed to confirm their work — the failure most likely to
make students stop trusting the validation cells. Each scaffold now seeds every
validated name with a comment saying what it should hold.

---

## 14. Issues found, severity and resolution

### Found by the independent reviewer, re-tested, and fixed

These are the defects that a green test suite does not catch: a tutorial whose
prose contradicts the numbers printed directly above it, and an experiment that
runs cleanly while measuring nothing.

| ID | Location | Issue | Severity | Verification | Resolution |
|---|---|---|---|---|---|
| **R-01** | `tutorials/_sources/10`, `perturbed_network` | The PV "scenarios" never perturbed the optimization. `add_OPF` bounds `psG` above by `max_p_mw`, which the perturbation never touched, so twelve scenarios solved one identical model while the text discussed the spread | **Critical** | Objective spread across 12 scenarios `3.87e-13`; solved `psG` pinned at `0.05` pu in every scenario regardless of `p_mw` | Perturbs `max_p_mw` — the binding quantity — alongside `p_mw`. Spread is now `7.79e-02`; scenarios run mean 0.2327, min 0.1889, max 0.2740 against a deterministic 0.2273, so "the deterministic solution is NOT the mean of the scenarios" is now true rather than printed over identical numbers |
| **R-02** | `networks.two_generator_system` | T01 Ex 1.2's equal-incremental-cost conclusion was false for its own data: with `c2 = 0.08` the equal-marginal-cost point is at 163.6 MW, far outside the 60 MW ceiling, so the QP optimum was the **same vertex as the LP** | **High** | QP optimum `[60, 40]`, marginal costs 34.6 and 57.4, spread 22.80 EUR/MWh, while the text said "both units share the load so that their MARGINAL costs are equal" | `c2 = 0.40`, which puts the equal-MC point at 41.86 MW, strictly interior. Optimum is now `[41.86, 58.14]` with both marginal costs 58.488 EUR/MWh, spread `1.24e-11`. The validation cell now asserts the marginal-cost spread, which is the exercise's actual claim |
| **R-03** | `tutorials/_sources/02`, KKT section | The "cheap unit interior" case used D=70, where the only feasible point is `[60, 10]` — the cheap unit exactly **at** its ceiling. The two contrasted cases had identical duals, and the interpretation printed "mu_max = [30. 0.] — all zero, because no upper limit is binding" | **High** | At D=70: `p* = [60, 10]`, `lambda = 55`, `mu_max = [30, 0]` — identical to the D=130 "binding" case | D=60, where `p* = [50, 10]`, `lambda = 25`, `mu_max = [0, 0]`. The contrast between an inactive and an active bound is now real |
| **R-04** | `tutorials/_sources/02`, Ex 2.3 | The integrality gap did not widen with start-up cost. The sweep varied only the *cheap* unit's start cost, while the entire gap came from the *other* unit's fractional start-ups | **High** | Absolute gap exactly invariant at 77.2785 EUR across a 0 → 10,000 sweep; relative gap *shrinking*, and the notebook printed "the gap is 0.30%; at 2000 EUR it has grown to 0.28%" | Sweeps every unit's start cost. Gap is now 0.00% at zero start cost (so "nothing to cheat on" is literally true) rising monotonically to 3.53%. The validation cell now asserts tightness at zero and monotone widening |
| **R-05** | `tutorials/_sources/07` | The wind forecast error entered the recourse with the **same sign** as the demand errors, so the system did not balance in any realisation — contradicting the notebook's explicit claim that `sum(alpha) = 1` balances every realisation | **High** | Mismatch is exactly `2*xi_wind` in every draw. The safety margin was also 51% too large: `sqrt(b'.Sigma.b)` is 20.474 MW with `(1,1,1)` against 13.565 MW with the correct `(-1,+1,+1)` | A single `BALANCE = (-1, +1, +1)` vector used at all seven sites, with the formulas in the markdown updated to `b` and the correlation discussion rewritten (with the correct sign the total is now *below* the independent equivalent, because positively-correlated terms of opposite sign partly cancel) |
| **R-06** | `tutorials/_sources/07` | The reported Boole guarantee counted `N` constraints while the model imposes `2N` — an upper **and** a lower limit per generator — halving the stated bound | **High** | Internally inconsistent in the same notebook: the allocation uses `n_constraints = 2 * N_BUS` and the interpretation says "all 6 limits", while the reported guarantee used 3 entries | `boole_joint_bound(np.repeat(..., 2))` at both reporting sites, matching the allocation |
| **R-07** | `tutorials/_sources/09` | The explanation for `z_LP == z_DW` invoked the integrality property, which two of the three subproblems measurably lack | **Medium** | 300 random price vectors per generator: worst MILP − LP is 204.80 (coal), 218.45 (gas), 0.00 (peak) | Text and takeaway rewritten to give the instance-specific reason and show the counterexample. **This also corrects an overclaim in this report** — see §10 |
| **R-08** | `tutorials/_sources/05` (×2), `10` (×2) | Four `relaxation_gap` call sites sorted their arguments with `min`/`max`, defeating the function's own guard and relabelling the AC value as the SOC lower bound whenever solver noise crossed them | **Medium** | `relaxation_gap(100, 90)` raises; `relaxation_gap(min(100,90), max(100,90))` returns 0.1. On this instance the crossing fires every run | All four pass their arguments in the stated order. `relaxation_gap` now also clamps its result at zero, for the same reason `CertifiedInterval.width` does |
| **R-09** | `tutorials/_sources/08` | The culminating certificate chain printed `LB_SOC = 0.22935436 <= z* <= UB_AC = 0.22935428` — an **empty** interval — with `abs()` hiding the sign in the width | **Medium** | Full-run output shows the inverted chain and a "width" of `7.4e-08` that is really a crossing | Routed through `certified_interval`, whose guard raises on a genuine violation and whose width is clamped. Now prints width `0.000e+00` |
| **R-10** | `tutorials/_sources/07` | The scenario sweep drew an **independent** sample per size (`seed = 1000 + n`), so the sets were not nested and the monotonicity its validation cell asserts did not follow from anything | **High** | Found while fixing R-05: the previous numbers (3910.77 → 4020.99 → 4079.76 → 4396.78) were monotone by luck; the corrected wind sign changed the draws and the assertion failed immediately | Draws the largest sample once and takes prefixes, so `S_20 ⊂ S_50 ⊂ S_200 ⊂ S_1000` and monotonicity is a theorem. Costs now rise 3759 → 3761 → 3890 → 4074 and out-of-sample violation falls 0.0625 → 0.0274 → 0.0017 → 0.0001 |

### Found by the independent reviewer, re-tested, and NOT confirmed

| Reviewer finding | Re-test | Disposition |
|---|---|---|
| "Every IPOPT solve reports `gap=inf%`" | A fresh NLP solve returns `SolveRecord.gap = None`, not infinite | Not reproduced as stated. Recorded rather than acted on; a defensive `isfinite` guard would be harmless but was not applied on the strength of an unreproduced report |

### Found by this audit

| ID | Location | Issue | Severity | Verification | Resolution |
|---|---|---|---|---|---|
| **A-01** | `tutorials/_sources/03`, `05` | Three validation cells assert on names the student notebook never introduces (`ptdf_record`, `exact_residuals`, `broken_record`, `broken_residuals`); correct work raises `NameError` | **High** | Each name occurs only inside its own validation cell in the generated exercise notebook | Scaffolds seed every validated name with a comment naming the deliverable; `audit_exercises.py` now enforces this |
| **A-02** | `src/psopt_course/metrics.py` | `CertifiedInterval.width` could return a negative number, and `relative_width` a negative gap that satisfies any "gap < tol" test | **Medium** | Fires on two real course instances: T05 (`z_SOC - z_BFM = +7.56e-08`) and T08 (`z_SOC - z_AC = +7.40e-08`) | `width` clamped at zero, reasoning documented in the property; guard moved to `tolerances.bound_holds` |
| **A-03** | `src/psopt_course/validation.py` | `bounds_table` guarded `lower <= upper` at `1e-9`, tighter than either solve, raising `ValueError` on numerically correct bounds | **Medium** | Direct call with T05's measured pair raised | Uses `tolerances.bound_holds`, the same guard as `metrics` |
| **A-04** | `tutorials/_sources/10` | Certified interval built from `min(z_soc, z_ac), max(...)`, silently repairing a violation of the ordering the section teaches | **Medium** | Code inspection: a misspecified relaxation above the feasible point would be swapped into place and still print a tidy interval | Passes `(z_soc, z_ac)` in the stated order; the now tolerance-aware guard raises on a genuine violation. Re-executed: `z* in [0.227275, 0.227275]`, width 0, residual `4.241e-07` |
| **A-05** | `metrics.py`, `validation.py` | Tolerance literals duplicated rather than drawn from the central module | **Low** | `grep` found `1e-6` in three places and `1e-9` in one, for two concepts | Sourced from `tolerances.py` |

### Found in the audit's own instruments

Recorded because three of these produced **false failures against the course**,
and an audit whose instruments are unexamined is not evidence.

| ID | Instrument | Defect | Effect | Resolution |
|---|---|---|---|---|
| **A-06** | `scripts/audit.py` | `zip(costs, costs[1:], strict=True)` | `ValueError`, harness would not run | `itertools.pairwise` |
| **A-07** | `scripts/audit_exercises.py` | Difficulty vocabulary omitted "Basic" | 6 false failures; all 32 tasks are in fact labelled | Vocabulary corrected |
| **A-08** | `scripts/audit_exercises.py` | Builtins not excluded; interpretation answers treated as orphaned solutions | 34 false failures (`map`, plus 33 legitimate interpretation answers) | `builtins` module used; interpretation answers now checked for a visible question instead |
| **A-09** | one-off SOC/AC ordering check | Raw `1e-9` guard, tighter than either solve | Falsely reported a violated bound at `7.4e-08` | Re-checked with `tolerances.bound_holds` — the mistake `tolerances.py` exists to prevent |

### Found during development, before this audit

Listed because the brief asks that issues discovered and fixed not be hidden.
Each was found by a check that failed, not by inspection.

| ID | Location | Issue | Severity |
|---|---|---|---|
| **D-01** | T01 | By-hand merit-order fill capped the cheap unit without passing the remainder on, dispatching 70 MW for 100 MW of demand; the notebook printed "all four methods agree: **False**" beneath prose claiming they agreed | **Critical** |
| **D-02** | T01 | Text named the cheapest unit as marginal while the measured shadow price was 55, the gas unit's cost | **High** |
| **D-03** | `networks.py` | `susceptance` returned per-unit `1/x` while powers were in MW; DC-OPF needed ~10-radian angles and was silently infeasible | **Critical** |
| **D-04** | T03 | Exercise asserted angle differences grow with load; measured 5.35° → 4.58° → 3.82°, i.e. they shrink, because load is met locally | **High** |
| **D-05** | `solvers.py` | Legacy `gurobi`/`gurobi_direct` returned wrong objectives on a rotated cone with an unmapped `other` status | **High** |
| **D-06** | `solvers.py` | `appsi_gurobi` returns `unknown` on 4 of 8 SOCP patterns | **Medium** |
| **D-07** | T05 | Bound check used `1e-9` absolute, stricter than either solve; printed `False` for objectives agreeing to `7.56e-07` | **Medium** |
| **D-08** | T07 | Monotonicity assertion assumed the sweep arrived in order | **Low** |
| **D-09** | T07 | `in-sample <= out-of-sample` fails at `N = 1000` where both are ~`1e-3` | **Medium** |
| **D-10** | T09 | Two integer-recovery approaches fail; **retained deliberately** as the section's content | — |
| **D-11** | T10 | An absolute local path leaked into a PGLib message | **Low** |
| **D-12** | `networks.py` | `net.deepcopy()` removed in pandapower 3.x | **Low** |
| **D-13** | T04 | Zero-width histogram when all 20 multi-starts found one optimum | **Low** |
| **D-14** | CI | `coinor-libipopt-dev` ships the library, not the `ipopt` executable; the idaes binary then needed `liblapack3`/`libblas3`/`libgfortran5` | **Low** |
| **D-15** | `tests/` | A test asserted `susceptance > 100 * susceptance_pu` when the ratio is exactly 100 | **Low** |

### Open, recorded, not resolved

The reviewer also reported the following, at Medium or below. They are recorded
rather than fixed, and the course ships with them.

| Location | Issue | Severity |
|---|---|---|
| `tutorials/_sources/06` | `dt` is hard-coded to 1.0 while `daily_profiles(n)` always spans 24 h and the objective carries no `dt`, so a 96-period run prices a 96-hour day. Knock-on: fast mode then changes the **physics** (a 12-hour day), which contradicts the guarantee stated in `config.py`, the README and `course_overview.md` §8 | **Medium** |
| `tutorials/_sources/10` §7 | The scalability sweep labels a point "16 scenarios" while `N_SCENARIOS` caps it at 12, so the reported `S^0.92` is sub-linear partly because the last point does 25% less work than its label | **Medium** |
| `tutorials/_sources/10` §5 | The chance-constrained voltage experiment is vacuous on this feeder: deterministic and chance-constrained give identical cost and zero violations, because the tightened floor sits ~10 sigma from the realised minimum | **Medium** |
| `src/psopt_course/validation.py` | `constraint_residuals` reports `feasible` for an **unsolved** model: `pyo.value` raises on uninitialised variables and the exception is swallowed, leaving the worst violation at 0.0. It also never checks variable bounds | **Medium** |
| `src/psopt_course/config.py` | `set_seed()` seeds `random` and the legacy `np.random` global state, neither of which anything in the repo uses — every draw goes through `default_rng(<explicit seed>)`. Reproducibility is real (§12) but not for the reason the docstring gives | **Medium** |
| `docs/course_overview.md` | Asserts two T04 measurements T04 does not make — that multi-start finds *different* local optima (it finds one, every time) and that a converged local solution can be worse than a good LP (no such experiment exists). `instructor_guide.md` states the first correctly, so the two docs contradict each other | **Medium** |
| `src/psopt_course/relaxations.py` | The branch-flow model silently ignores line charging susceptance (`BR_B`) and shunts. `case33bw` has neither, so nothing in the course is affected, but the model should refuse such a network the way it refuses transformers | **Medium** |
| `src/psopt_course/relaxations.py` | With `allow_meshed=True` the `"exact"` variant is **itself a relaxation** of AC: the cycle angle-consistency condition is absent, so recovered angle differences need not close around a loop. The docstring implies otherwise | **Medium** |
| `tutorials/_sources/07` | The "within budget" column compares one combined two-sided residual per generator against one epsilon, so a generator whose two limits each carry probability can exceed its budget and be reported correctly as out of budget for the wrong reason | **Medium** |
| `tutorials/_sources/07` | In the deterministic case every margin is zero, so `alpha` appears in no constraint but `sum(alpha) = 1` and is fully degenerate. The headline "violates 49.6% of the time" is therefore a solver tie-break, not a property of the model | **Medium** |
| `tutorials/_sources/03` | The markdown gives the PTDF as `B_d A (A' B_d A)^+` while the code uses the reference-deleted inverse. These are different matrices (the pseudo-inverse gives a distributed-slack PTDF); they agree on flows for balanced injections, which is why nothing downstream breaks | **Medium** |
| `tutorials/_sources/05` §6 | The Slater diagnostic counts branches "strictly inside the cone" at a `1e-9` threshold, far below IPOPT's convergence tolerance, so 32/32 is an artefact of the barrier rather than a property of the model | **Medium** |

### Resolution status

All **Critical** and **High** issues are resolved: **R-01** (Critical);
**R-02**, **R-03**, **R-04**, **R-05**, **R-06**, **R-10**, **A-01**, **D-02**,
**D-04**, **D-05** (High); **D-01**, **D-03** were Critical and were resolved
during development. The Medium issues found by this audit's own harnesses
(**A-02** to **A-05**) and the Medium reviewer findings **R-07** to **R-09**
are resolved. The twelve Medium items in the table immediately above are
**open**. **D-10** is retained by design and labelled as such in the notebook.

---

## 15. Verification matrix

| Requirement | Method | Evidence | Status |
|---|---|---|---|
| Units, signs, per-unit | Branch-flow model vs independent power flow | `3.20e-09` pu; losses `1.38e-08` MW | ✅ |
| Convexity claims | Grep every claim, read in context, check against definitions | §5; T08 table correct entry by entry | ✅ |
| Relaxation vs approximation | Bound, residual sign, tightness, constraint-block identity, destructive test | `z_SOC <= z_BFM`; residuals `+1.39e-08` to `4.24e-07`; identical blocks; 69× losses when broken | ✅ |
| SOC residuals | Direct computation over all branches | min `+1.39e-08`, max `4.24e-07` | ✅ |
| Bound orderings | Measured at every stated relationship | §8 | ✅ |
| Duals by finite difference | Perturb each right-hand side, re-solve | λ rel. err `5.29e-12`; LMPs `2.18e-11` | ✅ |
| KKT residuals | Computed, not asserted | all four `0.00e+00` | ✅ |
| MILP/LP bounds | Solve both; check integrality | `17041.711142 <= 17122.281643`; integrality `0.00e+00` | ✅ |
| Benders (LP) vs monolithic | Full enumeration of `Q(y)` | difference `0.00e+00`, y = 000 | ✅ |
| Benders (SOCP) vs monolithic | Full enumeration of `Q(y)` | difference `0.00e+00`, y = 101 | ✅ |
| Benders (AC) vs monolithic | Full enumeration of `Q(y)` | difference `0.00e+00`, y = 101 — agreement, **not** certification | ✅ |
| Benders cut validity | Every cut at every vertex, 8×8 | worst violation `+0.000e+00` | ✅ |
| Benders cut signs | Coefficients vs measured `dQ` | signs consistent; gap explained by convexity of `Q` | ✅ |
| Chance constraints | Monte Carlo, `N = 200,000` | 0.10054 / 0.05040 / 0.01009 vs 0.10 / 0.05 / 0.01 | ✅ |
| Joint vs individual | Monte Carlo at two correlations | joint 0.1425 (ρ=0), 0.1111 (ρ=0.6), Boole 0.15 | ✅ |
| Covariance propagation | Analytical vs NumPy vs sampling | `0.00e+00`; sampler rel. err `0.0004` | ✅ |
| Column-generation reduced costs | Both directions: termination and improvement | `+0.000e+00` at termination; three columns each lower the master | ✅ |
| Column feasibility | Re-check all 29 columns independently | all 29 satisfy limits, on/off and min up-time | ✅ |
| Column generation vs full pool | LP over all 111 feasible schedules | `18314.76190476` both, difference `7.28e-12` | ✅ |
| Dantzig-Wolfe bound claim | Measure `z_LP`, `z_DW`, `z_MILP` | coincide to `3.638e-12`; integrality gap 1.055% | ✅ |
| potpourri API | Import against pinned 0.7.0 | 6/6 symbols; `potpourri.research` not importable | ✅ |
| Central tolerances | One module, one constant per concept | `tolerances.py`; **A-03**, **A-05** resolved | ✅ |
| Solver statuses | Infeasible instance; legacy-interface comparison | `infeasible`, `ok=False`; **D-05**, **D-06** resolved | ✅ |
| Reproducibility | Repeat seeded sampling and solves | `0.00e+00`; objectives identical | ✅ |
| Physical plausibility | Balance, loss sign, voltage range | imbalance `7.69e-08`; losses positive; `|V|` in `[0.9131, 1.0000]` | ✅ |
| Notebooks from scratch | Regenerate from output-free sources, execute all | 10/10, 0 failures, `fast_mode() = False` | ✅ |
| Exercise/solution consistency | 9 structural checks over the tagged sources | 9/9 after **A-01** fixed | ✅ |
| Pedagogical statements | Manual review of every strong claim | §5, §6, §9 | ✅ |
| Literature | Bibliographic check of every dated citation | §3, all correct | ✅ |
| Independent reviewer | Commissioned with a concrete assignment; every finding re-tested before acting | 10 findings confirmed and fixed, 1 not reproduced; see §2 | ✅ |
| Narrative matches output | Reviewer read each tutorial's prose against its own printed numbers | 6 contradictions found (**R-01** to **R-06**); this audit's harnesses missed all of them | ✅ |
| Scenario studies actually vary | Objective spread across scenarios | was `3.87e-13`, now `7.79e-02` (**R-01**) | ✅ |

---

## What this audit does not establish

- **Twelve Medium issues are open**, listed in §14. The course ships with them.
  The most consequential is T06's `dt`, which makes fast mode change the
  physics rather than only the scale — a guarantee the course states in three
  places.
- **Checking numbers is not checking narratives.** This audit's 57 harness
  checks verified the library and the algorithms and found four defects. They
  did not find any of **R-01** to **R-06**, because those are failures of prose
  against output, and a harness that recomputes a quantity cannot notice that
  the sentence above it says something else. Every one was found by a reader.
  That is the clearest evidence here for why an independent review was worth
  commissioning, and it is a limitation of the method, not a one-off.
- **Tutorials 01, 04 and 06 and most of `docs/` received a single pass.** The
  reviewer's parallel sub-agent covered them; its findings are recorded but
  were not independently re-tested by this audit except where noted.
- **Exactness is instance-specific.** The SOC relaxation is tight *on the
  feeders and objectives used here*. Tutorial 05's Exercise 5.2 exists to show
  it failing. Nothing in this audit generalises that tightness.
- **The AC Benders level is not certified.** It agreed with enumeration on this
  feeder. Its cuts rest on local KKT multipliers and carry no global guarantee.
- **Correctness is not pedagogy.** This audit establishes that the numbers are
  right and the claims about them are accurate. Whether the course teaches well
  is a separate question that measurement does not answer.
