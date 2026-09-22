# Glossary

Where a term is built rather than merely used, the tutorial is named.

## Problem classes

**LP** — linear objective, linear constraints. Convex. *T01, T03*
**QP** — quadratic objective, linear constraints. Convex iff the form is PSD. *T01*
**QCQP** — quadratic constraints. Convex only in special cases. *T05*
**SOCP** — second-order cone constraints. Convex by construction. *T05*
**SDP** — positive-semidefinite matrix variables. Convex. Mentioned in *T05*
**NLP** — general smooth nonlinear. Convex or not. *T04*
**MILP** — LP with integer variables. *T02, T09*
**MISOCP** — SOCP with integer variables. *T08*
**MINLP** — NLP with integer variables.

## Optimization

**feasible** — satisfies every constraint.
**optimal** — feasible, and no feasible point is better *for this model*.
**local optimum** — no *nearby* feasible point is better.
**global optimum** — no feasible point anywhere is better. Equal to the local one for convex problems.
**convex** — a set whose segments stay inside it; a function whose epigraph is such a set.
**constraint qualification** — a regularity condition making KKT necessary (e.g. Slater's).
**KKT** — stationarity, primal and dual feasibility, complementary slackness. *T02*
**Lagrange multiplier** — the price attached to a constraint.
**shadow price** — the same number, read as ∂(optimal value)/∂(right-hand side).
**primal / dual** — the original problem and its multiplier problem.
**weak duality** — dual ≤ primal, always. The basis of Benders. *T02, T08*
**strong duality** — dual = primal, under conditions.
**Slater's condition** — a strictly feasible point exists. *T05*
**complementary slackness** — either a constraint binds or its price is zero. *T02*

## Bounds and gaps

**relaxation** — enlarges the feasible set. Gives a bound. *T02, T05*
**approximation** — changes the equations. Gives **no** bound. *T03*
**LP relaxation** — drop integrality. A lower bound for minimization. *T02*
**convex relaxation** — replace a nonconvex set by a convex superset. *T05*
**McCormick envelope** — the convex hull of a bilinear term over a box. *T05*
**integrality gap** — MILP optimum vs its LP relaxation. A property of the *formulation*.
**relaxation gap** — a relaxed optimum vs a feasible point of the original. *T05, T08*
**optimality gap** — incumbent vs best bound. "Has the search finished?"
**incumbent** — the best feasible solution found so far; an upper bound.
**certified interval** — [lower bound, upper bound] containing the true optimum. *T08, T10*

## Algorithms

**simplex** — moves between vertices of an LP.
**interior point** — approaches the optimum through the interior.
**branch-and-bound** — split the discrete space, prune with bounds. *T02*
**branch-and-price** — branch-and-bound with column generation at each node. *T09*
**Benders decomposition** — project out the subproblem, add cuts. *T08*
**Generalized Benders** — the same for convex nonlinear subproblems. *T08*
**Benders cut** — an inequality on the value function, built from duals. *T08*
**feasibility cut** — removes master decisions with no feasible operation.
**optimality cut** — improves the master's approximation of operational cost.
**Dantzig-Wolfe** — reformulate over convex combinations of patterns. *T09*
**column generation** — solve the LP by generating variables on demand. *T09*
**restricted master** — the master with only the columns generated so far. *T09*
**pricing problem** — finds a column of negative reduced cost. *T09*
**reduced cost** — a column's cost minus its value at the current prices. *T09*
**Big-M** — a constant linking a binary to a continuous bound. Loose values weaken the relaxation.

## Power systems

**OPF** — optimal power flow.
**DC-OPF** — the linear approximation. Four assumptions, no losses, no voltage. *T03*
**AC-OPF** — the full nonconvex formulation. *T04*
**branch-flow / DistFlow** — a formulation in squared voltage and current. *T05*
**LinDistFlow** — its linearisation. An approximation. *T08, T10*
**SOC-BFM** — the second-order cone relaxation of the branch-flow model. *T05*
**PTDF** — how an injection at one bus distributes across lines. *T03*
**LMP** — locational marginal price: the dual of nodal balance. *T03*
**congestion** — the dual of a binding line limit, and the reason prices differ. *T03*
**unit commitment** — the on/off scheduling problem. *T02, T09*
**per unit** — normalisation by a base. *T03*
**state of charge** — stored energy; the variable that couples periods. *T06*
**exactness** (of a relaxation) — the relaxed constraint binds at the optimum, so the point is physical. *T05*

## Uncertainty

**scenario** — one sampled realisation. *T07*
**recourse** — a decision made after uncertainty resolves. *T07*
**participation factor** — a generator's share of the imbalance. *T07*
**chance constraint** — Pr[constraint holds] ≥ 1−ε. *T07*
**joint chance constraint** — Pr[**all** constraints hold] ≥ 1−ε. Not the same. *T07*
**violation probability** (ε) — the allowed failure rate.
**risk allocation** — splitting a joint budget across constraints. *T07*
**Boole's inequality** — the union bound. Distribution-free, often conservative. *T07*
**out-of-sample validation** — measuring on samples the optimizer never saw. Mandatory. *T07*
