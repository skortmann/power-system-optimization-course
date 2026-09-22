# Mathematical background

Enough to start Tutorial 01 without a separate optimization textbook. Terse on
purpose: each item says what it is and where the course uses it.

## Vectors, matrices, norms

* $x \in \mathbb{R}^n$, inner product $a^\top x = \sum_i a_i x_i$.
* $\lVert x \rVert_2 = \sqrt{x^\top x}$. The **second-order cone** is
  $\{(t, x) : \lVert x \rVert_2 \le t\}$ — Tutorial 05.
* A matrix $Q$ is **positive semidefinite** ($Q \succeq 0$) when
  $x^\top Q x \ge 0$ for all $x$, equivalently all eigenvalues $\ge 0$. This is
  what decides whether a QP is convex.

## Gradients and Hessians

* $\nabla f(x)$ is the vector of first partials; $\nabla^2 f(x)$ the matrix of
  second partials.
* $f$ is convex on a convex set iff $\nabla^2 f \succeq 0$ there.

## Convex sets and functions

A set $C$ is **convex** when the segment between any two of its points stays
inside it. A function is convex when the region above its graph is a convex set.

Three facts the course leans on:

1. A **linear equality** defines a convex (flat) set. A **nonlinear equality**
   generally does not — this is why AC-OPF is nonconvex (Tutorial 04).
2. An intersection of convex sets is convex. Adding convex constraints keeps a
   problem convex.
3. Integrality is not convex: $\{0, 1\}$ is two points.

**Why it matters:** for a convex problem, a local optimum is global. Nothing
else in optimization buys you as much.

## Lagrangian and duality

For $\min f(x)$ s.t. $g(x) \le 0$, $h(x) = 0$:

$$L(x, \mu, \lambda) = f(x) + \mu^\top g(x) + \lambda^\top h(x)$$

The **dual function** $\inf_x L$ is concave regardless of the primal, and gives

$$z_{dual} \le z^\star \qquad \text{(weak duality, always)}$$

**Strong duality** ($z_{dual} = z^\star$) holds for feasible bounded LPs
automatically, and for convex problems under a regularity condition —
**Slater's**: some point is *strictly* feasible.

Weak duality is what makes a Benders cut valid (Tutorial 08). Strong duality is
what makes it tight.

## KKT conditions

At an optimum of a well-behaved problem:

| | |
|---|---|
| stationarity | $\nabla_x L = 0$ |
| primal feasibility | $g(x) \le 0$, $h(x) = 0$ |
| dual feasibility | $\mu \ge 0$ |
| complementary slackness | $\mu_i\, g_i(x) = 0$ |

Complementary slackness in words: **either a constraint binds, or its price is
zero.** Tutorial 02 verifies all four numerically.

For a **convex** problem KKT is necessary and sufficient. For a nonconvex one it
is only necessary — which is exactly the gap between "IPOPT says optimal" and
"this is the global optimum".

## Shadow prices

The multiplier on a constraint is the derivative of the optimal value with
respect to that constraint's right-hand side. In a power system the multiplier
on nodal balance is the locational marginal price (Tutorial 03).

## Probability

* Mean $\mu$, covariance $\Sigma$ (symmetric, PSD). The **correlation** matrix
  normalises it by the standard deviations.
* For $\xi \sim \mathcal{N}(\mu, \Sigma)$, a linear functional $b^\top \xi$ is
  Gaussian with mean $b^\top\mu$ and variance $b^\top \Sigma b$.
* The **quantile** $\Phi^{-1}(1-\epsilon)$ is the safety factor a chance
  constraint buys: 1.645 at $\epsilon = 0.05$, 2.326 at $\epsilon = 0.01$.

This gives the exact reformulation of a linear chance constraint:

$$\Pr[a^\top x + b^\top\xi \le c] \ge 1-\epsilon
  \iff a^\top x + b^\top\mu + \Phi^{-1}(1-\epsilon)\sqrt{b^\top\Sigma b} \le c$$

Note $\sqrt{b^\top\Sigma b}$ is a second-order cone term — which is why
Tutorial 07 needs the same solvers as Tutorial 05.

**Exact only for linear $g$ and Gaussian $\xi$.** State the assumption whenever
you report the result.

### Boole's inequality

$$\Pr\Big[\bigcup_i A_i\Big] \le \sum_i \Pr[A_i]$$

Distribution-free, and therefore safe and often loose. It converts a joint
chance constraint into individual budgets (Tutorial 07).

## Per unit

Electrical quantities are normalised by a base: $z_{pu} = z / z_{base}$ with
$z_{base} = V_{base}^2 / S_{base}$. It keeps numbers near 1, which matters for
solver conditioning.

Mixing per-unit and physical units is the classic bug. It produced a silently
infeasible DC-OPF during this course's development — susceptance in per unit
against powers in MW needed angles of ten radians. `psopt_course.networks.Branch`
now carries its base explicitly.
