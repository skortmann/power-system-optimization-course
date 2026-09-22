"""Figures that clarify the mathematics, and a house style.

Every function here answers a question the prose cannot answer as well — a
feasible polytope, the gap between a set and its relaxation, a cone. Nothing
here decorates.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "COLORS",
    "plot_feasible_region",
    "plot_mccormick_envelope",
    "plot_relaxation_vs_approximation",
    "plot_second_order_cone",
    "use_course_style",
]

#: A small, stable palette so the same concept keeps the same colour across
#: ten notebooks: primal/feasible, relaxed/bound, infeasible, emphasis.
COLORS = {
    "feasible": "#2b6cb0",
    "relaxed": "#dd6b20",
    "infeasible": "#c53030",
    "optimum": "#2f855a",
    "neutral": "#4a5568",
    "accent": "#805ad5",
}


def use_course_style() -> None:
    """Consistent, readable defaults. Called in every notebook's setup cell."""
    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 4.2),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
        }
    )


def plot_feasible_region(
    constraints,
    objective,
    *,
    bounds=((0.0, 100.0), (0.0, 100.0)),
    ax=None,
    title="Feasible region and objective contours",
    n_contours: int = 12,
):
    r"""Shade $\{x : a^\top x \le b\}$ and draw the objective's contours.

    Parameters
    ----------
    constraints:
        ``[(a1, a2, b, label), ...]`` meaning :math:`a_1 x_1 + a_2 x_2 \le b`.
    objective:
        ``(c1, c2)``; contours of :math:`c^\top x` are drawn, and the optimum of
        an LP sits at a vertex, which the picture should make obvious.
    """
    if ax is None:
        _fig, ax = plt.subplots()

    (x_lo, x_hi), (y_lo, y_hi) = bounds
    grid = np.linspace(x_lo, x_hi, 400), np.linspace(y_lo, y_hi, 400)
    X, Y = np.meshgrid(*grid)

    feasible = np.ones_like(X, dtype=bool)
    for a1, a2, b, _label in constraints:
        feasible &= (a1 * X + a2 * Y <= b + 1e-9)

    ax.contourf(X, Y, feasible.astype(float), levels=[0.5, 1.5],
                colors=[COLORS["feasible"]], alpha=0.22)

    for a1, a2, b, label in constraints:
        if abs(a2) > 1e-12:
            xs = np.array([x_lo, x_hi])
            ax.plot(xs, (b - a1 * xs) / a2, lw=1.4, label=label)
        else:
            ax.axvline(b / a1, lw=1.4, label=label)

    c1, c2 = objective
    Z = c1 * X + c2 * Y
    ax.contour(X, Y, Z, levels=n_contours, colors=COLORS["neutral"],
               alpha=0.45, linewidths=0.7, linestyles="dashed")

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    return ax


def plot_relaxation_vs_approximation(ax=None):
    """The distinction the course refuses to let slide.

    A **relaxation** enlarges the feasible set, so its optimum is a bound. An
    **approximation** moves the set somewhere else, so its optimum is not a
    bound on anything. Students reliably believe DC-OPF gives a bound and a
    convex relaxation gives an approximate solution; both are backwards.
    """
    if ax is None:
        _fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.0))
    left, right = ax

    theta = np.linspace(0, 2 * np.pi, 400)
    # A nonconvex "true" set: a star-ish blob.
    r_true = 1.0 + 0.30 * np.cos(3 * theta)
    x_true, y_true = r_true * np.cos(theta), r_true * np.sin(theta)

    # Relaxation: a convex set that CONTAINS it.
    r_relax = np.full_like(theta, r_true.max())
    x_rel, y_rel = r_relax * np.cos(theta), r_relax * np.sin(theta)

    left.fill(x_rel, y_rel, color=COLORS["relaxed"], alpha=0.22,
              label=r"relaxed set $\mathcal{F}_{relax}$")
    left.fill(x_true, y_true, color=COLORS["feasible"], alpha=0.55,
              label=r"true set $\mathcal{F}$")
    left.set_title("Relaxation:  $\\mathcal{F} \\subseteq \\mathcal{F}_{relax}$\n"
                   "$\\Rightarrow z_{relax} \\leq z^\\star$  (a bound)", fontsize=10)

    # Approximation: a set of similar size, somewhere else.
    shift = 0.55
    x_app, y_app = 1.05 * np.cos(theta) + shift, 1.05 * np.sin(theta) - 0.25
    right.fill(x_true, y_true, color=COLORS["feasible"], alpha=0.55,
               label=r"true set $\mathcal{F}$")
    right.fill(x_app, y_app, color=COLORS["infeasible"], alpha=0.30,
               label=r"approximate set")
    right.set_title("Approximation: different equations\n"
                    "$\\Rightarrow$ no bound in either direction", fontsize=10)

    for axis in (left, right):
        axis.set_aspect("equal")
        axis.set_xlim(-1.8, 2.0)
        axis.set_ylim(-1.8, 1.8)
        axis.legend(fontsize=8, loc="lower left")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
    return ax


def plot_second_order_cone(ax=None, *, n: int = 60):
    r"""The Lorentz cone $\{(x_1,x_2,t) : \|x\|_2 \le t\}$.

    Worth seeing once. "SOCP is just another nonlinear solver" survives right up
    until someone looks at the feasible set and notices it is convex, and that
    its boundary is where the interesting solutions sit.
    """
    if ax is None:
        fig = plt.figure(figsize=(5.5, 4.6))
        ax = fig.add_subplot(111, projection="3d")

    t = np.linspace(0, 1.6, n)
    phi = np.linspace(0, 2 * np.pi, n)
    T, PHI = np.meshgrid(t, phi)
    X, Y = T * np.cos(PHI), T * np.sin(PHI)

    ax.plot_surface(X, Y, T, alpha=0.35, color=COLORS["relaxed"],
                    linewidth=0, antialiased=True)
    ax.plot([0], [0], [0], "o", color=COLORS["neutral"], ms=4)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$t$")
    ax.set_title(r"$\|x\|_2 \leq t$  —  convex, and a cone")
    return ax


def plot_mccormick_envelope(
    x_bounds=(1.0, 4.0), y_bounds=(1.0, 3.0), ax=None, *, n: int = 40
):
    r"""The convex envelope of $w = xy$ over a box.

    Shows the thing students most need to feel about relaxations: the envelope
    is only as tight as the variable *bounds*. Widen the box and the relaxation
    degrades, which is why bound tightening is not housekeeping — it is where
    most of the strength of a relaxation comes from.
    """
    if ax is None:
        fig = plt.figure(figsize=(6.0, 4.8))
        ax = fig.add_subplot(111, projection="3d")

    xl, xu = x_bounds
    yl, yu = y_bounds
    X, Y = np.meshgrid(np.linspace(xl, xu, n), np.linspace(yl, yu, n))

    ax.plot_surface(X, Y, X * Y, alpha=0.75, color=COLORS["feasible"],
                    linewidth=0, label="$w = xy$")

    # The two lower McCormick faces; their max is the convex envelope.
    lower = np.maximum(xl * Y + X * yl - xl * yl, xu * Y + X * yu - xu * yu)
    ax.plot_surface(X, Y, lower, alpha=0.30, color=COLORS["relaxed"], linewidth=0)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$w$")
    ax.set_title(
        f"McCormick envelope of $w=xy$\n"
        f"$x\\in[{xl:g},{xu:g}]$, $y\\in[{yl:g},{yu:g}]$ — tightness follows the bounds",
        fontsize=10,
    )
    return ax
