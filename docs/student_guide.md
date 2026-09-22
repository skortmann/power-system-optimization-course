# Student guide

## Which notebooks do I open?

```text
tutorials/
├── 01_economic_dispatch_exercise.ipynb     <- work in these
├── 01_economic_dispatch_solution.ipynb     <- check against these, afterwards
└── ...
```

Open the **`_exercise`** notebook. Every task has a scaffold marked
`# TODO: ...`, and a markdown cell above it saying what to implement, why it
matters and what to expect.

The `_solution` notebooks are complete worked versions. Whether you have them
from day one is your instructor's decision. If you do: **attempt the exercise
first.** Reading a solution feels like learning and is not — the gap between
recognising a correct answer and producing one is the entire skill.

## How to work through an exercise

1. **Read the formulation.** Every task states the mathematics before the code.
   If you cannot write the constraint on paper, writing it in Pyomo will not
   help.
2. **Implement the TODO.**
3. **Run the validation cell.** Most tasks have one.
4. **Inspect the result physically.** Are the voltages plausible? Is the
   dispatch in merit order? Does the cost move the way you expected?
5. **Answer the interpretation question.** These are the point. They are also
   the ones worth arguing about with someone else.

> **Passing an assertion does not mean the model is correct.** The checks verify
> shapes, signs and obvious invariants. They cannot tell you that you modelled
> the wrong thing. Several exercises in this course produce solutions that pass
> every check and are physically meaningless — deliberately.

## What the labels mean

**Difficulty: Basic** — a direct application of what the notebook just showed.

**Difficulty: Intermediate** — combine two ideas, or get an index right.

**Difficulty: Advanced** — open-ended, or the obvious answer is wrong.

## Before you start

```bash
uv sync
uv run python scripts/check_solvers.py
uv run jupyter lab
```

`check_solvers.py` tells you which parts of the course your machine can run.
**IPOPT is the one that catches people** — it is a system binary rather than a
Python wheel, and roughly half the course needs it. `docs/solver_guide.md` has
the installation routes.

Running short on time or memory?

```bash
PSOPT_FAST=1 uv run jupyter lab
```

Fast mode shrinks horizons, scenario counts and iteration caps. It **never**
changes a formulation — every mathematical claim in the course is tested in both
modes, just on a smaller instance.

## Habits this course is trying to build

1. **Name the problem class before choosing a solver.** LP, QP, MILP, NLP,
   SOCP. The class decides which algorithms apply; the application does not.
2. **Read the termination condition before the objective.** A number from an
   infeasible run is still a float, and it will propagate into your plot without
   complaint.
3. **Know whether your simplification is a relaxation or an approximation.** One
   gives a bound; the other gives nothing. They are routinely confused in both
   directions.
4. **Validate out of sample.** A violation probability you typed into a model is
   not a result.
5. **When two models disagree, suspect the models first.** An apparent
   "relaxation gap" is usually a limit or a sign convention that differs.

## If you get stuck

- Re-read the mathematics above the cell. The notebooks state every formulation
  before coding it.
- Check units. Per unit versus MW is the single most common bug in this subject,
  and it produces *silently infeasible* models rather than error messages.
- Check the termination condition.
- Look at `psopt_course/` — the helper package contains data handling, plotting
  and validation, deliberately **not** the formulations. If you find yourself
  wanting to import the answer, that is the exercise.

## Reading order

The tutorials are cumulative and the later ones genuinely depend on the earlier
ones. Tutorial 08 needs duality (02), AC-OPF (04) and the SOC relaxation (05)
simultaneously.

If you are short on time and want the argument rather than the full course:
**01 → 02 → 05 → 08**.
