"""Exercise/solution consistency audit (brief section 36).

Both notebooks are generated from one tagged source, so they cannot drift the
way hand-maintained pairs do. They can still be wrong in ways the generator
does not catch, and those are what this checks:

* a scaffold with no solution behind it, so the instructor has nothing;
* a scaffold that contains its own answer, so the exercise is not an exercise;
* a validation cell that asserts nothing, so it can never fail;
* a validation cell that checks a name the scaffold never tells the student to
  create -- the student writes correct code, calls the variable something else,
  and the check fails anyway. This is the failure that makes students distrust
  the validation cells, so it is checked by name;
* a task with no difficulty label or no stated deliverable.

Run it as-is::

    uv run python scripts/audit_exercises.py
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Configuration -- edit these; there are no command-line flags.
# --------------------------------------------------------------------------

#: Words that mark a difficulty label in the markdown introducing a task.
DIFFICULTY_WORDS = ("basic", "intermediate", "advanced", "research")

#: Text that would give the answer away inside a student scaffold.
GIVEAWAY_PATTERNS = ("# Solution:", "ANSWER.", "#: solution")

#: How many cells above a scaffold count as its introduction.
CONTEXT_CELLS = 4

# --------------------------------------------------------------------------

import ast
import builtins
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import jupytext

from psopt_course.config import PROJECT_ROOT

SOURCE_DIR = PROJECT_ROOT / "tutorials" / "_sources"
RESULTS: list[dict] = []


def record(check: str, verdict: str, evidence: str) -> None:
    RESULTS.append({"check": check, "verdict": verdict, "evidence": evidence})
    print(f"  [{verdict}] {check}\n         {evidence}")


def tags(cell) -> list[str]:
    return list(cell.get("metadata", {}).get("tags", []) or [])


def assigned_names(source: str) -> set[str]:
    """Names bound at any level of a code cell, via the AST rather than regex."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set(re.findall(r"^\s*(\w+)\s*=", source, re.M))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            found.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                found.add(alias.asname or alias.name.split(".")[0])
    return found


def asserted_names(source: str) -> set[str]:
    """Top-level names an ``assert`` statement depends on."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            for inner in ast.walk(node.test):
                if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load):
                    found.add(inner.id)
    return found


#: Builtins plus the course's standard imports. None of these is ever
#: something a student is asked to produce, so an assertion mentioning one
#: says nothing about whether the task named its deliverable.
IGNORED = set(dir(builtins)) | {
    "np", "pd", "plt", "pyo", "tol", "math", "itertools", "copy", "warnings",
    "Path", "pytest", "nan", "inf",
}


def audit() -> int:
    sources = sorted(SOURCE_DIR.glob("[0-9][0-9]_*.py"))
    print(f"auditing {len(sources)} tagged sources\n")

    n_tasks = 0
    per_tutorial: dict[str, int] = {}
    orphan_scaffolds: list[str] = []
    orphan_solutions: list[str] = []
    giveaways: list[str] = []
    empty_validations: list[str] = []
    unnamed_targets: list[str] = []
    missing_difficulty: list[str] = []
    missing_deliverable: list[str] = []
    n_validations = 0
    n_interpretations = 0
    silent_interpretations: list[str] = []

    for path in sources:
        cells = jupytext.read(path, fmt="py:percent").cells
        per_tutorial[path.stem.split("_")[0]] = sum(
            1 for c in cells if {"exercise", "advanced-exercise"} & set(tags(c))
        )
        n_tasks += per_tutorial[path.stem.split("_")[0]]

        # Names a STUDENT has, i.e. everything except the solution cells.
        student_names: set[str] = set()
        for cell in cells:
            if cell["cell_type"] == "code" and "solution" not in tags(cell):
                student_names |= assigned_names(cell["source"])

        for i, cell in enumerate(cells):
            t = set(tags(cell))

            if {"exercise", "advanced-exercise"} & t:
                after = [set(tags(c)) for c in cells[i + 1 : i + 4]]
                if not any("solution" in a for a in after):
                    orphan_scaffolds.append(f"{path.name}:{i}")
                for pattern in GIVEAWAY_PATTERNS:
                    if pattern in cell["source"]:
                        giveaways.append(f"{path.name}:{i} {pattern!r}")

                context = " ".join(
                    c["source"].lower()
                    for c in cells[max(0, i - CONTEXT_CELLS) : i]
                    if c["cell_type"] == "markdown"
                )
                if not any(word in context for word in DIFFICULTY_WORDS):
                    missing_difficulty.append(f"{path.name}:{i}")
                if not re.search(r"\b(task|exercise)\b", context):
                    missing_deliverable.append(f"{path.name}:{i}")

            if "solution" in t:
                # Two kinds of solution cell live in these sources: the worked
                # implementation behind a scaffold, and the answer to an
                # interpretation question. The second is legitimately unpaired,
                # but it is only useful if the STUDENT can see the question.
                previous_markdown = next(
                    (c for c in reversed(cells[:i]) if c["cell_type"] == "markdown"),
                    None,
                )
                is_interpretation = previous_markdown is not None and (
                    "interpretation" in previous_markdown["source"].lower()
                )
                if is_interpretation:
                    n_interpretations += 1
                    body = previous_markdown["source"]
                    if "?" not in body:
                        silent_interpretations.append(f"{path.name}:{i}")
                    continue
                before = [set(tags(c)) for c in cells[max(0, i - 3) : i]]
                if not any({"exercise", "advanced-exercise"} & b for b in before):
                    orphan_solutions.append(f"{path.name}:{i}")

            if "validation" in t:
                n_validations += 1
                if "assert" not in cell["source"]:
                    empty_validations.append(f"{path.name}:{i}")
                    continue
                # Every name the assertion leans on must be one the student's
                # own notebook could plausibly have bound.
                missing = {
                    name
                    for name in asserted_names(cell["source"])
                    if name not in student_names and name not in IGNORED
                }
                if missing:
                    unnamed_targets.append(f"{path.name}:{i} -> {sorted(missing)}")

    print("1. Task inventory")
    record(
        "every tutorial carries tasks",
        "PASS" if all(v > 0 for v in per_tutorial.values()) else "FAIL",
        f"{n_tasks} tasks, {n_validations} validation cells; per tutorial: "
        + ", ".join(f"{k}={v}" for k, v in sorted(per_tutorial.items())),
    )

    print("\n2. Pairing between the two notebooks")
    record(
        "every scaffold has a solution behind it",
        "PASS" if not orphan_scaffolds else "FAIL",
        "no orphans" if not orphan_scaffolds else "; ".join(orphan_scaffolds),
    )
    record(
        "every implementation solution has a scaffold in front of it",
        "PASS" if not orphan_solutions else "FAIL",
        "no orphans" if not orphan_solutions else "; ".join(orphan_solutions),
    )
    record(
        "every interpretation answer has a question the student can see",
        "PASS" if not silent_interpretations else "FAIL",
        f"all {n_interpretations} interpretation answers follow markdown that "
        "asks something"
        if not silent_interpretations
        else "; ".join(silent_interpretations),
    )

    print("\n3. The exercise does not contain its own answer")
    record(
        "no scaffold leaks a solution marker",
        "PASS" if not giveaways else "FAIL",
        "clean" if not giveaways else "; ".join(giveaways),
    )

    print("\n4. Validation cells")
    record(
        "every validation cell asserts something",
        "PASS" if not empty_validations else "FAIL",
        f"all {n_validations} contain an assert"
        if not empty_validations
        else "; ".join(empty_validations),
    )
    record(
        "every validated name is one the scaffold asks the student to create",
        "PASS" if not unnamed_targets else "FAIL",
        f"all {n_validations} validation cells check names the student's own "
        "notebook binds"
        if not unnamed_targets
        else "; ".join(unnamed_targets),
    )

    print("\n5. Task descriptions")
    record(
        "every task carries a difficulty label",
        "PASS" if not missing_difficulty else "FAIL",
        f"all {n_tasks} labelled" if not missing_difficulty
        else "; ".join(missing_difficulty),
    )
    record(
        "every task states what to produce",
        "PASS" if not missing_deliverable else "FAIL",
        f"all {n_tasks} state a deliverable" if not missing_deliverable
        else "; ".join(missing_deliverable),
    )

    print("\n" + "=" * 78)
    counts: dict[str, int] = {}
    for row in RESULTS:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print(f"{len(RESULTS)} checks: "
          + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    print("=" * 78)
    return 1 if any(r["verdict"] == "FAIL" for r in RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(audit())
