"""Build the solution and exercise notebooks from one tagged source.

Each tutorial is authored once, as a jupytext percent file under
``tutorials/_sources/``. Cells carry tags that say who they are for::

    provided            infrastructure; appears verbatim in both notebooks
    solution            the worked implementation; solution notebook only
    exercise            the scaffold with TODOs; exercise notebook only
    advanced-exercise   as above, for optional harder material
    validation          a lightweight assert; appears in BOTH notebooks

A ``solution`` cell and the ``exercise`` cell before it are a pair: the student
sees the scaffold, the instructor sees the implementation, and both read the
same markdown above them. That is what keeps the two notebooks synchronized —
they are not two files being maintained in parallel.

Run it as-is::

    uv run python scripts/build_exercise_notebooks.py

or import it::

    from build_exercise_notebooks import build, check
    build(only=("01_",), execute=False)
    stale = check()          # CI uses this
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jupytext
import nbformat

from psopt_course.config import PROJECT_ROOT, fast_mode

# --------------------------------------------------------------------------
# Configuration — edit these; there are no command-line flags.
# --------------------------------------------------------------------------

#: Restrict the build to sources whose filename starts with one of these.
#: Empty tuple means "all of them".
ONLY: tuple[str, ...] = ()

#: Execute the solution notebooks after building. The exercise notebooks are
#: never executed past their first incomplete cell, by design.
EXECUTE = True

#: Seconds before a single cell is abandoned.
TIMEOUT = 1800

# --------------------------------------------------------------------------

SOURCE_DIR = PROJECT_ROOT / "tutorials" / "_sources"
OUTPUT_DIR = PROJECT_ROOT / "tutorials"

SOLUTION_SUFFIX = "_solution.ipynb"
EXERCISE_SUFFIX = "_exercise.ipynb"

#: Text that must never appear in a student notebook's code cells.
LEAK_MARKERS = ("ANSWER.", "#: solution")


def _tags(cell) -> list[str]:
    return list(cell.get("metadata", {}).get("tags", []) or [])


def _scaffold_for(solution_cell) -> str:
    """What the student sees where the solution would be.

    Only used when an author wrote a ``solution`` cell without pairing it with
    an ``exercise`` cell. A bare ``# TODO`` with no context is exactly the
    "partially deleted instructor notebook" the brief warns against, so this
    says what is missing and points at the markdown above it.
    """
    return (
        "# TODO: Implement your solution here.\n"
        "#\n"
        "# The task, the expected result and a hint are in the markdown cell\n"
        "# directly above this one.\n"
    )


def _read_source(path: Path) -> nbformat.NotebookNode:
    notebook = jupytext.read(path, fmt="py:percent")
    notebook.metadata.setdefault(
        "kernelspec",
        {"display_name": "Python 3", "language": "python", "name": "python3"},
    )
    notebook.metadata.setdefault("language_info", {"name": "python", "version": "3.12"})
    return notebook


def _solution_notebook(source: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Everything except the student scaffolds."""
    cells = [c for c in source.cells if "exercise" not in _tags(c) or "solution" in _tags(c)]
    # An `exercise`-tagged cell is the scaffold; drop it from the solution.
    cells = [c for c in cells if "exercise" not in _tags(c)]
    out = nbformat.v4.new_notebook(cells=cells, metadata=dict(source.metadata))
    return out


def _exercise_notebook(source: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Everything except the worked implementations.

    A ``solution`` cell is dropped when an ``exercise`` scaffold already stands
    in for it, and replaced by a TODO stub when it does not — so a student can
    never be left with a task and nowhere to write.
    """
    cells: list = []
    pending_scaffold = False

    for cell in source.cells:
        tags = _tags(cell)
        if "exercise" in tags or "advanced-exercise" in tags:
            cells.append(cell)
            pending_scaffold = True
            continue
        if "solution" in tags:
            if not pending_scaffold:
                stub = nbformat.v4.new_code_cell(_scaffold_for(cell))
                stub.metadata["tags"] = ["exercise"]
                cells.append(stub)
            pending_scaffold = False
            continue
        cells.append(cell)
        if cell.cell_type == "markdown":
            pending_scaffold = False

    # No outputs anywhere: a student notebook that ships results has leaked.
    for cell in cells:
        if cell.cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    return nbformat.v4.new_notebook(cells=cells, metadata=dict(source.metadata))


def _check_no_leaks(notebook: nbformat.NotebookNode, name: str) -> list[str]:
    """Look for solutions that survived into the student version."""
    problems: list[str] = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        if cell.get("outputs"):
            problems.append(f"{name} cell {index}: has outputs")
        for marker in LEAK_MARKERS:
            if marker in cell.source:
                problems.append(f"{name} cell {index}: contains {marker!r}")
    return problems


def _execute(notebook: nbformat.NotebookNode, workdir: Path, timeout: int) -> float:
    from nbclient import NotebookClient

    started = time.perf_counter()
    NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(workdir)}},
        allow_errors=False,
    ).execute()
    return time.perf_counter() - started


def _sources(only: tuple[str, ...]) -> list[Path]:
    found = sorted(SOURCE_DIR.glob("[0-9][0-9]_*.py"))
    if only:
        found = [p for p in found if any(p.name.startswith(prefix) for prefix in only)]
    return found


def build(
    *,
    only: tuple[str, ...] = ONLY,
    execute: bool = EXECUTE,
    timeout: int = TIMEOUT,
) -> int:
    """Generate both notebooks for each source. Returns a process exit code."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = _sources(only)
    if not sources:
        print(f"no sources matched {only or '(all)'} in {SOURCE_DIR}")
        return 1

    failures = 0
    for path in sources:
        stem = path.stem
        print(f"\n=== {stem} ===")
        source = _read_source(path)

        exercise = _exercise_notebook(source)
        leaks = _check_no_leaks(exercise, stem)
        if leaks:
            print("  LEAK CHECK FAILED:")
            for problem in leaks:
                print(f"    {problem}")
            failures += 1
        exercise_path = OUTPUT_DIR / f"{stem}{EXERCISE_SUFFIX}"
        nbformat.write(exercise, exercise_path)
        print(f"  wrote {exercise_path.name} ({len(exercise.cells)} cells, not executed)")

        solution = _solution_notebook(source)
        if execute:
            try:
                seconds = _execute(solution, OUTPUT_DIR, timeout)
                print(f"  executed in {seconds:.0f}s"
                      f"{' (reduced configuration)' if fast_mode() else ''}")
            except Exception as exc:
                print(f"  EXECUTION FAILED: {type(exc).__name__}: {str(exc)[:400]}")
                failures += 1
        solution_path = OUTPUT_DIR / f"{stem}{SOLUTION_SUFFIX}"
        nbformat.write(solution, solution_path)
        print(f"  wrote {solution_path.name} ({len(solution.cells)} cells)")

    print(f"\n{len(sources)} tutorial(s), {failures} failure(s)")
    return 1 if failures else 0


def check(*, only: tuple[str, ...] = ONLY) -> list[str]:
    """Report notebooks that no longer match their source.

    CI calls this so a committed notebook cannot drift away from the source it
    was generated from. Compares cell *sources*, ignoring outputs and execution
    counts, which change on every run.
    """
    stale: list[str] = []
    for path in _sources(only):
        stem = path.stem
        source = _read_source(path)
        for suffix, builder in (
            (EXERCISE_SUFFIX, _exercise_notebook),
            (SOLUTION_SUFFIX, _solution_notebook),
        ):
            target = OUTPUT_DIR / f"{stem}{suffix}"
            if not target.exists():
                stale.append(f"{target.name}: missing")
                continue
            expected = [c.source for c in builder(source).cells]
            actual = [c.source for c in nbformat.read(target, as_version=4).cells]
            if expected != actual:
                stale.append(
                    f"{target.name}: {len(actual)} cells on disk vs "
                    f"{len(expected)} expected — regenerate it"
                )
    return stale


if __name__ == "__main__":
    sys.exit(build())
