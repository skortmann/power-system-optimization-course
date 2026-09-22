"""From Economic Dispatch to Large-Scale Grid Optimization.

Reusable pieces for the course notebooks. The rule for what belongs here:
*boilerplate yes, mathematics no.* Data loading, plotting, validation and
solver bookkeeping live in this package; every formulation a tutorial teaches
is written out in the notebook where it is taught.
"""

from __future__ import annotations

import logging
import warnings

__version__ = "0.1.0"


def _quieten() -> None:
    """Silence noise that would otherwise fill every notebook's output.

    pandapower prints a multi-line numba warning on EVERY power flow, and the
    course runs hundreds of them. Installing numba is not worth a build-tool
    dependency for models of this size, and the warning tells a student nothing
    about optimization.

    Deliberately narrow: only these two sources, and only at import of this
    package. Nothing here hides a solver status or a convergence failure.
    """
    logging.getLogger("pandapower").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore", message=".*numba cannot be imported.*", category=UserWarning
    )


_quieten()
