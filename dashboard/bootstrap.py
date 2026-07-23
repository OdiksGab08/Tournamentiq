"""Bootstrap package discovery for the Streamlit dashboard entry point.

``streamlit run dashboard/app.py`` executes ``app.py`` as a script.  Python
then discovers sibling dashboard modules but does not always include the
repository root, which is where the top-level ``src`` package lives.  This
single entry-point bootstrap keeps imports stable without relying on the
current working directory.
"""

from __future__ import annotations

# Use ``Path`` to derive locations from this file instead of the working directory.
from pathlib import Path
# Use ``sys.path`` only at the application entry point to expose the repository package root.
import sys


def ensure_project_root() -> Path:
    """Add the repository root to Python's import search path exactly once."""
    # ``dashboard/bootstrap.py`` is one level below the repository root.
    project_root = Path(__file__).resolve().parents[1]
    # Verify both required project folders before changing Python's import search path.
    required_directories = (project_root / "src", project_root / "dashboard")
    if not all(path.is_dir() for path in required_directories):
        raise RuntimeError(
            f"TournamentIQ project root is incomplete: {project_root}"
        )
    root_text = str(project_root)
    # Insert the root once so absolute imports such as ``src.simulator`` work
    # when Streamlit executes ``dashboard/app.py`` as a standalone script.
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return project_root
