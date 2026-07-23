"""Bootstrap package discovery for the Streamlit dashboard entry point.

``streamlit run dashboard/app.py`` executes ``app.py`` as a script.  Python
then discovers sibling dashboard modules but does not always include the
repository root, which is where the top-level ``src`` package lives.  This
single entry-point bootstrap keeps imports stable without relying on the
current working directory.
"""

from __future__ import annotations

from pathlib import Path
import sys


def ensure_project_root() -> Path:
    """Add the repository root to Python's import search path exactly once."""
    project_root = Path(__file__).resolve().parents[1]
    required_directories = (project_root / "src", project_root / "dashboard")
    if not all(path.is_dir() for path in required_directories):
        raise RuntimeError(
            f"TournamentIQ project root is incomplete: {project_root}"
        )
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return project_root
