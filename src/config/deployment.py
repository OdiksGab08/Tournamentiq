"""Deployment-safe project paths and artifact diagnostics.

These helpers deliberately use stdout so the same diagnostics appear in local
terminals and Streamlit Community Cloud logs without changing normal UI output.
"""

from __future__ import annotations

from pathlib import Path
import traceback


def find_project_root(anchor: str | Path) -> Path:
    """Locate the repository root without relying on the current directory."""
    current = Path(anchor).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "models").is_dir() and (candidate / "dashboard").is_dir():
            return candidate
    raise RuntimeError(
        f"Could not locate TournamentIQ project root from {Path(anchor).resolve()}."
    )


def log_artifact(path: str | Path, *, label: str) -> Path:
    """Print a complete, safe diagnostic record before an artifact is read."""
    artifact = Path(path).resolve()
    exists = artifact.is_file()
    try:
        size = artifact.stat().st_size if exists else None
    except OSError as error:
        size = None
        print(f"[TournamentIQ] {label} stat error: {type(error).__name__}: {error}")
    print(f"[TournamentIQ] {label} path: {artifact}")
    print(f"[TournamentIQ] {label} exists: {exists}")
    print(f"[TournamentIQ] {label} size_bytes: {size}")
    if not exists:
        raise FileNotFoundError(f"Required {label} is missing: {artifact}")
    return artifact


def log_exception(context: str) -> None:
    """Print the active exception's full traceback for deployment diagnostics."""
    print(f"[TournamentIQ] {context} failed:\n{traceback.format_exc()}")
