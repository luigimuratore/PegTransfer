"""Resolve a training run by its full timestamped name or unique suffix."""

from __future__ import annotations

import os


def resolve_run_name(log_root_path: str, requested: str) -> str:
    runs = sorted(entry.name for entry in os.scandir(log_root_path) if entry.is_dir()) if os.path.isdir(log_root_path) else []
    if requested in runs:
        return requested
    matches = [run for run in runs if run.endswith(requested)]
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(runs) if runs else "none"
    reason = "Ambiguous run suffix" if matches else "Run not found"
    raise ValueError(f"{reason}: {requested!r}. Available runs: {available}")
