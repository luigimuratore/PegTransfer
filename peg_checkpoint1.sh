#!/usr/bin/env bash
# Replay a preserved milestone without using the task under development.
set -euo pipefail
PEG_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PEG_PYTHON:-python}" "$PEG_PROJECT_DIR/tools/replay_checkpoint1.py" "$@"
