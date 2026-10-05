#!/usr/bin/env bash
# Short copyable commands. Simulation/training runs only when the user invokes it.
set -euo pipefail
PEG_ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PEG_ROOT_DIR"
export PYTHONPATH="$PEG_ROOT_DIR/workflows/robotic_surgery/scripts${PYTHONPATH:+:$PYTHONPATH}"
PEG_SEQUENCE_DIR="$PEG_ROOT_DIR/workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/sac_sequence"
PEG_LOG_DIR="$PEG_ROOT_DIR/logs/sac_sequence"
PEG_STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
PEG_COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$PEG_COMMAND" in
  tests)
    exec python "$PEG_SEQUENCE_DIR/test_cpu.py" "$@" ;;
  probe)
    exec python "$PEG_SEQUENCE_DIR/probe.py" --baseline --source L5 --skill full --guidance 1 --num_envs 1 --episodes 3 --headless --trace --output "$PEG_LOG_DIR/baseline_full_$PEG_STAMP.json" "$@" ;;
  collect)
    exec python "$PEG_SEQUENCE_DIR/collect.py" --source L5 --skill full --num_envs 1 --episodes 2 --headless --guidance_levels 1 0.5 0 --output "$PEG_LOG_DIR/collect_full_$PEG_STAMP.json" --demonstrations "$PEG_LOG_DIR/full_demos_$PEG_STAMP.npz" "$@" ;;
  bc)
    if [[ $# -gt 0 && "$1" != --* ]]; then PEG_DEMOS="$1"; shift
    else PEG_DEMOS="$(python "$PEG_SEQUENCE_DIR/paths.py" demos)"; fi
    echo "[SEQUENCE] Demonstrations: $PEG_DEMOS"
    exec python "$PEG_SEQUENCE_DIR/train.py" --source L5 --skill full --guidance 0 --num_envs 16 --headless --demonstrations "$PEG_DEMOS" --bc_steps 6000 --imitation_only --run_name sequence_bc "$@" ;;
  eval)
    if [[ $# -gt 0 && "$1" != --* ]]; then PEG_CHECKPOINT="$1"; shift
    else PEG_CHECKPOINT="$(python "$PEG_SEQUENCE_DIR/paths.py" checkpoint)"; fi
    echo "[SEQUENCE] Checkpoint: $PEG_CHECKPOINT"
    exec python "$PEG_SEQUENCE_DIR/evaluate.py" --checkpoint "$PEG_CHECKPOINT" --source L5 --skill full --guidance 0 --num_envs 16 --episodes 100 --seed 123 --headless --output "$PEG_LOG_DIR/eval_full_unguided_$PEG_STAMP.json" "$@" ;;
  curriculum)
    if [[ $# -ge 3 && "$1" != --* ]]; then
      PEG_BASELINE="$1"; PEG_DEMOS="$2"; PEG_CHECKPOINT="$3"; shift 3
    elif [[ $# -eq 0 || "$1" == --* ]]; then
      PEG_BASELINE="$(python "$PEG_SEQUENCE_DIR/paths.py" baseline)"
      PEG_DEMOS="$(python "$PEG_SEQUENCE_DIR/paths.py" demos)"
      PEG_CHECKPOINT="$(python "$PEG_SEQUENCE_DIR/paths.py" checkpoint)"
    else echo 'Uso: bash peg_sequence.sh curriculum [BASELINE.json DEMO.npz CHECKPOINT.zip]' >&2; exit 2; fi
    echo "[SEQUENCE] Baseline: $PEG_BASELINE; demonstrations: $PEG_DEMOS; checkpoint: $PEG_CHECKPOINT"
    exec python "$PEG_SEQUENCE_DIR/curriculum.py" --baseline_report "$PEG_BASELINE" --demonstrations "$PEG_DEMOS" --initialize_from "$PEG_CHECKPOINT" --source L5 --output_dir "$PEG_LOG_DIR/curriculum_$PEG_STAMP" "$@" ;;
  tune)
    if [[ $# -ge 3 && "$1" != --* ]]; then
      PEG_BASELINE="$1"; PEG_DEMOS="$2"; PEG_CHECKPOINT="$3"; shift 3
    elif [[ $# -eq 0 || "$1" == --* ]]; then
      PEG_BASELINE="$(python "$PEG_SEQUENCE_DIR/paths.py" baseline)"
      PEG_DEMOS="$(python "$PEG_SEQUENCE_DIR/paths.py" demos)"
      PEG_CHECKPOINT="$(python "$PEG_SEQUENCE_DIR/paths.py" checkpoint)"
    else echo 'Uso: bash peg_sequence.sh tune [BASELINE.json DEMO.npz CHECKPOINT.zip] --skill SKILL --guidance G' >&2; exit 2; fi
    echo "[SEQUENCE] Baseline: $PEG_BASELINE; demonstrations: $PEG_DEMOS; checkpoint: $PEG_CHECKPOINT"
    exec python "$PEG_SEQUENCE_DIR/tune.py" --baseline_report "$PEG_BASELINE" --demonstrations "$PEG_DEMOS" --initialize_from "$PEG_CHECKPOINT" --output_dir "$PEG_LOG_DIR/tuning_$PEG_STAMP" "$@" ;;
  play|video)
    if [[ $# -gt 0 && "$1" != --* ]]; then PEG_CHECKPOINT="$1"; shift
    else PEG_CHECKPOINT="$(python "$PEG_SEQUENCE_DIR/paths.py" checkpoint)"; fi
    echo "[SEQUENCE] Checkpoint: $PEG_CHECKPOINT"
    PEG_EXTRA=()
    if [[ "$PEG_COMMAND" == video ]]; then PEG_EXTRA=(--headless --video --trace --episodes 1); fi
    exec python "$PEG_SEQUENCE_DIR/play.py" --checkpoint "$PEG_CHECKPOINT" --source L5 --skill full --guidance 0 --num_envs 1 --output "$PEG_LOG_DIR/inference_$PEG_STAMP.json" "${PEG_EXTRA[@]}" "$@" ;;
  *)
    echo 'Comandi: tests, probe, collect, bc, eval, curriculum, tune --skill SKILL --guidance G, play, video'
    echo 'bc/eval/curriculum/tune/play/video scelgono gli ultimi file compatibili; i percorsi espliciti restano disponibili.'
    echo 'Ordine: tests -> probe (baseline completa) -> collect -> bc -> eval -> curriculum se necessario.' ;;
esac
