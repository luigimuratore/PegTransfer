#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mode="${1:-help}"
if (( $# )); then shift; fi
case "$mode" in
  residual-tests) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/test_residual.py "$@" ;;
  residual-pipeline) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/residual_pipeline.py "$@" ;;
  residual-baseline|residual-train|residual-evaluate|residual-play)
    exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/residual_main.py "${mode#residual-}" "$@" ;;
  residual-video) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/residual_main.py play --headless --video --num_envs 1 --episodes 1 "$@" ;;
  inspect) exec "${PEG_PYTHON:-python}" tools/peg_grasp_v2/inspect_mesh.py "$@" ;;
  tests) exec "${PEG_PYTHON:-python}" tools/peg_grasp_v2/test_cpu.py "$@" ;;
  full-tests) exec "${PEG_PYTHON:-python}" tools/peg_grasp_v2/test_full.py "$@" ;;
  gpu-tests) exec "${PEG_PYTHON:-python}" tools/peg_grasp_v2/test_gpu.py "$@" ;;
  probe) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/replay.py "$@" ;;
  video) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/replay.py --video "$@" ;;
  collect) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/replay.py --mode wall --episodes 6 "$@" ;;
  full) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/full_task.py "$@" ;;
  full-replay) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/replay_full.py "$@" ;;
  full-export) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/full_export.py "$@" ;;
  full-train) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/full_train.py "$@" ;;
  full-play) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/full_play.py "$@" ;;
  full-learning-tests) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/test_full_learning.py "$@" ;;
  sac-export) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/rl_data.py "$@" ;;
  sac-train) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/rl_main.py train "$@" ;;
  sac-evaluate) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/rl_replay.py evaluate "$@" ;;
  sac-play) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/rl_replay.py play "$@" ;;
  sac-smoke) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/rl_main.py smoke "$@" ;;
  sac-tests) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/test_rl.py "$@" ;;
  autonomy-collect) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_main.py collect "$@" ;;
  autonomy-train) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_main.py train "$@" ;;
  autonomy-fit-jaws) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_main.py fit_jaws "$@" ;;
  autonomy-fit-router) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_main.py fit_router "$@" ;;
  autonomy-fit-contact) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_main.py fit_contact "$@" ;;
  autonomy-full-certify) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_full_certify.py "$@" ;;
  autonomy-assistance-curriculum) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_assistance_curriculum.py "$@" ;;
  autonomy-constraints-curriculum) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_constraints_curriculum.py "$@" ;;
  autonomy-motion-curriculum) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_motion_curriculum.py "$@" ;;
  autonomy-position-fit) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_position_fit.py "$@" ;;
  autonomy-motion-play) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_motion_play.py "$@" ;;
  autonomy-receiver-fit) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_receiver_fit.py "$@" ;;
  autonomy-receiver-curriculum) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_receiver_curriculum.py "$@" ;;
  autonomy-preserve) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/preserve_autonomy.py "$@" ;;
  autonomy-preserve-assisted) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/preserve_assisted_autonomy.py "$@" ;;
  autonomy-evaluate) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_replay.py evaluate "$@" ;;
  autonomy-play) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_replay.py play "$@" ;;
  autonomy-tests) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/test_autonomy.py "$@" ;;
  autonomy-curriculum) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_curriculum.py "$@" ;;
  autonomy-repackage) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_main.py repackage "$@" ;;
  checkpoint5-play|checkpoint5-video)
    checkpoint_output="logs/grasp_v2/${mode}_$(date +%Y%m%d_%H%M%S_%N)"
    checkpoint_visual=(--realtime)
    if [[ "$mode" == checkpoint5-video ]]; then checkpoint_visual=(--headless --video); fi
    exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_replay.py play \
      --policy .checkpoint_runs/checkpoint5-native-pickup-bc/model/policy.pt \
      --num_envs 1 --episodes 1 --seed 123 --output "$checkpoint_output" "${checkpoint_visual[@]}" "$@" ;;
  checkpoint7-play|checkpoint7-video)
    checkpoint_output="logs/grasp_v2/${mode}_$(date +%Y%m%d_%H%M%S_%N)"
    checkpoint_visual=(--realtime)
    if [[ "$mode" == checkpoint7-video ]]; then checkpoint_visual=(--headless --video); fi
    exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_replay.py play \
      --policy .checkpoint_runs/checkpoint7-assisted-full-bc-pilot/model/policy.pt \
      --milestone milestones/checkpoint7-assisted-full-bc-pilot.json \
      --skill full --grasp_support 1 --num_envs 1 --episodes 1 --episode_steps 10000 \
      --sources L5 --targets R2 --jitter_mm 0 --yaw_deg 0 --seed 123 \
      --output "$checkpoint_output" "${checkpoint_visual[@]}" "$@" ;;
  autonomy-transit-curriculum) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_transit_curriculum.py "$@" ;;
  autonomy-domain-curriculum) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_domain_curriculum.py "$@" ;;
  autonomy-transit-play) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_transit_play.py "$@" ;;
  checkpoint10-play|checkpoint10-video)
    checkpoint_output="logs/grasp_v2/${mode}_$(date +%Y%m%d_%H%M%S_%N)"
    checkpoint_visual=(--realtime)
    if [[ "$mode" == checkpoint10-video ]]; then checkpoint_visual=(--headless --video); fi
    exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_replay.py play \
      --policy .checkpoint_runs/checkpoint10-assisted-full-bc-transit/model/policy.pt \
      --milestone milestones/checkpoint10-assisted-full-bc-transit.json \
      --skill full --grasp_support 1 --support_profile firm --placement_support \
      --motion_profile travel --free_speed_scale 2 --transport_speed_scale 2 \
      --num_envs 1 --episodes 1 --episode_steps 10000 \
      --sources L5 --targets R2 --jitter_mm 0.5 --yaw_deg 2 --seed 2468 \
      --output "$checkpoint_output" "${checkpoint_visual[@]}" "$@" ;;
  checkpoint11-play|checkpoint11-video)
    checkpoint_output="logs/grasp_v2/${mode}_$(date +%Y%m%d_%H%M%S_%N)"
    checkpoint_visual=(--realtime)
    if [[ "$mode" == checkpoint11-video ]]; then checkpoint_visual=(--headless --video); fi
    exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_replay.py play \
      --policy .checkpoint_runs/checkpoint11-assisted-full-bc-yaw5/model/policy.pt \
      --milestone milestones/checkpoint11-assisted-full-bc-yaw5.json \
      --skill full --grasp_support 1 --support_profile firm --placement_support \
      --motion_profile travel --free_speed_scale 2 --transport_speed_scale 2 \
      --num_envs 1 --episodes 1 --episode_steps 10000 \
      --sources L5 --targets R2 --jitter_mm 0.5 --yaw_deg 5 --seed 2468 \
      --output "$checkpoint_output" "${checkpoint_visual[@]}" "$@" ;;
  checkpoint13-play|checkpoint13-video)
    checkpoint_visual=play
    if [[ "$mode" == checkpoint13-video ]]; then checkpoint_visual=video; fi
    exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_checkpoint_play.py checkpoint13 "$checkpoint_visual" "$@" ;;
  checkpoint12-play|checkpoint12-video)
    checkpoint_output="logs/grasp_v2/${mode}_$(date +%Y%m%d_%H%M%S_%N)"
    checkpoint_visual=(--realtime)
    if [[ "$mode" == checkpoint12-video ]]; then checkpoint_visual=(--headless --video); fi
    exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_replay.py play \
      --policy .checkpoint_runs/checkpoint12-three-posts-assisted/model/policy.pt \
      --milestone milestones/checkpoint12-three-posts-assisted.json \
      --skill full --grasp_support 1 --support_profile firm --placement_support \
      --gripper_load_guard --upright_grasp_support \
      --motion_profile travel --free_speed_scale 2 --transport_speed_scale 2 \
      --num_envs 1 --episodes 3 --episode_steps 10000 \
      --sources L5 L6 L4 --targets R2 R1 R3 --paired_posts --jitter_mm 0 --yaw_deg 0 --seed 59003 \
      --output "$checkpoint_output" "${checkpoint_visual[@]}" "$@" ;;
  checkpoint9-play|checkpoint9-video)
    checkpoint_output="logs/grasp_v2/${mode}_$(date +%Y%m%d_%H%M%S_%N)"
    checkpoint_visual=(--realtime)
    if [[ "$mode" == checkpoint9-video ]]; then checkpoint_visual=(--headless --video); fi
    exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_replay.py play \
      --policy .checkpoint_runs/checkpoint9-assisted-full-bc-randomized/model/policy.pt \
      --milestone milestones/checkpoint9-assisted-full-bc-randomized.json \
      --skill full --grasp_support 1 --support_profile firm --placement_support \
      --num_envs 1 --episodes 1 --episode_steps 10000 \
      --sources L5 --targets R2 --jitter_mm 0.5 --yaw_deg 2 --seed 2468 \
      --output "$checkpoint_output" "${checkpoint_visual[@]}" "$@" ;;
  checkpoint8-play|checkpoint8-video)
    checkpoint_output="logs/grasp_v2/${mode}_$(date +%Y%m%d_%H%M%S_%N)"
    checkpoint_visual=(--realtime)
    if [[ "$mode" == checkpoint8-video ]]; then checkpoint_visual=(--headless --video); fi
    exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/autonomy_replay.py play \
      --policy .checkpoint_runs/checkpoint8-assisted-full-bc-constraints/model/policy.pt \
      --milestone milestones/checkpoint8-assisted-full-bc-constraints.json \
      --skill full --grasp_support 1 --support_profile firm --placement_support \
      --num_envs 1 --episodes 1 --episode_steps 10000 \
      --sources L5 --targets R2 --jitter_mm 0 --yaw_deg 0 --seed 123 \
      --output "$checkpoint_output" "${checkpoint_visual[@]}" "$@" ;;
  train) exec "${PEG_PYTHON:-python}" -u tools/peg_grasp_v2/train.py "$@" ;;
  *) printf '%s\n' 'Usage: bash peg_grasp_v2.sh {inspect|tests|full-tests|gpu-tests|probe|video|collect|train|full|full-replay|full-export|full-train|full-play|full-learning-tests|sac-export|sac-train|sac-evaluate|sac-play|sac-smoke|sac-tests|autonomy-collect|autonomy-train|autonomy-fit-jaws|autonomy-fit-router|autonomy-fit-contact|autonomy-full-certify|autonomy-assistance-curriculum|autonomy-constraints-curriculum|autonomy-motion-curriculum|autonomy-motion-play|autonomy-position-fit|autonomy-preserve|autonomy-preserve-assisted|autonomy-evaluate|autonomy-play|autonomy-tests|autonomy-curriculum|autonomy-repackage|checkpoint5-play|checkpoint5-video|checkpoint7-play|checkpoint7-video|checkpoint8-play|checkpoint8-video|checkpoint9-play|checkpoint9-video|checkpoint10-play|checkpoint10-video|checkpoint11-play|checkpoint11-video|checkpoint12-play|checkpoint12-video|checkpoint13-play|checkpoint13-video|autonomy-transit-curriculum|autonomy-transit-play|autonomy-domain-curriculum} [options]' ;;
esac
