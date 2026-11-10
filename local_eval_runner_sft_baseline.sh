#!/bin/bash
# SFT baseline: the two flow-matching bases with NO PPO update, on the same 100 held-out eval
# seeds and the same eval config as local_eval_runner_student_env.sh. Without this the PPO
# step-40 numbers have nothing to be read against -- the 2026-10-31 sweep evaluated only PPO
# checkpoints, so no step-0 number exists on this seed set.
#
# The only difference from the PPO runner is that `runner.resume_dir` is not passed, so the
# actor keeps the SFT weights `actor.model.model_path` loads.
#
#   bash local_eval_runner_sft_baseline.sh n100 n30
#
# Needs BOTH GPUs (actor+rollout on 0, env on 1); do not start it beside a training run.
set -uo pipefail

(( $# )) || { echo "usage: $0 <n30|n100> [...]"; exit 2; }

R=/data/vxiao/bimanual_manipulation/RLinf-eval
cd "$R"
source .venv/bin/activate

export PATH=$HOME/.local/bin:/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

export ROBOTWIN_PATH=$R/RoboTwin-RLinf-rgb
export ROBOT_PLATFORM=ALOHA
export EMBODIED_PATH=$R/examples/embodiment
export REPO_PATH=$R
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=$R/curobo07b:$REPO_PATH:$ROBOTWIN_PATH:${PYTHONPATH:-}

export RAY_raylet_start_wait_time_s=900
export RAY_worker_register_timeout_seconds=3600
export RAY_ADDRESS=local
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ASSETS_PATH=$R/RoboTwin-RLinf
LOG_ROOT=$R/eval_logs_sft_baseline
mkdir -p "$LOG_ROOT"
SUMMARY=$LOG_ROOT/summary.log

ATTEMPT_TIMEOUT=16200
# MAX_EPOCHS is the runner's step budget, and it decides how much runs AFTER the
# eval. The loop body is rollout -> update -> eval, so the first eval is always the
# number wanted; everything past it is PPO training this script has no use for. The
# PPO arm got away with the default because resume_dir set global_step=40 and 41-40
# leaves one iteration. Starting from 0, MAX_EPOCHS=1 gives the same single pass.
MAX_ATTEMPTS=2

base_for() {
  case "$1" in
    n30)  echo "$R/fm_actiononly_30demo_ckpt/200.ckpt" ;;
    n100) echo "$R/fm_actiononly_100demo_ckpt/200.ckpt" ;;
    *) echo "" ;;
  esac
}

cleanup_ray_scoped() {
  local tmpdir=$1 p env
  for p in $(pgrep -f 'ray::|raylet|gcs_server|plasma_store|log_monitor|dashboard' 2>/dev/null); do
    env=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null)
    if printf '%s\n' "$env" | /usr/bin/grep -qx "RAY_TMPDIR=$tmpdir" \
       || tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | /usr/bin/grep -qF -- "$tmpdir"; then
      kill -9 "$p" 2>/dev/null
    fi
  done
}

preflight_fail=0
for l in "$@"; do
  m=$(base_for "$l")
  [ -n "$m" ] || { echo "PREFLIGHT: unknown label $l (expected n30 or n100)" | tee -a "$SUMMARY"; preflight_fail=1; continue; }
  [ -e "$m" ] || { echo "PREFLIGHT MISSING: $m" | tee -a "$SUMMARY"; preflight_fail=1; }
done
[ -e "$ASSETS_PATH/assets/embodiments/aloha-agilex/curobo_left.yml" ] \
  || { echo "PREFLIGHT MISSING: curobo left config" | tee -a "$SUMMARY"; preflight_fail=1; }
[ "$preflight_fail" -eq 0 ] || { echo "PREFLIGHT FAILED, nothing launched" | tee -a "$SUMMARY"; exit 1; }
echo "=== $(date) preflight OK for: $* ===" | tee -a "$SUMMARY"

for l in "$@"; do
  model_path=$(base_for "$l")
  log_dir=$LOG_ROOT/$(date +'%Y%m%d-%H%M%S')-sft_${l}_eval
  mkdir -p "$log_dir"
  echo "=== $(date) starting SFT baseline $l (model_path=$model_path) ===" | tee -a "$SUMMARY"
  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    export RAY_TMPDIR=/tmp/ray_sft_${l}_a${attempt}
    mkdir -p "$RAY_TMPDIR"
    echo "  [$l] attempt $attempt/$MAX_ATTEMPTS" | tee -a "$SUMMARY"
    timeout --kill-after=15 "$ATTEMPT_TIMEOUT" python "$EMBODIED_PATH/train_embodied_agent.py" \
      --config-path "$EMBODIED_PATH/config/" --config-name robotwin_scan_object_gap_flow_baseline_eval2_local \
      runner.logger.log_path="$log_dir" \
      runner.max_epochs=${MAX_EPOCHS:-41} \
      env.train.total_num_envs=2 \
      env.eval.total_num_envs=5 \
      env.eval.start_round=0 \
      env.eval.rollout_epoch=20 \
      env.train.assets_path="$ASSETS_PATH" \
      env.eval.assets_path="$ASSETS_PATH" \
      actor.global_batch_size=16 actor.micro_batch_size=4 \
      actor.model.model_path="$model_path" \
      actor.model.gap_flow.gap_root="$R/gap_root_local" \
      > "$log_dir/run_attempt${attempt}.log" 2>&1
    rc=$?
    cleanup_ray_scoped "$RAY_TMPDIR"
    rm -rf "$RAY_TMPDIR"
    sleep 3
    if [ "$rc" -eq 0 ]; then
      echo "=== $(date) SFT $l SUCCEEDED on attempt $attempt, log_dir=$log_dir ===" | tee -a "$SUMMARY"
      break
    fi
    echo "  [$l] attempt $attempt failed/timed out, rc=$rc" | tee -a "$SUMMARY"
  done
done
echo "SFT_BASELINE_EVALS_DONE" | tee -a "$SUMMARY"
