#!/bin/bash
# Local 100-seed eval for the student env-track PPO checkpoints (step 40), on the same
# held-out seed set and the same eval config as the 2026-11-01 10-checkpoint sweep, so the
# numbers are directly comparable to it. Labels are passed as arguments; each needs a
# ppo_checkpoints_step40/_resume/<label>/global_step_40 symlink.
#
#   bash local_eval_runner_student_env.sh student_env_n100L0p5 [student_env_n30L0p5 ...]
#
# The eval config places actor+rollout on GPU 0 and env on GPU 1, so this needs BOTH cards
# idle; do not start it while a training run holds one.
set -uo pipefail

(( $# )) || { echo "usage: $0 <label> [<label> ...]"; exit 2; }

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
# curobo07b first so it shadows the pip-installed nvidia-curobo package.
export PYTHONPATH=$R/curobo07b:$REPO_PATH:$ROBOTWIN_PATH:${PYTHONPATH:-}

export RAY_raylet_start_wait_time_s=900
export RAY_worker_register_timeout_seconds=3600
export RAY_ADDRESS=local
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ASSETS_PATH=$R/RoboTwin-RLinf
CKPT_BASE=$R/ppo_checkpoints_step40
LOG_ROOT=$R/eval_logs_student_env
mkdir -p "$LOG_ROOT"
SUMMARY=$LOG_ROOT/summary.log

FM30=$R/fm_actiononly_30demo_ckpt/200.ckpt
FM100=$R/fm_actiononly_100demo_ckpt/200.ckpt

ATTEMPT_TIMEOUT=16200  # 4h30m; the 2026-11-01 sweep measured 2h02m (n100) to 3h11m (n30)
MAX_ATTEMPTS=2         # one retry in case the first attempt hits the intermittent init race

# The host is shared, so a bare `pkill -f 'ray::'` would kill other users' jobs. Clean up
# only this attempt's own ray processes, identified by the per-attempt RAY_TMPDIR.
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

base_for() {
  case "$1" in
    *n30*)  echo "$FM30" ;;
    *n100*) echo "$FM100" ;;
    *) echo "" ;;
  esac
}

run_eval() {
  local label=$1 model_path=$2
  local ckpt_dir=$CKPT_BASE/_resume/$label/global_step_40
  local log_dir=$LOG_ROOT/$(date +'%Y%m%d-%H%M%S')-ppo_${label}_ckpt40_eval
  mkdir -p "$log_dir"
  echo "=== $(date) starting $label (model_path=$model_path) ===" | tee -a "$SUMMARY"

  local attempt rc
  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    # Ray's plasma socket path is RAY_TMPDIR + "/ray/session_<timestamp>_<pid>/sockets/plasma_store",
    # and AF_UNIX caps that at 107 bytes. /tmp/ray_eval_sparse_heldout_n100L0p5_a1 left it one
    # byte over and the n100 eval died in ray.init twice (2026-11-11 05:53). A short hashed
    # directory keeps the label unique without spending the budget on it.
    export RAY_TMPDIR=/tmp/ray_$(printf '%s' "${label}_a${attempt}" | md5sum | cut -c1-10)
    mkdir -p "$RAY_TMPDIR"
    echo "  [$label] attempt $attempt/$MAX_ATTEMPTS (timeout=${ATTEMPT_TIMEOUT}s)" | tee -a "$SUMMARY"
    timeout --kill-after=15 "$ATTEMPT_TIMEOUT" python "$EMBODIED_PATH/train_embodied_agent.py" \
      --config-path "$EMBODIED_PATH/config/" --config-name robotwin_scan_object_gap_flow_baseline_eval2_local \
      runner.logger.log_path="$log_dir" \
      runner.resume_dir="$ckpt_dir" \
      runner.max_epochs=41 \
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
      echo "=== $(date) $label SUCCEEDED on attempt $attempt, log_dir=$log_dir ===" | tee -a "$SUMMARY"
      return 0
    fi
    echo "  [$label] attempt $attempt failed/timed out, rc=$rc" | tee -a "$SUMMARY"
  done
  echo "=== $(date) $label GAVE UP after $MAX_ATTEMPTS attempts, log_dir=$log_dir ===" | tee -a "$SUMMARY"
}

# Paths this sweep depends on. Each has been wrong at least once after moving machines, and
# the failures are silent or look like a simulator hang, so check them first.
preflight_fail=0
check_path() {
  if [ ! -e "$1" ]; then echo "PREFLIGHT MISSING: $2 -> $1" | tee -a "$SUMMARY"; preflight_fail=1; fi
}
check_path "$ASSETS_PATH/assets/embodiments/aloha-agilex/curobo_left.yml" "curobo left config"
check_path "$R/gap_root_local/thirdparty/dinov2/hubconf.py" "dinov2 repo"
check_path "$R/gap_root_local/pretrained/dinov2_vitl14_reg4_pretrain.pth" "dinov2 weights"
for l in "$@"; do
  check_path "$CKPT_BASE/_resume/$l/global_step_40/actor" "resume dir $l"
  [ -n "$(base_for "$l")" ] || { echo "PREFLIGHT: cannot tell the SFT base for label $l (expected n30 or n100 in the name)" | tee -a "$SUMMARY"; preflight_fail=1; }
  check_path "$(base_for "$l")" "fm_actiononly base for $l"
done
if /usr/bin/grep -rqs project_data "$ASSETS_PATH/assets/embodiments/"; then
  echo "PREFLIGHT: stale cross-cluster paths in embodiment configs; run RoboTwin-RLinf-rgb/script/update_embodiment_config_path.py from $ASSETS_PATH" | tee -a "$SUMMARY"
  preflight_fail=1
fi
if [ "$preflight_fail" -ne 0 ]; then echo "PREFLIGHT FAILED, nothing launched" | tee -a "$SUMMARY"; exit 1; fi
echo "=== $(date) preflight OK for: $* ===" | tee -a "$SUMMARY"

for l in "$@"; do
  run_eval "$l" "$(base_for "$l")"
done

echo "STUDENT_ENV_EVALS_DONE" | tee -a "$SUMMARY"
