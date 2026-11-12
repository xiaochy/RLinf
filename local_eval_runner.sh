#!/bin/bash
# Local (no-SLURM) 100-seed PPO eval runner for all 10 step=40 checkpoints.
# Mirrors grogu's ppo_ckpt_eval_overlay3.sh invocation, adapted for a single
# local machine: no sbatch/trap, assets_path and actor.model.model_path
# overridden to local paths, continues to the next checkpoint on failure.
set -uo pipefail

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
LOG_ROOT=$R/eval_logs_local
mkdir -p "$LOG_ROOT"
SUMMARY=$LOG_ROOT/summary.log

FM30=$R/fm_actiononly_30demo_ckpt/200.ckpt
FM100=$R/fm_actiononly_100demo_ckpt/200.ckpt

ATTEMPT_TIMEOUT=12000  # 3h20m per attempt (a 20-round 100-seed eval measures ~2h10m)
MAX_ATTEMPTS=2        # one retry in case the first attempt hits the intermittent init race

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
    rm -rf "$RAY_TMPDIR"
    pkill -9 -f 'ray::' 2>/dev/null
    pkill -9 -f 'raylet' 2>/dev/null
    sleep 3
    if [ "$rc" -eq 0 ]; then
      echo "=== $(date) $label SUCCEEDED on attempt $attempt, log_dir=$log_dir ===" | tee -a "$SUMMARY"
      return 0
    fi
    echo "  [$label] attempt $attempt failed/timed out, rc=$rc" | tee -a "$SUMMARY"
  done
  echo "=== $(date) $label GAVE UP after $MAX_ATTEMPTS attempts, log_dir=$log_dir ===" | tee -a "$SUMMARY"
}

# Paths this sweep depends on. Each of these has been wrong at least once after moving
# machines, and the failures are silent or look like a simulator hang, so check them first.
preflight_fail=0
check_path() {
  if [ ! -e "$1" ]; then echo "PREFLIGHT MISSING: $2 -> $1" | tee -a "$SUMMARY"; preflight_fail=1; fi
}
check_path "$ASSETS_PATH/assets/embodiments/aloha-agilex/curobo_left.yml" "curobo left config"
check_path "$R/gap_root_local/thirdparty/dinov2/hubconf.py" "dinov2 repo"
check_path "$R/gap_root_local/pretrained/dinov2_vitl14_reg4_pretrain.pth" "dinov2 weights"
check_path "$FM30" "fm_actiononly 30demo ckpt"
check_path "$FM100" "fm_actiononly 100demo ckpt"
for l in sparse_n30L0p3 sparse_n30L0p5 sparse_n30L0p7 dense_n30L0p5 \
         sparse_n100L0p3 sparse_n100L0p5 sparse_n100L0p7 \
         dense_n100L0p3 dense_n100L0p5 dense_n100L0p7; do
  check_path "$CKPT_BASE/_resume/$l/global_step_40/actor" "resume dir $l"
done
if /usr/bin/grep -rqs project_data "$ASSETS_PATH/assets/embodiments/"; then
  echo "PREFLIGHT: stale cross-cluster paths in embodiment configs; run RoboTwin-RLinf-rgb/script/update_embodiment_config_path.py from $ASSETS_PATH" | tee -a "$SUMMARY"
  preflight_fail=1
fi
if [ "$preflight_fail" -ne 0 ]; then echo "PREFLIGHT FAILED, nothing launched" | tee -a "$SUMMARY"; exit 1; fi
echo "=== $(date) preflight OK ===" | tee -a "$SUMMARY"

echo "=== $(date) LOCAL_EVAL_RUNNER starting all 10 checkpoints ===" | tee -a "$SUMMARY"

run_eval sparse_n30L0p3   "$FM30"
run_eval sparse_n30L0p5   "$FM30"
run_eval sparse_n30L0p7   "$FM30"
run_eval dense_n30L0p5    "$FM30"
run_eval sparse_n100L0p3  "$FM100"
run_eval sparse_n100L0p5  "$FM100"
run_eval sparse_n100L0p7  "$FM100"
run_eval dense_n100L0p3   "$FM100"
run_eval dense_n100L0p5   "$FM100"
run_eval dense_n100L0p7   "$FM100"

echo "ALL_10_EVALS_DONE" | tee -a "$SUMMARY"
