#!/bin/bash
# Local 100-seed eval of a converted pi0.5 scan_object checkpoint in RLinf's harness.
#
#   bash local_eval_runner_pi05_scanobject.sh [<model_path>]
#
# Why this exists: the pi0.5 finetunes were only ever evaluated through openpi's own JAX
# server. Running one through RLinf is both the end-to-end check on the orbax->safetensors
# conversion (experiments/2026-11-13/pi05-lora-to-rlinf-conversion.md) and the pre-PPO
# baseline. The eval seed file here is byte-identical to
# pi05-scanobject-rl/scan_object_eval_seeds.json, so the scenes are the same 100 the
# openpi-side rate (0.639 / 0.663 on the 83 the expert solves) was measured on.
#
# It uses .venv-openpi, NOT .venv: RLinf's openpi wrapper imports openpi, which the gap_flow
# venv does not have, and installing jax[cuda12] into that venv would put the whole student
# pipeline at risk.
set -uo pipefail

R=/data/vxiao/bimanual_manipulation/RLinf-eval
cd "$R"
source .venv-openpi/bin/activate

export PATH=$HOME/.local/bin:/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

export ROBOTWIN_PATH=$R/RoboTwin-RLinf-rgb
export ROBOT_PLATFORM=ALOHA
export EMBODIED_PATH=$R/examples/embodiment
export EVALUATIONS_PATH=$R/evaluations
export REPO_PATH=$R
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
# curobo07b first so it shadows the pip-installed nvidia-curobo package.
export PYTHONPATH=$R/curobo07b:$REPO_PATH:$ROBOTWIN_PATH:${PYTHONPATH:-}

export RAY_raylet_start_wait_time_s=900
export RAY_worker_register_timeout_seconds=3600
export RAY_ADDRESS=local
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=robotwin_scan_object_openpi_pi05_eval_local
MODEL_PATH=${1:-$R/pi05_rlinf_ckpt/pi05_scan_object_lora_100_step6000_openpi}
ASSETS_PATH=$R/RoboTwin-RLinf
LOG_ROOT=$R/eval_logs_pi05_scanobject
mkdir -p "$LOG_ROOT"
SUMMARY=$LOG_ROOT/summary.log
LABEL=$(basename "$MODEL_PATH")
LOG_DIR=$LOG_ROOT/$(date +'%Y%m%d-%H%M%S')-${LABEL}
ATTEMPT_TIMEOUT=${ATTEMPT_TIMEOUT:-16200}

# Each of these has been wrong at least once after moving machines, and the failure is either
# silent or looks like a simulator hang.
preflight_fail=0
check_path() { [ -e "$1" ] || { echo "PREFLIGHT MISSING: $2 -> $1" | tee -a "$SUMMARY"; preflight_fail=1; }; }
check_path "$ASSETS_PATH/assets/embodiments/aloha-agilex/curobo_left.yml" "curobo left config"
check_path "$MODEL_PATH/model.safetensors" "converted pi0.5 weights"
check_path "$MODEL_PATH/norm_stats.json" "norm stats beside the weights"
check_path "$R/evaluations/robotwin/$CONFIG.yaml" "eval config"
check_path "$R/evaluations/eval_embodied_agent.py" "eval entry script"
check_path "$R/.venv-openpi/bin/python" "openpi venv"
if /usr/bin/grep -rqs project_data "$ASSETS_PATH/assets/embodiments/"; then
  echo "PREFLIGHT: stale cross-cluster paths in embodiment configs; run RoboTwin-RLinf-rgb/script/update_embodiment_config_path.py from $ASSETS_PATH" | tee -a "$SUMMARY"
  preflight_fail=1
fi
python -c "import openpi.models.pi0_config" 2>/dev/null || {
  echo "PREFLIGHT: openpi is not importable in .venv-openpi" | tee -a "$SUMMARY"; preflight_fail=1; }
[ "$preflight_fail" -eq 0 ] || { echo "PREFLIGHT FAILED, nothing launched" | tee -a "$SUMMARY"; exit 1; }

# Ray's plasma socket path is RAY_TMPDIR + a long session dir, and AF_UNIX caps it at 107
# bytes; a hashed short directory keeps the label unique without spending that budget.
export RAY_TMPDIR=/tmp/ray_$(printf '%s' "pi05so_$LABEL" | md5sum | cut -c1-10)
mkdir -p "$RAY_TMPDIR" "$LOG_DIR"

echo "=== $(date) eval $LABEL -> $LOG_DIR ===" | tee -a "$SUMMARY"
timeout --kill-after=15 "$ATTEMPT_TIMEOUT" python "$R/evaluations/eval_embodied_agent.py" \
  --config-path "$R/evaluations/robotwin/" --config-name "$CONFIG" \
  runner.logger.log_path="$LOG_DIR" \
  env.eval.assets_path="$ASSETS_PATH" \
  rollout.model.model_path="$MODEL_PATH" \
  > "$LOG_DIR/run.log" 2>&1
rc=$?
echo "=== $(date) $LABEL exit $rc, log_dir=$LOG_DIR ===" | tee -a "$SUMMARY"

# Only this run's own ray processes: the host is shared, so a bare pkill would hit others.
for p in $(pgrep -f 'ray::|raylet|gcs_server|plasma_store|log_monitor|dashboard' 2>/dev/null); do
  if tr '\0' '\n' < /proc/$p/environ 2>/dev/null | /usr/bin/grep -qx "RAY_TMPDIR=$RAY_TMPDIR"; then
    kill -9 "$p" 2>/dev/null
  fi
done
rm -rf "$RAY_TMPDIR"
exit $rc
