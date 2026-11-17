#!/bin/bash
# Local 100-seed eval for the student env-track PPO checkpoints, on the same held-out seed
# set and the same eval config as the 2026-11-01 10-checkpoint sweep, so the numbers are
# directly comparable to it. Labels are passed as arguments; each needs a
# ppo_checkpoints_step40/_resume/<label>/global_step_<step> symlink.
#
#   bash local_eval_runner_student_env.sh student_env_n100L0p5 [student_env_n30L0p5 ...]
#
# A label may carry a non-default checkpoint step as "<label>:<step>"; without one the step is
# 40. This exists because the 2026-11-15 80-step extension of the sparse lr-2e-6 arm produces a
# global_step_80, and the alternative -- symlinking step 80 under the name global_step_40 -- puts
# a wrong step in the eval dir name, which is the only place the step is recorded once the run
# is over.
#
#   bash local_eval_runner_student_env.sh sparse_lr2e6_ext80_n100L0p5:80
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

# max_epochs is step+1, not a constant: the eval resumes <step> and runs exactly one more
# epoch, whose val_check_interval=1 is what produces the eval. With the old hardcoded 41 a
# step-80 checkpoint had nothing left to run, so the process exited 0 after FSDP init, wrote
# zero scalars, and the runner recorded it as SUCCEEDED (2026-11-15 16:31, sparse_lr2e6_ext80).
run_eval() {
  local label=$1 model_path=$2 step=$3
  local ckpt_dir=$CKPT_BASE/_resume/$label/global_step_$step
  local log_dir=$LOG_ROOT/$(date +'%Y%m%d-%H%M%S')-ppo_${label}_ckpt${step}_eval
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
      runner.max_epochs=$((step + 1)) \
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
for spec in "$@"; do
  l=${spec%%:*}; st=${spec##*:}; [ "$st" = "$spec" ] && st=40
  check_path "$CKPT_BASE/_resume/$l/global_step_$st/actor" "resume dir $l (step $st)"
  [ -n "$(base_for "$l")" ] || { echo "PREFLIGHT: cannot tell the SFT base for label $l (expected n30 or n100 in the name)" | tee -a "$SUMMARY"; preflight_fail=1; }
  check_path "$(base_for "$l")" "fm_actiononly base for $l"
done
if /usr/bin/grep -rqs project_data "$ASSETS_PATH/assets/embodiments/"; then
  echo "PREFLIGHT: stale cross-cluster paths in embodiment configs; run RoboTwin-RLinf-rgb/script/update_embodiment_config_path.py from $ASSETS_PATH" | tee -a "$SUMMARY"
  preflight_fail=1
fi
# The requirement is ASYMMETRIC, measured 2026-11-16 23:40 on a live eval, steady over 24 s:
#
#   card 0   ray::EmbodiedFSDPActor            1426 MiB
#   card 0   ray::MultiStepRolloutWorker       1944 MiB    -> ~3.4 GiB total
#   card 1   ray::EnvWorker.evaluate          19962 MiB    -> ~20 GiB
#
# The earlier comment here said "~15 GiB on each" and the gate asked 18 GiB of both. That was an
# inference from the 2026-11-15 18:41 OOM, not a measurement, and it is wrong in BOTH directions:
#
#  - card 0 never needs 18 GiB. Asking for it refuses launches that would have been fine, which
#    is how the gate came to block for hours with a card sitting idle.
#  - card 1 needs MORE than 18 GiB. 19962 > 18000, so the old gate would PASS with 18 GiB free
#    and then the EnvWorker would OOM inside the first SubEnv reset -- exactly the failure the
#    comment was written to prevent.
#
# Why the 11-15 OOM looked symmetric: a PPO arm was holding the box, and a TRAINING job puts its
# big EnvWorker on card 0 (the rawv_s3456 OOM at 17:40 on 11-16 reported "GPU 0 ... Process
# 1118040 has 12.34 GiB"). So under contention card 0 is where the TRAINING job's worker sits,
# and the eval's modest 3.4 GiB fails for lack of room rather than for lack of a 15 GiB appetite.
NEED_0=6000      # 3.4 GiB measured + headroom
NEED_1=22000     # 20.0 GiB measured + headroom
while read -r idx free; do
  case "$idx" in 0) need=$NEED_0 ;; 1) need=$NEED_1 ;; *) need=$NEED_1 ;; esac
  if [ "$free" -lt "$need" ]; then
    echo "PREFLIGHT: GPU $idx has only ${free} MiB free; this eval needs ${need} MiB there" \
      | tee -a "$SUMMARY"
    preflight_fail=1
  fi
done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ',')
if [ "$preflight_fail" -ne 0 ]; then echo "PREFLIGHT FAILED, nothing launched" | tee -a "$SUMMARY"; exit 1; fi
echo "=== $(date) preflight OK for: $* ===" | tee -a "$SUMMARY"

for spec in "$@"; do
  l=${spec%%:*}; st=${spec##*:}; [ "$st" = "$spec" ] && st=40
  run_eval "$l" "$(base_for "$l")" "$st"
done

echo "STUDENT_ENV_EVALS_DONE" | tee -a "$SUMMARY"
