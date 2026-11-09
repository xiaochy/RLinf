#!/bin/bash
# Local PPO run with the student dense reward on the simulator's own track (track_source=env),
# noise_level 0.5, on top of a GAP flow-matching SFT base. Mirrors grogu's gap_ppo_student_*.sbatch:
# auto-resume from the newest global_step_N, up to MAX_STEP. The committed configs carry grogu paths,
# so everything machine-specific is overridden here.
#
#   bash train_student_env_track_local.sh n30|n100 <accelerator_rank>
#
# <accelerator_rank> is the GPU this run gets, as cluster.component_placement. Exporting
# CUDA_VISIBLE_DEVICES does NOT work: RLinf hands each worker the GPUs its placement assigns and
# overwrites CUDA_VISIBLE_DEVICES with an index counted from the full device list, so a run started
# with CUDA_VISIBLE_DEVICES=1 still lands on physical GPU 0. Hydra cannot override the placement key
# from the command line either (its grammar rejects the comma in "actor, env, rollout"), so the run
# gets a generated config that differs from the committed one only in that line.
set -uo pipefail

WHICH=${1:?usage: $0 n30|n100 <accelerator_rank>}
RANK=${2:?usage: $0 n30|n100 <accelerator_rank>}

W=/data/vxiao/bimanual_manipulation/RLinf-student-track
E=/data/vxiao/bimanual_manipulation/RLinf-eval
B=/data/vxiao/bimanual_manipulation

cd "$W"
source "$E/.venv/bin/activate"

export PATH=$HOME/.local/bin:/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

export ROBOTWIN_PATH=$E/RoboTwin-RLinf-rgb
export ROBOT_PLATFORM=ALOHA
export EMBODIED_PATH=$W/examples/embodiment
export REPO_PATH=$W
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# $E is on the path for gap_policy, which is untracked and lives in the RLinf-eval checkout.
export PYTHONPATH=$E/curobo07b:$REPO_PATH:$E:$ROBOTWIN_PATH:${PYTHONPATH:-}

export RAY_raylet_start_wait_time_s=900
export RAY_worker_register_timeout_seconds=3600
export RAY_ADDRESS=local
export RAY_TMPDIR=/tmp/ray_student_env_${WHICH}_r${RANK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RAY_TMPDIR"

BASE_CONFIG=robotwin_scan_object_ppo_gap_flow_student_env_${WHICH}L0p5
CONFIG=${BASE_CONFIG}__rank${RANK}
CFG_DIR=$EMBODIED_PATH/config
sed -E "s/^(    actor, env, rollout: )0-0$/\1${RANK}-${RANK}/" \
    "$CFG_DIR/${BASE_CONFIG}.yaml" > "$CFG_DIR/${CONFIG}.yaml"
/usr/bin/grep -qx "    actor, env, rollout: ${RANK}-${RANK}" "$CFG_DIR/${CONFIG}.yaml" \
    || { echo "could not set the accelerator rank in $CFG_DIR/${CONFIG}.yaml"; exit 2; }
ASSETS=$E/RoboTwin-RLinf
STUDENT=$B/SVM-cotrain-student
SVMREF=$B/SVM-reference
case "$WHICH" in
  n30)  FM=$E/fm_actiononly_30demo_ckpt/200.ckpt ;;
  n100) FM=$E/fm_actiononly_100demo_ckpt/200.ckpt ;;
  *) echo "first argument must be n30 or n100, got $WHICH"; exit 2 ;;
esac

LOG_ROOT=$W/ppo_student_env_runs
SUMMARY=$LOG_ROOT/summary_${WHICH}.log
MAX_STEP=40
mkdir -p "$LOG_ROOT"

for p in "$ASSETS/assets/embodiments/aloha-agilex/curobo_left.yml" \
         "$E/gap_root_local/pretrained/dinov2_vitl14_reg4_pretrain.pth" \
         "$E/gap_root_local/thirdparty/dinov2/hubconf.py" \
         "$FM" "$STUDENT/checkpoints/student/student.pt" "$SVMREF/svm/preprocess.py"; do
    [ -e "$p" ] || { echo "PREFLIGHT MISSING: $p" | tee -a "$SUMMARY"; exit 1; }
done
echo "=== $(date) preflight OK ($WHICH, accelerator rank $RANK, base=$FM) ===" | tee -a "$SUMMARY"

latest_ckpt() {
    ls -d "$LOG_ROOT"/*-${CONFIG}/*/checkpoints/global_step_* 2>/dev/null \
        | sed -E 's#(.*)global_step_([0-9]+)$#\2 \1global_step_\2#' | sort -n | tail -1 | cut -d' ' -f2
}

# Clean up only this run's own ray processes: the machine is shared, so a bare `pkill -f ray::`
# would take down other jobs. The per-run RAY_TMPDIR identifies ours.
cleanup_ray_scoped() {
    local p env
    for p in $(pgrep -f 'ray::|raylet|gcs_server|plasma_store' 2>/dev/null); do
        env=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null)
        printf '%s\n' "$env" | /usr/bin/grep -qx "RAY_TMPDIR=$RAY_TMPDIR" && kill -9 "$p" 2>/dev/null
    done
}

prev=""
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    ck=$(latest_ckpt)
    step=0; [ -n "$ck" ] && step=${ck##*global_step_}
    echo "=== $(date) attempt $attempt: latest ckpt=${ck:-none} (step $step) ===" | tee -a "$SUMMARY"
    [ "$step" -ge "$MAX_STEP" ] && { echo "all $MAX_STEP steps done" | tee -a "$SUMMARY"; break; }
    [ -n "$ck" ] && [ "$ck" = "$prev" ] && { echo "no new checkpoint since last attempt; stopping" | tee -a "$SUMMARY"; break; }
    prev=$ck

    LOG_DIR=$LOG_ROOT/$(date +'%Y%m%d-%H%M%S')-${CONFIG}
    mkdir -p "$LOG_DIR"
    SR=env.train.student_reward
    OV=(runner.logger.log_path="$LOG_DIR" actor.model.model_path="$FM")
    [ -n "$ck" ] && OV+=(runner.resume_dir="$ck")

    python "$EMBODIED_PATH/train_embodied_agent.py" \
      --config-path "$EMBODIED_PATH/config/" --config-name "$CONFIG" \
      "${OV[@]}" \
      env.train.assets_path="$ASSETS" \
      env.eval.assets_path="$ASSETS" \
      $SR.student_repo="$STUDENT" \
      $SR.svm_repo="$SVMREF" \
      $SR.student_ckpt="$STUDENT/checkpoints/student/student.pt" \
      $SR.dino_repo_dir="$E/gap_root_local/thirdparty/dinov2" \
      $SR.dino_weights="$E/gap_root_local/pretrained/dinov2_vitl14_reg4_pretrain.pth" \
      actor.model.gap_flow.gap_root="$E/gap_root_local" \
      > "$LOG_DIR/run_embodiment.log" 2>&1
    rc=$?
    echo "=== $(date) attempt $attempt exited rc=$rc, log=$LOG_DIR/run_embodiment.log ===" | tee -a "$SUMMARY"
    cleanup_ray_scoped
    sleep 15
    [ "$rc" -eq 0 ] && { echo "clean exit" | tee -a "$SUMMARY"; break; }
done
echo "=== $(date) STUDENT_ENV_TRACK_${WHICH}_DONE ===" | tee -a "$SUMMARY"
