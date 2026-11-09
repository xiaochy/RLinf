#!/bin/bash
# Local smoke test for student_reward.track_source=env: does the simulator ground-truth track reach the
# student, and is the reward it produces sane? One short rollout epoch, no checkpoint writes.
# The committed configs carry grogu paths; everything machine-specific is overridden here.
set -uo pipefail

W=/data/vxiao/bimanual_manipulation/RLinf-student-track   # this worktree (branch student-reward)
E=/data/vxiao/bimanual_manipulation/RLinf-eval            # venv, RoboTwin, curobo, assets, gap_root
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
export RAY_TMPDIR=/tmp/ray_student_smoke
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RAY_TMPDIR"

ASSETS=$E/RoboTwin-RLinf
FM30=$E/fm_actiononly_30demo_ckpt/200.ckpt
STUDENT=$B/SVM-cotrain-student
SVMREF=$B/SVM-reference
LOG_DIR=$W/smoke_logs/$(date +'%Y%m%d-%H%M%S')-student_env_track
mkdir -p "$LOG_DIR"

for p in "$ASSETS/assets/embodiments/aloha-agilex/curobo_left.yml" \
         "$E/gap_root_local/pretrained/dinov2_vitl14_reg4_pretrain.pth" \
         "$FM30" "$STUDENT/checkpoints/student/student.pt" "$SVMREF/svm/preprocess.py"; do
    [ -e "$p" ] || { echo "PREFLIGHT MISSING: $p"; exit 1; }
done
echo "=== $(date) preflight OK, log_dir=$LOG_DIR ==="

SR=env.train.student_reward
python "$EMBODIED_PATH/train_embodied_agent.py" \
  --config-path "$EMBODIED_PATH/config/" \
  --config-name robotwin_scan_object_ppo_gap_flow_student_env_n30L0p5 \
  runner.logger.log_path="$LOG_DIR" \
  runner.max_epochs=1 \
  runner.save_interval=1000 \
  env.train.rollout_epoch=1 \
  env.train.total_num_envs=2 \
  env.train.max_steps_per_rollout_epoch=64 \
  env.train.assets_path="$ASSETS" \
  env.eval.assets_path="$ASSETS" \
  env.eval.total_num_envs=2 \
  env.eval.max_steps_per_rollout_epoch=64 \
  $SR.student_repo="$STUDENT" \
  $SR.svm_repo="$SVMREF" \
  $SR.student_ckpt="$STUDENT/checkpoints/student/student.pt" \
  $SR.dino_repo_dir="$E/gap_root_local/thirdparty/dinov2" \
  $SR.dino_weights="$E/gap_root_local/pretrained/dinov2_vitl14_reg4_pretrain.pth" \
  actor.global_batch_size=8 actor.micro_batch_size=2 \
  actor.model.model_path="$FM30" \
  actor.model.gap_flow.gap_root="$E/gap_root_local" \
  > "$LOG_DIR/smoke.log" 2>&1
rc=$?
echo "=== $(date) smoke exited rc=$rc ==="
echo "log: $LOG_DIR/smoke.log"
exit $rc
