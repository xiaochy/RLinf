#!/bin/bash
# openpi JAX (orbax) finetune -> the bare `Pi0` safetensors layout RLinf's PyTorch openpi
# package loads, so the pi0.5 scan_object SFT can be PPO post-trained with the student
# dense reward.
#
#   bash convert_pi05_scan_object_ckpt.sh 30|100 [step]
#
# `step` defaults to the newest numeric checkpoint dir under the run. Writes
#   pi05-scanobject-rl/converted/pi05_scan_object_ee_lora_<n>/{model.safetensors,config.json}
#   .../physical-intelligence/behavior/norm_stats.json
# which is what actor.model.model_path points at in
# robotwin_scan_object_ppo_openpi_pi05_student_env_ee_n<n>L0p5.yaml.
#
# The LoRA tensors are carried through, NOT merged, so the RLinf side must build the model
# with paligemma_variant=gemma_2b_lora / action_expert_variant=gemma_300m_lora (the configs
# do). Merging instead would also be valid and would let the plain gemma_2b variants load;
# it is not done here because nothing in this repo needs it yet.
set -euo pipefail

N=${1:?usage: $0 30|100 [step]}
case "$N" in 30|100) ;; *) echo "first argument must be 30 or 100"; exit 2 ;; esac

B=/project_data/held/vxiao/Bimanual_Manipulation_Project
P=$B/RoboTwin/policy/pi05
CFG=pi05_scan_object_ee_lora_$N
EXP=scan_object_ee_lora_$N
RUN=$P/checkpoints/$CFG/$EXP
OUT=$B/pi05-scanobject-rl/converted/$CFG

STEP=${2:-}
if [ -z "$STEP" ]; then
    STEP=$(ls "$RUN" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
    [ -n "$STEP" ] || { echo "no numeric checkpoint under $RUN"; exit 1; }
fi
SRC=$RUN/$STEP
STATS=$P/assets/$CFG/scan_object_ee_$N/norm_stats.json

for p in "$SRC/params" "$STATS"; do
    [ -e "$p" ] || { echo "MISSING: $p"; exit 1; }
done
echo "=== converting $SRC (step $STEP) -> $OUT ==="

mkdir -p "$OUT"
cd /data/vxiao/bimanual_manipulation/RLinf-student-track 2>/dev/null || cd "$B/RLinf-setup"
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
    --input-model       "$SRC" \
    --input-norm-stats  "$STATS" \
    --output-model      "$OUT" \
    --output-norm-stats "$OUT/physical-intelligence/behavior/norm_stats.json"

echo "=== done ==="
ls -la "$OUT"
