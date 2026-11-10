#!/bin/bash
# Wait for both matched sparse-reward controls (same held-out train seeds as the student
# arm, same lr, same noise_level, same 40 steps) to finish, then eval their step-40
# checkpoints on the same 100 held-out seeds as every other arm. The eval config spans both
# cards (actor+rollout on 0, env on 1), so it cannot share the host with a training run.
set -uo pipefail

R=/data/vxiao/bimanual_manipulation/RLinf-eval
W=/data/vxiao/bimanual_manipulation/RLinf-student-track
RUNS=$W/ppo_sparse_heldoutseeds_runs
LOG=$R/eval_logs_student_env/watcher_sparse_heldout.log
mkdir -p "$(dirname "$LOG")"

NEED=22500   # MiB free on each card before starting; actor+rollout is the big one
POLL=180
STABLE=3

echo "=== $(date) sparse-heldout watcher started ===" | tee -a "$LOG"

for w in n30 n100; do
    until /usr/bin/grep -q "STUDENT_ENV_TRACK_${w}_DONE" "$RUNS/summary_${w}.log" 2>/dev/null; do
        sleep "$POLL"
    done
    echo "=== $(date) $w sparse control finished ===" | tee -a "$LOG"
done

# Each arm gets a label the eval runner can resolve: base_for() picks the pre-PPO
# flow-matching checkpoint from n30 / n100 in the label.
LABELS=()
for w in n30 n100; do
    ck=$(ls -d "$RUNS"/*-robotwin_scan_object_ppo_gap_flow_sparse_heldoutseeds_${w}L0p5*/*/checkpoints/global_step_40 2>/dev/null | tail -1)
    if [ -n "$ck" ]; then
        mkdir -p "$R/ppo_checkpoints_step40/_resume/sparse_heldout_${w}L0p5"
        ln -sfn "$ck" "$R/ppo_checkpoints_step40/_resume/sparse_heldout_${w}L0p5/global_step_40"
        LABELS+=("sparse_heldout_${w}L0p5")
        echo "    $w -> $ck" | tee -a "$LOG"
    else
        echo "=== $(date) $w has no global_step_40; skipping it ===" | tee -a "$LOG"
    fi
done
[ "${#LABELS[@]}" -gt 0 ] || { echo "=== $(date) nothing to evaluate ===" | tee -a "$LOG"; exit 1; }

ok=0
while [ "$ok" -lt "$STABLE" ]; do
    sleep "$POLL"
    free0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    free1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
    if [ "$free0" -ge "$NEED" ] && [ "$free1" -ge "$NEED" ]; then
        ok=$((ok + 1))
    else
        ok=0
    fi
    echo "$(date +%H:%M:%S) free0=${free0} free1=${free1} stable=${ok}/${STABLE}" >> "$LOG"
done

echo "=== $(date) both GPUs free, launching eval for: ${LABELS[*]} ===" | tee -a "$LOG"
exec bash "$R/local_eval_runner_student_env.sh" "${LABELS[@]}"
