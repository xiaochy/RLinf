#!/bin/bash
# Report when the local eval chain finishes OR dies. The earlier watcher waited only for
# the STUDENT_ENV_EVALS_DONE marker, so a killed runner would have left it blocked forever.
set -uo pipefail
R=/data/vxiao/bimanual_manipulation/RLinf-eval
cd "$R"
while true; do
    ppo_done=0; sft_done=0
    grep -q STUDENT_ENV_EVALS_DONE eval_logs_student_env/summary.log 2>/dev/null && ppo_done=1
    grep -q SFT_BASELINE_EVALS_DONE eval_logs_sft_baseline/summary.log 2>/dev/null && sft_done=1
    alive=$(pgrep -fc "local_eval_runner_(student_env|sft_baseline)\.sh" 2>/dev/null || echo 0)
    if { [ "$ppo_done" = 1 ] && [ "$sft_done" = 1 ]; } || [ "$alive" = 0 ]; then
        echo "=== $(date) local eval report (ppo_done=$ppo_done sft_done=$sft_done runners_alive=$alive) ==="
        for f in eval_logs_student_env/summary.log eval_logs_sft_baseline/summary.log; do
            echo "--- $f"; cat "$f" 2>/dev/null || echo "(absent)"
        done
        echo "--- per-run success numbers"
        for d in eval_logs_student_env/*/ eval_logs_sft_baseline/*/; do
            [ -d "$d" ] || continue
            l=$(ls "$d"/run_attempt*.log 2>/dev/null | tail -1); [ -n "$l" ] || continue
            echo "== $d"
            tr -d '\000' < "$l" | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*m//g' \
              | grep -E "success_rate|success_once|eval/|GAVE UP|Traceback|Error" | tail -6
        done
        exit 0
    fi
    sleep 300
done
