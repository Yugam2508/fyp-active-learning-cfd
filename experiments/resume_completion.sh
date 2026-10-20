#!/usr/bin/env bash
# resume_completion.sh — continue run_completion_campaign.sh after a machine
# restart: resume the stage that was started but not finished (finished seeds
# are skipped by every campaign), and re-arm the CFD helper of stage 2a.
# Does nothing if the runner is already running or all stages are finished.
set -u
cd "$(dirname "$0")/.."
LOG=results/logs/completion_master.log
RUNNER='bash experiments/run_completion_campaign\.sh 4 [12][ab]'
if pgrep -fx "$RUNNER" > /dev/null; then echo "runner already running"; exit 0; fi
if grep -q "ALL STAGES FINISHED" "$LOG"; then echo "all stages finished"; exit 0; fi
st=$(grep -oE "STAGE [0-9][ab] START" "$LOG" | tail -n 1 | awk '{print $2}')
st=${st:-1a}
if grep -q "STAGE $st DONE" "$LOG"; then
  st=$(awk -v s="$st" 'BEGIN{n["1a"]="1b";n["1b"]="2a";n["2a"]="2b";n["2b"]="";print n[s]}')
fi
[ -n "$st" ] || { echo "nothing to resume"; exit 0; }
echo "RESTART $(date -u +%FT%TZ) (container restart; resumed at stage $st)" >> "$LOG"
setsid nohup bash experiments/run_completion_campaign.sh 4 "$st" >> "$LOG" 2>&1 < /dev/null &
case $st in 1b|2a)
  pgrep -fx "bash experiments/cfd_precompute_helper\.sh --designs-only --design=maximin --design=sobol --ref2" > /dev/null || \
    setsid nohup bash experiments/cfd_precompute_helper.sh "--designs-only --design=maximin --design=sobol --ref2" \
      >> results/logs/cmp_pre_helper_2a.log 2>&1 < /dev/null &
esac
echo "resumed at stage $st"
