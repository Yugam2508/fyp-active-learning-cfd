#!/usr/bin/env bash
# run_selection_campaign.sh — campaign_selection.py on all ten models (N = 20).
#   bash experiments/run_selection_campaign.sh [n_parallel]     (default 2)
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/logs
run_one() {
  python experiments/campaign_selection.py "$1" > "results/logs/sel_$1.log" 2>&1 \
    && echo "done   $1" || echo "FAILED $1"
}
export -f run_one
printf "%s\n" cfd plume borehole otl piston borehole_gum plume_gum physics ishigami flood \
  | xargs -P "${1:-2}" -L 1 bash -c 'run_one "$@"' _
echo "ALL JOBS FINISHED"
