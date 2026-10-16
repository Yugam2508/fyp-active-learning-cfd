#!/usr/bin/env bash
# run_coverage_campaign.sh — every campaign_coverage.py job, 30 seeds, N = 20.
#   bash experiments/run_coverage_campaign.sh [n_parallel]     (default 4)
# Logs: results/logs/cov_<benchmark>_<surrogate>.log
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
POS="plume borehole otl piston plume_gum borehole_gum"
jobs() {
  for b in physics plume borehole ishigami otl piston borehole_gum plume_gum flood; do
    for s in gp_orig gp_m52 gp_quad spce; do echo "$b $s"; done
    [[ " $POS " == *" $b "* ]] && for s in gp_log spce_log; do echo "$b $s"; done
  done
}
mkdir -p results/logs
run_one() {
  if python experiments/campaign_coverage.py "$1" "$2" 0 30 > "results/logs/cov_$1_$2.log" 2>&1
  then echo "done   $1 $2"; else echo "FAILED $1 $2"; fi
}
export -f run_one
jobs | xargs -P "${1:-4}" -L 1 bash -c 'run_one "$@"' _
echo "ALL JOBS FINISHED"
