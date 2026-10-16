#!/usr/bin/env bash
# run_cfd_campaign.sh — every job of the CFD case, 30 seeds, N = 20.
# Run experiments/cfd_precompute.py first: passive arms and the reference then
# only read cached CFD runs. Adaptive arms run CFD inline and are split into
# three blocks of 10 seeds (results/parts/), merged at the end.
#   bash experiments/run_cfd_campaign.sh [n_parallel]     (default 4)
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CFD_WORKDIR=${CFD_WORKDIR:-/tmp}
jobs() {
  for a in al_var al_gp ak_log; do for s in 0 10 20; do echo "gen $a $s"; done; done
  for a in gp_orig gp_amp gp_m52 gp_quad gp_log gp_quadlog spce spce_log pas_orig pas_ard; do
    echo "gen $a 0"; done
  for s in gp_orig gp_m52 gp_quad gp_log spce spce_log; do echo "cov $s 0"; done
}
mkdir -p results/logs results/parts
run_one() {
  local kind=$1 a=$2 s=$3 log="results/logs/cfd_${1}_${2}_${3}.log"
  if [[ $kind == gen && $a =~ ^(al_|ak_) ]]; then
    RESULTS_DIR=results/parts/cfd_${a}_${s} python experiments/campaign_generalisation.py cfd $a $s 10 > $log 2>&1
  elif [[ $kind == gen ]]; then
    python experiments/campaign_generalisation.py cfd $a 0 30 > $log 2>&1
  else
    python experiments/campaign_coverage.py cfd $a 0 30 > $log 2>&1
  fi && echo "done   $kind $a $s" || echo "FAILED $kind $a $s"
}
export -f run_one
jobs | xargs -P "${1:-4}" -L 1 bash -c 'run_one "$@"' _
python experiments/merge_parts.py
echo "ALL JOBS FINISHED"
