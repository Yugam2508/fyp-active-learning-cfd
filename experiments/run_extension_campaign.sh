#!/usr/bin/env bash
# run_extension_campaign.sh — the final extension of the journal study:
#   E1  higher-dimensional models (wingweight, d = 10; otl20, d = 20): all
#       main arms at N = 20 and the budget sweep N = 10, 40, 80
#   E2  AK-MCS targeted at the reference 95 % quantile (akq_*) on the models
#       whose physical threshold is not in a tail (plume, plume_gum, cfd)
#   E3  coverage of the log-scale GP with trend (gp_quadlog, the remaining
#       selection candidate) and of every surrogate on the new models
#   E4  LOO selection records for the new models
# CFD runs of akq_log are split into seed blocks (results/parts/), merged at
# the end. Resumable.   bash experiments/run_extension_campaign.sh [n_parallel]
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CFD_WORKDIR=${CFD_WORKDIR:-/tmp}
NEW="wingweight otl20"
jobs() {
  for s in 0 10 20; do echo "part cfd akq_log $s"; done                      # E2, CFD
  for b in $NEW; do echo "gen $b al_var 20"; done
  for b in plume plume_gum; do for a in akq_m52 akq_log; do echo "gen $b $a 20"; done; done
  for b in $NEW; do                                                          # E1
    for a in gp_orig gp_amp gp_m52 gp_quad gp_log gp_quadlog spce spce_log pas_ard al_gp \
             ak_m52 ak_log; do echo "gen $b $a 20"; done
    for N in 10 40 80; do
      for a in gp_orig gp_m52 gp_quad gp_log spce spce_log pas_ard al_gp; do echo "gen $b $a $N"; done
    done
  done
  for b in plume borehole otl piston borehole_gum plume_gum cfd; do echo "cov $b gp_quadlog"; done  # E3
  for b in $NEW; do
    for s in gp_orig gp_m52 gp_quad gp_log gp_quadlog spce spce_log; do echo "cov $b $s"; done
    echo "sel $b"                                                            # E4
  done
}
mkdir -p results/logs results/parts
run_one() {
  local kind=$1 b=$2 a=$3 x=${4:-}
  local log="results/logs/ext_${kind}_${b}_${a}${x:+_$x}.log"
  case $kind in
    part) RESULTS_DIR=results/parts/cfd_${a}_${x} python experiments/campaign_generalisation.py $b $a $x 10 > $log 2>&1 ;;
    gen)  python experiments/campaign_generalisation.py $b $a 0 30 $x > $log 2>&1 ;;
    cov)  python experiments/campaign_coverage.py $b $a 0 30 > $log 2>&1 ;;
    sel)  python experiments/campaign_selection.py $b > "results/logs/ext_sel_${b}.log" 2>&1 ;;
  esac && echo "done   $*" || echo "FAILED $*"
}
export -f run_one
jobs | xargs -P "${1:-4}" -L 1 bash -c 'run_one "$@"' _
python experiments/merge_parts.py
echo "ALL JOBS FINISHED"
