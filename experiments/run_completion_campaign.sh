#!/usr/bin/env bash
# run_completion_campaign.sh — the runs of docs/RESEARCH_PLAN.md (Tiers 1 and 2).
# Stages run one after the other; within a stage, jobs run in parallel.
#   1a  T1.2 Bayesian/recalibrated GP intervals on the N = 20 coverage records
#       (--augment), T1.3 OpenTURNS cross-check and PC-Kriging, T1.6 bump model,
#       T1.1 coverage/selection at N = 10, 40 (analytical), CFD designs for
#       N = 10, 40 (one core)
#   1b  T1.1 CFD at N = 10, 40; T1.1 coverage/selection at N = 80 (analytical)
#   2a  T2.1 design types, T2.2 IMSE design, T2.3 stacking, T2.4 vector output,
#       T2.5 correlated inputs, T2.6 CFD grid convergence index; CFD runs for
#       the design types and the second reference (one core)
#   2b  T2.1/T2.2/T2.3 on the CFD case, T2.6 second reference
# Resumable (finished seeds are skipped).
#   bash experiments/run_completion_campaign.sh [n_parallel] [first_stage]
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CFD_WORKDIR=${CFD_WORKDIR:-/tmp}
P=${1:-4}; FIRST=${2:-1a}
ANA="physics plume borehole ishigami otl piston wingweight otl20 borehole_gum plume_gum flood"
POS="plume borehole otl piston wingweight otl20 borehole_gum plume_gum borehole_corr cfd"
is_pos() { [[ " $POS " == *" $1 "* ]]; }
cands() { if is_pos "$1"; then echo "gp_m52 gp_quad spce gp_log gp_quadlog spce_log"; else echo "gp_m52 gp_quad spce"; fi; }
cands80() { if is_pos "$1"; then echo "gp_log gp_quadlog spce_log"; else echo "gp_m52 gp_quad spce"; fi; }
trend() { if is_pos "$1"; then echo gp_quadlog; else echo gp_quad; fi; }

jobs_1a() {
  echo "pre --designs-only --N=10,40"
  for f in results/results_cov_*_gp_*.json; do                     # T1.2
    b=$(basename "$f" .json); b=${b#results_cov_}
    [[ $b == *_N* || $b == *_d* ]] && continue
    python - "$f" <<'EOF' && echo "aug ${b%_gp_*} gp_${b##*_gp_}"
import json, sys
sys.exit(0 if any("pw_gpt_cov" not in r for r in json.load(open(sys.argv[1]))) else 1)
EOF
  done
  for b in physics plume borehole ishigami otl piston; do echo "ot $b"; done   # T1.3
  for N in 20 40; do                                                         # T1.6
    for a in gp_m52 gp_quad spce pas_ard al_gp ak_m52; do echo "gen bump $a $N"; done
    for s in gp_m52 gp_quad spce; do echo "cov bump $s $N"; done
    echo "sel bump $N"
  done
  for N in 10 40; do                                                         # T1.1
    for b in $ANA; do
      for s in $(cands $b); do echo "cov $b $s $N"; done
      is_pos $b && echo "gen $b gp_quadlog $N"
      echo "sel $b $N"
    done
  done
}
jobs_1b() {
  for N in 10 40; do                                                         # T1.1, CFD
    for a in gp_orig gp_m52 gp_quad gp_log gp_quadlog spce spce_log pas_ard; do echo "gen cfd $a $N"; done
    if [ $N = 40 ]; then for s in 0 10 20; do echo "part cfd al_gp 40 $s"; done
    else echo "gen cfd al_gp 10"; fi
    for s in $(cands cfd); do echo "cov cfd $s $N"; done
    echo "sel cfd $N"
  done
  for b in $ANA; do                                                          # T1.1, N = 80
    for s in $(cands80 $b); do echo "cov $b $s 80"; done
    is_pos $b && echo "gen $b gp_quadlog 80"
    echo "sel $b 80"
  done
}
jobs_2a() {
  echo "pre --designs-only --design=maximin --design=sobol --ref2"
  echo "gci"
  echo "vec"                                                                 # T2.4
  for b in $ANA bump; do
    for D in maximin sobol; do                                               # T2.1
      for a in $(cands $b); do echo "gen $b $a 20 --design=$D"; done
      echo "cov $b $(trend $b) 20 --design=$D --nobayes"
    done
    for N in 20 40; do echo "gen $b al_imse $N"; done                        # T2.2
    echo "stack $b 20"                                                       # T2.3
  done
  for a in $(cands borehole_corr) pas_ard al_gp ak_log; do echo "gen borehole_corr $a 20"; done   # T2.5
  for s in $(cands borehole_corr); do echo "cov borehole_corr $s 20"; done
  echo "sel borehole_corr 20"; echo "stack borehole_corr 20"
}
jobs_2b() {
  for D in maximin sobol; do
    for a in $(cands cfd); do echo "gen cfd $a 20 --design=$D"; done
    echo "cov cfd gp_quadlog 20 --design=$D --nobayes"
  done
  for s in 0 10 20; do echo "part cfd al_imse 20 $s"; done
  echo "stack cfd 20"
  echo "ref2"
}

mkdir -p results/logs results/parts
run_one() {
  local kind=$1; shift
  local tag; tag=$(echo "$kind $*" | tr ' =/' '___' | tr -s '_')
  local log="results/logs/cmp_${tag}.log"
  case $kind in
    pre)   CFD_WORKERS=1 python experiments/cfd_precompute.py "$@" > $log 2>&1 ;;
    aug)   python experiments/campaign_coverage.py $1 $2 0 30 --augment > $log 2>&1 ;;
    gen)   python experiments/campaign_generalisation.py $1 $2 0 30 $3 "${@:4}" > $log 2>&1 ;;
    cov)   python experiments/campaign_coverage.py $1 $2 0 30 $3 "${@:4}" > $log 2>&1 ;;
    sel)   python experiments/campaign_selection.py $1 $2 > $log 2>&1 ;;
    stack) python experiments/campaign_stacking.py $1 $2 > $log 2>&1 ;;
    ot)    python experiments/crosscheck_openturns.py $1 > $log 2>&1 ;;
    vec)   python experiments/campaign_vector.py > $log 2>&1 ;;
    gci)   python experiments/cfd_gci.py > $log 2>&1 ;;
    ref2)  python experiments/cfd_reference2.py > $log 2>&1 ;;
    part)  RESULTS_DIR=results/parts/cfd_${2}_N${3}_${4} \
             python experiments/campaign_generalisation.py $1 $2 $4 10 $3 > $log 2>&1 ;;
  esac && echo "done   $kind $*" || echo "FAILED $kind $*"
}
export -f run_one
started=0
for st in 1a 1b 2a 2b; do
  [ "$st" = "$FIRST" ] && started=1
  [ $started = 1 ] || continue
  echo "STAGE $st START $(date -u +%FT%TZ)"
  jobs_$st | xargs -P "$P" -L 1 bash -c 'run_one "$@"' _
  python experiments/merge_parts.py
  echo "STAGE $st DONE $(date -u +%FT%TZ)"
done
echo "ALL STAGES FINISHED"
