#!/usr/bin/env bash
# run_journal_campaign.sh — every campaign_generalisation.py job of the journal
# study, 30 seeds each, run in parallel (resumable: finished seeds are skipped).
#
#   bash experiments/run_journal_campaign.sh [n_parallel]     (default 4)
#   bash experiments/run_journal_campaign.sh --list           (print the jobs)
#
# Groups
#   A  new baselines (quadratic-trend GP, sparse PCE, AK-MCS design) on the
#      four benchmarks of the conference paper, N = 20
#   R  --augment of the published N = 20 records: regenerates every record,
#      checks that all stored fields are reproduced and adds the thr_* fields
#   B  all arms on the OTL circuit and piston models, N = 20
#   C  non-uniform input models (borehole_gum, plume_gum, flood), N = 20
#   D  budget sweep, N = 10, 40, 80 (adaptive committee: N = 10, 40 only)
# Logs: results/logs/<benchmark>_<arm>_N<N>[_augment].log
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

ORIG="physics plume borehole ishigami"
POS="plume borehole otl piston plume_gum borehole_gum"      # positive outputs
is_pos() { [[ " $POS " == *" $1 "* ]]; }

jobs() {
  # longest jobs first, so that the parallel slots stay balanced
  for b in $ORIG; do echo "$b al_var 40"; done
  for b in otl piston; do echo "$b al_var 20"; echo "$b al_ueigv 20"; done
  for b in $ORIG; do echo "$b al_var 10"; done
  for b in $ORIG; do                                                   # R
    for a in al_var al_ueigv pas_orig pas_ard al_gp gp_orig gp_amp gp_ardnoamp \
             gp_ard gp_m52; do echo "$b $a 20 --augment"; done
    is_pos $b && echo "$b gp_log 20 --augment"
  done
  for b in $ORIG; do                                                   # A
    for a in gp_quad spce ak_m52; do echo "$b $a 20"; done
    is_pos $b && for a in gp_quadlog spce_log ak_log; do echo "$b $a 20"; done
  done
  for b in otl piston; do                                              # B
    for a in gp_orig gp_amp gp_ardnoamp gp_ard gp_m52 gp_quad gp_log gp_quadlog \
             spce spce_log pas_orig pas_ard al_gp ak_m52 ak_log; do echo "$b $a 20"; done
  done
  for b in borehole_gum plume_gum flood; do                            # C
    for a in gp_orig gp_amp gp_m52 gp_quad spce pas_ard al_gp ak_m52; do echo "$b $a 20"; done
    is_pos $b && for a in gp_log gp_quadlog spce_log ak_log; do echo "$b $a 20"; done
  done
  for N in 10 40 80; do                                                # D
    for b in $ORIG otl piston; do
      for a in gp_orig gp_m52 gp_quad spce pas_ard al_gp; do echo "$b $a $N"; done
      is_pos $b && for a in gp_log spce_log; do echo "$b $a $N"; done
    done
  done
}

if [[ "${1:-}" == "--list" ]]; then jobs; exit 0; fi
mkdir -p results/logs
run_one() {
  local b=$1 a=$2 N=$3 flag=${4:-}
  local log="results/logs/${b}_${a}_N${N}${flag:+_augment}.log"
  if python experiments/campaign_generalisation.py "$b" "$a" 0 30 "$N" $flag > "$log" 2>&1
  then echo "done   $b $a N=$N $flag"; else echo "FAILED $b $a N=$N $flag (see $log)"; fi
}
export -f run_one
jobs | xargs -P "${1:-4}" -L 1 bash -c 'run_one "$@"' _
echo "ALL JOBS FINISHED"
