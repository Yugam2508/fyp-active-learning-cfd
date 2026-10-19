#!/usr/bin/env bash
# cfd_precompute_helper.sh — when the analytical jobs of a completion stage
# have finished and only the single-core CFD precompute job is left, use the
# idle cores: a second cfd_precompute.py instance with 3 workers works through
# the same flow states from the other end (every run checks the cache first).
#   bash experiments/cfd_precompute_helper.sh "<precompute options>"
set -u
cd "$(dirname "$0")/.."
OPTS=$1
busy() { pgrep -f "experiments/(campaign_|crosscheck_)" > /dev/null; }
pre_alive() { pgrep -f "experiments/cfd_precompute.py $OPTS" > /dev/null; }
while pre_alive && busy; do sleep 120; done
sleep 120
if pre_alive && ! busy; then
  echo "helper start $(date -u +%FT%TZ)"
  CFD_WORKERS=3 OMP_NUM_THREADS=1 CFD_WORKDIR=${CFD_WORKDIR:-/tmp} \
    python experiments/cfd_precompute.py $OPTS --reverse
  echo "helper done $(date -u +%FT%TZ)"
fi
