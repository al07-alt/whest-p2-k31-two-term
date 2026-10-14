#!/bin/bash
# run from the repo root: REPO=$(pwd) bash v29_experiments/<script>.sh
export REPO=${REPO:-$(pwd)}
cd "${STARTERKIT:-../whest-starterkit}"
EST=$REPO/v29_experiments/estimator_v29_ef.py
run(){ echo "== $1"; env $2 .venv/bin/python $REPO/v29_experiments/fsrun.py $EST 1,2,3 2>&1 | grep final; }
run "kink3" "EF_KINK=3"
run "dead2" "EF_KINK=2 EF_KINK_DEADONLY=1"
run "kink2.5" "EF_KINK=2.5"
run "kink2" "EF_KINK=2"
