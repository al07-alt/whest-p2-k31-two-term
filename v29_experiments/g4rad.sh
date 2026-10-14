#!/bin/bash
# run from the repo root: REPO=$(pwd) bash v29_experiments/<script>.sh
export REPO=${REPO:-$(pwd)}
cd "${STARTERKIT:-../whest-starterkit}"
EST=$REPO/v29_experiments/estimator_v29_ef.py
run(){ echo "== $1"; env $2 .venv/bin/python $REPO/v29_experiments/fsrun.py $EST 1,2,3 2>&1 | grep final; }
run "g4rad1.0" "EF_G4RAD=1.0"
run "g4rad1.1" "EF_G4RAD=1.1"
