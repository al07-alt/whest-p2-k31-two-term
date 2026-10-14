#!/bin/bash
# run from the repo root: REPO=$(pwd) bash v29_experiments/<script>.sh
export REPO=${REPO:-$(pwd)}
cd "${STARTERKIT:-../whest-starterkit}"
EST=$REPO/v29_experiments/estimator_v29_ef.py
run(){ echo "== $1"; env $2 .venv/bin/python $REPO/v29_experiments/fsrun.py $EST 1,2,3 2>&1 | grep final; }
run "g4last1.3" "EF_G4LAST=1.3"
run "g4last0.7" "EF_G4LAST=0.7"
run "d3last1.1" "EF_D3LAST=1.1"
