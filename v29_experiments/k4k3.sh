#!/bin/bash
export REPO=${REPO:-$(pwd)}; cd "${STARTERKIT:-../whest-starterkit}"
EST=$REPO/v29_experiments/estimator_v29_ef.py
run(){ echo "== $1"; env $2 .venv/bin/python $REPO/v29_experiments/fsrun.py $EST 1,2,3 2>&1 | grep final; }
run "k4k3 0.7 all" "EF_K4K3=0.7"
run "k4k3 0.7 last" "EF_K4K3=0.7 EF_K4K3_LASTONLY=1"
run "k4k3 0.35 all" "EF_K4K3=0.35"
