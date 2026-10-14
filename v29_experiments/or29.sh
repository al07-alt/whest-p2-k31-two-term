#!/bin/bash
# One V29 run per network (each run is the first predict() of its process), with [EF] switches.
# usage, from the repo root:  REPO=$(pwd) bash v29_experiments/or29.sh TAG "ENV=... ENV=..." 0,1,2
# EF_K31OR=<net> turns on the oracle slice for that net (set automatically below; pass EF_K31OR=-1 to disable).
# Normalisation: the injected slot is 0.5 * EF_K31OR_SCALE * K^T, so EF_K31OR_SCALE=2 is the TRUE slice (report §2.1).
export REPO=${REPO:-$(pwd)}
cd "${STARTERKIT:-$REPO/../whest-starterkit}"
TAG=$1; EXTRA=$2; NETS=${3:-0,1,2}
mkdir -p $REPO/data/pred
for i in ${NETS//,/ }; do
  out=$(env EF_K31OR=$i $EXTRA .venv/bin/python $REPO/v29_experiments/fsrun.py $REPO/v29_experiments/estimator_v29_ef.py $i 2>&1)
  echo "$out" | grep -E "final" | sed "s/^/$TAG /"
  echo "$out" | grep -q final && cp $REPO/data/fspred_$(printf %03d $i).npy $REPO/data/pred/${TAG}_$i.npy
done
