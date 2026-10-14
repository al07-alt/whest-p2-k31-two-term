#!/bin/bash
# usage: or29b.sh TAG "ENV..." nets   (as or29.sh, but keeps the full run log in runlogs/TAG_i.log)
cd ${STARTERKIT:-../whest-starterkit}
TAG=$1; EXTRA=$2; NETS=${3:-0,1,2}
mkdir -p ${REPO:-.}/logs/runlogs
for i in ${NETS//,/ }; do
  out=$(env EF_K31OR=$i $EXTRA .venv/bin/python ${REPO:-.}/v29_experiments/fsrun.py ${REPO:-.}/v29_experiments/estimator_v29_ef.py $i 2>&1)
  echo "$out" > ${REPO:-.}/logs/runlogs/${TAG}_$i.log
  echo "$out" | grep -E "final" | sed "s/^/$TAG /" || true
  echo "$out" | grep -q final && cp ${REPO:-.}/data/fspred_$(printf %03d $i).npy ${REPO:-.}/data/pred/${TAG}_$i.npy
  echo "$out" | grep -q final || echo "$TAG $i FAILED: $(echo "$out" | tail -3)"
done
