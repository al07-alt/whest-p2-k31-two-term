#!/bin/bash
# Report §4: V29 vs V29 + two-term closure on held-out public networks 3-22 (one process per run).
# usage, from the repo root: REPO=$(pwd) bash v29_experiments/validate_two_term.sh   -> logs/or29_u.log format
R="bash v29_experiments/or29.sh"
for i in $(seq 3 22); do
  $R base "EF_K31OR=-1" $i
  $R two1 "EF_K31OR=-1 EF_K31TWO=1 EF_K31TWO_B2=1.0" $i
done
# the submitted file v29_two_term/estimator.py contains only this change and reproduces the 'two1' numbers.
# mechanism ablation (report Table 3), same nets
for i in $(seq 3 22); do
  $R abl_b1only "EF_K31OR=-1 EF_K31TWO=1 EF_K31TWO_B2=0.0" $i
  $R abl_v29k3t "EF_K31OR=-1 EF_K31K3T=1.0" $i
  $R abl_k3075 "EF_K31OR=-1 EF_WK431_K3=0.75" $i
done
# statistics: python src/ablstats.py  -> logs/ablstats.log
