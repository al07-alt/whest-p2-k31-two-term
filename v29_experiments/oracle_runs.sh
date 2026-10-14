#!/bin/bash
# Every V29 configuration quoted in report §2, §5.2 and Appendix C (three networks each, ~70 s per run).
# Log: logs/or29_p.log (main grid), logs/or29_l..o.log. CIs: python src/v29boot.py <tags>.
R="bash v29_experiments/or29.sh"
$R base "EF_K31OR=-1"
$R s2all "EF_K31OR_SCALE=2"                                     # true slice, both uses
$R s2nonradall "EF_K31OR_SCALE=2 EF_K31OR_MODE=10"              # true non-radial part + radial rebuilt on V29's Sigma
$R s2radall "EF_K31OR_SCALE=2 EF_K31OR_MODE=3"                  # radial part alone, LS scale
for r in 1 4 16 64; do $R rank$r "EF_K31OR_SCALE=2 EF_K31OR_MODE=11 EF_K31OR_RANK=$r"; done
$R mem1-8 "EF_K31OR_SCALE=2 EF_K31OR_LAYERS=1-8"
$R mem1-11 "EF_K31OR_SCALE=2 EF_K31OR_LAYERS=1-11"
$R s2_815 "EF_K31OR_SCALE=2 EF_K31OR_LAYERS=8-15"
for sc in 1 1.5 2 3; do for ch in all cov k3; do $R g${sc}_$ch "EF_K31OR_SCALE=$sc EF_K31OR_CH=$ch"; done; done   # §2.3 grid (scale = sc/2)
$R wk0 "EF_K31OR=-1 EF_WK431=0"                                 # no (3,1) term
# §5.2 memoryless models (both uses)
$R zrad2all "EF_K31OR=-1 EF_K31RAD=2 EF_K31RAD_CH=all"
$R eta2all "EF_K31OR=-1 EF_K31ETA=2 EF_K31ETA_CH=all"
$R k3t43 "EF_K31OR=-1 EF_K31K3T=1.3333333"
$R tree1all "EF_K31OR=-1 EF_TREE=1"
$R s2w1 "EF_K31OR_SCALE=2 EF_K31OR_MODE=8 EF_K31OR_NF=1"
$R s2fit4 "EF_K31OR_SCALE=2 EF_K31OR_MODE=4"
$R s2fit5 "EF_K31OR_SCALE=2 EF_K31OR_MODE=5"
$R s2fit6 "EF_K31OR_SCALE=2 EF_K31OR_MODE=6"
$R s2w5 "EF_K31OR_SCALE=2 EF_K31OR_MODE=8 EF_K31OR_NF=5"
# Appendix C: V29's own (3,1) term rescaled, in both uses or per use
for c in 0.5 1.5 2; do $R wk$c "EF_K31OR=-1 EF_WK431=$c"; done
$R cov125 "EF_K31OR=-1 EF_WK431_COV=1.25"
$R cov15 "EF_K31OR=-1 EF_WK431_COV=1.5"
$R k3075 "EF_K31OR=-1 EF_WK431_K3=0.75"
$R cov125k3075 "EF_K31OR=-1 EF_WK431_COV=1.25 EF_WK431_K3=0.75"
$R k305 "EF_K31OR=-1 EF_WK431_K3=0.5"
$R k3085 "EF_K31OR=-1 EF_WK431_K3=0.85"
# placebo controls (report §2): another network's slice; permuted non-radial remainder
$R xnet "EF_K31OR_SCALE=2 EF_K31OR=1" 0
$R xnet "EF_K31OR_SCALE=2 EF_K31OR=2" 1
$R xnet "EF_K31OR_SCALE=2 EF_K31OR=0" 2
$R perm "EF_K31OR_SCALE=2 EF_K31OR_MODE=12"
# two-term closure on nets 0-2 (report §4, Table 2)
$R two1 "EF_K31OR=-1 EF_K31TWO=1 EF_K31TWO_B2=1.0"
$R two "EF_K31OR=-1 EF_K31TWO=1"
