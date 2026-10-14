# Logs

Raw outputs of the scripts quoted in the report.

| log | content | report |
|---|---|---|
| `or29_p.log` | V29 oracle grid at the correct normalisation (tags as in `v29_experiments/oracle_runs.sh`; `g{sc}_{ch}` = slice scale sc/2) | §2.2, §2.3 |
| `or29_l.log`, `or29_m.log`, `or29_n.log`, `or29_o.log` | V29: channel splits, memoryless models (§5.2), second-chaos term | §2.3, §5.2 |
| `v29boot.log` | paired bootstrap CIs (`src/v29boot.py`) | §2.2, §5.2, App. C |
| `or29_q.log`, `or29_r.log`, `or29_s.log` | reruns with saved predictions (§5.2 models, V29 coefficient rescaling, per-use scales) | §5.2, App. C |
| `or29_b.log` … `or29_k.log` | earlier runs at **half** the true slice (`EF_K31OR_SCALE=1`), before the normalisation check; only the 0.5 row of §2.3 uses them (`or29_b` "base"/"or1-15" lines = half slice both uses, `or29_f` = half slice per use) | §2.3 |
| `v29cov1_n*.log` | offline replica of V29's covariance program fed the exact state (f = 0, 0.5, 1) | §2.1 |
| `k31v29ratio.log` | V29's (3,1) coefficient against the LS coefficient | §2.4 |
| `k31alpha3.log`, `k31alpha4.log` | row-wise radial scale by α bin | §2.4 |
| `theory{0,1,2}.log` | λ identity | §4.2 |
| `skewid0b.log`, `skewid1.log`, `skewid2.log` | skewness / kurtosis identities | §4.3, §4.4 |
| `k31coef.log`, `k31eta_n*.log`, `k31res_n*.log`, `k31res2_n*.log` | structure of the slice | §4.4 |
| `tree31_n1.log`, `tree31_n2.log` | tree coefficients, nets 1–2 (net 0 in the earlier release) | §5.1 |
| `onestep10_n1.log`, `onestep10_n2.log` | one-step audit, nets 1–2 | §3.4 |

Note on `or29_c.log`: the lines tagged `radv 0` and `fit1 0` were produced by a bug that zeroed the slice; they equal
the "no (3,1) term" run (`wk0`, re-run in `or29_p.log`).

## Added for the v6 report

| log | content | report |
|---|---|---|
| `or29_t.log` | two-term closure on nets 0–2 (`two` = per-layer b₂ table, `two1` = b₂ = 1) | §4, Table 2 |
| `or29_u.log` | V29 vs two-term closure (`two1`) on held-out nets 3–22 | §4, Figure 3 |
| `or29_v.log` | placebo controls (`xnet`, `perm`) and the submitted file on nets 0–2 (`clean_v5`) | §2, §4 |
| `v5multi.log` | the submitted file, three networks in one process (later calls use deeper Strassen) | §4 |
| `b2check.log` | quadrature check of b₂ = 1 | §3 |

## Added for the v7 report (methodology overhaul; all runs strictly serial, one process per network unless stated)

`src/runstats.py` relabels `or29_b.log`'s `base` lines (they are the half-slice oracle) as `or1-15_dup`. It also
reports every tag/network conflict between files.

**Runs on networks 0–99.**

| log | tags (configuration) | report |
|---|---|---|
| `or29_y.log` | nets 3–22: `s2all` (exact slice), `s2radall` (radial part alone, global LS scale), `s2nonradall` (Σ-swap check, 3–11), `perm` (permuted remainder) | §2, Table 1 |
| `or29_ext.log` | nets 3–22: see the list below | §2, Table 1 |
| `or29_t21.log` | orientation fix `EF_K31T21=1` on nets 0–2 (`base_t21`, `two1t21`, `s2allt21`, `s2k3t21`, `s15allt21`) and `two1t21` on nets 3–22 | §5, Table 4 |
| `or29_drift.log` | `base_rerun` on nets 3–4 (bit-identical to `or29_u.log`) | §4 |
| `v29orig_3.log` | the unmodified V29 file on net 3 (predictions bit-identical to the test copy's `base`) | §1 |
| `or29_sealed.log` | pre-registered stage 1, nets 23–49: `base`, `two1`, `k3075c09`, `abl_v29k3t`, `abl_b1only`, `two1r`; amendment 1: `two1t21` | §4, Tables 2–3 |
| `or29_sealed2.log` | pre-registered stage 2, nets 50–99: `base`, `two1`; amendment 1: `two1t21` | §4 |
| `or29_z.log`, `or29_x.log` | nets 3–22: `two` (b₂ table) and `two1r` (refitted b₁) | §4 |
| `multi_base.json`, `multi_two1.json` (`.log`) | nets 3–22 predicted in one process each | §4 |
| `runlogs.tar.gz` | full output of every run above (extract with `tar xzf runlogs.tar.gz`) (oracle diagnostics such as `[EF] li … true-radial b`; `s15all_*` contain V29's coefficient prints) | §2 |

The `or29_ext.log` tags are:

* `rowrad`: radial part alone, one LS scale per row.
* `rowrad4`, `rowrad16`, `rowrad64`: the row-wise radial part plus the top-k directions of its remainder.
* `wrad`: radial part alone at the F₃F₁-weighted scale.
* `perms1`, `perms2`: remainder permuted within |α| bins, two seeds.
* `donor`: another network's remainder.
* `s15all`: the exact slice ×0.75, printing V29's coefficients.

**Offline checks.**

| log | content | report |
|---|---|---|
| `v29cov1o_n*.log` | one covariance step from the exact state: slot = 0, Kᵀ, or K (wrong orientation); nets 0–5 | §2 |
| `v29cov1s_n*.log`, `covs_summary.log` | symmetric / antisymmetric parts, two-term law, c·Σ at the LS scale; squared-unit shares | §3 |
| `v29cov1f_n*.log`, `covf_summary.log` | in-sample optimal coefficients for 17 smooth features; V29's form at its best local scale | §3 |
| `v29coef_local.log`, `v29ratio20.log` | V29's adapted coefficient against the local optimum and the LS coefficient | §2 |

**Laws, noise and summaries.**

| log | content | report |
|---|---|---|
| `law23.log`, `law_summary.log`, `law_oos.log` | the two-term law on 23 networks; fixed-coefficient transfer | §3 |
| `noise3.log` | three-way noise decomposition (truth, dumps, bake) | §1 |
| `onestep10_n0.log`, `onestep_sq.log` | the T-PCP one-step audit for net 0; all three nets in squared units | §6 |
| `skewid_n*.log`, `skewstats.log` | the skewness identity on nets 0–9, and seeds 5–7 on nets 0–2 | App. E |
| `t21_sanity.log` | V29 with and without the orientation switch (agreement to 8e-7) | §5 |
| `sealed_stage1.log`, `sealed_all.log` | the pre-registered analysis (`src/sealed_analysis.py`) | §4 |
| `runner.log` | start and end time of every queued run | — |
