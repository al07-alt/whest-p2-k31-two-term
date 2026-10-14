### Table A (anatomy). Change in final MSE against V29, % — mean [95 % t-CI over networks], networks improved

| replacement of V29's (3,1) term, both uses | networks 0–2 | networks 3–22 | all |
|---|---|---|---|
| exact slice | -15.8 [-39.3, +7.8], 3/3 | -18.5 [-20.4, -16.6], 20/20 | -18.1 [-20.2, -16.0], 23/23 |
| exact slice at scale 0.75 | -16.0 [-27.6, -4.5], 3/3 | -16.9 [-18.6, -15.2], 20/20 | -16.8 [-18.4, -15.2], 23/23 |
| exact non-radial part + radial part rebuilt on V29's Σ | -16.0 [-39.4, +7.4], 3/3 | -18.7 [-21.4, -16.0], 9/9 | -18.0 [-21.3, -14.7], 12/12 |
| radial part alone, global LS scale | +4.8 [-6.7, +16.3], 0/3 | +3.4 [+2.4, +4.4], 1/20 | +3.6 [+2.5, +4.7], 1/23 |
| radial part alone, $F_3F_1$-weighted scale | +0.5 [-6.9, +7.9], 1/3 | +0.4 [-0.2, +1.1], 9/20 | +0.4 [-0.3, +1.1], 10/23 |
| radial part alone, one LS scale per row | — | -1.0 [-1.9, -0.2], 13/20 | -1.0 [-1.9, -0.2], 13/20 |
| row-wise radial + top-4 of its remainder | — | -4.3 [-7.2, -1.4], 9/10 | -4.3 [-7.2, -1.4], 9/10 |
| row-wise radial + top-16 of its remainder | — | -9.2 [-10.7, -7.7], 20/20 | -9.2 [-10.7, -7.7], 20/20 |
| row-wise radial + top-64 of its remainder | — | -14.9 [-17.3, -12.5], 10/10 | -14.9 [-17.3, -12.5], 10/10 |
| placebo: remainder with labels permuted | +21.1 [+13.4, +28.7], 0/3 | +27.7 [+24.7, +30.7], 0/20 | +26.8 [+24.0, +29.6], 0/23 |
| placebo: remainder permuted within $|\alpha|$ bins, seed 1 | — | +21.8 [+19.8, +23.9], 0/20 | +21.8 [+19.8, +23.9], 0/20 |
| placebo: remainder permuted within $|\alpha|$ bins, seed 2 | — | +19.1 [+16.6, +21.5], 0/20 | +19.1 [+16.6, +21.5], 0/20 |
| placebo: another network's remainder (same layer) | — | +25.3 [+20.8, +29.9], 0/20 | +25.3 [+20.8, +29.9], 0/20 |
| placebo: another network's whole slice | +87.2 [-62.4, +236.8], 0/3 | — | +87.2 [-62.4, +236.8], 0/3 |
| no (3,1) term at all | +17.4 [-4.8, +39.6], 0/3 | — | +17.4 [-4.8, +39.6], 0/3 |

**Fractions of the exact-slice gain** (ratio of mean gains over networks 3–22, network bootstrap 95 % CI):

* two-term closure: 17 % [12, 21] (n = 20)
* radial part alone (global): -18 % [-24, -13] (n = 20)
* row-wise radial alone: 5 % [1, 10] (n = 20)
* row-wise radial + top-4: 23 % [11, 35] (n = 10)
* row-wise radial + top-16: 50 % [42, 59] (n = 20)
* row-wise radial + top-64: 79 % [73, 86] (n = 10)
* exact non-radial + radial on V29 Σ: 101 % [101, 101] (n = 9)
* exact slice at 0.75: 91 % [88, 95] (n = 20)

### Table B (every non-oracle closure variant run on the development networks 0–2 before the choice, by mean change)

| variant | tag | nets 0 / 1 / 2 (%) | mean | improved |
|----------------------------------------------|-------------|----------------------|-------|---------|
| two-term, $b_2=1$, joint $b_1$ table (**chosen, submitted**) | `two1` | -6.2 / -0.9 / -1.6 | -2.9 | 3/3 |
| two-term, $b_2$ per-layer table | `two` | -6.8 / +0.2 / -1.4 | -2.7 | 2/3 |
| V29 term ×0.75 higher-cumulant, ×0.9 covariance | `k3075c09` | -1.7 / -0.2 / -2.5 | -1.4 | 3/3 |
| V29 term ×0.75 higher-cumulant | `k3075` | -3.3 / +1.2 / -1.9 | -1.4 | 2/3 |
| V29 term ×0.85 higher-cumulant | `k3085` | -2.3 / +0.4 / -1.4 | -1.1 | 2/3 |
| V29 term + Gaussian tree births, both uses | `tree1all` | +0.7 / -1.3 / -2.7 | -1.1 | 2/3 |
| V29 term + (4/3) κ3T/v, both uses | `k3t43` | -4.6 / +1.9 / -0.0 | -0.9 | 2/3 |
| V29 term + tree births, covariance only | `treecov` | +1.2 / -1.2 / -2.3 | -0.8 | 2/3 |
| V29 term ×0.5 higher-cumulant | `k305` | -4.3 / +5.2 / -1.6 | -0.2 | 2/3 |
| single latent ×½, both uses | `eta1all` | +2.2 / -0.5 / -2.2 | -0.2 | 2/3 |
| measured-λ radial ×½, both uses | `rad1all` | +1.9 / -0.6 / — | +0.6 | 1/2 |
| V29 term ×1.25 covariance | `cov125` | -3.1 / +4.7 / +2.8 | +1.4 | 1/3 |
| ×1.25 covariance, ×0.75 higher-cumulant | `cov125k3075` | -3.8 / +8.7 / +2.9 | +2.6 | 1/3 |
| V29 term ×0.5, both uses | `wk05` | +7.8 / +2.9 / -1.6 | +3.0 | 1/3 |
| measured-λ radial ×½, covariance only | `rad1cov` | +11.2 / +0.9 / +1.6 | +4.5 | 0/3 |
| single latent ×½, covariance only | `eta1cov` | +10.3 / +1.4 / +2.2 | +4.6 | 0/3 |
| single latent (κ4/v)Σ, both uses | `eta2all` | -0.1 / +7.9 / +10.8 | +6.2 | 1/3 |
| measured-λ radial, z-space, both uses | `zrad2all` | -1.3 / +9.3 / +15.4 | +7.8 | 1/3 |
| V29 term ×1.5 covariance | `cov15` | -1.0 / +15.2 / +10.2 | +8.2 | 1/3 |
| V29 term ×1.5, both uses | `wk15` | +2.5 / +10.4 / +12.5 | +8.5 | 0/3 |
| V29's coefficient on the radial form $v_i\Sigma_{ij}$, both uses | `radv` | +18.0 / +11.6 / +2.8 | +10.8 | 0/3 |
| V29 term ×2, both uses | `wk2` | +15.5 / +34.2 / +35.9 | +28.5 | 0/3 |
| V29 term + (4/3) κ3T/v, covariance only | `k3t43cov` | +58.3 / +108.8 / — | +83.6 | 0/2 |

Run after the choice (not part of the selection): `two1t21` (orientation fix, §5): -5.9 / -1.3 / -2.3 %.

### Table C (two-term closure against V29 on every set)

| networks | role | n | mean (%) | 95 % t-CI | bootstrap CI | improved | Σ MSE (%) | p, t (1-sided) | p, sign (2-sided) |
|-------|----------------------|----|---------|----------------|----------------|--------|---------|--------|-------|
| 0–2 | development | 3 | -2.90 | [-10.14, +4.33] | [-6.24, -0.86] | 3/3 | -2.90 | 0.1 | 0.2 |
| 3–22 | held out | 20 | -3.07 | [-4.10, -2.04] | [-4.02, -2.18] | 19/20 | -3.19 | 3e-06 | 4e-05 |
| 3–7 | held out, in V29's fit range | 5 | -2.26 | [-5.79, +1.27] | [-4.75, -0.42] | 4/5 | -2.43 | 0.08 | 0.4 |
| 8–22 | held out, outside it | 15 | -3.34 | [-4.44, -2.24] | [-4.36, -2.41] | 15/15 | -3.44 | 7e-06 | 6e-05 |
| 23–49 | pre-registered, stage 1 | 27 | -3.62 | [-4.59, -2.64] | [-4.52, -2.71] | 26/27 | -3.51 | 2e-08 | 4e-07 |
| 50–99 | pre-registered, stage 2 | 50 | -2.87 | [-3.42, -2.31] | [-3.41, -2.34] | 47/50 | -2.85 | 3e-14 | 4e-11 |
| 23–99 | stages 1 and 2 | 77 | -3.13 | [-3.62, -2.64] | [-3.61, -2.65] | 73/77 | -3.09 | 9e-21 | 2e-17 |

### Table D1 (held-out networks 3–22; not pre-registered)

Each row: mean change against V29 [95 % t-CI], networks improved; paired change of the variant against the full closure; two-sided p of that paired change, Holm-adjusted over the rows shown.

| variant | vs V29 | improved | variant vs closure | Holm p |
|----------------------------------------|----------------------|---------|----------------------|--------|
| full closure | -3.1 [-4.1, -2.0] | 19/20 | — | — |
| V29 term + κ3T/v (no table) | -2.2 [-3.0, -1.5] | 19/20 | +0.9 [+0.2, +1.5] | 0.02 |
| $b_1$ table alone (no κ3T/v) | -0.5 [-1.2, +0.1] | 12/20 | +2.6 [+1.8, +3.5] | 1e-05 |
| V29 term ×0.75 in the higher-cumulant use | -1.8 [-2.6, -1.0] | 17/20 | +1.4 [+0.6, +2.2] | 0.008 |
| closure with refitted $b_1$ | -2.6 [-3.7, -1.5] | 18/20 | +0.5 [+0.2, +0.8] | 0.008 |
| closure with the measured $b_2$ table (dropped on nets 0–2) | -3.3 [-4.5, -2.1] | 19/20 | -0.3 [-0.5, -0.0] | 0.02 |

### Table D2 (pre-registered stage 1, networks 23–49: every configuration against V29)

| configuration | vs V29 | improved |
|------------------------------------------------------------|----------------------|---------|
| full closure (submitted) | -3.6 [-4.6, -2.6] | 26/27 |
| V29 term + κ3T/v | -2.0 [-2.7, -1.2] | 24/27 |
| $b_1$ table alone | -1.4 [-2.0, -0.8] | 23/27 |
| equal-budget comparator: V29 term ×0.75 higher-cumulant, ×0.9 covariance | -1.8 [-2.2, -1.4] | 26/27 |
| closure with refitted $b_1$ | -2.7 [-3.6, -1.7] | 22/27 |

**Pre-registered secondary family** (stage 1; statistic exactly as registered; two-sided; Holm over the four):

| contrast | MSE ratio − 1 | mean [95 % t-CI] | < 0 | p | Holm p |
|----------------------------------|------------------------------|--------------------|-------|-------|-------|
| S1 V29 term + κ3T/v vs V29 | `abl_v29k3t` / `base` | -2.0 [-2.7, -1.2] | 24/27 | 2e-05 | 5e-05 |
| S2 $b_1$ table alone vs V29 | `abl_b1only` / `base` | -1.4 [-2.0, -0.8] | 23/27 | 5e-05 | 9e-05 |
| S3 closure vs equal-budget comparator | `two1` / `k3075c09` | -1.8 [-2.6, -1.0] | 21/27 | 0.0001 | 0.0001 |
| S4 refitted $b_1$ vs closure | `two1r` / `two1` | +1.0 [+0.7, +1.2] | 1/27 | 9e-08 | 4e-07 |

### Table E (orientation of the (3,1) slot in the (2,1) update; networks 0–2, %)

| configuration | as in V29 | with $K$ in the transposed term |
|----------------------------------------|--------------------------|--------------------------|
| V29 (symmetric slot) | +0.0 / +0.0 / +0.0 | +0.0 / — / — |
| two-term closure | -6.2 / -0.9 / -1.6 | -5.9 / -1.3 / -2.3 |
| exact slice, both uses | -26.6 / -9.0 / -11.7 | -25.9 / -10.4 / -13.1 |
| exact slice, higher-cumulant use only | +66.8 / +49.8 / +60.4 | +68.5 / +54.1 / +55.7 |
| exact slice ×0.75, both uses | -21.3 / -12.7 / -14.0 | -20.8 / -13.2 / -15.3 |

### Table F (one process for all networks 3–22: later predict() calls use deeper Strassen levels)

* closure vs V29, both in one process: -3.1 [-4.1, -2.0], 19/20 improved (n = 20).
* V29 in one process vs V29 one process per network: mean +0.026 %, max |·| 0.435 %.
* FLOPs added by the closure per network: [44097536]; V29 FLOP fractions in the process: [0.2671, 0.254, 0.2526, '…'] (first call first).

