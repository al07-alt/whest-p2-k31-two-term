# Pre-registration: confirmation of the two-term (3,1) closure on untouched public networks

Written 2026-10-09, before any per-network run on networks 23-99 for any configuration below.
Its sha256 is recorded in `prereg_sealed.sha256` and quoted in the write-up. (This is an internal
pre-registration: there is no third-party timestamp. The hash only shows that the file was not edited afterwards.)

## What has and has not touched networks 23-99

* All closure parameters were fixed on networks 0-2. That covers the b1 table (joint fit), b2 = 1 (theory) and the
  rescaling grid. Networks 3-22 served as a first held-out set, which was then also used for ablations.
* No per-network V29 run has been made on networks 23-99.
* The graded submission #334821 did see them, but only as part of a single aggregate score; no per-network numbers
  were available to us.
* V29's own constants (504aldo) were fitted on their data, not ours. Some of our offline measurements (the
  EF_LAMG table, marginals) used networks up to 29, but none of them enters the configurations below.

## Configurations (exact environment strings, test copy `estimator_v29_ef.py`)

| tag | env | role |
|---|---|---|
| base | `EF_K31OR=-1` | unchanged V29 |
| two1 | `EF_K31OR=-1 EF_K31TWO=1 EF_K31TWO_B2=1.0` | the submitted closure (= `v29_two_term/estimator.py`) |
| k3075c09 | `EF_K31OR=-1 EF_WK431_K3=0.75 EF_WK431_COV=0.9` | equal-budget comparator: the best rescaling of V29's own term on networks 0-2 (mean −1.4 %, improved all three) |
| abl_v29k3t | `EF_K31OR=-1 EF_K31K3T=1.0` | V29 + the derived κ3T/v term only |
| abl_b1only | `EF_K31OR=-1 EF_K31TWO=1 EF_K31TWO_B2=0.0` | the fitted radial table only |
| two1r | `EF_K31OR=-1 EF_K31TWO=1 EF_K31TWO_B2=1.0 EF_K31TWO_TAB=refit` | b1 refitted with b2 fixed at 1 (variant) |

## Sets

* **Stage 1:** networks 23-49 (27 networks), all six configurations.
* **Stage 2:** networks 50-99 (50 networks), base and two1 only.

## Metric and tests

Per network i, d_i = MSE_final(config)/MSE_final(base) − 1, where MSE_final is the final-layer MSE against the
organisers' truth means.

* **Primary (H1)**, two1 vs base on stage 1 and then on stage 2, each separately:
  * one-sided t-test of mean d_i < 0 at α = 0.05;
  * report the mean, its 95 % network-level t-CI, an exact sign test and a 10 000-resample network bootstrap CI;
  * also report the ratio of summed MSEs.
  * H1 is confirmed on a stage if the one-sided p < 0.05.
* **Secondary (stage 1, Holm-corrected as a family of four, two-sided)**, each on paired per-network differences:
  * S1: abl_v29k3t vs base;
  * S2: abl_b1only vs base;
  * S3: two1 vs k3075c09, comparing MSE_two1/MSE_k3075c09 − 1;
  * S4: two1r vs two1.
* **Descriptive:** a code-drift check (base re-run on networks 3 and 4, compared with `or29_u.log`), and results
  split by network block.

## Reporting rule

Every configuration above is reported whatever the outcome. If H1 fails on a stage, the write-up says so in the
abstract.

## Amendment 1 (written 2026-10-09, before any run on networks 23-99 has finished)

* **What we found.** One of the four terms in which V29 uses the (3,1) slot (the second (2,1) entry, Wick pair
  (3,2) × (1,1)) reads the slot with the transposed orientation. For V29's own symmetric term this makes no
  difference; we checked that V29's predictions agree to float32 rounding. For an asymmetric slot, such as the
  closure, the exact slice, or any K with K_ij ≠ K_ji, the correct object is K = slotᵀ.
* **Configuration added.** `two1t21` = `EF_K31OR=-1 EF_K31TWO=1 EF_K31TWO_B2=1.0 EF_K31T21=1`. On networks 0-2 it gave
  −5.9 / −1.3 / −2.3 %, against two1's −6.2 / −0.9 / −1.6 %.
* **Analysis.** It is added to stage 1 and stage 2, with two new exploratory contrasts that are not part of the
  Holm family above:
  * H1b: two1t21 vs base, one-sided, same test as H1;
  * S5: two1t21 vs two1, two-sided.
* **Unchanged.** The primary hypothesis H1 and the four secondary contrasts are not changed.
