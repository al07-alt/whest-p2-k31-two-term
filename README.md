# The (3,1) slice of κ₄ in cumulant propagation: code, logs, pre-registration and measurements

This is the companion repository for the Phase 2 write-up *"The (3,1) slice of κ₄ in cumulant propagation"* (ARC
White-Box Estimation Challenge 2026). The report is `writeup/report.pdf`; its source is `writeup/report.md`, and
`writeup/report_template.md` holds the version with table placeholders.

**Paired submission.** `v29_two_term/estimator.py` is team 504aldo's public V29 estimator (MIT) with one change, marked
`[EF]`: the two-term closure of the (3,1) slice of κ₄ (report §4).

**Pre-registration.** `prereg/prereg_sealed.md` fixes the configurations, the metric and the tests for networks 23–99.
It was written before any run on those networks. Its hashes are in `prereg/prereg_sealed.sha256`; check the
original with `head -n 54 prereg/prereg_sealed.md | sha256sum`.

| version | sha256 | time |
|---|---|---|
| original (first 54 lines) | `eca7755479d0162a222dcede8f577c4421d99f0a177a95235f41d4edc2a4f59a` | 09:00 UTC |
| with amendment 1 | `037c6a642da05debc13da54e7fe017c1a05dc0034ceb7a54b8467ac788d32a20` | 09:26 UTC |

The analysis is `src/sealed_analysis.py`. Its output is in `logs/sealed_all.log` (full precision) and
`logs/sealed_all_4digit.log` (4-digit logged values, `PRECISE=0`).

## Contents

* **`src/`** holds the analysis scripts. Every table of report §4–§5 is regenerated from the raw logs and saved
  predictions by `src/make_tables.py`, which writes `writeup/tables_v7.md`. `writeup/assemble_v7.py` fills the report
  template with those tables.
* **`v29_experiments/estimator_v29_ef.py`** is a V29 test copy with environment switches for every oracle, placebo and
  closure experiment. With all switches off, it reproduced the unmodified V29 file bit for bit on network 3
  (`logs/v29orig_3.log`).
* **The runners** are `v29_experiments/or29b.sh` (one process per network) and `v29_experiments/runner.sh` (a serial
  job queue). The full list of runs, in order, is in `v29_experiments/jobs_v7_all.txt`.
* **`logs/`** holds the raw outputs. Its files are:

  | file | contents |
  |---|---|
  | `or29_*.log` | one line per run: `TAG NET final MSE …` |
  | `logs/runlogs.tar.gz` | full per-run logs, including oracle diagnostics (extract with `tar xzf logs/runlogs.tar.gz -C logs`) |
  | `logs/README.md` | the tag → configuration map |

* **T-PCP**, our float64 research chain, is in `src/pcp3.py`, `src/fac3.py`, `src/mix2.py` and `src/mix.py`.

## Setup

```
python -m venv .venv && .venv/bin/pip install numpy scipy huggingface_hub pyarrow matplotlib
.venv/bin/python src/fetch_mini.py && .venv/bin/python src/convert.py            # data/npy, data/truth.npz
.venv/bin/python src/fetchmarg.py 0,1,...,29                                      # N=1e8 marginals (data/kp)
.venv/bin/python src/fetchpair2.py 0,1,...,22                                     # dense pre_M11/M21/M31 (data/kp)
.venv/bin/python src/fetchpair2.py 0,1,2,3,4,5 M11,pre_M22                        # for the local covariance-step checks
.venv/bin/python src/fetch_bakev2.py 0,1,2,3,4,5                                  # independent bake (data/bk)
```

Run the analysis scripts from the repository root. They read and write under `./data`.

The V29 experiments need the official starter kit (`whest-starterkit`, flopscope 0.12.1). Put it next to this folder
or point `STARTERKIT=/path` at it, and set `REPO=$(pwd)`. Then each line of `v29_experiments/jobs_v7_all.txt` is one
run, for example `REPO=$(pwd) bash v29_experiments/or29b.sh two1 "EF_K31OR=-1 EF_K31TWO=1 EF_K31TWO_B2=1.0" 23`.

**Normalisation.** V29's (3,1) slot holds Kᵀ, and the oracle injects 0.5 · `EF_K31OR_SCALE` · Kᵀ, so
`EF_K31OR_SCALE=2` is the exact slice. `src/v29cov1o.py` checks both the scale and the orientation.

**Precision.** MSEs are recomputed at full precision from the saved predictions whenever the prediction file belongs
to the logged run (`src/runstats.py`). Otherwise the 4-digit logged value is used.

## Claim → script → log

| § | claim | script | log |
|---|---|---|---|
| 1 | noise of truth, dumps and bake | `noise3.py` | `noise3.log` |
| 1 | the test copy reproduces V29 bit for bit; no code drift | runner (`v29orig`, `base_rerun`) | `v29orig_3.log`, `or29_drift.log` |
| 2 | units and orientation of the slot (one covariance step) | `v29cov1o.py <net> 5,10,14` | `v29cov1o_n*.log` |
| 2 | anatomy, Table 1, Figure 1 | runner (`s2all`, `s15all`, `s2radall`, `wrad`, `rowrad*`, `perm*`, `donor`) | `or29_p.log`, `or29_y.log`, `or29_ext.log` |
| 2 | fractions of the exact-slice gain | `make_tables.py` | `writeup/tables_v7.md` |
| 2 | row-wise radial scale by \|α\| (Figure 2) | `law23.py`, `law_summary.py` | `law23.log`, `law_summary.log` |
| 2 | V29's coefficient vs LS and vs local optimum | `v29ratio20.py`, `v29coef_local.py` | `v29ratio20.log`, `v29coef_local.log` |
| 3 | two-term law on 23 networks; b₂ range; R² | `law23.py`, `law_summary.py` | `law23.log`, `law_summary.log` |
| 3 | fixed-coefficient transfer | `law_oos.py` | `law_oos.log` |
| 3 | b₂ = 1 for a quadratic perturbation | `b2check.py` | `b2check.log` |
| 3 | local value of slot contents (symmetric part, law, V29 form, features, orientation) | `v29cov1s.py`, `v29cov1f.py`, `covs_summary.py`, `covf_summary.py` | `v29cov1s_n*.log`, `v29cov1f_n*.log`, `covs_summary.log`, `covf_summary.log` |
| 4 | selection history (Table F1) | `make_tables.py` | `or29_*.log` |
| 4 | closure on every set (Table 2) and the pre-registered tests | `make_tables.py`; `sealed_analysis.py` | `or29_u.log`, `or29_sealed.log`, `or29_sealed2.log`, `sealed_all.log` |
| 4 | ablations and comparators (Table 3) | `make_tables.py`; `sealed_analysis.py` | `or29_w.log`, `or29_x.log`, `or29_sealed.log` |
| 4 | residual-time equivalence | `sealed_analysis.py` | `sealed_all.log` |
| 4 | one process for many networks | runner (`multi_*`) | `multi_base.json`, `multi_two1.json` |
| 4 | graded score #334821 (5.26e-9, raw 2.07e-8, 0 failed) | AIcrowd submission page | — |
| 5 | orientation fix: sanity check, Table 4, held-out and pre-registered contrasts | `t21_sanity.py`; `make_tables.py`; `sealed_analysis.py` | `t21_sanity.log`, `or29_t21.log`, `sealed_all.log` |
| 6 | T-PCP oracle, one-step audit in squared units, tree births | `ortest.py`, `onestep10.py`, `onestep_sq.py`, `tree31.py` | `or0/1/2.log`, `onestep10_n*.log`, `onestep_sq.log`, `tree31_n*.log` |
| E | skewness identity with seed and network spread | `skewid.py <net> 40000 [seed]`, `skewstats.py` | `skewid_n*.log`, `skewstats.log` |

## Attribution

* Wu et al. (ARC, 2026) and their MIT reference code: the framework.
* Team 504aldo (`whest-p2-cumulant-k3`, MIT): the V29 estimator. Both `v29_two_term/estimator.py` and
  `v29_experiments/estimator_v29_ef.py` are their file with our marked changes.
* Team EscAI: the eps² law, oracle studies, accuracy by cancellation.
* Keenan Pepper: the public N = 1e8 moment datasets.
* Development used Claude (Anthropic) under the author's direction, including LLM-run reviews and methodology audits.
