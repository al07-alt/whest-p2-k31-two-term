---
title: "The (3,1) slice of κ₄ in cumulant propagation"
subtitle: "Where the strongest public estimator loses accuracy, and a two-term closure that recovers part of it, confirmed in a pre-registered test"
author: "kai_leal_gonzalez · ARC White-Box Estimation Challenge 2026, Phase 2 · paired submission: #334821"
date: "October 2026"
geometry: margin=2cm
fontsize: 10pt
mainfont: FreeSerif
mathfont: DejaVu Math TeX Gyre
colorlinks: true
---

**Abstract.** Cumulant-propagation estimators for wide random ReLU MLPs stall near a raw final-layer MSE of
$2\times10^{-8}$. Using public Monte-Carlo dumps of dense pair cumulants ($N=10^8$), we replace the closure objects
inside 504aldo's V29, the strongest public estimator, one at a time. All numbers are changes in final-layer MSE against
V29 (negative is better), averaged over networks, with 95 % intervals over networks.

* **Where the error is.** Replacing V29's model of the $(3,1)$ slice of the fourth cumulant,
  $K_{ij}=\kappa(z_i,z_i,z_i,z_j)$, by the exact slice gives −18.1 % [−20.2, −16.0]. It helps on all 23 networks with
  dumps.
  * Every placebo we tried makes V29 worse on every network it was tested on.
  * The slice's radial part recovers at most 5 % of this gain, even with one oracle scale per row.
* **A two-term law.** $K_{ij}\approx b_1 v_i\Sigma_{ij}+b_2\,\kappa_{3,i}T_{iij}/v_i$ explains $R^2$ 0.65–0.81 of
  the slice at $\ell=14$, against 0.45–0.65 for the radial term alone, on all 23 networks.
  * $b_2$ lies between 1.17 and 1.41 at every layer from 4 to 14. For a quadratic perturbation of a Gaussian it is 1
    to leading order.
  * One step from the exact state, the law gives 39–59 % of the slice's value in V29's covariance update, against
    28–56 % for V29's own symmetric form at its best scale. Our 17-feature fit reaches at most 62 %.
* **A one-change improvement, tested as pre-registered.** In V29 we replaced the slice term by the law, with $b_2=1$
  and a $b_1$ table taken from the free fit on networks 0–2 (where $b_2\approx1.3$). We then froze this estimator and
  compared it with V29 on networks 23–99, on which no V29 variant had been run.
  * Stage 1 (27 networks): −3.62 % [−4.59, −2.64], 26/27 improved.
  * Stage 2 (50 networks): −2.87 % [−3.42, −2.31], 47/50 improved.
  * All 77: −3.13 % [−3.62, −2.64], 73/77 improved, one-sided $p\approx10^{-20}$.

  The closure recovers about a sixth of the exact slice's gain (17 % [12, 21] on networks 3–22). The derived
  $\kappa_3T/v$ term carries more than half of it. The closure beats the rescaling of V29's own term that was chosen
  on the same development networks (by 1.8 % [1.0, 2.6]). The organisers' graded score moves from $5.42$ to
  $5.26\times10^{-9}$ (−3.0 %).

We also find that one of the four terms in which V29 uses the slice reads it transposed. This is harmless for V29's
symmetric term; fixing it is a correctness change with negligible effect on our closure (§5).
Our paired submission is V29 with the single change of §4.

# 1. Problem, data and statistics

**Task.** The input is $x\sim\mathcal N(0,I_n)$. Each layer computes $z_\ell=W_\ell^\top h_{\ell-1}$ and
$h_\ell=\mathrm{relu}(z_\ell)$, for $\ell=0,\dots,15$, with $h_{-1}=x$, $n=1024$ and He-initialised weights. The
estimand is $\mathbb E\,h_{15}$. The score is the final-layer MSE times the fraction of the $2^{41}$-FLOP budget used,
floored at 0.1. "Raw MSE" means the MSE before that factor.

* *FLOPs.* All variants below except the tree births of §6 use the same FLOPs to four digits, so score ratios equal
  MSE ratios.
* *FLOP fraction.* A process's first `predict()` call uses 0.267 of the budget. Later calls use deeper Strassen levels
  and 0.253–0.254, close to the grader's implied fraction of 0.2545 ($5.42\times10^{-9}/2.13\times10^{-8}$). The
  closure's MSE ratio is the same in both settings (§4).

**Cumulant chains.** A cumulant chain propagates the mean $m$ and covariance $\Sigma$ of $z_\ell$, together with models
of the third and fourth cumulants, through each ReLU (Wu et al., 2026). For a neuron with mean $m_i$, variance
$v_i=s_i^2$ and $\alpha_i=m_i/s_i$, the Gaussian expectations of the ReLU derivatives are $F_1=\Phi(\alpha)$,
$F_2=\varphi(\alpha)/s$ and $F_3=-\alpha\varphi(\alpha)/s^2$. The pair cumulants are
$$T_{ij}=\kappa(z_i,z_i,z_j),\qquad Q_{ij}=\kappa(z_i,z_i,z_j,z_j),\qquad K_{ij}=\kappa(z_i,z_i,z_i,z_j),$$
with the repeated index first; we write $T_{iij}\equiv T_{ij}$ to show which index is doubled. To first order in each
cumulant,
$$\mathrm{Cov}(h_i,h_j)=\Sigma_{ij}F_{1,i}F_{1,j}+\tfrac12\Sigma_{ij}^2F_{2,i}F_{2,j}+\tfrac12\big(T_{ij}F_{2,i}F_{1,j}+T_{ji}F_{1,i}F_{2,j}\big)+\tfrac14Q_{ij}F_{2,i}F_{2,j}+\tfrac16\big(K_{ij}F_{3,i}F_{1,j}+K_{ji}F_{1,i}F_{3,j}\big)+\dots$$
The tripled index of $K$ always carries the third derivative. The slice also enters the updates of the
post-activation $(2,1)$ and $(2,2)$ slices, which carry $\kappa_3$ and $\kappa_4$ forward. These are the slice's **two
uses**: the covariance update and the higher-cumulant updates.

**V29.** V29 is 504aldo's public estimator (MIT) and the strongest public chain. It scores $5.42\times10^{-9}$ (raw
$2.13\times10^{-8}$).

* It carries $\kappa_3$ exactly in factored form.
* It models the slice as $c_\ell\Sigma_{ij}$, a *symmetric* matrix with one adaptively scaled coefficient per layer.
* We changed a copy of V29 only through switches marked `[EF]` in the code. With all switches off, the copy
  reproduced the unmodified V29 file bit for bit on the network we checked (network 3).

**Data.**

* *Networks.* The 100 networks of the organisers' public `mini` split (indices 0–99), with Monte-Carlo ground truth.
* *Dumps.* Keenan Pepper's dumps give dense moment matrices at $N=10^8$ for networks 0–22 ($\mathbb E z_iz_j$,
  $\mathbb E z_i^2z_j$ and $\mathbb E z_i^3z_j$ at every layer), from which every exact slice is computed.
* *Noise.* Three independent estimates of $\mathbb E h_{15}$ exist for networks 0–5: the organisers' truth, the dumps
  and an independently seeded $N=10^8$ run ("bake").
  * Their pairwise MSEs give the noise of each: $7.4\times10^{-11}$ (truth), $8.6\times10^{-10}$ (dumps) and
    $7.5\times10^{-10}$ (bake).
  * The truth's noise is 0.35 % of V29's MSE. Because every variant is scored against the same truth, it only dilutes
    relative changes, by about 0.35 % of their size.

**Splits and order of events.**

| networks | role |
|--------------|--------------------------------------------------------------------------------|
| 0–2 | development: every coefficient, and every choice among variants (Table F1) |
| 3–22 | first held-out set for the closure; later also used for ablations and for the oracle anatomy |
| 23–49, 50–99 | pre-registered confirmation, stages 1 and 2 |

* **What was decided first.** The closure and its comparators were frozen on 8 October. The pre-registration
  (configurations, metric, tests) was hashed on 9 October at 09:00 UTC, with an amendment at 09:26 UTC that adds the
  §5 variant as an exploratory contrast. The first run on networks 23–99 finished at 10:08 UTC.
* **What could have leaked, before sealing.**
  * The graded score of our submission was known before sealing. If the grader's public set contains networks
    23–99, they entered that one aggregate number.
  * The marginal moments of networks 23–29 entered a measured κ₄/v² table (networks 0–29). That table was used by
    three of the rejected variants in Table F1 and by one Appendix A experiment, but by no tested configuration.
  * No V29 variant had been run on any of these networks, network by network, and no coefficient of any tested
    configuration came from them.
* **V29's own fitting range.** V29's tables were fitted on public dumps 0–7, so networks 0–7 are not independent of
  V29 itself.

**Statistics.**

* *Per-network metric.* For each network $i$, $d_i=\mathrm{MSE}_{\rm variant}/\mathrm{MSE}_{\rm V29}-1$ (final layer).
  We report its mean over networks with a 95 % $t$-interval, the number of networks improved, and (for the
  pre-registered tests) a network bootstrap, a sign test and the change of summed MSE.
* *Fractions of a gain.* These are ratios of mean gains over networks, with network-bootstrap intervals.
* *Precision.* MSEs are recomputed at full precision from the saved predictions where available (⟨PREC⟩ runs); the
  rest use 4-digit logged values. We changed the analysis code after stage 2 had finished, in one respect only: full
  precision and exact $t$ quantiles instead of the 4-digit logs. The conclusions are the same under both versions,
  and both outputs are in the repository.
* *Neuron-level intervals.* Bootstrap intervals over output neurons, used only in Appendix D, understate the
  uncertainty and are descriptive.

# 2. Where V29's error lives

We switch V29's slice term to an exact or placebo object built from the dumps, in both uses and at every layer, and
measure the change in final MSE.

**Units and orientation of the slot.** An offline replica of V29's covariance program, fed with the exact state of
one layer, fixes both (three networks, $\ell=5,10,14$).

* The slot holds $K^\top$.
* With it, the next-layer variance error falls from $2.1$–$5.0\times10^{-4}$ (no slice) to $0.4$–$1.2\times10^{-4}$.
* With $K$ in place of $K^\top$, the error is back to $1.6$–$5.2\times10^{-4}$.

All anatomy runs use V29's own handling of the second use, including the orientation issue of §5.

**Table 1.** Change in final MSE against V29 when its slice term is replaced in both uses (mean [95 % CI over
networks]; networks improved).

| replacement | networks | change | improved |
|--------------------------------------------------------------|---------|----------------------------|-----------|
| exact slice | 0–22 | **−18.1 % [−20.2, −16.0]** | 23/23 |
| exact slice ×0.75 | 0–22 | −16.8 % [−18.4, −15.2] | 23/23 |
| radial part alone, global least-squares scale | 0–22 | +3.6 % [+2.5, +4.7] | 1/23 |
| radial part alone, $F_3F_1$-weighted scale | 0–22 | +0.4 % [−0.3, +1.1] | 10/23 |
| radial part alone, one least-squares scale per row | 3–22 | −1.0 % [−1.9, −0.2] | 13/20 |
| row-wise radial + top-16 singular directions of its remainder | 3–22 | −9.2 % [−10.7, −7.7] | 20/20 |
| row-wise radial + top-4 / top-64 of its remainder | 3–12 | ⟨RL⟩ |
| two-term closure of §4 (for reference) | 0–22 | −3.1 % [−4.0, −2.1] | 22/23 |
| *placebo:* remainder with neuron labels permuted | 0–22 | +26.8 % [+24.0, +29.6] | 0/23 |
| *placebo:* remainder permuted within $\lvert\alpha\rvert$ bins (seed 1 / seed 2) | 3–22 | +21.8 % / +19.1 % | 0/20, 0/20 |
| *placebo:* another network's remainder, same layer | 3–22 | +25.3 % [+20.8, +29.9] | 0/20 |
| *placebo:* another network's whole slice | 0–2 | +30 … +150 % | 0/3 |
| no $(3,1)$ term at all | 0–2 | +8 … +26 % | 0/3 |

* **"Radial part".** This is the projection of the exact slice onto $v_i\Sigma_{ij}$, made on the true covariance
  and rebuilt on V29's $\Sigma$. The remainder is the exact slice minus that projection.
* **Σ-swap check.** Rebuilding only the radial part on V29's $\Sigma$, while keeping the exact remainder, reproduces
  the exact-slice gain (−18.0 %, 12/12). This is a consistency check only. The swapped piece is small, so it says
  nothing about leakage.

![Change in final MSE against V29 when its slice term is replaced by exact, partial, placebo or closure objects. Grey points are networks; coloured points and bars are means with 95 % $t$-intervals over networks.](fig/anatomy_forest.pdf){width=86%}

**What Table 1 shows.**

* **The headroom is real and specific.** The exact slice helps on every network, and so does 0.75 of it. Every
  placebo hurts on every network, including:
  * a remainder that keeps the $\lvert\alpha\rvert$ structure but scrambles neuron identities within it;
  * the true remainder of another network.
* **The value is not in the radial part.** On networks 3–22, measured as fractions of the exact slice's mean gain:
  * the radial projection at its least-squares scale recovers −18 % [−24, −13]: it hurts;
  * with importance weights it does nothing;
  * with an oracle scale for every row it recovers 5 % [1, 10];
  * adding the top 16 directions of the row-wise remainder brings it to 50 % [42, 59];
  * the closure of §4 recovers 17 % [12, 21].
* **It is spread over many directions.** After the row-wise projection, the top 4 / 16 / 64 singular directions of the
  remainder recover 23 / 50 / 79 % of the exact gain (networks 3–12 for 4 and 64; 3–22 for 16). On networks 0–2, the
  global remainder's top 64 directions hold 87–88 % of its energy at $\ell=14$ but recover only 45–71 % of its gain
  (Appendix D). A cheap low-rank correction would therefore capture only part of the headroom.

**The two uses tolerate different scales** (networks 0–2; Appendix D).

* Given to both uses, scales 0.5–1 help on all three networks.
* Given to the higher-cumulant updates alone, the exact slice hurts at every scale.
* We have no mechanism for this. The orientation fix of §5 does not remove it (Table 4).

**Why one radial coefficient cannot fit both places.**

* *The fit is anchored far from the kink.* Fitting the radial scale separately for rows binned by the tripled
  neuron's $\lvert\alpha\rvert$ ($\ell=14$, all 23 networks) gives:
  * near the kink ($\lvert\alpha\rvert<0.75$), 0.56–0.67 of the global least-squares scale;
  * for $\lvert\alpha\rvert>4$, 1.15–1.36 (Figure 2).

  The far rows dominate the global fit.
* *The covariance update reads the slice near the kink.* It weights $K$ through $F_3\propto\alpha\varphi(\alpha)$,
  which peaks at $\lvert\alpha\rvert\approx1$.
* *At depth, V29's coefficient is close to what the update wants.*
  * Regressed against the exact slice, V29's coefficient is 0.56–0.83 of the least-squares coefficient of its own form
    (20 networks, $\ell=4,8,11,14$).
  * It is 0.95–1.19 of the coefficient that is optimal for one covariance step from the exact state at $\ell=10$
    and 14, but 0.75–0.82 at $\ell=5$ (networks 3–5).

![Row-wise radial scale of the exact slice at $\ell=14$, by $\lvert\alpha\rvert$ of the tripled neuron, relative to the global least-squares scale (23 networks; median in blue).](fig/k31_ushape.pdf){width=52%}

# 3. A two-term law for the slice

**The law.** For each layer and network we regress the exact slice on two features. The first is radial,
$v_i\Sigma_{ij}$. The second is $\kappa_{3,i}T_{iij}/v_i$, the off-diagonal analogue of
$\kappa_4\approx\lambda v^2+c\,\kappa_3^2/v$:
$$K_{ij}\approx b_1\,v_i\Sigma_{ij}+b_2\,\frac{\kappa_{3,i}\,T_{iij}}{v_i}.$$

* **$b_2$ is stable.** It lies between 1.17 and 1.41 at every layer $\ell=4$–14 on all 23 networks, with medians of
  1.25–1.33. It is higher at $\ell=1$–2.
* **$b_1$ saturates.** Its median levels off at 0.024–0.027 for $\ell\ge7$, while a radial-only fit keeps growing.
* **The fit improves.** At $\ell=14$, $R^2$ rises from 0.45–0.65 (radial only) to 0.65–0.81, on all 23 networks.
* **Coefficients transfer.** We take the $b_1$ table of §4 (from networks 0–2) with $b_2=1$. On networks 3–22 at
  $\ell=14$ it reaches $R^2$ 0.63–0.79 (median 0.73). That is 0.013–0.044 below a per-network fit of the same two
  features.

**The second coefficient in the simplest case.** Let $z_i=g+a(g^2-1)$ and $z_j=\rho g+\sqrt{1-\rho^2}\,e$, with $g,e$
independent standard normals. To leading order in $a$,
$$K_{ij}=24a^2\rho,\qquad\kappa_{3,i}=6a,\qquad T_{iij}=4a\rho,\qquad v_i=1,$$
so $K_{ij}=\kappa_{3,i}T_{iij}/v_i$ and $b_2=1$ for every $\rho$.

* Quadrature gives 1.0001 at $a=0.01$.
* Perturbing $z_j$ as well, with the same sign, raises the ratio (1.05–1.44 in our cases). With the opposite sign it
  falls below 1.
* Real neurons have skewness of the sign of their mean (Appendix E), so measured $b_2>1$ is the expected direction.

**What the law is worth locally.** We feed V29's covariance program the exact state of one layer and score each slot by
the share of the exact slice's reduction of the squared next-layer variance error that it achieves. The b₁ table
was fitted on networks 0–2, so networks 3–5 are out of sample for it.

| slot content (in-sample coefficients where marked) | nets 0–2 | nets 3–5 |
|----------------------------------------------------------------|--------------|--------------|
| exact slice | 100 % | 100 % |
| its symmetric part only | 72–87 % | 70–85 % |
| all 17 smooth features of the current state (in-sample) | 50–61 % | 49–62 % |
| two-term law (in-sample coefficients) | 41–59 % | 40–59 % |
| **two-term law as used in §4** ($b_1$ table, $b_2=1$) | **40–58 %** | **39–59 %** |
| V29's form $c\,\Sigma_{ij}$ at its best local scale (in-sample) | 33–55 % | 28–56 % |
| V29's form at its entry-wise least-squares scale | 16–54 % | 8–56 % |
| exact slice with the wrong orientation ($K$ for $K^\top$) | −12 … +46 % | −20 … +42 % |

The ranges run over networks and $\ell=5,10,14$. The 17 features are radial and latent terms (also binned by
$\lvert\alpha\rvert$), mean terms, $\kappa_4\Sigma/v$, transposes and $c\,\Sigma$.

* **The fixed law is near the best its features allow.** It is 0–4 points below the in-sample optimum for its two
  features. The locally preferred $b_2$ (1.06–1.49) brackets the entry-wise fits.
* **Our smooth features stop near 60 %.** In this test, the rest of the slice's local value appears to sit in
  entry-level structure, consistent
  with §2.
* **The law's advantage over V29's form is largest at $\ell\ge10$**, where the slice is largest.
* **Asymmetry matters.** The antisymmetric part holds only 12–20 % of the slice's squared norm. Yet orienting it
  wrongly costs half or more of the slice's local value, and can make the slice worse than none. V29's form is
  symmetric, so it cannot represent this part at all.

# 4. A two-term closure inside V29

**The change.** In both uses we replace V29's term $c_\ell\Sigma_{ij}$ by
$$K_{ij}=b_{1,\ell}\,v_i\Sigma_{ij}+\kappa_{3,i}T_{iij}/v_i .$$

* **Inputs.** $v$, $\Sigma$, $\kappa_3$ and $T$ are V29's own propagated quantities. $T$ is V29's $(2,1)$ slice, and
  $\kappa_3$ its diagonal.
* **The $b_1$ table and $b_2=1$ are not a joint fit.** $b_{1,\ell}$ is the radial coefficient of the free two-term fit
  of §3 (where $b_2\approx1.3$), averaged over networks 0–2. $b_2=1$ comes from the calculation of §3.
  * We did not refit $b_1$ with $b_2=1$. That self-consistent refit is 5–34 % larger at $\ell\ge6$, growing with depth,
    and it is worse end to end (Table 3, S4).
  * So the pairing is partly an empirical choice made on networks 0–2.
  * Hindsight from networks 3–22 (Table 3): the fully fitted law, with the measured $b_2$ table ($\approx1.3$), does
    0.3 % better than the submitted closure. The self-consistent $b_2=1$ refit does 0.5 % worse.
  * The theory value $b_2=1$ is therefore a slight underestimate for real networks, as §3 anticipates for neurons
    whose partners are also skewed. We kept the submitted closure frozen for the pre-registered test.
* **Cost.** Three elementwise $n^2$ operations per layer: 44,097,536 FLOPs per network, as counted in the
  multi-process logs. That is 0.002 % of the budget.
  * Residual time changes by ⟨RT⟩ on networks 23–99. On networks 3–22 it changes by $+0.4$ ms ($[-7.2,+8.0]$).
  * Both intervals lie inside a ±20 ms equivalence margin, which we chose after the fact.
  * On our machine V29 itself runs at 560–690 ms of residual time, against the 400 ms cap. The graded run had no
    failures.

**How the variant was chosen.** Table F1 (Appendix F) lists all 23 non-oracle variants of the slice term run on networks 0–2 before
the choice. We chose the two-term law with $b_2=1$, which had the best mean and improved all three networks. The best
rescaling of V29's own term, ×0.75 in the higher-cumulant use and ×0.9 in the covariance use, also improved all three.
It was chosen from the same development networks, so we pre-registered it as the comparator.

**Table 2.** The closure against V29 on every set.

⟨TC⟩

![Per-network change in final MSE from the closure on the held-out networks and both pre-registered stages (blue: better than V29; red: worse). Dashed lines are means.](fig/two_term_sets.pdf){width=92%}

* **The pre-registered test confirms the closure.** On stage 1 the primary hypothesis holds, with one-sided
  $p=2\times10^{-8}$. Stage 2: ⟨S2b⟩.
* **Graded result.** The organisers graded the submitted file (#334821) at $5.26\times10^{-9}$ (raw
  $2.07\times10^{-8}$) with no failed MLPs, against $5.42\times10^{-9}$ for unchanged V29 (#334406), a change of −3.0 %.
  The graded set may overlap the networks above, so this is a consistency check, not an independent test.
* **Robustness.**
  * *Later calls in one process.* Later `predict()` calls in a process use deeper Strassen levels. With networks 3–22
    predicted in one process, the closure gives ⟨MP⟩.
  * *Code drift.* Re-running the test copy on two held-out networks reproduces the earlier predictions bit for bit.
  * *V29's fitting range.* On networks 3–7 the change is −2.3 % [−5.8, +1.3] (4/5), and on 8–22 it is −3.3 %
    [−4.4, −2.2] (15/15). These five networks cannot separate the two.

**Which part does the work.**

**Table 3.** Ablations and comparators. The held-out part is not pre-registered; the stage-1 part is.

⟨TD⟩

* **The derived term carries more than half of the gain.** Added to V29's own term, $\kappa_3T/v$ alone gives 73 % of
  the closure's mean gain on networks 3–22 and 54 % on stage 1.
* **The $b_1$ table alone helps a little.** It gives −1.4 % on stage 1; on networks 3–22 its −0.5 % was not
  significant.
* **The parts are complementary.** The full closure beats each part alone (paired, Holm-adjusted).
* **The closure beats the pre-registered rescaling comparator** by 1.8 % [1.0, 2.6] (S3).

**What the closure does not reach.** It recovers 17 % [12, 21] of the exact slice's gain on networks 3–22. Appendix D
lists closures with coefficients fitted to the scored network itself; even those reach only about a quarter of it.

# 5. An orientation inconsistency in V29's $(2,1)$ update

**The derivation.** Take $f=h_i^2h_j$. A fourth cumulant changes $\mathbb E f$, to first order, by
$\frac1{24}\sum\kappa_{abcd}\,\mathbb E\,\partial_a\partial_b\partial_c\partial_d f$. The two $(3,1)$ arrangements
give
$$\tfrac16K_{ij}\,\mathbb E\,(h_i^2)'''\,\mathbb E\,h_j'+\tfrac16K_{ji}\,\mathbb E\,(h_i^2)'\,\mathbb E\,h_j''' .$$

**What V29 does.** V29's term table uses the slot $w=K^\top$ (identified in §2) in four entries. We checked each
against the Wick expansion.

* The covariance and $(2,2)$ entries carry coefficient $\tfrac13$ and are symmetrised, so both arrangements come out
  right.
* The $(2,1)$ update is not symmetrised. Its first entry, $\tfrac16w_{ij}\,\mathbb E(h_i^2)'F_{3,j}$, is right. Its
  second, $\tfrac16w_{ij}\,\mathbb E(h_i^2)'''F_{1,j}$, gives $K_{ji}$ where the derivation needs $K_{ij}$.
* For V29's symmetric term this changes nothing: with the fix, V29's predictions agree to float32 rounding (relative
  $8\times10^{-7}$).
* For any asymmetric slice — the closure, or the exact slice — the entry is wrong. §3 shows that the orientation is
  not a detail: swapping it in the covariance update removes half or more of the slice's local value.
* The switch `EF_K31T21` puts $w^\top$ in that entry. An independent review of the term table reached the same
  conclusion.

**Effect.**

**Table 4.** Change in final MSE against V29 on networks 0–2, with V29's orientation and with the fix (%).

⟨TE⟩

* On networks 3–22 the fix changes the closure by −0.10 % [−0.19, −0.00] (13/20), at full precision.
* The pre-registered exploratory contrast on networks 23–99 gives ⟨T21S⟩.
* The fix is therefore a correctness change, not a source of gain. For the exact slice it changes the gain by
  +0.7 / −1.4 / −1.4 points on networks 0–2 (Table 4). This is too few networks to say whether §2's anatomy is
  conservative.
* It does not explain why the higher-cumulant use dislikes the exact slice at full scale.

# 6. The same object in a float64 chain

This section uses T-PCP, our float64 research chain, which carries exact third-cumulant sources (Appendix C), on
networks 0–2.

**The oracle.** Replacing T-PCP's radial slice by the exact one lowers the final MSE by 19.6 / 14.7 / 26.9 %. It does
so only together with Edgeworth-corrected Hermite coefficients,
$\hat f_k=f_k+\kappa_3 f_{k+3}/(6s^3)+\kappa_4f_{k+4}/(24s^4)$; without them, the exact slice costs +21.4 % on net 0.

**The pairing, to first order.** Take $K_{ij}=\lambda v_i\Sigma_{ij}$ and $\kappa_{4,i}=\lambda v_i^2$.

* The cross term contributes $-\tfrac{\lambda}{6}\alpha_i\varphi_i\Sigma_{ij}F_{1,j}$.
* The Edgeworth shift of $F_{1,i}$ contributes $\big(\tfrac{\lambda}{8}\alpha_i-\tfrac{\lambda}{24}\alpha_i^3\big)\varphi_i\Sigma_{ij}F_{1,j}$.
* The sum is $-\tfrac{\lambda}{24}\alpha_i(1+\alpha_i^2)\varphi_i\Sigma_{ij}F_{1,j}$.

For $\lvert\alpha\rvert\le1$ the cross term alone is 2–4 times the net effect, and the shift alone has the opposite
sign. So the two corrections must come together.

**One step from the exact state.** With the exact slice, T-PCP's next-layer variances reach 1.01–1.07× the bake's
noise floor at $\ell\le5$ and 1.22–1.34× at $\ell=14$. In squared units, the radial slice achieves 40–45 % of the
exact slice's error reduction at $\ell=5$ and 21–30 % at $\ell=14$; a constant noise floor cancels in these ratios.

**Tree births.** For Gaussian pre-activations, the leading connected fourth cumulant of the post-activations is a sum
over 16 labelled trees (Price's theorem). Contracted with the next weights, it gives the slice's births in six $n^3$
products. Write $G=\Sigma_{\rm off}\,\mathrm{diag}(F_1)W$, $D=\mathrm{diag}(F_2)W$ and $H=\Sigma_{\rm off}(D\circ G)$.
Then
$$K^{\rm tree}=6(H\circ D)^\top G+6(H\circ G)^\top D+3\left(\mathrm{diag}(F_3)W\circ G\circ G\right)^\top G+(G\circ G\circ G)^\top\mathrm{diag}(F_3)W .$$

* Regressed together with the radial term, its coefficient is 0.92–1.11 (theory: 1).
* Its size is $\|K^{\rm tree}\|/\|K\|=0.12$–0.21.
* It improves T-PCP by 2.9 / 5.5 / 4.6 %.
* Inside V29 it gives +0.7 / −1.3 / −2.7 % at +31 % FLOPs (Table F1).

# 7. Limitations

* **Scope.** All networks share the task architecture. The anatomy covers networks 0–22, the networks with dumps.
  The scale grid of the two uses, the T-PCP results and part of the local analysis cover networks 0–5 or 0–2.
* **Size of the gain.** The closure recovers about a sixth of the exact slice's gain. Even the full exact slice would
  move V29's raw MSE by about 18 %, roughly to $4.4\times10^{-9}$ in score. The leaders are near
  $1.7\times10^{-9}$, so the slice is one real piece of the remaining error, not the whole.
* **Oracle noise.** The injected slices carry the dumps' Monte-Carlo noise, so oracle gains are, if anything,
  understated.
* **Selection.**
  * The closure was chosen among the 23 variants of Table F1.
  * Networks 3–22 served as a held-out set and then for ablations and anatomy.
  * The pairing of $b_2=1$ with an unrefit $b_1$ was made on networks 0–2 (§4).
  * The variant we dropped there, with the measured $b_2$, turned out slightly better on networks 3–22.
  * Only the pre-registered stages are free of these choices.
  * The pre-registration is internal: its hashes show it was not edited after sealing, but no third party
    timestamped it.
* **Open mechanisms.**
  * Why the two uses tolerate different scales.
  * Whether the slice's remaining value is carried memory, which is a hypothesis. The natural test is a transported
    $\kappa_4$ source in factored form, built like V29's $\kappa_3$ sources and fed to both uses. Its cost is the open
    question.

# Appendix A. Calibration inside V29 (net 0 unless stated)

**V29's moments against the bake.**

* *$\kappa_3$.* Slope 0.98–1.00 and correlation 0.999 at every layer.
* *$\kappa_4$.* The slope falls from 0.99 at $\ell=1$ to 0.75 at $\ell=15$.
* *Variance.* The relative error grows from $1.4\times10^{-4}$ to $1.4$–$1.5\times10^{-3}$.

**Corrections that are right in isolation hurt V29** (networks 0–2).

* Scaling the readout $\kappa_4$ by 1.3 corrects its slope but costs +10 %.
* Using a measured $\lambda_\ell v^2$ table (networks 0–29) as the readout $\kappa_4$ costs +2–3 %.
* Rescaling V29's own slice term by 1.5 costs +2.5 / +10.4 / +12.5 %, and by 2 costs +15.5 / +34.2 / +35.9 %.

We read this as joint calibration of V29's fitted pieces around a $\kappa_4$ channel that under-accumulates. This is a
hypothesis.

# Appendix B. Old third-cumulant content and the output Jacobian (net 0, T-PCP)

* **Old content is radial.** The $(2,1)$ content carried by sources aged five transports or more is fitted by
  $r_1\,\mathrm{Sym}(m\otimes\Sigma)+r_2\,m^{\otimes3}$ with $R^2$ 0.57–0.90, rising with depth. Its share of $T$
  rises from 0.24 to 0.69.
* **But replacing it is costly.** Replacing it by that fit raises the final MSE from $2.5\times10^{-8}$ to
  $5.9\times10^{-8}$, $1.3\times10^{-7}$ and $3.2\times10^{-7}$, for sources older than 10, 8 and 6 layers.
* **The output Jacobian is low-rank.** The Jacobian of $\mathbb E h_{15}$ with respect to the layer-$\ell$ means puts
  99.7, 99.2, 96.7 and 84.7 % of its energy in its top 128 directions, at $\ell=1,4,8,12$.

# Appendix C. T-PCP and further measurements

**T-PCP.** T-PCP carries $m$, $\Sigma$, $T$ and $Q$ with a fused Mehler/Hermite transfer of order 6, and transports each
third-cumulant source exactly with two dense $n\times n$ factors per source and layer. On net 0 its raw MSE is
$2.48\times10^{-8}$.

**A "feedback" term.** A mixed term derived from a Monte-Carlo residual has fitted slope 0.93 (theory: 1) and improves
T-PCP by 16 % on net 0. Its content resembles the rank-16 "D21 feedback" factors that 504aldo introduced
empirically; we have not checked the correspondence formally.

**Kink-sparsity masks in V29 (networks 0–2).** Masking columns or rows of far-from-kink neurons costs +3.9 % to +94 %.

**Measured negatives.**

| idea | result |
|------------------------------|------------------------------------------------------------|
| keep only the youngest one or two source ages | final MSE ×8.5–13 |
| learned per-neuron correction of the local error | held-out $R^2\le0.06$ |
| second-order 1-D Edgeworth term in T-PCP | +2–4 % |
| exact instead of rank-16 births in T-PCP | −5 % |

# Appendix D. Development-network tables (networks 0–2; descriptive)

Intervals are paired bootstraps over output neurons. They understate the uncertainty (§1).

**D1. Exact-slice variants** (both uses; change in final MSE against V29, %).

| replacement | net 0 | net 1 | net 2 |
|--------------------------------------------|--------------|--------------|--------------|
| exact slice | −26.6 [−31.6, −21.4] | −9.0 [−15.7, −1.7] | −11.7 [−17.2, −6.3] |
| Σ-swap check (radial part rebuilt on V29's $\Sigma$) | −26.8 [−31.6, −21.6] | −9.3 [−16.3, −1.6] | −11.9 [−17.1, −6.4] |
| radial part alone (global LS scale) | +0.1 [−2.6, +2.9] | +5.0 [+2.3, +7.9] | +9.3 [+6.5, +12.3] |
| radial + top-1 / 4 / 16 / 64 of the remainder | −1.7 / −5.6 / −8.4 / −17.6 | +5.3 / −0.3 / −4.0 / −6.6 | +10.3 / +7.6 / +1.1 / −5.3 |
| exact slice in layers 1–8 / 1–11 / 8–15 only | −12.3 / −22.9 / −20.1 | −3.9 / −7.7 / −12.3 | −4.6 / −9.6 / −6.4 |

**D2. Scale × use grid** (exact slice; nets 0 / 1 / 2, %). The 0.5 row comes from 4-digit logs.

| scale | both uses | covariance only | higher-cumulant only |
|--------|------------------------------|------------------------------|------------------------------|
| 0.5 | −10.9 / −9.1 / −11.6 | −12.0 / −5.7 / −13.9 | +5.2 / +2.7 / +7.1 |
| 0.75 | −21.3 / −12.7 / −14.0 | −14.9 / +12.8 / −3.5 | +28.9 / +17.6 / +26.3 |
| 1 | −26.6 / −9.0 / −11.7 | +16.8 / +68.4 / +32.9 | +66.8 / +49.8 / +60.4 |
| 1.5 | −20.8 / +18.5 / +7.2 | +182 / +285 / +184 | +186 / +157 / +169 |

**D3. Closures with coefficients fitted to the exact slice of the scored network** (per layer, in-sample, so
optimistic by construction; %).

| closure | net 0 | net 1 | net 2 |
|----------------------------------------|--------------|--------------|--------------|
| two-term law, both coefficients fitted, exact $\kappa_3$ and $T$ | −6.9 | +0.3 | −2.5 |
| two-term law, both coefficients fitted, V29's $\kappa_3$ and $T$ (+ transpose) | −7.0 | +0.1 | −2.4 |
| radial + three mean terms | −5.3 | +1.5 | +0.1 |
| five features, $F_3F_1$-weighted fit | −6.6 | +0.3 | +0.6 |
| radial, $F_3F_1$-weighted scale | −2.6 | +1.0 | +3.3 |
| radial alone, LS scale | +0.1 | +5.0 | +9.3 |

# Appendix E. Skewness from the mean: a fixed-input identity

Hold the distribution of the layer input $h$ fixed, and average only over the current layer's weights.

* *Notation.* Write $\mu=\mathbb E h$, $\hat\mu=\mu/\|\mu\|$ and $d=h-\mu$, with $u=\hat\mu^\top d$ and
  $d_\perp=d-u\hat\mu$.
* *Weights.* Split neuron $a$'s weights as $w_a=t_a\hat\mu+w_{a\perp}$, so that $m_a=\|\mu\|t_a$, and let
  $\omega_a=\|w_{a\perp}\|^2/(n-1)$.

Averaging over the direction of $w_{a\perp}$, which is isotropic under He initialisation, gives
$$\mathbb E[\kappa_3(z_a)]=t_a^3\kappa_3(u)+3t_a\omega_a\,\mathrm{Cov}\!\left(u,\|d_\perp\|^2\right),\qquad \mathbb E[v_a]=t_a^2\mathrm{Var}\,u+\omega_a\mathrm{tr}\,C_\perp,$$
so to leading order $\gamma_a\approx\alpha_a\cdot3\,\mathrm{Cov}(u,\|d_\perp\|^2)/(\|\mu\|\,\mathrm{tr}\,C_\perp)$: each
neuron's skewness is its standardized mean times one network scalar.

**How well it holds** (networks 0–9, $4\times10^4$ forward passes each).

* The slope against the $N=10^8$ truth is 0.94–0.97 on average across networks (range 0.91–0.99). $R^2$ is 0.81 at
  $\ell=4$ and 0.86–0.88 at $\ell\ge8$.
* Across three seeds the slope varies by at most 0.02.

**What it does not do.** A table $\gamma=c_\ell\alpha$ fitted on the other nine networks predicts better out of sample,
with $R^2$ higher by 0.007–0.020 on every network. So the identity explains, without a fitted parameter and to within
3–6 %, why $\gamma\propto\alpha$ baselines work. Its fourth-order analogue gives the known fixed-input kurtosis
factor $\lambda=3[\mathrm{Var}\|d\|^2-2\|C\|_F^2]/(\mathrm{tr}\,C)^2$, the finite-width four-point vertex of Roberts,
Yaida & Hanin. It does not improve an estimator.

# Appendix F. Selection history

**Table F1.** Every non-oracle variant of V29's slice term run on networks 0–2 before the choice (change in final MSE
against V29, %).

⟨TB⟩

# Appendix G. Glossary

| term | meaning |
|------------------------------|------------------------------------------------------------|
| slice $T$, $Q$, $K$ | pair cumulants $\kappa(z_i,z_i,z_j)$, $\kappa(z_i,z_i,z_j,z_j)$, $\kappa(z_i,z_i,z_i,z_j)$ |
| slot | V29's array for the $(3,1)$ slice; it holds $K^\top$ |
| radial part | projection of $K$ on $v_i\Sigma_{ij}$; remainder = $K$ minus it |
| two uses | the covariance update, and the $(2,1)$/$(2,2)$ higher-cumulant updates |
| exact slice, oracle | the object from the $N=10^8$ dumps, injected into a chain |
| placebo | an object of the same kind with the wrong neuron identities |
| bake | an independently seeded $N=10^8$ run of the same networks |
| T-PCP | "transported pairwise cumulant propagation", our float64 research chain |
| $\gamma$, $\lambda$ | skewness $\kappa_3/s^3$; global kurtosis factor $n\sum\kappa_4/(\sum v)^2$ |

# Reproducibility, credits and disclosure

**Code and logs.** Code, raw logs, the pre-registration and a claim → script → log map are at

<https://github.com/al07-alt/whest-p2-k31-two-term>


* All V29 experiments use one test file with environment switches (marked `[EF]`). Every table in §4–§5 is
  regenerated from the raw logs and saved predictions by `make_tables.py`. The pre-registered analysis is
  `sealed_analysis.py`.
* The paired submission contains only the change of §4. It reproduces the test file's numbers exactly.
* **Pre-registration hashes** (sha256). The first is for the original text (the first 54 lines of
  `prereg_sealed.md`, 09:00 UTC); the second is for the text with amendment 1 (09:26 UTC).

```
eca7755479d0162a222dcede8f577c4421d99f0a177a95235f41d4edc2a4f59a
037c6a642da05debc13da54e7fe017c1a05dc0034ceb7a54b8467ac788d32a20
```

**Data.** Keenan Pepper's public datasets `arc-whestbench-p2-higher-moments-2026` and `whest-p2-bakev2-mini-supp`, and
the organisers' public `mini` split.

**Credits.**

* Wu et al. (ARC, 2026), for the framework.
* 504aldo, for V29 (MIT). Our submission is V29 with one marked change.
* Team EscAI, for the eps² law, the oracle studies and "accuracy by cancellation".
* Yaida, and Roberts, Yaida & Hanin, for finite-width theory.
* Price (1958), for the tree expansion.

**LLM disclosure.** The code, the analysis and this text were developed with Claude (Anthropic) under the author's
direction. Reviews of drafts were run as separate Claude instances. They contributed the $\gamma\propto\alpha$ baseline,
the network-level $\lambda$ identity, the $b_2=1$ calculation, the requests for rank, memory and placebo tests, and two
methodology audits. The audits led to the network-level analysis, the pre-registration, the new placebos and
projections, and an independent check of the orientation argument. Every number is produced by a script in the
repository, and untested interpretations are marked as hypotheses.
