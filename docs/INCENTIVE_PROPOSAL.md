<div class="titleblock">
<h1>Fair Incentives for Federated Mental-Health Learning</h1>
<p class="subtitle">A Shapley-Driven, Noise-Robust Incentive Mechanism Extending ShapFed, with a
Tabular Depression-Screening Case Study</p>
<p class="meta"><b>Project:</b> FedPowerShap &rarr; FedShapIn (working title) &nbsp;|&nbsp;
<b>Advisor:</b> Prof. Lekha &nbsp;|&nbsp; <b>Date:</b> July 2026 &nbsp;|&nbsp; <b>Revision 2</b></p>
<p class="meta"><b>Document type:</b> Research proposal (pre-implementation) &nbsp;|&nbsp;
<b>Setting:</b> Cross-silo federated learning on tabular mental-health survey data</p>
<p class="meta"><b>Revision 2 addresses the review comment</b> &ldquo;<i>but this methodology we failed
in our health dataset, right?</i>&rdquo; &mdash; see &sect;5, which states the prior negative result in full,
and &sect;10, which pre-registers falsifiable success criteria.</p>
</div>

<div class="abstract">
<p><b>Abstract.</b> Federated learning (FL) lets multiple institutions train a shared model without
pooling raw data &mdash; a natural fit for privacy-sensitive <b>mental-health screening</b>, where
clinics, universities, and health apps hold small, heterogeneous, and noisily-labelled cohorts. But
FL is only sustainable if participants are <b>fairly rewarded for genuine contribution</b>; otherwise
high-quality data holders drop out and free-riders dilute the model. The state-of-the-art contribution
estimator, <b>ShapFed</b> (Tastan et al., IJCAI 2024), computes class-specific Shapley values and uses
them for weighted aggregation, but it <i>stops at measurement</i>: it defines no reward rule and proves
no economic property (individual rationality, incentive-compatibility, budget balance). We propose
<b>FedShapIn</b>, an incentive layer that converts a <b>noise-robust</b> contribution signal into rewards
with stated economic properties, evaluated on a public <b>tabular depression-screening</b> benchmark
partitioned across simulated institutions with injected free-riders.</p>
<p><b>We state the central risk to this proposal up front.</b> Our own prior study
(&sect;5) tested contribution-aware <i>aggregation</i> on three medical-imaging datasets and found that
under 40% label noise it <b>did not beat the strongest baseline (FedDyn) on any of them</b> &mdash; a
tie on OCTMNIST and losses on ISIC and APTOS. That is a real negative result, and this proposal does not
hide behind it. What the same study <i>did</i> establish is that the underlying quality signal
<b>correctly ranks clients</b>: corrupted clients received 0.69&times; the aggregation weight of clean
clients, reliably and from round 1. FedShapIn depends on the <b>ranking</b> property, not on the accuracy
property that failed. &sect;5 argues this distinction precisely, &sect;10 converts it into
<b>pre-registered, falsifiable</b> success criteria that do not include beating FedDyn on accuracy, and
&sect;10.3 commits us to reporting the aggregation comparison on the new dataset <i>even if it fails
again</i>. <b>No results are reported here &mdash; this is a proposal.</b></p>
</div>

## 1. Introduction and motivation

Mental-health conditions such as depression and anxiety are widespread and under-diagnosed, and machine
learning on screening data (questionnaires, lifestyle, and wearable-derived features) is a promising
triage aid. The data, however, is **siloed and sensitive**: a single university counselling centre, a
clinic, or a wellness-app operator each holds only a few hundred to a few thousand records, and privacy
regulation plus stigma make centralised pooling impractical. Federated learning (FL) is the standard
answer: each site trains locally and shares only model updates. A dedicated survey of FL for
**mental health in education** [11] documents exactly this setting — high-school and college cohorts held
by institutions that cannot legally pool them.

Two properties of the mental-health setting make plain FL insufficient:

1. **Participation is voluntary and costly.** Institutions spend compute, curation effort, and accept
   privacy risk to join. Without a credible, *fair* reward for genuine contribution, the best data
   holders have no reason to stay, and **free-riders** — sites that submit low-effort or fabricated
   updates to harvest the shared model — are not deterred. This is the classic FL *incentive* problem [2,3].
2. **Contribution is heterogeneous and hard to measure honestly.** Cohorts differ in size, class balance,
   and label reliability (self-report scales are noisy). A reward mechanism must therefore rest on a
   contribution estimate that is **robust to noise and to strategic manipulation**, not merely a proxy
   such as dataset size or self-declared quality.

<div class="callout">
<b>Thesis.</b> The FedPowerShap project has a <i>verified</i> per-client quality signal (&sect;5.2) but
could <i>not</i> convert it into an accuracy win over FedDyn (&sect;5.1). The natural next step is
therefore <b>not</b> another attempt at the accuracy claim. It is to apply the signal to the problem it
actually solves &mdash; <b>allocating reward in proportion to contribution</b> &mdash; where correct
ranking is the requirement and accuracy parity is sufficient.
</div>

This reframing is the core intellectual move of the proposal, and it is a *narrowing* of the claim, not a
widening. It is deliberately falsifiable (&sect;10).

## 2. Background

**Federated learning.** A server coordinates *K* clients over communication rounds. Each round, a subset
of clients trains on local data and returns updates; the server aggregates them (classically **FedAvg**,
a size-weighted average). Data is typically **non-IID** across clients, which degrades naive averaging
and motivates robust aggregators (FedProx, SCAFFOLD, FedDyn, MOON, FedBN).

**Shapley value for contribution.** From cooperative game theory, the Shapley value fairly divides the
"value" of a coalition among its members by averaging each member's *marginal contribution* over all
orderings. In FL it is the principled way to answer *"how much did client k help?"*, and it uniquely
satisfies efficiency, symmetry, null-player, and additivity. Its cost (exponential in *K*) drives
approximations — Monte-Carlo permutation sampling, gradient/last-layer proxies (as in ShapFed [1]),
and recent accelerations: **FedOwen** uses Owen sampling to cut estimation cost at fixed budget [8], and
**DPVS-Shapley** targets the same bottleneck [9].

**Incentive-mechanism properties.** A reward/payment mechanism is judged on well-defined economic
criteria, which this proposal adopts as its evaluation axes:

- **Individual Rationality (IR):** every honest participant's utility (reward minus cost) is non-negative
  — nobody is worse off for joining.
- **Incentive Compatibility (IC) / truthfulness:** a participant's best strategy is to behave honestly;
  manipulation does not pay.
- **Budget Balance (BB):** total rewards stay within a fixed budget (weak BB) or exactly exhaust it
  (strong BB).
- **Fairness:** reward is monotone in — and ideally proportional to — true contribution; measurable via
  the reward–contribution correlation and Jain's index on payoffs.
- **Free-rider / Byzantine resistance:** clients contributing little or adversarial updates receive
  little or no reward and are down-weighted in aggregation.

## 3. Related work (literature survey)

**(a) Shapley-based contribution in FL.** ShapFed [1] computes **Class-Specific Shapley Values (CSSV)**
from last-classifier-layer gradients and aggregates via contribution-weighted averaging (ShapFed-WA),
improving class-imbalanced performance and enabling contribution-proportional personalization. A
large-scale experimental evaluation of contribution estimators [10] benchmarks this family systematically.

**(b) The fragility of single-round gradient signals — external corroboration of our negative result.**
Two recent papers independently identify the failure mode we measured (&sect;5.2). **CoAst** [12] states
that existing validation-free methods "assess the contribution based on the parameters and gradients of
local models and the global model in a **single training round**, which is easily compromised by the
stochasticity of model training," and fixes it with cross-round valuation. **SpectralFed / SpectralFuse**
[13] motivates a data-free spectral-entropy signal precisely because existing signals are "susceptible to
manipulation." Our anti-discriminative CSSV finding is therefore not an implementation artefact — it is a
known-and-independently-reported weakness of cosine-of-update scoring, which strengthens rather than
weakens the case for the blended, temporally-smoothed signal in &sect;7.1.

**(c) FL incentive mechanisms.** Surveys [2,3] group mechanisms into **game theory** (Stackelberg, Nash),
**auction theory** (reverse and budget-constrained auctions), and **contract/reputation theory**.
Representative systems: **FIFL** [5] (fair, attack-aware rewards), **FedFAIM** [6] (model-performance-based
fair incentive), **Shapley-UCB** for data-market seller selection [7], and reward-timing work [14].
**FedEach** [4] is notable for computing contributions and selecting reliable clients **without any
server-side validation set**, tolerating a high fraction of malicious clients — directly relevant to our
robustness goal.

<div class="warn">
<b>Closest competitor &mdash; stated honestly.</b> <b>FRIMFL</b> [15] already combines a reverse auction,
a <b>reputation mechanism</b>, and <b>Shapley-based contribution evaluation</b> to reward reliable clients
under a budget. Our mechanism is therefore <i>not</i> the first to join Shapley values, reputation, and
budgeted reward. Three differences remain, and the proposal must stand on these and not on a false
novelty claim:
<ul>
<li><b>Class-specific</b> rather than scalar Shapley contribution (CSSV), which matters when sites differ
in prevalence rather than in size.</li>
<li>A <b>measured, published failure analysis</b> of the contribution signal under partial participation
(&sect;5.2) and a signal designed against that specific failure &mdash; where FRIMFL assumes its Shapley
signal is reliable, we have evidence that the naive version is not.</li>
<li>An <b>explicit adversary model with graded free-rider strength</b> (&sect;9.3) and pre-registered
falsification thresholds (&sect;10), rather than a demonstration on a favourable setting.</li>
</ul>
</div>

**(d) FL for mental health.** The CMES survey [16] documents the privacy motivation and data-silo problem
for computational mental healthcare; the education-focused survey [11] covers the student cohort
specifically. Applied work includes **FedMental** [17] (depression detection from social-media text, which
reports FL within ~2.5 F1 of centralized but a severe differential-privacy trade-off), on-device FL for
depression detection from Reddit posts [18], and privacy-preserving **EEG-based MDD screening** on MODMA
[19]. None of these define or evaluate an incentive mechanism.

<div class="callout">
<b>Gap in the literature.</b> Shapley-based FL work measures contribution but mostly stops short of a
reward rule with economic guarantees; FL incentive work that does close the loop (FRIMFL, FIFL, FedFAIM)
uses coarse or assumed-reliable contribution signals and is not validated under the partial-participation
failure mode of &sect;5.2; and FL-for-mental-health work uses no incentive mechanism at all.
<b>FedShapIn targets that intersection</b> &mdash; a noise-robust, class-specific contribution signal
driving a budget-balanced reward rule, instantiated on mental-health data.
</div>

### 3.1 Positioning table

| Method | Contribution signal | Reward rule | IR / IC / BB | Noise-robust | Mental-health validated |
|:--|:--|:--|:--:|:--:|:--:|
| FedAvg | size only | none | – / – / – | no | no |
| ShapFed [1] | class-specific Shapley (CSSV) | none (aggregation only) | – / – / – | **no** (see §5.2) | no |
| FedCE | gradient alignment | none | – / – / – | partial | no |
| CoAst [12] | cross-round similarity | none | – / – / – | yes | no |
| SpectralFuse [13] | spectral entropy of updates | none | – / – / – | yes | no |
| FIFL [5] / FedFAIM [6] | accuracy / performance | monetary | partial | partial | no |
| **FRIMFL [15]** | **Shapley + reputation** | **auction + reward** | **partial** | **partial** | no |
| FedEach [4] | evaluator score + reputation | penalty/priority | – / partial / – | **yes** | no (HAR) |
| **FedShapIn (ours)** | **CSSV + loss detector + reputation EMA** | **budget-balanced Shapley payoff** | **yes / approx / yes** | **yes (§5.2 fix)** | **yes (this proposal)** |

## 4. Base paper deep-dive: ShapFed

**What it does.** ShapFed treats the rows of each client's last-layer weight update as class-specific
gradient signals. The server computes, per client *k* and class *c*, the cosine similarity between that
client's class-*c* update and a reference (consensus) update — the **Class-Specific Shapley Value**
CSSV<sub>k,c</sub>. Aggregated per client, these become normalized aggregation weights (**ShapFed-WA**),
and can be used to hand each client a **personalized** model scaled to its contribution. Reported wins are
strongest under **class imbalance**, where size-weighted FedAvg under-serves minority classes.

**Why it is the right base paper.** It is already the intellectual anchor of FedPowerShap; extending it
keeps continuity with the group's codebase and prior report, and its per-class contribution scores are
exactly the substrate an incentive layer needs.

<div class="warn">
<b>Critique / limitations we target.</b>
<ul>
<li><b>No incentive mechanism.</b> ShapFed converts contribution into <i>aggregation weights and
personalized models</i>, never into an explicit <b>reward/payment</b>, and proves no economic property
(IR, IC, budget balance). Contribution measurement &ne; incentive mechanism.</li>
<li><b>Fragile signal under realistic conditions.</b> Our measurements (&sect;5.2) found the CSSV cosine
signal <b>anti-discriminative under partial participation and label noise</b>: corrupted clients receive
<i>higher</i> scores. A reward built on raw CSSV is therefore manipulable. Independent work [12,13]
reports the same class of weakness.</li>
<li><b>No mental-health / tabular validation.</b> ShapFed is evaluated on generic image benchmarks;
tiny-cohort, self-report-noisy mental-health data is untested.</li>
</ul>
</div>

## 5. Prior results, including what failed (scope of the negative result)

This section exists because it is the first question any reviewer — or advisor — will ask. It reports the
prior study's outcome without softening, then delimits exactly what it does and does not imply.

**Prior protocol.** Three medical-imaging FL benchmarks (OCTMNIST retinal OCT, ISIC dermoscopy, APTOS-2019
diabetic retinopathy), Dirichlet α=0.3 non-IID, ResNet-18 @ 64px, 10 clients with 5 active per round, 2
local epochs, SGD lr=1e-3 cosine-decayed, 40 rounds, 20k-sample train cap, last-round metric, 3–5 seeds.
Noise model: a fraction ρ of clients have 60% of training labels flipped; test labels stay clean.
Frozen in `scripts/run_noise_sweep.py`; full tables in `results/crossdataset_tables.md`.

### 5.1 What failed: the accuracy claim under noise

| Dataset (40% noisy clients) | FedAvg | FedDyn | Proposed | Δ vs FedDyn |
|:--|:--:|:--:|:--:|:--:|
| OCTMNIST | 0.462 | **0.523** | 0.518 | −0.006 (tie) |
| ISIC | 0.669 | **0.672** | 0.645 | −0.027 (loss) |
| APTOS | 0.577 | **0.649** | 0.588 | −0.061 (loss) |

FedDyn is at least as good on **every** dataset. Fairness (Jain) and worst-client accuracy at 40% also
favour FedDyn (e.g. APTOS worst-client 0.152 vs 0.048). On clean data the proposed method reaches parity
(OCTMNIST 0.708 vs 0.688, n=5; APTOS 0.728 vs 0.715; ISIC 0.719 vs 0.734) — all within seed variance.

The diagnosis was itself a finding: define the **"opening"** as how much 40% noise costs FedDyn. It is
large only on OCTMNIST (0.158) and small on APTOS (0.066) and ISIC (0.062). FedDyn's dynamic
regularization already absorbs *moderate* label noise, so down-weighting mostly discards usable data and
loses. **Conclusion recorded in the prior report: there is no general robustness win from
contribution-aware aggregation in this regime.** We do not re-litigate that here.

### 5.2 What worked: the ranking claim

Two results survive, and both are about *identifying* clients rather than about accuracy.

- **The loss-based detector ranks correctly.** On OCTMNIST at 40% noise (3 seeds), corrupted clients
  received **0.156** mean aggregation weight versus **0.228** for clean clients — a **0.69 ratio**, stable
  from round 1 onward (`figures/paper/mechanism_weights_n40.png`). The signal is
  `w ∝ exp(−β·(loss_k − min_loss))` where `loss_k` is the current global model's loss on client *k*'s own
  data (`federated/server.py::_loss_weights`).
- **A reportable negative result about CSSV.** Scoring clients by cosine similarity of the classifier
  update against the consensus is **anti-discriminative** here: corrupted clients score *higher*
  (noisy/clean weight ratio ≈ 1.1–1.3), even with a coordinate-wise-median reference. Cause: with 5 of 10
  clients active and 40% corrupted, the active set typically contains 2–3 corrupted clients, so the
  consensus direction is itself pulled toward the corrupted updates and the robust-reference assumption
  breaks. This is why the method switched to the loss detector, and why &sect;7.1 does not use raw CSSV.

### 5.3 Why the incentive claim is a different claim

<div class="novelty">
<b>The two claims are logically independent.</b>
<ul>
<li><b>Failed claim (&sect;5.1):</b> re-weighting clients by estimated quality yields a <i>more accurate
global model</i> than FedDyn. This requires that suppressing a noisy client's data helps more than losing
that data hurts &mdash; a net-benefit condition that depends on the baseline's own noise tolerance, and
which does not hold when the baseline already absorbs the noise.</li>
<li><b>Proposed claim (this document):</b> the estimated quality <i>ranks clients correctly</i>, so a
reward proportional to it pays honest contributors and starves free-riders. This requires only
<b>monotonicity of the score in true contribution</b>. &sect;5.2 is direct evidence for it, measured on
the same health data where the accuracy claim failed.</li>
</ul>
A mechanism can allocate reward correctly while the same weights fail to improve accuracy &mdash; because
reward allocation is a <b>ranking/normalization</b> problem, whereas accuracy is a
<b>bias&ndash;variance</b> problem in which discarding noisy-but-informative gradients has a real cost.
</div>

Two further reasons the new setting is more favourable than the failed one — both testable, not assumed:

1. **Free-riders carry no signal; noisy labels do.** A 60%-label-flip client still produces partially
   informative gradients, which is precisely why suppressing it cost accuracy in &sect;5.1. A free-rider
   submitting a near-zero or randomised update contributes no usable information, so denying it weight and
   reward has no offsetting cost. The adversary this proposal targets is strictly easier to exclude.
2. **The metric no longer needs a margin.** &sect;5.1 needed to *beat* a strong baseline on accuracy.
   &sect;10 asks only for **non-inferiority** on accuracy, and moves the burden of proof onto incentive
   metrics for which no competitive baseline exists in this domain.

<div class="warn">
<b>Residual risk we accept.</b> If the score's monotonicity also degrades on tabular data, FedShapIn
fails &mdash; the &sect;5.2 evidence is from one modality (medical images) and one corruption type (label
flips). &sect;10.2 makes this the primary falsification test and &sect;10.3 commits to publishing the
outcome either way.
</div>

## 6. What we already have (reuse from FedPowerShap)

The existing codebase substantially de-risks the engineering, verified against the source:

- **Three contribution/valuation signals:** CSSV (`federated/shapley.py::compute_cssv`, with `mean` /
  `median` / `trimmed` consensus references), the loss-based quality detector plus a persistent
  **reputation EMA** (`federated/server.py::_loss_weights`, `_shapfed_weights`; EMA state `self.cssv_ema`,
  momentum `cssv_ema=0.5`), and FedCE gradient-alignment (`federated/contribution.py::fedce_weights`).
  Per-client weights, scores, and probe losses are logged every round.
- **Nine baselines** as `--method` presets in `config.py::METHOD_PRESETS`: FedAvg, FedProx, SCAFFOLD,
  FedDyn, FedBN, MOON, PoC-FedAvg, FedCE, and `proposed`.
- **An "unreliable client" regime:** Dirichlet non-IID partitioning (`data/dataset.py::dirichlet_partition`)
  and label-noise injection via `--noisy-client-frac` / `--label-noise-rate`, plus the frozen multi-seed
  sweep driver `scripts/run_noise_sweep.py`.
- **Fairness and stability tooling:** Jain's index over per-client accuracies
  (`server.py::evaluate_per_client`), weight capping to prevent aggregation collapse
  (`shapley.py::_cap_weights`, `cssv_max_weight=0.30`), and a table/figure pipeline
  (`scripts/aggregate_results.py`, `make_paper_figures.py`).

<div class="novelty">
<b>Net new work (next phase, after this proposal is approved):</b> (1) a <b>tabular dataset loader</b> and
a small tabular MLP whose final layer is named per <code>CLASSIFIER_LAYER_NAME</code> (currently
<code>"classifier"</code> in <code>models/cnn.py</code>) so CSSV keys on it &mdash; <code>data/dataset.py</code>
today loads only image datasets, so this is a new code path, not a config change; (2) the <b>FedShapIn
incentive/payment module</b> consuming the already-logged per-round scores; (3) a <b>graded free-rider
attack harness</b> (&sect;9.3); (4) incentive-specific metrics and a <b>leave-one-out ground-truth
contribution</b> oracle.
</div>

## 7. Problem statement and research questions

We study cross-silo FL for binary depression screening across *K* institutions, a fraction of which are
**unreliable** (label-noisy or free-riding). The server holds a fixed per-round **reward budget** *B* and
must both (a) train an accurate, fair global model and (b) allocate rewards that reflect true contribution.

- **RQ1 (fidelity — primary):** Does FedShapIn's reward correlate with each client's *true* contribution
  (leave-one-out held-out marginal value) more faithfully than size-proportional, equal-split, and
  raw-CSSV rewards?
- **RQ2 (robustness / IC — primary):** Under graded free-riders and label noise, does FedShapIn deny
  reward to and down-weight bad actors — i.e. is honesty the best response across the attack ladder?
- **RQ3 (non-inferiority — secondary):** Is FedShapIn's screening performance statistically
  **non-inferior** to FedDyn, and how does it trade fairness (Jain on accuracy *and* on payoff) against
  the nine baselines? *Note: this is deliberately posed as non-inferiority, not superiority — see §5.1.*
- **RQ4 (scope of the prior negative result):** Does the &sect;5.1 finding (no aggregation win under
  noise) **replicate on tabular mental-health data**? A negative answer would extend the prior
  characterization to a new modality; a positive answer would bound it to imaging. Either is reportable.

## 8. Proposed method: FedShapIn

**Overview.** FedShapIn adds an incentive layer on top of ShapFed's contribution substrate, in three parts.

**8.1 Noise-robust contribution signal.** Instead of raw CSSV, define a per-client per-round score

<div class="eq">s_k = &alpha;&middot;CSSV_k + &beta;&middot;q_k(loss) + &gamma;&middot;rep_k ,&nbsp;&nbsp; &alpha;+&beta;+&gamma;=1</div>

where q<sub>k</sub>(loss) is the loss-based quality weight — the &sect;5.2 fix for CSSV's
anti-discriminative failure — and rep<sub>k</sub> is the reputation EMA, which rewards *consistent*
contribution and damps single-round manipulation. The EMA term is also the mechanism CoAst [12] identifies
as the cure for single-round instability, so its inclusion is theoretically motivated and not merely
smoothing. All three components already exist in the codebase (&sect;6). The (α,β,γ) weights are ablated
(&sect;9.4); given &sect;5.2 we expect β to dominate and **α may be driven to 0** — an outcome we will
report plainly, since "the class-specific Shapley term contributes nothing" is itself a finding about
ShapFed.

**8.2 Reward rule (budget-balanced Shapley payoff).** Given budget *B* per round and non-negative scores
s<sub>k</sub> over active clients, allocate

<div class="eq">r_k = B &middot; s_k / &Sigma;_j s_j</div>

as the primary rule (**strong budget balance** by construction; **fairness** since reward is monotone in
contribution). Two comparison designs: a **reputation-gated** variant zeroing reward below a reputation
floor (free-rider deterrence), and a **reverse-auction** variant in the style of FRIMFL [15]. Rewards
accumulate across rounds into a per-client payoff used for settlement and for contribution-proportional
personalization (inherited from ShapFed).

**8.3 Claimed properties (to be argued formally + shown empirically).**

- **Budget balance:** exact by the normalization in 8.2.
- **Individual rationality:** honest clients receive reward monotone in genuine contribution; with a cost
  model, IR holds above a minimum contribution threshold (to be characterised).
- **Approximate incentive-compatibility:** because the score blends a manipulation-resistant loss signal
  with a slow reputation term, single-round gaming (gradient scaling, label flipping, update replay)
  *lowers* rather than raises reward. The empirical test is RQ2. We scope the claim to the attack classes
  in &sect;9.3 and do not claim IC in general.
- **Fairness:** high reward–contribution correlation and high Jain payoff index (RQ1, RQ3).

## 9. Dataset and experimental design

### 9.1 Dataset (facts verified, July 2026)

**Modality choice.** A **public tabular mental-health survey** dataset: it plugs into the existing harness
(no image/EEG/audio pipeline), needs no data-use agreement (unlike DAIC-WOZ), and is large enough to
partition across simulated institutions (unlike the ~55-subject MODMA EEG set).

**Primary — Student Depression Dataset** (Kaggle, `adilshamim8` / `hopesb` mirrors) [20]:
**27,901 records**, reducing to ~27,826 after dropping missing values. Features: Gender, Age, City,
Profession, Academic Pressure, Work Pressure, CGPA, Study Satisfaction, Job Satisfaction, Sleep Duration,
Dietary Habits, Degree, *Have you ever had suicidal thoughts?*, Work/Study Hours, Financial Stress,
Family History of Mental Illness. Target: **binary `Depression`**.

<div class="warn">
<b>Two dataset corrections from revision 1</b>, both of which change the experiment design:
<ul>
<li><b>The target is majority-positive, not a minority class.</b> Mean of <code>Depression</code> is
<b>0.586</b> &mdash; 58.6% positive, 41.4% negative. Revision 1 described a "realistic minority class";
that was wrong. Two consequences: (i) 58.6% is far above clinical prevalence (~15&ndash;35%), so this is a
self-selected survey sample and we will say so rather than calling it epidemiological; (ii) ShapFed's
reported advantage is strongest under <b>class imbalance</b>, which a near-balanced global target does not
provide. The imbalance must therefore come from the <b>federated partition</b> &mdash; Dirichlet
&alpha;=0.3 over the label gives per-site prevalence skew, which is the realistic mechanism anyway
(clinics and counselling centres differ in case mix). We will report per-site prevalence explicitly.</li>
<li><b>No PHQ-9 scores exist in this dataset.</b> Revision 1 said labels "follow the standard PHQ-9 /
GAD-7 &ge; 10 threshold." The Kaggle set ships a pre-computed binary label with no item-level
questionnaire scores, so that claim is removed. For a genuinely PHQ-9-scored cross-check we add the
<b>PHQ-9 Student Depression Dataset</b> (Mendeley Data, <code>kkzjk253cy</code>) [21] as a secondary set,
where the &ge;10 threshold <i>is</i> applicable.</li>
</ul>
</div>

<div class="warn">
<b>Task-saturation risk (new, and the most serious design threat).</b> Published work on this dataset
reaches <b>77.7% (Naive Bayes)</b> to <b>88.8% accuracy / 93.4% AUC (GLNet)</b> centrally [22], and SHAP
analysis ranks <b>suicidal ideation as the single most influential feature</b> [22]. Since suicidal
ideation is a core diagnostic symptom rather than an independent predictor, that feature makes the task
close to saturated &mdash; and <b>a saturated task cannot discriminate between aggregators</b>, which
would render RQ3/RQ4 uninformative regardless of the mechanism. Mitigation: run every experiment in
<b>two feature regimes</b> &mdash; (a) <b>full</b> features, and (b) <b>screening</b> features with the
suicidal-ideation item removed &mdash; and treat (b) as the headline regime. We will report the
centralized ceiling in both regimes before running any FL comparison, so the discriminative headroom is
known in advance rather than discovered afterwards.
</div>

**Backups.** `ikynahidwin/depression-student-dataset` (smaller, cleaner cross-check) and
`shariful07/student-mental-health` (Malaysia cohort, cross-population robustness). Licensing for all sets
is Kaggle/Mendeley-public and will be cited explicitly in the final paper.

### 9.2 Federated setup

Partition records across *K* = 10 simulated institutions with **Dirichlet non-IID** sampling over the
label (α = 0.3, matching the frozen protocol), reusing `dirichlet_partition`. Model: a small tabular MLP
(2–3 hidden layers) with its final linear layer named `classifier` so CSSV reads it unchanged. Protocol
adapted from the frozen `PROTOCOL` in `run_noise_sweep.py`: 10 clients, 5 active/round, ~40 rounds,
**5 seeds** (up from 3 — the prior study's ±0.13 variance on APTOS showed 3 is too few for a
non-inferiority test).

### 9.3 Adversary model: graded free-riders

Revision 1 specified a single free-rider mode, which risks a trivially easy result. We instead define an
**attack ladder** of increasing subtlety, and report detection and reward capture at each rung:

| Rung | Free-rider behaviour | Expected difficulty |
|:--|:--|:--|
| F0 | Honest (control) | — |
| F1 | Random / Gaussian-noise update | trivial to detect |
| F2 | Zero-effort: returns the global model unchanged | easy (no update signal) |
| F3 | Trains on a small random subset (10% of local data) | moderate |
| F4 | Trains honestly then scales the update by a factor *c* > 1 to inflate apparent contribution | hard — targets the cosine term |
| F5 | Label-noise client (60% flips), i.e. the &sect;5.1 adversary | hardest; partially informative |

Sweep free-rider fraction ∈ {0%, 20%, 40%} independently of label-noise fraction ∈ {0%, 20%, 40%}.
Reporting F1–F5 separately is what makes RQ2 a real test: a mechanism that only defeats F1–F2 is not
robust, and we will say so if that is what we find.

### 9.4 Baselines, metrics, ablations

**Baselines.** The nine existing presets, plus three incentive controls — **no-incentive**,
**equal-split**, **size-proportional** — plus **raw-CSSV reward** (the ablation that isolates §5.2), all
versus **FedShapIn** and its reputation-gated / reverse-auction variants.

**Metrics.**

- *Model:* balanced accuracy and AUC (headline, given the 58.6% base rate — plain accuracy is misleading
  here), per-class accuracy, and a majority-class-only reference (0.586) as a sanity floor.
- *Fairness:* Jain's index on per-client accuracy **and** on cumulative payoff.
- *Incentive fidelity (RQ1):* Spearman correlation between cumulative reward and a ground-truth
  contribution, computed by **leave-one-out retraining** on held-out data. Feasible here: *K*=10 and a
  small MLP make 10 LOO retrains cheap, unlike the ResNet setting.
- *Robustness (RQ2):* per-rung free-rider **detection AUC**, **share of cumulative budget captured** by
  free-riders (target ≈ 0), and global-accuracy degradation vs clean.
- *Economic:* budget-balance residual (should be exactly 0), and per-client utility under a stated cost
  model for IR.

**Ablations.** Signal composition (α,β,γ: CSSV-only vs +loss vs +reputation); reward rule (proportional
vs reputation-gated vs auction); budget size *B*; reputation floor; EMA momentum; feature regime
(full vs screening).

## 10. Pre-registered success criteria and falsification

<div class="callout">
<b>Why this section exists.</b> Revision 1's expected outcome was "competitive/better global accuracy" plus
higher reward correlation. The accuracy half re-opened the comparison that &sect;5.1 already lost. These
criteria are fixed <b>before</b> any run, and the accuracy criterion is non-inferiority only.
</div>

### 10.1 Primary criteria (the proposal succeeds only if all hold)

| # | Criterion | Threshold | Maps to |
|:--|:--|:--|:--:|
| C1 | Reward–contribution fidelity | Spearman ρ(reward, LOO value) exceeds **each** of size-proportional, equal-split, and raw-CSSV by **≥ 0.15**, with non-overlapping 95% bootstrap CIs over 5 seeds | RQ1 |
| C2 | Free-rider starvation | Free-riders capture **≤ 5%** of cumulative budget at a 40% free-rider fraction (equal-split captures ~40%), for rungs **F1–F4** | RQ2 |
| C3 | Detection quality | Free-rider detection **AUC ≥ 0.85** on rungs F1–F4 | RQ2 |
| C4 | Accuracy non-inferiority | FedShapIn balanced accuracy ≥ FedDyn − **0.01**, one-sided TOST at α=0.05 over 5 seeds | RQ3 |
| C5 | Budget balance | Residual \|Σr_k − B\| = 0 exactly, every round | §8.3 |

### 10.2 Falsification conditions (the proposal fails, and we report it)

- **C1 fails** → the blended score does not rank clients better than dataset size. This is the sharpest
  test, because it directly probes whether the &sect;5.2 monotonicity transfers to tabular data. If it
  fails, the honest conclusion is that the FedPowerShap contribution signal is **modality-specific**, and
  the paper becomes a negative result extending &sect;5.1 rather than an incentive mechanism.
- **C2/C3 fail on F1–F2** (the easy rungs) → the mechanism is not viable at all.
- **C2/C3 fail only on F4** (update scaling) → report as a **characterised limitation** with the specific
  attack that defeats it; a mechanism robust to F1–F3 but not F4 is still publishable if stated as such.
- **C4 fails** → down-weighting costs more accuracy than the incentive layer justifies; report the
  fairness–accuracy trade-off curve and recommend the reputation-gated variant (which touches reward only,
  not aggregation weights) as the deployable configuration.

### 10.3 Commitment on the replication of the prior negative result

We will run the **FedDyn vs proposed-aggregation** comparison on the tabular data (RQ4) and report it
**whether or not down-weighting wins**. If &sect;5.1 replicates — no aggregation win on a fourth,
non-imaging dataset — that strengthens the prior characterization and will be reported as such in the same
paper. The incentive results (C1–C3, C5) stand or fall independently of it, because they do not depend on
the aggregation weights improving accuracy.

## 11. Timeline, deliverables, and risks

**Indicative timeline (post-approval).** Weeks 1–2: tabular loader + MLP + EDA, including the centralized
ceiling in both feature regimes (&sect;9.1). Weeks 3–4: FedShapIn incentive module + graded free-rider
harness + LOO oracle + metrics. Weeks 5–7: full sweep (baselines + variants + ablations, 5 seeds).
Weeks 8–9: analysis, figures, paper draft. **Deliverable:** a conference/journal paper with reproducible
code, extending the existing FedPowerShap report.

**Risks & mitigations.**

| Risk | Severity | Mitigation |
|:--|:--:|:--|
| The §5.2 signal does not transfer to tabular data (the core risk) | **High** | Made the primary falsification test (C1, §10.2); measured early in Weeks 3–4 so the project can pivot to the negative-result paper before the full sweep is spent |
| Task saturation makes all methods tie | **High** | Two feature regimes; centralized ceiling measured *before* the FL sweep; balanced accuracy/AUC as headline (§9.1) |
| Overlap with FRIMFL [15] weakens novelty | Medium | Stated openly in §3(c); differentiate on class-specific signal, the measured failure analysis, and the graded adversary — and implement the FRIMFL-style auction as a **baseline** rather than ignoring it |
| Ground-truth contribution (LOO) is expensive | Low | *K*=10 with a small MLP makes 10 retrains cheap; Owen/Monte-Carlo sampling [8] if *K* grows |
| Formal IC is hard to prove in general | Medium | Scope the theoretical claim to the F1–F5 attack classes; support empirically (RQ2); position general guarantees as future work |
| Self-selected, non-clinical sample (58.6% prevalence) | Medium | State explicitly; cross-check on the PHQ-9-scored Mendeley set [21] and the Malaysia cohort |

## 12. References

1. Tastan, N., et al. *Redefining Contributions: Shapley-Driven Federated Learning* (ShapFed). **IJCAI 2024**, pp. 5009–5017. arXiv:2406.00569. Code: github.com/tnurbek/shapfed.
2. Zhan, Y., et al. *A Comprehensive Survey of Incentive Mechanism for Federated Learning.* arXiv:2106.15406.
3. *Incentive Mechanisms for Federated Learning: From Economic and Game-Theoretic Perspective.* arXiv:2111.11850.
4. Lim, H.W., Tanjung, S.Y., Iwan, I., Yahya, B.N., Lee, S.-L. *FedEach: Federated Learning with Evaluator-Based Incentive Mechanism for Human Activity Recognition.* **Sensors** 25(12):3687, 2025. doi:10.3390/s25123687.
5. *FIFL: A Fair Incentive Mechanism for Federated Learning.* **ICPP 2021**. doi:10.1145/3472456.3472469.
6. *FedFAIM: A Model Performance-Based Fair Incentive Mechanism for Federated Learning.* IEEE Trans., 2022.
7. Chen, K., Xu, Z. *Federated Learning for Data Market: Shapley-UCB for Seller Selection and Incentives.* arXiv:2410.09107.
8. KhademSohi, H., Hemmati, H., Zhou, J., Drew, S. *Owen Sampling Accelerates Contribution Estimation in Federated Learning* (FedOwen). **ECAI 2025**. arXiv:2508.21261.
9. *DPVS-Shapley: Faster and Universal Contribution Evaluation Component in Federated Learning.* arXiv:2410.15093.
10. *Contributions Estimation in Federated Learning: A Comprehensive Experimental Evaluation.* **PVLDB**. doi:10.14778/3659437.3659459.
11. Ebrahimi, M., et al. *The Transition from Centralized Machine Learning to Federated Learning for Mental Health in Education: A Survey of Current Methods and Future Directions.* arXiv:2501.11714 (abridged version at the **AAAI-2025** AI for Education workshop).
12. *CoAst: Validation-Free Contribution Assessment for Federated Learning based on Cross-Round Valuation.* **ACM MM 2024**. arXiv:2409.02495.
13. Ukaye, A., et al. *Data-Free Contribution Estimation in Federated Learning using Gradient von Neumann Entropy* (SpectralFed / SpectralFuse). **CVPR 2026 Workshops (FedVision)**. arXiv:2604.22562.
14. *Right Reward Right Time for Federated Learning.* arXiv:2503.07869.
15. *FRIMFL: A Fair and Reliable Incentive Mechanism in Federated Learning.* **Electronics** 12(15):3259, 2023.
16. *A Comprehensive Survey on Federated Learning Applications in Computational Mental Healthcare.* **CMES** 142(1):49–90, 2025.
17. *FedMental: Evaluating Federated Learning for Mental Health Detection from Social Media Data.* **ACL 2026**. arXiv:2605.18936.
18. *On-device Federated Learning in Smartphones for Detecting Depression from Reddit Posts.* arXiv:2410.13709.
19. Gupta, et al. *Privacy-preserving EEG-based Major Depressive Disorder screening via Federated Learning (MODMA).* 2024.
20. Student Depression Dataset. Kaggle: `adilshamim8/student-depression-dataset` (mirror: `hopesb/student-depression-dataset`); 27,901 records, binary `Depression` target, mean 0.586. Backups: `ikynahidwin/depression-student-dataset`; `shariful07/student-mental-health`.
21. PHQ-9 Student Depression Dataset. **Mendeley Data**, `kkzjk253cy`.
22. *Explaining factors influencing students' depression with a deep learning approach* (GLNet; 88.84% accuracy, 93.36% AUC; SHAP ranks suicidal ideation first). **Frontiers in Psychology** 16:1684274, 2025. See also *Predicting Student Depression Using Naive Bayes* (77.66% accuracy), JENTIK, 2025.
23. Cho, Y.J., et al. *Client Selection in Federated Learning: Power-of-Choice.* 2020.
24. FedPowerShap internal reports: *When Does Contribution-Aware Aggregation Help?* (`docs/FedPowerShap_Report.pdf`), `docs/SYSTEM_OVERVIEW.md` §7 (the §5.1/§5.2 results), `docs/ROBUST_SHAPFED_RESULTS.md` §2.2 (the CSSV negative result), `results/crossdataset_tables.md` (full tables).
