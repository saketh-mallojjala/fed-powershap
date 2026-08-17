# DCFI-UPA — proposed methodology and dataset coverage

Companion to [`DCFI_TECHNICAL_REPORT.md`](DCFI_TECHNICAL_REPORT.md). The diagrams
below render natively on GitHub.

---

## 1. The proposed system: one communication round

The design principle is a **strict one-way boundary**. The reliability signal may
decide *who gets paid*; it may not decide *whose update enters the model*. That
separation is what makes the incentive layer free of accuracy cost — it was
measured, not assumed (see §4 and the technical report §9.3).

```mermaid
flowchart TB
    subgraph SEL["① SELECTION"]
        direction TB
        S1["Score every non-quarantined client<br/>utility + reliability + under-participation bonus"]
        S2["Sample m of N <b>proportional to score</b><br/>(not top-m: top-m locks out mid-ranked clients)"]
        S1 --> S2
    end

    subgraph TRAIN["② TRAINING CHANNEL — optimises accuracy"]
        direction TB
        T1["Each selected client trains locally with FedDyn"]
        T2["Aggregate with FedDyn using <b>UNIFORM weights</b><br/>every selected update enters untouched"]
        T3["Updated global model w<sub>t+1</sub>"]
        T1 --> T2 --> T3
    end

    subgraph RISK["③ RISK ASSESSMENT — three independent views"]
        direction LR
        R1["<b>Probe loss</b><br/>L(w<sub>t</sub>; D<sub>k</sub>)<br/><i>data quality</i><br/>blind to effort"]
        R2["<b>Effort density</b><br/>‖Δ<sub>k</sub>‖ / n<sub>k</sub><br/><i>did they work?</i><br/>blind to small inflation"]
        R3["<b>Temporal self-similarity</b><br/>cos(Δ<sub>k</sub><sup>t</sup>, Δ<sub>k</sub><sup>t−1</sup>)<br/><i>replay / staleness</i>"]
    end

    subgraph REWARD["④ REWARD CHANNEL — off the training path"]
        direction TB
        W1["Approximate leave-one-out on a <b>held-out validation split</b><br/>utility = drop in balanced accuracy without client k<br/>fairness = drop in minority-class recall without client k"]
        W2["Eligibility from risk tier<br/>low → pay full · medium → pay half · high → pay nothing"]
        W3["r<sub>k</sub> = B · [ w<sub>base</sub>/|S| + w<sub>util</sub>·utility + w<sub>fair</sub>·fairness ]"]
        W4["Cap 40% per client · hold back 20%<br/>released once reliability recovers"]
        W1 --> W2 --> W3 --> W4
    end

    REL["⑤ Update reliability (asymmetric EMA)<br/>falls fast, recovers slowly<br/>repeated high risk → quarantine from SELECTION"]

    SEL --> TRAIN
    TRAIN -.->|"client updates Δ<sub>k</sub>"| RISK
    RISK --> REWARD
    TRAIN -.->|"aggregate model"| W1
    RISK --> REL
    REL -.->|"next round"| SEL
    REWARD --> PAY(["Payment to client k"])

    NOTE{{"⛔ NO PATH from reward back into the model.<br/>Paying a client less can never degrade accuracy."}}
    REWARD -.-x NOTE
    NOTE -.-x TRAIN

    classDef train fill:#e9f6ee,stroke:#0a7d3f,stroke-width:2px
    classDef rew fill:#eef4f9,stroke:#0b3d63,stroke-width:2px
    classDef risk fill:#fdf6e3,stroke:#b8860b,stroke-width:2px
    classDef note fill:#fdf0ee,stroke:#b3261e,stroke-width:2px,stroke-dasharray:4 3
    class TRAIN,T1,T2,T3 train
    class REWARD,W1,W2,W3,W4,PAY rew
    class RISK,R1,R2,R3 risk
    class NOTE note
```

### Why the boundary is one-way

Our earlier method let the contribution signal set **aggregation weights**. A
controlled ablation showed the better that signal worked, the *worse* the model
became: the variant that detected nothing matched FedDyn (0.695 vs 0.697), and
the variant that detected well lost 0.030. A label-noise client still holds
usable gradient signal, and FedDyn's dynamic regularisation already bounds its
damage — so down-weighting discards information for protection already present.

Measured effect of moving the gate (5 seeds, balanced accuracy):

| Scenario | FedDyn | Gate on **aggregation** | Gate on **reward only** |
|:--|:--:|:--:|:--:|
| clean | 0.768 | 0.735 | **0.737** |
| zero_update | 0.768 | 0.647 | **0.752** |
| copied_update | 0.777 | 0.643 | **0.762** |
| random_update | 0.765 | 0.637 | **0.750** |

Adversary reward capture under reward-only gating stays at 0.00% / 1.97% / 6.80%
against a 40% fair share.

---

## 2. Threat model

```mermaid
flowchart LR
    A["Adversary<br/>types"] --> B["<b>Structurally impossible</b><br/>updates an honest client<br/>could never produce"]
    A --> C["<b>Inside the honest distribution</b><br/>performs some genuine training"]

    B --> B1["zero_update — 0.8%"]
    B --> B2["copied_update — 3.3%"]
    B --> B3["scaled_global — 6.3%"]
    B --> B4["random_update — 9.1%"]

    C --> C1["stale_model — 31.8%"]
    C --> C2["low_effort — 33.5%"]
    C --> C3["norm_inflation — 37.1%"]
    C --> C4["label_flip — 37.4%"]

    B1 & B2 & B3 & B4 --> D(["✅ <b>In scope</b> — starved<br/>0.8–9.1% of budget<br/>vs 40% fair share"])
    C1 & C2 & C3 & C4 --> E(["⚠️ <b>Out of scope</b> — not starved<br/>near their full 40% share<br/>reported as limitations"])

    classDef ok fill:#e9f6ee,stroke:#0a7d3f,stroke-width:2px
    classDef bad fill:#fdf0ee,stroke:#b3261e,stroke-width:2px
    class B,B1,B2,B3,B4,D ok
    class C,C1,C2,C3,C4,E bad
```

The split is not arbitrary and it is not a convenience. An adversary that does
*some* real training produces an update lying inside the natural spread of
honest behaviour, and §7.6 of the technical report quantifies why that is
unfixable by magnitude scoring: honest effort density has log-sd 1.014, so a 3×
inflation sits 1.08 sd inside normal variation. Separating at 2 sd would demand
inflation beyond 7.6×, which no attacker is obliged to use.

---

## 3. Experimental methodology

```mermaid
flowchart TB
    D1["<b>Step 0 — Centralized ceiling</b><br/>Pooled-data training, both feature regimes.<br/>Purpose: confirm discriminative headroom EXISTS<br/>before any federated comparison is run."] --> D2

    D2{"Is the task<br/>saturated?"} -->|"yes — 0.96 acc / 0.996 AUC<br/>with suicidal-ideation feature"| D3["Drop that feature.<br/>A saturated task cannot<br/>discriminate between aggregators."]
    D2 -->|"no — 0.794 acc / 0.862 AUC"| D4
    D3 --> D4

    D4["<b>Step 1 — Frozen protocol</b><br/>10 clients · 5 active/round · 2 local epochs · 40 rounds<br/>SGD lr=0.005 cosine · Dirichlet α=0.3 · last-round metric<br/>LR calibrated on FedAvg ONLY, then frozen for every method"] --> D5

    D5["<b>Step 2 — Baselines</b><br/>FedAvg · FedProx · FedDyn · FedCE · SCAFFOLD<br/>FedBN · MOON · PoC-FedAvg"] --> D6
    D6["<b>Step 3 — Proposed system</b><br/>DCFI-UPA, gate on reward only"] --> D7
    D7["<b>Step 4 — Threat model</b><br/>8 adversaries × adversary fraction"] --> D8
    D8["<b>Step 5 — Ground truth</b><br/>Leave-one-out under FedAvg (mechanism-neutral,<br/>so truth is not defined by the rule under test)"] --> D9
    D9["<b>Step 6 — Metrics (15, three families)</b>"] --> D10
    D10["<b>Step 7 — Ablations + parameter sweeps</b>"]

    classDef step fill:#eef4f9,stroke:#0b3d63,stroke-width:2px
    classDef gate fill:#fdf6e3,stroke:#b8860b,stroke-width:2px
    class D1,D4,D5,D6,D7,D8,D9,D10 step
    class D2,D3 gate
```

### The fifteen metrics

| Performance | Reward fairness | Fairness |
|:--|:--|:--|
| balanced accuracy | Spearman ρ(reward, LOO) | Jain — useful-client rewards |
| macro-F1 | reward–contribution MAE | worst-client balanced accuracy |
| AUC | reward share, positive vs negative contributors | selection-frequency Jain |
| convergence rounds | reward-to-cost ratio | ρ(reward, client size) |
| collapsed seeds | individual-rationality rate | minority-class recall |

---

## 4. Datasets tried

### Used in reported results

| Dataset | Modality | Task | Runs | Where it appears |
|:--|:--|:--|:--:|:--|
| **Student Depression** (Kaggle `hopesb`, 27,901 rows) | tabular survey | binary depression screening, 58.6% positive | **869** | **Primary — all current DCFI-UPA results** |
| **OCTMNIST** | retinal OCT | 4-class | 38 | Earlier robustness study; the CSSV negative result |
| **ISIC / DermaMNIST** | dermoscopy | 7-class | 20 | Cross-dataset generalisation |
| **APTOS-2019** | fundus photography | 5-class ordinal (QWK) | 20 | Cross-dataset generalisation |
| **Depression Student** (Kaggle `ikynahidwin`, 502 rows) | tabular survey | binary | ceiling only | Saturation check — showed 0.96 acc / 0.996 AUC with the suicidal-ideation feature, which is why the screening regime exists |

**Coverage: 4 datasets across 2 modalities with full federated results**, plus one
used as a saturation control. The core negative result (contribution-weighted
aggregation does not beat FedDyn) replicates on all four.

### Present in the codebase, not used for results

`chestxray14` (2 exploratory runs only), `cifar10`, `mnist`, `fmnist`, `brats` —
loaders exist and work, but none are part of any reported study.

### Why the tabular mental-health dataset is primary

- Matches the incentive setting: small institutional cohorts, self-reported and
  therefore noisy labels, genuine privacy motivation.
- Cheap enough to run 864 federations — the leave-one-out ground truth alone
  needs 390, which is infeasible on ResNet-scale imaging.
- Non-saturated in the screening regime, so aggregators are actually
  distinguishable (verified in Step 0 before any comparison was run).

### Honest limitation

All current DCFI-UPA results are on **one dataset**. The earlier negative result
is 4-dataset; the incentive mechanism is not yet. Porting DCFI to OCTMNIST or
APTOS is the clearest way to strengthen external validity, and the code path is
dataset-agnostic — it needs runs, not new implementation.
