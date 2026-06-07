# Molecular SSINet: Applying Self-Supervised Discovering of Interpretable Features to Molecule Generation

**Reinforcement Learning - A.A. 2025/26 · Pimpinelli Francesco (2214340), Sapienza Università di Roma**

This project tackles the black-box problem of reinforcement-learning–driven *de novo*
molecule generation. It takes the **MoleGuLAR** generation pipeline (Goel et al.) and
augments it with an interpretability module adapted from **SSINet** (Shi et al.), so that
on top of *generating* drug-like molecules the system also *explains* which atoms of a
generated molecule the policy attends to most. The original SSINet was designed for
image-based RL; the contribution here, **Molecular SSINet**, ports that idea from the
2D image domain to molecular graphs / SMILES.

---

## 1. Motivation

RL agents that design molecules optimise a reward, but give no insight into *why* a given
structure was produced. For a chemist this is a serious limitation: the generated SMILES
is accepted or rejected on faith. The goal of this work is a tool that highlights the most
important substructures of a generated molecule, the atoms the policy most relies on,
turning an opaque generator into something a non-expert can read at a glance.

The approach is inspired by two papers (in `papers/`):

- **MoleGuLAR** - Goel, Raghunathan, Laghuvarapu & Priyakumar, *Molecule Generation Using
  Reinforcement Learning with Alternating Rewards*.
- **SSINet** - Shi et al., *Self-Supervised Discovering of Interpretable Features for
  Reinforcement Learning*, IEEE TPAMI (2020).

---

## 2. Method

### 2.1 MoleGuLAR generation pipeline

| Stage | Component |
|-------|-----------|
| Molecule representation | **SMILES** strings, tokenised character-by-character |
| Generator (policy) | **Stack-Augmented RNN** (`StackAugmentedRNN`) |
| RL algorithm | **REINFORCE** policy gradient (`Reinforcement`) |
| Binding-affinity (BA) predictor | **GIN** - Graph Isomorphism Network (`GINPredictor`) over pretrained `gin_supervised_infomax` embeddings |
| Target protein | **4BTK** (TTBK1) |
| Multi-objective properties | Binding Affinity, **LogP**, **QED**, **TPSA** |

The generator is pre-trained on a SMILES corpus, then biased by policy gradient toward
molecules with desired properties. Each property contributes a shaped reward; four reward
shapes are available (`exponential`, `linear`, `logarithmic`, `squared`), with the
exponential shape giving the strongest push toward low binding energy.

Multiple objectives are combined in one of two ways:

- **Weighted Sum** - all property rewards added every step.
- **Alternating Rewards** - the active objective is cycled every *N* iterations
  (`SWITCH_MODE`, `SWITCH_FREQUENCY`), the strategy that gives MoleGuLAR its name.

### 2.2 SSINet interpretability

SSINet learns, *self-supervised*, a mask that selects the features a trained policy
actually uses. It is a two-stage scheme: **(1)** train the actor to convergence so it
becomes an expert policy that generates state–action pairs; **(2)** freeze that policy and
train a lightweight **mask network** on those pairs, balancing two core properties:
*behaviour preservation* (the masked input must reproduce the policy's decisions, via an
InfoNCE objective) and *sparsity* (the mask keeps as few features as possible). The output
is an importance map over the input.

### 2.3 Molecular SSINet - the novelty

The adaptation moves SSINet from image pixels to **molecular graphs**:

- **`AtomMaskMLP`** - a per-atom importance scorer operating on the frozen GIN node
  embeddings; only this small head carries gradients.
- **`MolecularSSINet`** - the graph-domain analog of SSINet's masked-encoder / InfoNCE +
  sparsity training. The GIN predictor is frozen and shared; the mask co-evolves with the
  RL molecule distribution through a **warm-start** phase on pre-training SMILES followed
  by **concurrent updates** (`update_mask`) on each RL iteration's fresh molecules.
- **Output** - per-atom importances (per-molecule min-max normalised) rendered as 2D
  **atom-importance heatmaps** on the molecular structure, plus a **batch interpretation**
  that aggregates importance across the top-N generated molecules to surface recurring
  pharmacophoric features rather than single-molecule quirks.

---

## 3. Repository layout

```
.
├── MoleGuLAR_SSINet.ipynb   Main self-contained notebook (Colab/Kaggle): GIN + Molecular SSINet
├── MoleGuLAR/               Upstream Goel et al. code & assets
│   ├── Optimizer/           Predictors, target PDBs (4BTK, 6LU7), reward & switch scripts
│   ├── models/              gin / docking model definitions
│   ├── Analysis/            original analysis notebook
│   ├── environment.yml      conda environment (reference)
│   └── Dockerfile
├── outputs/                 Experiment runs (see §5)
│   ├── weighted_sum/
│   ├── alternate_rewards/ , alternate_rewards_15/ , alternate_rewards_35/
│   └── alternate_rewards_new_mask/
├── analyses/                "Chemist-in-the-loop" evaluation reports (.docx)
├── papers/                  Source papers + project proposal
├── presentation/            Presentation.pptx + Architecture.PNG
├── keys.env                 API secrets - git-ignored (see §4)
└── README.md
```

Each run folder under `outputs/` contains `checkpoints/` (model weights, training
metrics), `images/` (SSINet atom-importance + batch top-5 heatmaps), `plots/` (training
curves, property distributions, violin / element-frequency plots), `new_molecules/`
(generated SMILES CSV) and a `metadata.txt` snapshot of the run configuration.

---

## 4. Running it

The whole pipeline lives in **`MoleGuLAR_SSINet.ipynb`** and is designed to run
top-to-bottom on **Google Colab or Kaggle** (GPU runtime). The notebook self-installs its
stack (PyTorch 2.3, DGL, DGL-LifeSci, RDKit, scikit-learn, W&B), detects the platform,
mounts storage and clones the upstream repo for data / predictor files.

1. Open the notebook on a GPU runtime.
2. Adjust the **Configuration** cell. Key knobs:

   | Setting | Default | Meaning |
   |---------|---------|---------|
   | `PREDICTOR` | `"gin"` | binding-affinity predictor (this notebook supports GIN only) |
   | `PROTEIN` | `"4BTK"` | target protein |
   | `REWARD_FUNCTION` | `"exponential"` | reward shape |
   | `NUM_ITERATIONS` / `N_POLICY` | `175` / `15` | RL iterations / policy steps each |
   | `USE_LOGP` / `USE_QED` / `USE_TPSA` | `True` | active property objectives |
   | `SWITCH_MODE` / `SWITCH_FREQUENCY` | `True` / `15` | alternating-reward cycling |
   | `USE_INTERPRET` | `True` | enable the Molecular SSINet interpretability block |
   | `USE_WANDB` / `PUSH_TO_GITHUB` | `True` | optional logging / checkpoint push |

3. Run all cells. Training takes **~3 h** in the reference configuration and writes
   checkpoints, plots, generated molecules and SSINet heatmaps under `outputs/`.

---

## 5. Experiments & results

Two aggregation strategies were studied, plus alternating-reward variants differing in
switch frequency:

- **`weighted_sum/`** - baseline multi-objective sum.
- **`alternate_rewards_35/` (AR35)** and **`alternate_rewards_15/` (AR15)** - alternating
  rewards switching every 35 vs 15 iterations.

Representative generated compounds include para-phenylene-sulfone oligomers; the
alternating-reward setting let the model cap a chain with a **primary sulfonamide group**,
the moiety responsible for the hinge bond that inhibits the target. All chemical
assessments were performed with a **"Chemist-in-the-Loop"** (Claude Opus prompted as a
medicinal chemist); the resulting evaluations live in `analyses/`.

**Takeaways (pros / cons).** The mask reached a good level of generalisation and produced
visual interpretations legible to non-experts, faithfully reproducing SSINet's two core
properties in the molecular domain, at a modest ~3 h training cost. On the cons side, mask
quality is **training-dependent**, and because importance is a **per-molecule min-max
normalisation** every molecule is forced to have "important" atoms, which can overstate
salience on otherwise flat structures.

---

## 6. Future work

- Validate binding-affinity rewards against an **AutoDock-GPU** baseline.
- Extend the importance visualisation to **3D** molecular conformers.
- Improve mask reliability by **relaxing the loss constraints** appropriately.
- Move the interpretation module **into the per-token SMILES generation** itself.

---

## 7. References

1. M. Goel, S. Raghunathan, S. Laghuvarapu, U. D. Priyakumar. *MoleGuLAR: Molecule
   Generation Using Reinforcement Learning with Alternating Rewards.* (upstream:
   <https://github.com/devalab/MoleGuLAR>)
2. W. Shi et al. *Self-Supervised Discovering of Interpretable Features for Reinforcement
   Learning.* IEEE TPAMI, 2020.
