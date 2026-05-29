# Toy SAE-as-VAE Lab

This is a small, fully-local (CPU, seconds–minutes) ground-truth-rigorous lab for
the **SAE ≈ amortized MAP / VAE ≈ amortized VI** thesis. The experiment roadmap
lives in [`sae_vae_experiments.md`](../../../sae_vae_experiments.md); this README
is the primary planning document for the toy track. A toy "LLM" (identical to the
ARENA / Anthropic superposition paper) compresses five ground-truth features into a
2-D bottleneck, creating superposition. Four inference families — SAE (MAP), Gaussian
VAE, Laplace VAE, and spike-and-slab VAE — are then trained to recover the original
features from those 2-D activations. Because the ground truth is fully known, every
"VAE beats SAE" claim (calibration, ambiguity, OOD, prior sensitivity, amortization
gap) can be verified rigorously — something that is impossible on GPT-2 activations.
The GPT-2 activation track has been parked under `experiments/old_code/` (see
[Parked work](#parked-work)).


## Why a toy lab

- **Ground truth is known.** The generating `W` matrix is accessible; calibration
  and ambiguity claims can be checked against actual feature presence, not
  heuristic proxies.
- **CPU-friendly.** Full sweeps across all four families complete in seconds to
  minutes on a laptop; no GPU or Colab session required.
- **Identical model contract.** The toy models share the same instance-dimension
  trick, decoder-column normalisation, and evaluation API as the parked GPT-2 side,
  so findings transfer cleanly.
- **ARENA / Anthropic superposition geometry.** Five features compressed to a 2-D
  hidden space force genuine superposition, making the recovery problem non-trivial.
- **Cheap ablations.** `n_inst` parallelism lets a full sparsity/prior sweep train
  in a single forward pass on CPU.


## Layout

```
experiments/
├── saevae/toy/                        # the toy subpackage (import root)
│   ├── __init__.py                    # re-exports the public API
│   ├── models.py                      # ToyModel, ToySAE, GaussianVAE,
│   │                                  #   LaplaceVAE, SpikeSlabVAE
│   ├── train.py                       # train_toy_model, train_sae, train_vae
│   ├── metrics.py                     # align_features (sign-aware), evaluate
│   └── experiments/
│       ├── __init__.py
│       ├── pareto.py                  # MSE vs L0 across all 4 families
│       ├── calibration.py             # detection AUROC + calibration curves
│       ├── ambiguity.py               # interpolation entropy (commit vs spread)
│       ├── ood.py                     # ELBO/KL OOD detection
│       ├── posterior_entropy.py       # prior sensitivity (π sweep)
│       └── amortization_gap.py        # encoder vs ISTA optimum
│
├── figures/toy/                       # generated plots (PDF + PNG)
│   ├── pareto.pdf / .png
│   ├── calibration.pdf / .png
│   ├── ambiguity.pdf / .png
│   ├── ood.pdf / .png
│   ├── posterior_entropy_vs_prior.pdf / .png
│   └── amortization_gap.pdf / .png
│
└── reports/toy/                       # numerical results (CSV)
    ├── pareto.csv
    ├── calibration.csv
    ├── ood_auroc.csv
    ├── posterior_entropy.csv
    └── amortization_gap.csv
```

Note: `ambiguity.py` writes only figures (no CSV); the midpoint entropy values are
printed to stdout and visible in the figures.


## Quick start

```bash
pip install torch einops numpy matplotlib scipy tqdm

cd /Users/dhawaldixit/projects/saes-as-vaes/experiments
python -m saevae.toy.experiments.pareto              # Pareto MSE vs L0 (all 4 families)
python -m saevae.toy.experiments.calibration         # detection AUROC
python -m saevae.toy.experiments.ambiguity           # interpolation entropy
python -m saevae.toy.experiments.ood                 # ELBO/KL OOD
python -m saevae.toy.experiments.posterior_entropy   # prior sensitivity
python -m saevae.toy.experiments.amortization_gap    # encoder vs ISTA
```

Each script is self-contained: it trains its own models from scratch, writes outputs
to `figures/toy/` and `reports/toy/`, and prints a summary table.


## Models (core API)

All six experiment scripts import from a single entry point:

```python
from saevae.toy import (ToyModelConfig, ToyModel, ToySAE, GaussianVAE, LaplaceVAE,
                        SpikeSlabVAE, train_toy_model, train_sae, train_vae,
                        align_features, evaluate)
```

| Class | Latent dim | Inference / loss | Key params |
|---|---|---|---|
| `ToyModel` | n/a (it is the "LLM") | Importance-weighted MSE; generates hidden activations `h = Wx` | `n_inst`, `n_features=5`, `d_hidden=2` |
| `ToySAE` | `d_sae` (=5) | MAP: ReLU + L1; loss = MSE + λ‖z‖₁ | `sparsity_coeff` (λ) |
| `GaussianVAE` | `d_latent` (=5) | VI: Gaussian prior N(0,I); closed-form KL | `beta` (β-VAE weight) |
| `LaplaceVAE` | `d_latent` (=5) | VI: Laplace prior; KL has closed form via E[|z|] | `beta`, `laplace_b` (scale b) |
| `SpikeSlabVAE` | `d_latent` (=5) | VI: spike-and-slab prior (1−π)δ₀ + πN(0,1); Binary-Concrete gate | `beta`, `prior_pi`, `free_bits`, `temperature` |

`evaluate(model, method, n_eval)` returns `{mse, l0, expected_l0}` per-instance
numpy arrays. `align_features(model, method, inst=0)` returns `{perm, signs, cos}`;
`signs` is load-bearing — see the bug-fix section below. The core files
(`models.py`, `train.py`, `metrics.py`) are **frozen**; new experiments add scripts
only.


## What's built and verified (results table)

All numbers come from the CSVs in `reports/toy/`.

| Script | What it does | Headline result |
|---|---|---|
| `pareto.py` | Sweeps λ (SAE), β (Gaussian/Laplace VAE), and π/β (spike-slab) and plots MSE vs L0 | SAE dominates the Pareto frontier: at L0 ≈ 0.17, MSE ≈ 1e-3 (λ=0.1); Gaussian VAE collapses to L0=0 for β ≥ 0.1 with MSE ≈ 0.028; Laplace VAE achieves L0 > 0 only at b=5.0 (MSE=0.046); spike-slab hard-L0=0 under all β ≥ 0.3, but expected_L0 ∈ [0.78, 2.20] |
| `calibration.py` | Builds feature-detection AUROC for SAE (ReLU score) and Laplace VAE (|μ|); uses sign-aware alignment | Mean AUROC: SAE = 0.579, VAE = 0.840; anti-aligned SAE latents (features 0, 3) score at chance (0.500) because no ReLU latent detects those features with correct polarity |
| `ambiguity.py` | Interpolates 200 inputs between the two most-similar ground-truth features; measures activation entropy at the midpoint | SAE: entropy ≈ 0.000 at midpoint (ReLU commits entirely to one feature or stays silent); VAE: entropy ≈ 0.376 (spreads probability mass across both competing features) |
| `ood.py` | Scores in-dist, random noise, dense (80% active), and 5× scaled activations with SAE loss and VAE ELBO/KL | VAE KL is the unique OOD channel: AUROC 0.708 (random noise), 0.836 (dense), 0.587 (scaled 5×); SAE loss and VAE MSE provide overlapping but generally weaker signals; VAE KL flags latent-space anomalies that look fine to the decoder |
| `posterior_entropy.py` | Trains spike-slab VAE at π ∈ {0.5, 0.3, 0.2, 0.1, 0.05, 0.02}; records posterior entropy and E[L0] | E[L0] declines monotonically: 1.44 (π=0.5) → 0.70 (π=0.02); hard L0 = 0 throughout (spike-slab gate pathology); posterior entropy non-monotone (2.86 at π=0.5, 2.49 at π=0.02) — collapse is in E[L0], not entropy, under current free_bits regime |
| `amortization_gap.py` | Runs 200-iteration non-negative ISTA on frozen SAE decoder; compares encoder objective vs ISTA optimum | Gap = 5.44e-5 (encoder objective 0.007973, ISTA cold 0.007919); encoder is within ~0.7% of the iterative optimum — amortization cost is negligible on the toy |

**Note on ambiguity numbers.** The entropy values above are read from stdout/figure
annotations rather than a CSV (the script does not write a CSV). They reflect the
`LaplaceVAE` at β=0.05, b=0.3 vs `ToySAE` at λ=0.1 under feature_probability=0.1,
seed=42.

**Note on OOD table.** The full AUROC table from `ood_auroc.csv`:

| Signal | Random noise | Dense (80% active) | Scaled 5× |
|---|---|---|---|
| VAE neg-ELBO | 0.634 | 0.803 | 0.602 |
| VAE KL | 0.708 | 0.836 | 0.587 |
| VAE MSE | 0.531 | 0.695 | 0.627 |
| SAE loss | 0.715 | 0.837 | 0.644 |
| SAE MSE | 0.672 | 0.785 | 0.610 |


## Bug fixes already in place

**1. Calibration "SAE AUROC = 0.011" anti-alignment.**
The original notebook produced a mean SAE detection AUROC of ≈ 0.011, which looks
like complete failure. The actual symptom was near-perfect *anti-correlation*
(1 − 0.011 ≈ 0.989): the old `align_features` matched decoder columns by `|cosine|`
only, which silently paired latents whose decoder column pointed in the *opposite*
direction of the ground-truth feature. A ReLU latent anti-aligned to feature i fires
when the hidden activation is pushed *away* from feature i's direction — i.e., when
i is absent — producing an inverted detection score. The fix in `metrics.py` is that
`align_features` now returns a `signs` list alongside `perm` and `cos`. In
`calibration.py`, any matched latent with `signs[i] == -1` has its score replaced
by 0 (constant, AUROC ≈ 0.5) rather than flipped — reflecting the honest finding
that the SAE cannot detect that feature with correct polarity, not exploiting the
inversion. After the fix, features 0 and 3 score at chance (0.500); features 2
scores 0.902. Mean SAE AUROC rises to 0.579 vs the VAE's 0.840.

**2. L0 metric family mismatch.**
The notebooks used a single global threshold of "1% of the single largest activation
in the model" to compute L0. This number is not comparable across families: SAE ReLU
codes are large-magnitude; VAE posterior means are small; and for spike-and-slab it
silently read L0 = 0 for a collapsed gate without flagging it. `metrics.evaluate`
now applies per-family conventions: SAE uses `z > 0` (ReLU support); Gaussian and
Laplace VAE use `|μ| > 0.1` (declared threshold, noted in reports); spike-slab uses
the hard eval-time gate `(gate_logit > 0)` and *also* reports `expected_l0 =
Σ sigmoid(gate_logits)`. This makes the toy Pareto L0 axis commensurable with the
GPT-2 side and makes gate collapse visible rather than hiding it.


## Known open issue

**Spike-slab "soft-on / hard-off" pathology.** Under any non-trivial KL pressure
(β ≥ ~0.1), the eval-time hard gate `(gate_logit > 0)` reads hard L0 = 0 for every
spike-slab configuration tested, while `expected_l0` stays healthy (ranging from
~0.70 to ~2.20 in `pareto.csv`). The 0.5 hard threshold is too strict: the
posterior gate probabilities are consistently below 0.5 in the low-π regimes where
sparsity matters most, because the KL penalty drives the gate logits negative even
when the gates are doing useful work. The `expected_l0` metric shows the posterior
is not fully collapsed; the hard threshold just fails to capture it.

Three options, all consistent with the "SAE dominates Pareto" headline (which holds
under any of them):

- **(a) Report `expected_l0` for spike-slab on the Pareto axis.** Honest and
  meaningful; makes the spike-slab points visible on the plot. Requires a note in
  the figure caption.
- **(b) Threshold gates where posterior odds beat prior odds:** `gate_logit > log(π/(1−π))`.
  Principled Bayesian decision boundary; will give L0 > 0 for the well-trained
  configs while remaining 0 for truly collapsed ones.
- **(c) Call it a finding.** Report that clean point-estimate sparsity (hard-zero
  gates at eval) is not free from a sparse-prior VAE on a 2-D bottleneck —
  the KL term drives gates below the hard threshold even when the posterior is
  informative. This is itself evidence for the MAP/VI distinction.

This issue must be resolved before the final Pareto figure is used in a report or
the T1 decoder-stability sweep is interpreted. The decision is deferred to the user.


## Coverage map vs `sae_vae_experiments.md`

The roadmap lives at `/Users/dhawaldixit/projects/saes-as-vaes/sae_vae_experiments.md`.

| Roadmap experiment | Description | Toy coverage |
|---|---|---|
| **0.1** — Reproduce pretrained SAE | GPT-2 pipeline validation against sae_lens published numbers | N/A — GPT-2 only; parked |
| **1.1** — Train SAE from scratch | SAE λ sweep, Pareto frontier (MAP baseline) | ✅ `pareto.py` sweeps λ ∈ {0.01…1.0} |
| **2.1** — Gaussian-prior VAE | β-VAE sweep, overlaid on SAE Pareto curve | ✅ `pareto.py` sweeps β ∈ {0.01…5.0} |
| **2.2** — Sparse-prior VAE (spike-and-slab) | Main event: sparse prior VI vs MAP | ✅ `pareto.py` sweeps π and β; Laplace VAE also in scope |
| **3.1** — Posterior uncertainty as interpretability signal | Per-feature σ splits into ambiguous vs clear contexts | ⚠️ Spirit only — `ambiguity.py` tests commit-vs-spread on a synthetic interpolation rather than token-context inspection with an LLM judge; the GPT-2-specific `contexts.py` / `judge.py` path is parked |
| **3.2** — Prior sensitivity / posterior shape | Entropy vs prior strength; decoder column stability | ⚠️ Half done — `posterior_entropy.py` covers entropy + E[L0] axis; decoder-column stability (Hungarian-matched cosine vs π) is **missing** — that is T1 below |
| **3.3** — Amortization gap | Encoder vs ISTA optimum | ✅ `amortization_gap.py`; gap = 5.44e-5 |
| **Stretch 1** — Hierarchical prior | Group-structured prior | N/A — not implemented, out of scope for toy |
| **Stretch 2** — Different LLM layers | Early vs late layer pipeline | N/A — GPT-2 only; parked |
| **Stretch 3** — Causal interventions | Intervene on high/low-confidence VAE features; test downstream effect reliability | ⚠️ Not yet implemented on toy — that is T2 below; the toy decoder provides a clean ground-truth downstream task, unlike GPT-2 |


## Planned next work (T1, T2, T3)

These specs are lifted from the Addendum (May 2026) of
`~/.claude/plans/shimmying-spinning-lamport.md`. They are the implementation
contract for whoever runs them.

---

### T1 — Decoder-column stability vs π

**Gap closed:** the missing half of roadmap 3.2 — "do W_d columns stay stable or
reorganize as π shrinks?"

**File to create:** `experiments/saevae/toy/experiments/decoder_stability.py`

**Sweep:** Train one base ToyModel (`n_inst=1, n_features=5, d_hidden=2,
feature_probability=0.025`, 5000 steps). Then train `SpikeSlabVAE` over
π ∈ {0.5, 0.3, 0.2, 0.1, 0.05, 0.02} (β=0.3, free_bits=0.5, temperature=0.5,
8000 steps, kl_warmup=3000). Persist each trained model's `W_dec` (d_latent × d_in).

**Metrics (per π, via `scipy.optimize.linear_sum_assignment` on −|cosine|):**
- `mean_cos_to_ref`: mean matched |cosine| between this π's `W_dec` columns and
  the reference `W_dec` at π=0.5.
- `mean_cos_to_gt`: mean matched |cosine| between this π's `W_dec` columns and the
  toy model's ground-truth `W` columns (instance 0).
- `n_recovered`: count of |cos| > 0.9 against ground truth.
- `dead_count`: latents with all-zero decoder column.

**Plot:** `figures/toy/decoder_stability.pdf` + `.png`. Twin-axis vs π (log x): left
y = `mean_cos_to_ref`, right y = `mean_cos_to_gt`. Include a small heatmap inset of
the per-π Hungarian-matched |cosine| matrix for π=0.5 vs π=0.02 to show qualitative
reorganisation.

**Output CSV:** `reports/toy/decoder_stability.csv` with columns `pi, mean_cos_to_ref,
mean_cos_to_gt, n_recovered, dead_count`.

**Story:** Does the dictionary reorganise as π shrinks (drop in `mean_cos_to_ref`),
and does ground-truth recovery degrade with it?

**Report-back checklist:**
- [ ] `decoder_stability.csv` exists with 6 rows (one per π value)
- [ ] Both figures exist as PDF + PNG
- [ ] Numbers printed for `mean_cos_to_ref`, `mean_cos_to_gt`, `n_recovered` per π
- [ ] Spike-slab reports both `l0` and `expected_l0`
- [ ] Any collapse or anomaly flagged

---

### T2 — Causal interventions on the toy "LLM"

**Gap closed:** roadmap Stretch 3. The toy decoder (`out = ReLU(WᵀWx + b_final)`) is
a downstream task with ground truth, making per-feature interventions testable
rigorously. This is where the "does Bayesian framing buy us something?" question gets
a clean ground-truth answer: do high-confidence VAE features produce more reliable
downstream effects than low-confidence ones?

**File to create:** `experiments/saevae/toy/experiments/causal_interventions.py`

**Setup:** Train base toy model (feature_probability=0.1, 5000 steps) + one each of:
SAE (λ=0.1, 5000 steps), LaplaceVAE (β=0.05, b=0.3, 10000 steps, warmup=3000),
SpikeSlabVAE (β=0.3, π=0.1, free_bits=0.5, 10000 steps, warmup=3000).

**Procedure (5000 test samples):**
1. Encode `h → ẑ` (SAE: ReLU code; VAEs: μ for mean, σ/gate-prob for confidence).
2. Reconstruct `h_recon` via each method's decoder.
3. Compute toy LLM baseline output: `out_base = ReLU(h_recon @ model.W[0] + model.b_final[0])`.
4. For each ground-truth feature k ∈ {0..4}: build intervened code by setting
   `ẑ[:, perm[k]] = 0` (use sign-aware `align_features` permutation; skip
   anti-aligned latents). Re-decode → `h_int`. Compute `out_int`.
5. Per-sample intervention effect `Δ = out_int − out_base`.

**Metrics per feature k:**
- **Target concentration:** `|Δ[:, k]| / Σ_j |Δ[:, j]|` (mean over samples; high =
  effect lands on the right output dimension).
- **Reliability:** std of `Δ[:, k]` across samples (low = consistent).
- **Bayesian split for VAEs:** bucket samples by posterior confidence (Laplace:
  1/σ_k; SpikeSlab: gate prob γ_k). Compare target concentration in high- vs
  low-confidence buckets.

**Plot:** `figures/toy/causal_interventions.pdf` + `.png`. Three panels: (a) bar
chart of mean target concentration per method per gt feature; (b) scatter of target
concentration vs posterior confidence for the VAE with regression line and Spearman
ρ; (c) reliability (std of Δ) per method per feature.

**Output CSV:** `reports/toy/causal_interventions.csv` with per-(method, feature,
confidence-bucket) summary.

**Report-back checklist:**
- [ ] `causal_interventions.csv` exists
- [ ] Both figures exist as PDF + PNG
- [ ] Spearman ρ (confidence vs target concentration) reported for each VAE
- [ ] Headline: do high-confidence VAE interventions produce reliably more targeted
      downstream effects than SAE or all-VAE-mean?

---

### T3 — Identifiability axis on the Pareto sweep

**Gap closed:** the Pareto plot currently shows MSE vs L0 but says nothing about
whether the learned dictionary is finding the true generating directions. Hungarian-
matched mean |cosine| between learned `W_dec` columns and the toy's ground-truth `W`
is a clean per-run scalar orthogonal to MSE/L0.

**File to modify:** `experiments/saevae/toy/experiments/pareto.py` (additive only;
do not change existing logic or column order).

**Changes:**
- After each trained model is evaluated, compute via Hungarian matching:
  - `gt_mean_cos`: mean matched |cosine| between learned decoder columns and the
    toy's ground-truth `W` columns.
  - `gt_recovered`: count of |cos| > 0.9.
- Append both as new columns to `reports/toy/pareto.csv`. Existing columns
  (family, param_name, param_value, mse, l0, expected_l0) must remain unchanged.
- Add a new figure: `figures/toy/identifiability.pdf` + `.png`. Scatter plot, x = L0,
  y = `gt_mean_cos`, one color per family with same palette as `pareto.png`
  (SAE=red circles, Gaussian=blue triangles, Laplace=green squares,
  spike-slab=purple diamonds). Annotate with sweep values.
- Existing `figures/toy/pareto.pdf` content must remain unchanged.
- Print a per-family summary table (best-config row: MSE, L0, gt_mean_cos,
  gt_recovered) to stdout.

**Output CSV:** `reports/toy/pareto.csv` extended; `figures/toy/identifiability.pdf`
+ `.png` new.

**Report-back checklist:**
- [ ] `pareto.csv` retains original 6 columns with `gt_mean_cos` and `gt_recovered`
      appended
- [ ] `identifiability.pdf` + `.png` exist
- [ ] Original `pareto.pdf` content unchanged
- [ ] Per-family best-row summary printed


## Conventions

- **Run commands** always from `experiments/`: `python -m saevae.toy.experiments.<name>`.
- **CPU only.** `import matplotlib; matplotlib.use("Agg")` before pyplot. No MPS
  auto-select — PyTorch's MPS float32 backend produces silently corrupted gradients
  in this matmul-heavy loop (documented in `models.py`).
- **Seeds.** `pareto.py`, `posterior_entropy.py`, `amortization_gap.py` use
  `torch.manual_seed(0); np.random.seed(0)`. `calibration.py`, `ambiguity.py`,
  `ood.py` use seed 42. Keep runs deterministic; do not change seeds without noting
  it.
- **Outputs.** Figures to `figures/toy/`, CSVs to `reports/toy/`. Every figure is
  saved as both PDF and PNG (`dpi=150`).
- **Spike-slab.** Always report both `l0` (hard gate) and `expected_l0` (E[gate])
  in every sweep that includes spike-and-slab, per the open pathology above.
- **Core API is frozen.** Do not modify `models.py`, `train.py`, or `metrics.py`
  for T1–T3. Add new experiment files only.
- **Subagent import contract.**
  ```python
  from saevae.toy import (
      ToyModelConfig, ToyModel, ToySAE, GaussianVAE, LaplaceVAE, SpikeSlabVAE,
      train_toy_model, train_sae, train_vae, align_features, evaluate,
  )
  ```
  `align_features` returns `{perm, signs, cos}`; use `signs` to skip anti-aligned
  latents (don't flip). `evaluate` returns `{mse, l0, expected_l0}` per-instance
  numpy arrays. Ground-truth columns live in `model.W[inst]` (shape `d_hidden ×
  n_features`).


## Parked work

The GPT-2 activation track — activation extraction, SAE/VAE training on 5M × 768
fp16 activations, the `sae_lens` anchor, W&B sweep infrastructure, and the
`contexts.py` / `judge.py` LLM-as-judge pipeline — has been moved to
`experiments/old_code/`. The full implementation plan (including the exact file list,
sweep configs, and Colab driver notebooks) remains in
`~/.claude/plans/shimmying-spinning-lamport.md`. The move is reversible if GPU
compute opens up; the toy subpackage imports cleanly as `from saevae.toy import ...`
regardless.
