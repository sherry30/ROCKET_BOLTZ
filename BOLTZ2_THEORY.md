# ROCKET-Boltz2: Theory

A walkthrough of the crystallographic objective, how Boltz-2 is wired into it,
and how gradients propagate from diffraction data back to the learnable bias.

---

## 1. The Crystallographic Refinement Problem

### What we have

- **Experimental structure-factor amplitudes** |F_obs(h,k,l)| and their
  uncertainties σ, measured by X-ray diffraction and stored in an MTZ file.
  These are the only experimental data used during refinement.

- **A predicted protein structure** — atom coordinates, elements, and
  isotropic B-factors — from Boltz-2.

### What we want

Atom coordinates that simultaneously:
1. **Explain the X-ray data** — the diffraction pattern computed from the model
   matches the measured one.
2. **Stay chemically reasonable** — bond lengths, angles, and torsions near
   ideal values (enforced implicitly by keeping Boltz-2 as the structural prior).

### Why it's non-trivial

X-ray diffraction measures only **amplitudes** |F(h,k,l)| — the **phases**
φ(h,k,l) are lost in the experiment.  Without phases you cannot directly
invert the diffraction pattern to electron density.  Refinement is an
iterative optimisation that simultaneously fits amplitudes and estimates phases
from the current model.

---

## 2. Structure Factors and the LLG Objective

### Structure factor calculation

For each reflection **h** = (h, k, l), the computed structure factor is:

```
F_calc(h) = Σ_j  f_j(|s|) · exp(2πi h·r_j) · exp(−B_j |s|² / 4)
```

where r_j is the fractional coordinate of atom j, f_j is its element-dependent
scattering factor, B_j is its isotropic B-factor (Å²), and
|s| = sin(θ)/λ is the scattering vector magnitude.

In ROCKET, F_calc is computed by **SFcalculator** (SFC_Torch) — a
GPU-accelerated real-space summation that is fully differentiable with respect
to atom positions.

### Normalised structure factors

The MTZ produced by `rk.preprocess` contains **normalised** amplitudes:

```
E_obs = F_obs / <F²>^(1/2)_{resolution bin}
```

Normalisation removes the average fall-off with resolution, making the
likelihood function well-behaved across all resolution shells.

### The LLG objective

ROCKET maximises the **Log-Likelihood Gain** — the improvement in log-likelihood
of the observed amplitudes over a random (uninformative) model:

```
LLG = Σ_{h ∈ working set}  log p(|E_obs(h)| | σ_A, E_calc(h))
                          − log p(|E_obs(h)| | σ_A = 0)
```

The likelihood uses the **Rice distribution**:

```
p(|E_obs| | E_calc, σ_A) ∝  |E_obs| / σ²
    · exp(−(|E_obs|² + σ_A² |E_calc|²) / 2σ²)
    · I₀(σ_A |E_obs| |E_calc| / σ²)
```

- **σ_A ∈ [0, 1]** — phase reliability; correlation between E_calc and the
  true structure factors; estimated per resolution bin each iteration
- **I₀** — modified Bessel function, order 0
- **σ² = 1 − σ_A²**

Higher LLG means the model's diffraction pattern better explains the data.
ROCKET **minimises −LLG** as the training loss.

σ_A is estimated each iteration from the ratio of |F_calc| to |F_obs| in
resolution bins, on **detached** coordinates, so it does not enter the gradient.

---

## 3. Input Pipeline

### 3a. Crystallographic inputs

`rk.preprocess --model boltz2` produces two files in `ROCKET_inputs/`:

```
rk.preprocess --model boltz2 [--precomputed_alignment_dir alignments/]
  │
  ├─ boltz predict → {file_id}_boltz2_unrelaxed.pdb
  │     (Boltz-2 structure prediction from FASTA)
  │
  ├─ phasertng.picard (Phaser MR)
  │     places the prediction in the crystal unit cell
  │     → {file_id}-MRed.pdb
  │
  ├─ phenix.superpose_pdbs
  │     superposes the full Boltz-2 prediction onto the MR model
  │     → ROCKET_inputs/{file_id}-pred-aligned.pdb
  │
  └─ phasertng mtz_generator
        normalises |F_obs| per resolution bin, assigns R-free flags
        → ROCKET_inputs/{file_id}-Edata.mtz
```

These two files are the crystallographic inputs to the refinement loop.
The aligned PDB provides initial coordinates; the MTZ provides the target
diffraction amplitudes that are held fixed throughout refinement.

### 3b. Boltz-2 sequence features

`prepare_boltz2_feats` converts the aligned PDB into the tensor dictionary
that Boltz-2 expects.  This runs once at preprocessing time (CPU-only) and is
pickled to `ROCKET_inputs/feats_boltz2.pkl`.

```
prepare_boltz2_feats(pred-aligned.pdb, a3m_path=alignment.a3m)
  │
  ├─ gemmi.read_structure → per-chain amino-acid sequences
  │
  ├─ parse_boltz_schema(..., boltz_2=True)
  │     → Target: StructureV2 (Boltz-2 atom graph)
  │
  ├─ parse_a3m(a3m) → MSA object  [if an MSA was found in the alignment dir]
  │     mapped to {chain_asym_id → MSA} for all protein chains
  │     enables: z = z + msa_module(z, s, feats)  in the trunk
  │     without this: single-sequence fallback (dummy MSA)
  │
  └─ Boltz2Featurizer().process(tokenized, ...)
       → feats dict (batch=1, CPU tensors):
           token_pad_mask       [1, N_tokens]
           atom_pad_mask        [1, N_atoms]
           atom_to_token        [1, N_atoms, N_tokens]   one-hot
           residue_index        [1, N_tokens]
           asym_id              [1, N_tokens]
           res_type             [1, N_tokens, 32]         one-hot
           ref_atom_name_chars  [1, N_atoms, 4, 64]       one-hot chars
           token_bonds          [1, N_tokens, N_tokens]
           rel_pos              [1, N_tokens, N_tokens, D]
           … (and more)
```

`feats` is moved to GPU once at the start of `run_boltz2_xray_refinement` and
reused for every iteration.  Providing an MSA via `--precomputed_alignment_dir`
is strongly recommended: it activates the MSA module in the trunk and gives the
model richer inter-residue information, which translates to a better gradient
signal.

---

## 4. The Refinement Loop

Each gradient step performs this sequence:

```
──────────────────────────────────────────────────────────────────────────
FIXED (loaded once, never updated):
  feats_gpu       Boltz-2 sequence features          [1, …]   GPU
  sfc             SFcalculator with Edata.mtz                  GPU
  reference_pos   initial atom coords (pred-aligned)  [N, 3]  GPU

LEARNABLE (updated every step):
  w_pair   [128, 128]   identity-init   requires_grad=True   (16 384 params)
  b_pair   [128]        zeros-init      requires_grad=True   (128 params)
──────────────────────────────────────────────────────────────────────────
```

### Step 1 — Trunk recycling with ConForNets bias

The Boltz-2 trunk runs for `recycling_steps` iterations (default 3).
Only the **final** iteration retains the autograd graph; earlier ones are
detached to avoid backpropagating through the entire recycling stack.

The learnable bias is injected on the final recycling pass, **before the
PairFormer stack** (inspired by ConForNets, arxiv 2604.18559):

```
feats → input_embedder → s_inputs [1, N, 384]
                       → z_init   [1, N, N, 128]

for i = 0 … recycling_steps:
    s = s_init + s_recycle(s_norm(s_prev))
    z = z_init + z_recycle(z_norm(z_prev))
    z = z + msa_module(z, s_inputs, feats)    ← active if a3m provided
    |
    | no_grad for i < recycling_steps
    | enable_grad for i == recycling_steps   (final only)
    |
    if final iteration:
        z = z @ w_pair + b_pair              ← BIAS INJECTION (pre-PairFormer)
        s, z = pairformer_module(s, z, mask) ← activation-checkpointed per block
    else:
        s, z = pairformer_module(s, z, mask)
```

- `w_pair [128, 128]` — initialised to identity; same transform applied to
  all (i, j) positions, independent of protein length
- `b_pair [128]` — initialised to zeros; broadcast over all positions

Activation checkpointing is applied to every PairFormer block on the final pass
so gradients flow back through the whole PairFormer stack without storing all
intermediate activations.  Python closure late-binding is avoided by capturing
each block via default-argument in the per-block checkpoint closure.

`z_out` after PairFormer carries the gradient graph back to `w_pair` and
`b_pair` via `z @ w_pair + b_pair`.

### Step 2 — Diffusion conditioning

```
q, c, atom_enc_bias, atom_dec_bias, token_trans_bias
  = DiffusionConditioning(
        s_trunk    = s,
        z_trunk    = z_out,        ← grad flows: z_out → w_pair @ z → w_pair, b_pair
        rel_pos    = rel_pos(feats),
        feats      = feats,
    )
```

`q` and `c` are the token-level keys and context passed into the diffusion
transformer.  They carry the gradient that flows back through PairFormer → bias
injection → `w_pair` and `b_pair`.

### Step 3 — Truncated-backprop diffusion sampling

T = total sampling steps (default 200), K = truncated_backprop_steps (default 20).

```
x_T ~ N(0, σ_T²)     initial noisy coordinates     [1, N_atoms, 3]

for t = T−1 … 0:
    compute noise schedule (σ_tm, σ_t, γ)
    x_noisy = x.detach() + stochastic_noise

    if t < T − K:                    ← steps 0 … T−K−1
        with torch.no_grad():
            x_denoised = network(x_noisy, t, q, c, …)
        x = euler_step(…).detach()

    else:                            ← steps T−K … T−1 (last K steps)
        x_denoised = network(x_noisy, t, q, c, …)
        x = euler_step(…)            ← grad flows through q, c → z_biased

atom_coords = x     [1, N_atoms, 3]     grad-carrying
```

Truncating to the last K steps is a standard TBPTT approximation: storing the
full T=200-step unrolled graph would be prohibitively expensive and produce very
noisy gradients from stochastic noise injections deep in the chain.

**K=5 is insufficient**: empirical testing showed K=5 (2.5% of the trajectory)
gives gradients too weak to produce measurable Rwork improvement, and causes
catastrophic LLG collapse once `w_pair` has drifted ~0.5 Frobenius norm from
identity.  **K=20** (10% of the trajectory) gives 4× stronger and more
directionally consistent gradients and is the validated default.

Random coordinate augmentation (rotation/translation applied during normal
Boltz-2 inference) is disabled during ROCKET to avoid scrambling the gradient
signal.

### Step 4 — Coordinate extraction and Kabsch alignment

Boltz-2 uses a flat per-atom layout indexed by `feats["atom_to_token"]`.
`extract_allatoms_boltz2` maps atoms back to residues and reorders them to
match the SFcalculator atom topology (chain–residue–atomname ordering).

```
atom_coords [1, N_atoms, 3]
  ↓ extract_allatoms_boltz2(atom_coords, feats, cra_name_sfc)
      maps atoms → residues via atom_to_token
      filters by atom_pad_mask
      reorders to SFC cra_name ordering
  ↓
raw_xyz  [N_sfc_atoms, 3]   grad-carrying

  ↓ iterative_kabsch_alignment(raw_xyz, best_pos, …)
      finds optimal global R, t on detached coords
      re-applies R, t to raw_xyz (grad preserved)
  ↓
aligned_xyz  [N_sfc_atoms, 3]   grad-carrying
```

### Step 5 — Rigid-body refinement (RBR)

RBR finds the optimal 6-DOF placement of the model in the unit cell by
minimising −LLG over 3 rotations (quaternion) + 3 translations.

```
  ↓ rigidbody_refine_quat(aligned_xyz.detach(), llgloss_rbr, …)
      L-BFGS on DETACHED coords  →  optimal R, t

  ↓ apply_rotation_translation(R, t, aligned_xyz)
      R, t applied to GRAD-CARRYING tensor
  ↓
optimized_xyz  [N_sfc_atoms, 3]   grad-carrying
```

L-BFGS cannot itself propagate gradients through its inner loop.  The
detach-then-reapply pattern recovers the gradient: L-BFGS finds the best
rigid-body transform; that transform is re-applied to the original grad-carrying
coordinates so the gradient can still flow back through the alignment.

### Step 6 — B-factors (no grad)

```
# Preferred path (boltz2_conf.ckpt has predict_bfactor=True):
token_B = softmax(bfactor_module(s.detach())) · bin_centers   [1, N_tokens]

# Fallback if bfactor_module is absent:
token_B = plddt2pseudoB(plddt · 100)                          [1, N_tokens]

atom_B  = token_B[:, atom_to_token_idx]                        [1, N_atoms]
sfc.atom_b_iso = atom_B.detach().clamp(max=200.0)
```

Boltz-2's `bfactor_module` predicts a histogram over B ∈ [0, 100] Å²; we take
the expected value under that distribution.  If the checkpoint does not include
`bfactor_module` (i.e. `model.predict_bfactor=False`), the code falls back to
the pLDDT→pseudo-B conversion used in the original AF2-ROCKET.

B-factors modulate the Debye–Waller fall-off in F_calc but are always detached
before being set on the SFC — consistent with the original ROCKET design.
The 200 Å² clamp ensures F_calc values remain numerically meaningful.

### Step 7 — LLG computation

```
sfc.atom_pos_orth = optimized_xyz.detach()   update SFC geometry

llg, r_work, r_free = llgloss(
    optimized_xyz,              ← grad-carrying
    sub_ratio = 1.0,            all reflections (sub-sampling adds too much noise)
    solvent   = True,           bulk-solvent correction
    …
)
```

Inside `llgloss`:
1. `sfc.calc_fprotein(optimized_xyz)` — GPU structure factor summation;
   F_calc(h) for each reflection with grad flowing through → optimized_xyz
2. Bulk-solvent contribution added (flat model, scale refined on detached coords)
3. σ_A (pre-estimated this iteration, detached) applied to Rice likelihood
4. Sum over working reflections → LLG scalar

### Step 8 — Loss and backward

**Phase 1:**
```
L = −LLG  +  λ_L2 · Σ_j w_j · ||optimized_xyz_j − ref_pos_j||²
```

**Phase 2:**
```
L = −LLG
```

The L2 term in Phase 1 keeps the model close to the initial Boltz-2 prediction.
`w_j` (bfactor_weights) gives disordered atoms (high predicted B) lower penalty,
allowing more movement where the model is least confident.

```
L.backward()
  grad path: −LLG → optimized_xyz
           → RBR (reapply R, t) → aligned_xyz
           → Kabsch alignment → atom_coords
           → last K diffusion steps → DiffusionTransformer
           → q, c (from DiffusionConditioning)
           → z_biased → w_pair, b_pair
```

---

## 5. Gradient Flow at a Glance

```
w_pair [128,128], b_pair [128]    ← only learnable parameters
    │
    ▼  (final recycling pass only)
z_pre = z_init + z_recycle + msa_module   [B, N, N, 128]    (no_grad upstream)
    │
    ▼
z @ w_pair + b_pair               ← BIAS INJECTION
    │
    ▼  (activation-checkpointed per block)
PairFormer blocks × L
    │
    ▼
z_out [B, N, N, 128]
    │
    ▼
DiffusionConditioning → q, c, enc_bias, dec_bias, trans_bias
    │
    ▼  (last K reverse-diffusion steps; earlier T−K steps detached)
DiffusionTransformer
    │
    ▼
atom_coords  [1, N_atoms, 3]
    │
    ▼
extract_allatoms → Kabsch alignment → RBR (reapply R, t)
    │
    ▼
optimized_xyz  [N_sfc_atoms, 3]
    │
    ▼
SFcalculator.calc_fprotein → F_calc(h)
    │
    ▼
Rice likelihood → LLG → −LLG = loss
```

The input embedder, MSA module, and Boltz-2 weights are **frozen**; no gradient
reaches them.  Gradients do flow through PairFormer (via activation checkpointing)
because the bias is injected before the PairFormer stack.

---

## 6. Two-Phase Schedule

### Phase 1 — Exploration

Three independent runs with different diffusion seeds explore the bias
parameter space broadly.  Low-resolution reflections only (3.0 Å cutoff) give
a robust signal less sensitive to model errors at high resolution.

```yaml
algorithm:
  iterations: 100
  optimization:
    additive_learning_rate: 1e-4     # lr for b_pair  [128]
    multiplicative_learning_rate: 1e-4  # lr for w_pair  [128, 128]
    l2_weight: 1.0e-7                # regularise toward initial prediction
    smooth_stage_epochs: 80          # LR decay from iter 20 → iter 100
    phase2_final_lr: 1e-5            # target lr at end of smooth stage
    batch_sub_ratio: 1.0             # full reflections (no subset sampling)
data:
  min_resolution: 3.0
execution:
  num_of_runs: 3                     # independent traces
```

**Why lr=1e-4?** Adam normalises gradients to ≈1 per element per step.  With
`lr=1e-3`, even the very first Adam step overshoots (empirically: ΔLLG=−5 on
full reflections after one step at lr=1e-3).  With `lr=1e-4`, the bias stays
in the productive region near identity for ~20 iterations and then slowly drifts
away.  The smooth stage (last 80 of 100 iterations) decays lr from 1e-4 to 1e-5,
locking in the improvement and preventing post-peak drift.

**Why full reflections (batch_sub_ratio=1.0)?** Sub-sampling 70% of reflections
per step (the AF2-ROCKET default) adds ±23 LLG units of noise per step, vs. ±4
units with full reflections.  With a gradient signal of only ~50–150 LLG units
total, this noise/signal ratio is unfavourable and causes premature collapse.

**Seed pre-scan**: Before the 3 optimisation runs, `rk.preprocess` evaluates 9
diffusion seeds with an identity bias and ranks them by LLG.  The 3 best seeds
are used for the optimisation runs.  Seed-to-seed LLG variance (~370 units) far
exceeds the optimisation signal (~50–150 units), so starting from the best seeds
is critical.  The scan result is saved to `ROCKET_inputs/seed_scan.npy` and
loaded by `rk.refine` at startup.

At the end of Phase 1, the run with the highest LLG is selected.  Its
`best_w_pair` and `best_b_pair` are saved for Phase 2.

### Phase 2 — Exploitation

Single run, warm-started from the Phase-1 best bias.  All resolution shells
are used and the learning rate is small, giving precise fine-tuning.

```yaml
algorithm:
  iterations: 500
  optimization:
    additive_learning_rate: 1e-5     # small LR — fine-tune
    multiplicative_learning_rate: 1e-5
    l2_weight: 0.0                   # no regularisation
    smooth_stage_epochs: null        # constant LR
    batch_sub_ratio: 1.0
data:
  min_resolution: null               # all reflections
paths:
  starting_bias:    best_w_pair_*.pt
  starting_weights: best_b_pair_*.pt
execution:
  num_of_runs: 1
```

---

## 7. Design Choices

### Why a channel-wise [128, 128] transform before PairFormer?

This design follows ConForNets (arxiv 2604.18559): a channel-wise affine
transform of the pre-PairFormer pair latents.

Applying the transform **before PairFormer** (rather than after, on the output
`z`) means the PairFormer blocks can propagate and amplify the bias signal
through their full triangle-attention / outer-product machinery.  The gradient
path therefore runs through the entire PairFormer stack, giving a richer
training signal than a post-hoc additive shift.

Using a `[128, 128]` matrix instead of a per-position `[N, N, 128]` tensor
makes the parameter count **length-independent** (16,512 params vs. O(N²)), and
produces a smoother optimisation landscape because the same transformation is
enforced at every residue pair.

### Why truncated backprop through diffusion?

Full backprop through T = 200 steps would require storing the entire unrolled
computation graph and propagating gradients through stochastic noise injections
at every step — O(T) memory and very noisy gradients.  Truncating to K = 20 steps
(TBPTT, 10% of the trajectory) captures short-range denoising credit assignment
with manageable cost while providing enough gradient signal to reliably improve
the structure.  K=5 was found empirically to be insufficient — gradients from
only 2.5% of the trajectory are too weak to produce measurable Rwork improvement.

### Why detach and reapply in RBR?

L-BFGS cannot propagate gradients through its own inner optimisation loop.
The detach-reapply pattern gives the correct approximate gradient: "move the
atom coordinates in the direction that, after the best rigid-body alignment,
maximises LLG."

### Why are B-factors excluded from the gradient?

The B-factor prediction module (a histogram over discrete bins) is a secondary
output used to set the Debye–Waller fall-off in F_calc.  Differentiating
through it would add another path to `w_pair`/`b_pair` but the signal would
be noisy relative to the direct coordinate path.  This matches the original
ROCKET design and keeps the gradient path clean.
