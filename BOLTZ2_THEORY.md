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
rk.preprocess --model boltz2 [--a3m_path alignment.a3m]
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
  │     converts pLDDT → pseudo-B-factors
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
  ├─ parse_a3m(a3m_path) → MSA object  [if --a3m_path provided]
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
reused for every iteration.  Providing `--a3m_path` is strongly recommended:
it activates the MSA module in the trunk and gives the model richer inter-residue
information, which translates to a better gradient signal.

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
  w_pair   [1, N, N, 128]   ones-init    requires_grad=True
  b_pair   [1, N, N, 128]   zeros-init   requires_grad=True
──────────────────────────────────────────────────────────────────────────
```

### Step 1 — Trunk recycling

The Boltz-2 trunk runs for `recycling_steps` iterations (default 3).
Only the **final** iteration retains the autograd graph; earlier ones are
detached to avoid backpropagating through the entire recycling stack.

```
feats → input_embedder → s_inputs [1, N, 384]
                       → z_init   [1, N, N, 128]

for i = 0 … recycling_steps:
    s = s_init + s_recycle(s_norm(s_prev))
    z = z_init + z_recycle(z_norm(z_prev))
    z = z + msa_module(z, s_inputs, feats)    ← active if a3m provided
    s, z = pairformer_module(s, z, mask)
    |
    | no_grad for i < recycling_steps − 1
    | enable_grad for i == recycling_steps − 1   (final only)

z_biased = w_pair * z + b_pair     ← PAIR BIAS INJECTION
```

`z_biased` carries the gradient graph back to `w_pair` and `b_pair`.
Everything upstream (input embedder, PairFormer, MSA module) is frozen and
receives no gradient.

### Step 2 — Diffusion conditioning

```
q, c, atom_enc_bias, atom_dec_bias, token_trans_bias
  = DiffusionConditioning(
        s_trunk    = s,
        z_trunk    = z_biased,     ← grad flows: z_biased → w_pair, b_pair
        rel_pos    = rel_pos(feats),
        feats      = feats,
    )
```

`q` and `c` are the token-level keys and context passed into the diffusion
transformer.  They carry the gradient from `z_biased`.

### Step 3 — Truncated-backprop diffusion sampling

T = total sampling steps (default 200), K = truncated_backprop_steps (default 5).

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

Truncating to K=5 steps is a standard TBPTT approximation: storing the full
T=200-step unrolled graph would be prohibitively expensive and produce very
noisy gradients from stochastic noise injections deep in the chain.

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
token_B = softmax(bfactor_module(s.detach())) · bin_centers   [1, N_tokens]
atom_B  = token_B[:, atom_to_token_idx]                        [1, N_atoms]
sfc.atom_b_iso = atom_B.detach().clamp(max=200.0)
```

B-factors modulate the Debye–Waller fall-off in F_calc but are always detached
before being set on the SFC — consistent with the original ROCKET design.
The 200 Å² clamp ensures F_calc values remain numerically meaningful (Boltz-2
single-sequence pLDDT is lower than AF2, which can produce very large raw
pseudo-B values that would zero out F_calc at all resolutions).

### Step 7 — LLG computation

```
sfc.atom_pos_orth = optimized_xyz.detach()   update SFC geometry

llg, r_work, r_free = llgloss(
    optimized_xyz,              ← grad-carrying
    sub_ratio = 0.7,            stochastic subset of reflections
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
w_pair, b_pair                   ← only learnable parameters
    │
    ▼
z_biased = w_pair * z + b_pair   [1, N, N, 128]
    │
    ▼
DiffusionConditioning → q, c, enc_bias, dec_bias, trans_bias
    │
    ▼  (last K reverse-diffusion steps)
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

Everything upstream of the bias injection (input embedder, PairFormer,
MSA module, Boltz-2 weights) is **frozen** and receives no gradient.

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
    additive_learning_rate: 0.05     # large LR — explore
    l2_weight: 1.0e-7                # regularise toward initial prediction
    smooth_stage_epochs: 50          # cosine LR decay after iter 50
data:
  min_resolution: 3.0
execution:
  num_of_runs: 3                     # independent traces
```

At the end, the run with the lowest −LLG is selected.  Its `best_w_pair` and
`best_b_pair` are saved for Phase 2.

Diversity across runs comes from independent diffusion noise seeds (Boltz-2 has
no MSA to subsample as AF2-ROCKET does).

### Phase 2 — Exploitation

Single run, warm-started from the Phase-1 best bias.  All resolution shells
are used and the learning rate is small, giving precise fine-tuning.

```yaml
algorithm:
  iterations: 500
  optimization:
    additive_learning_rate: 0.0001   # small LR — fine-tune
    l2_weight: 0.0                   # no regularisation
    smooth_stage_epochs: null        # constant LR
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

### Why the pair representation z?

Boltz-2 has no MSA track.  `z` is the primary inter-residue feature passed from
the trunk into the diffusion conditioning module — the natural analogue to the
AF2 MSA cluster profile.  Biasing `z` directly modulates the geometric priors
(pairwise distances, orientations) fed into the diffusion process.

### Why truncated backprop through diffusion?

Full backprop through T = 200 steps would require storing the entire unrolled
computation graph and propagating gradients through stochastic noise injections
at every step — O(T) memory and very noisy gradients.  Truncating to K = 5 steps
(TBPTT) captures short-range denoising credit assignment with manageable cost.

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
