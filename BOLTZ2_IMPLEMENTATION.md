# ROCKET-Boltz2

ROCKET applied to Boltz-2: inference-time crystallographic refinement by
gradient descent through a learnable bias on Boltz-2's pair representation.
No Boltz-2 weights are modified.

---

## How it works

ROCKET introduces a small set of learnable parameters into an internal
representation of the structure predictor, then optimises those parameters to
maximise the fit of the predicted structure to X-ray diffraction data (measured
by the Log-Likelihood Gain, LLG).

For Boltz-2, the learnable bias applies a **channel-wise affine transform** to
the trunk pair representation z immediately **before** the PairFormer stack
(inspired by ConForNets, arxiv 2604.18559):

```
z  →  z @ w_pair + b_pair  →  PairFormer  →  z_out
```

- `w_pair`: `[128, 128]` — initialised to identity
- `b_pair`: `[128]` — initialised to zeros

Applied as `z [B, N, N, 128] @ w_pair [128, 128] + b_pair [128]` — the
same linear transform is broadcast over all (i, j) positions.  At iteration 0
the identity initialisation leaves the model prediction unchanged.

This formulation is **length-independent**: `w_pair` and `b_pair` have
`128² + 128 = 16,512` parameters regardless of protein size (vs. `2 × N² × 128`
for the old per-position bias).  Activation checkpointing is applied to each
PairFormer block so gradients flow through without storing all intermediate
activations.

Everything else — trunk weights, diffusion network weights, MSA module — is
frozen.

---

## Code structure

| File | Purpose |
|---|---|
| `rocket/boltz2_wrapper.py` | `Boltz2PairBias` — injects the pair bias, runs the trunk recycling loop, implements truncated-backprop diffusion sampling |
| `rocket/refinement_boltz2.py` | Full refinement loop + `prepare_boltz2_feats` + `precompute_boltz2_seeds` |
| `rocket/coordinates_boltz2.py` | Atom extraction from Boltz-2 output + Kabsch alignment; uses Boltz-2 direct B-factor prediction, falls back to pLDDT→pseudo-B |
| `rocket/refinement_config.py` | `Boltz2Config` fields; `gen_config_phase2` handles Boltz-2 bias filenames |
| `rocket/scripts/run_preprocess.py` | `rk.preprocess --model boltz2` — predict + MR + feats + seed scan + configs |
| `rocket/scripts/run_refine.py` | `rk.refine` — dispatches to Boltz-2 backend; loads feats and seed scan from config |
| `rocket/scripts/generate_msa.py` | `rk.generate_msa` — fetches MSA via ColabFold API |

---

## CLI workflow

### Prerequisites

Both `rk.preprocess --model boltz2` and `rk.refine` load model weights and
must run on the GPU node:

```bash
ssh shehryar@max-hpcgwg006
micromamba activate rocket-of
```

Boltz cache dir: `/data/dust/group/it/crystalsfirst/dev/shehry/boltz_cache/`

---

### Workflow — rk.preprocess + rk.refine

`rk.preprocess --model boltz2` runs Boltz-2 prediction, Phaser MR,
featurization, seed scan, and config generation in a single command.

**Expected input layout in the working directory:**
```
{file_id}_fasta/{file_id}.fasta      # protein FASTA
{file_id}_data/*.mtz                 # diffraction data (MTZ)
```

**Step 0 — Generate MSA (optional but recommended)**

```bash
rk.generate_msa \
  --fasta 1lj5_fasta/1lj5.fasta \
  --file_id 1lj5 \
  --output_dir alignments/
# outputs: alignments/1lj5/bfd_uniclust_hits.a3m   ← use for --a3m_path
```

**Step 1 — Preprocess (GPU node)**

```bash
rk.preprocess \
  --file_id 1lj5 \
  --method xray \
  --output_dir ./1lj5_processed \
  --model boltz2 \
  --boltz2_cache_dir /data/dust/group/it/crystalsfirst/dev/shehry/boltz_cache \
  --a3m_path alignments/1lj5/bfd_uniclust_hits.a3m \
  --n_seeds_to_scan 9
```

`--a3m_path` activates the Boltz-2 MSA module (`z = z + msa_module(...)`) —
recommended for better gradient signal.  `--n_seeds_to_scan` (default 9)
controls how many diffusion seeds are pre-evaluated during preprocessing.

What this does internally:
1. Parses FASTA → writes `{file_id}_boltz_input.yaml`
2. Runs `boltz predict` → `{file_id}_boltz2_unrelaxed.pdb`
3. Runs Phaser MR (phasertng) → `{file_id}-MRed.pdb`
4. Superposes Boltz-2 prediction onto MR model → `ROCKET_inputs/{file_id}-pred-aligned.pdb`
5. Generates `ROCKET_inputs/{file_id}-Edata.mtz` (normalised structure factors)
6. Runs `prepare_boltz2_feats` → `ROCKET_inputs/feats_boltz2.pkl`
7. Pre-evaluates 9 diffusion seeds → `ROCKET_inputs/seed_scan.npy`
8. Writes `ROCKET_config_phase1_boltz2.yaml` and `ROCKET_config_phase2_boltz2.yaml`

Output layout:
```
1lj5_processed/
  ROCKET_inputs/
    1lj5-pred-aligned.pdb
    1lj5-Edata.mtz
    feats_boltz2.pkl
    seed_scan.npy           ← precomputed seed ranking (loaded by rk.refine)
  ROCKET_config_phase1_boltz2.yaml
  ROCKET_config_phase2_boltz2.yaml
```

**Step 2 — Phase 1 refinement (GPU node)**

```bash
rk.refine 1lj5_processed/ROCKET_config_phase1_boltz2.yaml
```

Feats and seed scan paths are embedded in the config — no extra flags needed.
Phase 1 starts directly from the best precomputed seeds, runs `num_of_runs`
independent traces (default 3), and saves:

```
1lj5_processed/ROCKET_outputs/<uuid>/phase1_boltz2_1lj5/
  best_model_A_18.pdb       # best structure
  best_w_pair_A_18.pt       # channel-wise weight matrix  [128, 128]
  best_b_pair_A_18.pt       # channel-wise bias vector    [128]
  NEG_LLG_it_A.npy, rwork_it_A.npy, rfree_it_A.npy, …
  seed_scan.npy             # copy of the seed scan used
```

**Step 3 — Phase 2 refinement (GPU node)**

```bash
rk.refine 1lj5_processed/ROCKET_config_phase2_boltz2.yaml
```

Phase 2 warm-starts from the Phase-1 best `w_pair`/`b_pair`, lower LR, no L2,
all resolution shells.

---

## Config reference

Boltz-2-specific fields in the generated YAML:

```yaml
execution:
  model: boltz2              # selects Boltz-2 backend

boltz2:
  boltz2_checkpoint_path: /path/to/boltz2_conf.ckpt
  feats_path: /path/to/feats_boltz2.pkl         # auto-loaded by rk.refine
  truncated_backprop_steps: 20                   # K: grad steps in diffusion (K=5 too noisy)
  boltz2_recycling_steps: 3                      # trunk recycling iterations
  boltz2_num_sampling_steps: 200                 # total diffusion steps T
  precomputed_seed_scan: /path/to/seed_scan.npy  # written by rk.preprocess
```

**Truncated backprop note**: `K=5` (2.5% of the 200-step trajectory) gives gradients
that are too noisy — no Rwork improvement over 100 iterations, and catastrophic LLG
collapse at iter 60–70 once `w_pair` drifts ~0.5 Frobenius norm from identity.
`K=20` (10%) is the validated default: 4× stronger and more directionally consistent.

**Seed scan precomputation**: `rk.preprocess --model boltz2` runs
`precompute_boltz2_seeds()` and writes `ROCKET_inputs/seed_scan.npy`.  The path
is embedded in the generated configs; `rk.refine` loads it at startup instead of
re-scanning (saves ~9 model-forward calls per run).  `--n_seeds_to_scan` (default 9).

**B-factor note**: `coordinates_boltz2.py` now uses Boltz-2's direct `bfactor_module`
prediction when available (`model.predict_bfactor=True`, which is the case for
`boltz2_conf.ckpt`).  The module outputs a histogram over B ∈ [0, 100] Å²; the
expected value is taken via softmax and broadcast from tokens to atoms.  If the
checkpoint does not include a bfactor module, the code falls back to the
pLDDT→pseudo-B conversion used in the original AF2-ROCKET.

**Learning-rate note**: `lr=1e-3` overshoots even on the first Adam step
(debug T2: single step at `lr=1e-3` gives ΔLLG=−5 even on full reflections).
`lr=1e-4` is stable: no collapse, monotone improvement for ~20 iterations.
`rk.preprocess` defaults to `lr=1e-4` with `smooth_stage_epochs=80`
(decay starts at iter 20, lr decays from 1e-4 → 1e-5).

**Smooth stage formula fix**: the LR decay formula used `decay_rate^iteration`
(global index) instead of `decay_rate^stage_step` (steps into the smooth stage).
This caused the lr to drop to near-target in a single step and w_L2 to go
negative at the end of the stage.  Fixed: `stage_step = iteration - stage_start`.

Phase-2 config differences vs Phase-1:

| Field | Phase 1 | Phase 2 |
|---|---|---|
| `algorithm.iterations` | 100 | 500 |
| `algorithm.optimization.additive_learning_rate` | **0.0001** | **0.00001** |
| `algorithm.optimization.multiplicative_learning_rate` | **0.0001** | **0.00001** |
| `algorithm.optimization.l2_weight` | 1e-7 | 0.0 |
| `algorithm.optimization.smooth_stage_epochs` | 80 | null |
| `algorithm.optimization.batch_sub_ratio` | 1.0 | 1.0 |
| `data.min_resolution` | 3.0 Å | null (all) |
| `execution.num_of_runs` | 3 | 1 |
| `paths.starting_bias` | — | `best_w_pair_*.pt` |
| `paths.starting_weights` | — | `best_b_pair_*.pt` |

---

## Optimization details

### LR scheduling (smooth stage)

Phase 1 applies smooth-stage LR decay: the last `smooth_stage_epochs` (default
80) iterations decay the learning rate from `lr = 0.0001` down to
`phase2_final_lr = 0.00001`.  With `iterations=100` the smooth stage starts at
iter 20, right after typical LLG peak, preventing post-peak drift.

The decay formula is `lr × decay_rate^stage_step` where
`stage_step = iteration − (iterations − smooth_stage_epochs)`.  Using the
global `iteration` index was a bug that caused the lr to drop almost
immediately to the target value (instead of linearly over `smooth_stage_epochs`)
and made the L2 weight go negative near the end of the stage.

### Seed pre-scan

Before the main optimisation loops, `refinement_boltz2.py` scans
`max(num_of_runs × 3, 6)` diffusion seeds with an identity bias (no gradient).
It selects the `num_of_runs` seeds that give the highest initial LLG and uses
those for the optimisation runs.  This matters because seed-to-seed LLG variance
(~370 units with identity bias) exceeds the optimisation signal (~150 units of
genuine improvement), so starting from the best seeds is critical.

Scan results are saved to `{out_dir}/seed_scan.npy` as `(LLG, seed)` pairs.

### Rwork interpretation

ROCKET's Rwork formula uses complex Fcalc vs real FEFF with DOBS² weighting:

```
R = Σ DOBS² |FEFF − Fc_complex| / Σ DOBS² FEFF
```

This encodes **both amplitude and phase error**.  Random model → R ≈ 1.41.
Perfect model → R ≈ 0.  A starting model with R ≈ 1.27 corresponds to ~78°
average phase error, which is a plausible starting point for Phaser MR + Boltz-2.
This is NOT the same as conventional crystallographic Rwork (random ≈ 0.83).

## Python API

```python
import pickle
from rocket import run_boltz2_xray_refinement

feats = pickle.load(open("ROCKET_inputs/feats_boltz2.pkl", "rb"))
run_boltz2_xray_refinement("ROCKET_config_phase1_boltz2.yaml", feats)
```

`run_refinement` (used by `rk.refine`) also accepts the config path directly
and will auto-load feats from `config.boltz2.feats_path` if `feats=None`.
