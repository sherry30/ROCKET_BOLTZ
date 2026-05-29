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
| `rocket/refinement_boltz2.py` | Full refinement loop (incl. inline seed pre-scan) + `prepare_boltz2_feats` |
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
# outputs: alignments/1lj5/1lj5.a3m   ← merged MSA, auto-detected by --precomputed_alignment_dir
```

**Step 1 — Preprocess (GPU node)**

```bash
rk.preprocess \
  --file_id 1lj5 \
  --method xray \
  --output_dir ./1lj5_processed \
  --model boltz2 \
  --boltz2_cache_dir /data/dust/group/it/crystalsfirst/dev/shehry/boltz_cache \
  --precomputed_alignment_dir alignments/ \
  --sampling_mode ddim
```

`--precomputed_alignment_dir` (default `alignments/`) activates the Boltz-2 MSA
module.  rk.preprocess looks in `<alignment_dir>/<file_id>/` and auto-detects the
a3m, preferring the merged `<file_id>.a3m` written by `rk.generate_msa` (then the
AF2/OpenFold names `bfd_uniclust_hits.a3m` etc.).  The selected MSA is featurized
into `feats_boltz2.pkl` at preprocess time, so `rk.refine` picks it up
automatically.  `--sampling_mode` selects the diffusion gradient mode
(`ddim` recommended; see below).

What this does internally:
1. Parses FASTA → writes `{file_id}_boltz_input.yaml`
2. Runs `boltz predict` → `{file_id}_boltz2_unrelaxed.pdb`
3. Runs Phaser MR (phasertng) → `{file_id}-MRed.pdb`
4. Superposes Boltz-2 prediction onto MR model → `ROCKET_inputs/{file_id}-pred-aligned.pdb`
5. Generates `ROCKET_inputs/{file_id}-Edata.mtz` (normalised structure factors)
6. Runs `prepare_boltz2_feats` → `ROCKET_inputs/feats_boltz2.pkl`
7. Auto-detects the R-free **test-set value** from the Edata (the minority flag
   value) and writes it as `testset_value` in the config
8. Writes `ROCKET_config_phase1_boltz2.yaml` and `ROCKET_config_phase2_boltz2.yaml`

**R-free convention note**: programs disagree on which flag value is the test
set (CCP4 `FreeR_flag` uses 0, phenix `R-free-flags` uses 1).  SFcalculator
treats `R-free-flags == testset_value` as the held-out set, so the wrong
`testset_value` silently holds out the *work* set (Rfree≈0 or computed on the
majority).  Preprocessing therefore detects it from the data — the test set is
the held-out minority — rather than assuming a convention.

Output layout:
```
1lj5_processed/
  ROCKET_inputs/
    1lj5-pred-aligned.pdb
    1lj5-Edata.mtz
    feats_boltz2.pkl
  ROCKET_config_phase1_boltz2.yaml
  ROCKET_config_phase2_boltz2.yaml
```

**Step 2 — Phase 1 refinement (GPU node)**

```bash
rk.refine 1lj5_processed/ROCKET_config_phase1_boltz2.yaml
```

The feats path is embedded in the config — no extra flags needed.  At the start
of refinement a **diffusion seed pre-scan** runs inline (in the same sampling
mode as refinement; see "Seed pre-scan" below), then `num_of_runs` independent
traces run from the best seeds, saving:

```
1lj5_processed/ROCKET_outputs/<uuid>/phase1_boltz2_1lj5/
  best_model_A_18.pdb       # best structure
  best_w_pair_A_18.pt       # channel-wise weight matrix  [128, 128]
  best_b_pair_A_18.pt       # channel-wise bias vector    [128]
  NEG_LLG_it_A.npy, rwork_it_A.npy, rfree_it_A.npy, …
  seed_scan.npy             # scan results (LLG, seed) for this run
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
  feats_path: /path/to/feats_boltz2.pkl   # auto-loaded by rk.refine
  truncated_backprop_steps: 20            # K: grad steps for truncated_bptt mode
  boltz2_recycling_steps: 3               # trunk recycling iterations
  boltz2_num_sampling_steps: 200          # total diffusion steps T
  sampling_mode: ddim                     # truncated_bptt | single_step | ddim
  ddim_steps: 20                          # deterministic steps for ddim mode
```

**Sampling mode note**: `ddim` (deterministic N-step) gives the cleanest gradient
and the best real R-factor improvement; `single_step` (ConForNets-style) also
works; `truncated_bptt` (the original K-step stochastic mode) does not improve
real R-factors.  See `BOLTZ2_PAIR_BIAS_ANALYSIS.md` for the full comparison.

**Seed pre-scan**: runs **inline** at the start of `rk.refine`, in the same
sampling mode as refinement (see "Seed pre-scan" under Optimization details).
There is no precompute step or stored seed-scan file to keep in sync — the scan
and the refinement forward pass always agree by construction.

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
`max(num_of_runs × 3, 6)` diffusion seeds with an identity bias (no gradient) and
selects the `num_of_runs` seeds that give the highest initial LLG.  Seed-to-seed
LLG variance is large (σ ≈ 270 for TBPTT, ≈ 120 for DDIM), so starting from the
best seeds helps.

The scan runs **inline at the start of every `rk.refine`**, in the **same
sampling mode** as refinement.  This is deliberate: the seed → structure mapping
is mode-dependent (the seed that is best under full TBPTT sampling is not
necessarily best under DDIM), so a precomputed scan from a different mode would
select the wrong seeds.  Running it inline guarantees the scan and the refinement
forward pass always agree — there is no separate precompute step or stored
seed-scan file to keep in sync.

Scan results are saved to `{out_dir}/seed_scan.npy` as `(LLG, seed)` pairs for
the record.

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
