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

For Boltz-2, the learnable bias acts on the trunk **pair representation z**:

```
z_biased = w_pair ⊙ z  +  b_pair
```

`w_pair` and `b_pair` are both `[1, N_tokens, N_tokens, 128]`.  They are
initialised to ones and zeros respectively, so at iteration 0 the model
produces its unmodified prediction.  Gradients flow back through K steps of
reverse diffusion, through the diffusion conditioning module, through `z_biased`,
and into `w_pair` and `b_pair`.

Everything else — trunk weights, diffusion network weights, MSA module — is
frozen.

---

## Code structure

| File | Purpose |
|---|---|
| `rocket/boltz2_wrapper.py` | `Boltz2PairBias` — injects the pair bias, runs the trunk recycling loop, implements truncated-backprop diffusion sampling |
| `rocket/refinement_boltz2.py` | Full refinement loop + `prepare_boltz2_feats` helper |
| `rocket/coordinates_boltz2.py` | Atom extraction from Boltz-2 output + Kabsch alignment |
| `rocket/refinement_config.py` | `Boltz2Config` fields; `gen_config_phase2` handles Boltz-2 bias filenames |
| `rocket/scripts/run_preprocess.py` | `rk.preprocess --model boltz2` — unified preprocessing (predict + MR + feats + configs) |
| `rocket/scripts/run_prep_boltz2.py` | `rk.prep_boltz2` — standalone feats + config generation (for use with an existing PDB) |
| `rocket/scripts/run_refine.py` | `rk.refine` — dispatches to Boltz-2 backend; loads feats from config automatically |
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

### Workflow A — Unified (recommended, no AlphaFold2 dependency)

`rk.preprocess --model boltz2` runs Boltz-2 prediction, Phaser MR,
featurization, and config generation in a single command.

**Expected input layout in the working directory:**
```
{file_id}_fasta/{file_id}.fasta      # protein FASTA
{file_id}_data/*.mtz                 # diffraction data (MTZ)
```

**Step 0 — Generate MSA (optional but recommended)**

If you already have an a3m file from a previous AF2 run, skip this.
Otherwise generate one via the ColabFold API (~2–5 min, no local DB needed):

```bash
rk.generate_msa \
  --fasta 1lj5_fasta/1lj5.fasta \
  --file_id 1lj5 \
  --output_dir alignments/
# outputs: alignments/1lj5/bfd_uniclust_hits.a3m
#          alignments/1lj5/mgnify_hits.a3m
#          alignments/1lj5/1lj5.a3m   ← merged, use this for --a3m_path
```

**Step 1 — Preprocess (GPU node)**

```bash
rk.preprocess \
  --file_id 1lj5 \
  --method xray \
  --output_dir ./1lj5_processed \
  --model boltz2 \
  --boltz2_cache_dir /data/dust/group/it/crystalsfirst/dev/shehry/boltz_cache \
  --a3m_path alignments/1lj5/bfd_uniclust_hits.a3m
```

`--a3m_path` activates the Boltz-2 MSA module (`z = z + msa_module(...)`)
during refinement — recommended for better gradient signal.

What this does internally:
1. Parses FASTA → writes `{file_id}_boltz_input.yaml` (includes `msa:` key if `--a3m_path` given)
2. Runs `boltz predict` → `{file_id}_boltz2_unrelaxed.pdb`
3. Runs Phaser MR (phasertng) → `{file_id}-MRed.pdb`
4. Superposes Boltz-2 prediction onto MR model → `ROCKET_inputs/{file_id}-pred-aligned.pdb`
5. Generates `ROCKET_inputs/{file_id}-Edata.mtz` (normalised structure factors)
6. Runs `prepare_boltz2_feats` → `ROCKET_inputs/feats_boltz2.pkl`
7. Writes `ROCKET_config_phase1_boltz2.yaml` and `ROCKET_config_phase2_boltz2.yaml`

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

The feats path is embedded in the config by `rk.preprocess`; no `--feats` flag
needed.  To use a different feats file explicitly: `--feats /path/to/feats.pkl`.

Phase 1 runs `num_of_runs` independent traces (default 3) with different
diffusion seeds, selects the best by LLG, and saves:

```
1lj5_processed/ROCKET_outputs/<uuid>/phase1_boltz2_1lj5/
  best_model_A_42.pdb       # best structure
  best_w_pair_A_42.pt       # multiplicative bias  [1, N, N, 128]
  best_b_pair_A_42.pt       # additive bias        [1, N, N, 128]
  NEG_LLG_it_A.npy, rwork_it_A.npy, rfree_it_A.npy, …
```

**Step 3 — Phase 2 refinement (GPU node)**

```bash
rk.refine 1lj5_processed/ROCKET_config_phase2_boltz2.yaml
```

Phase 2 warm-starts from the Phase-1 best `w_pair`/`b_pair`, uses a lower
learning rate, no L2 regularisation, and includes all resolution shells.

---

### Workflow B — With an existing aligned PDB

If you already have a `{file_id}-pred-aligned.pdb` from a previous AF2 run and
want to add Boltz-2 refinement, use `rk.prep_boltz2` for the featurization step
(CPU only — no GPU required):

```bash
# Step 1 — standard AF2 preprocessing (GPU)
rk.preprocess \
  --file_id 1lj5 \
  --method xray \
  --output_dir ./1lj5_processed \
  --precomputed_alignment_dir alignments/ \
  --max_recycling_iters 20 \
  --use_deepspeed_evoformer_attention

# Step 2 — generate Boltz-2 feats + configs (CPU, login node OK)
rk.prep_boltz2 \
  --output_dir ./1lj5_processed \
  --file_id 1lj5 \
  --cache_dir /data/dust/group/it/crystalsfirst/dev/shehry/boltz_cache \
  --a3m_path alignments/1lj5/bfd_uniclust_hits.a3m

# Steps 3–4 — same rk.refine commands as Workflow A
```

`rk.prep_boltz2` key options:
```
--a3m_path               path to a3m MSA file (recommended)
--truncated_backprop_steps   5    # K: steps with grad in reverse diffusion
--sampling_steps           200    # total reverse-diffusion steps
--recycling_steps            3    # trunk recycling iterations
--num_of_runs                3    # Phase-1 independent traces
--phase1_iterations        100
--phase2_iterations        500
```

---

## Config reference

Boltz-2-specific fields in the generated YAML:

```yaml
execution:
  model: boltz2              # selects Boltz-2 backend

boltz2:
  boltz2_checkpoint_path: /path/to/boltz2_conf.ckpt
  feats_path: /path/to/feats_boltz2.pkl   # auto-loaded by rk.refine
  truncated_backprop_steps: 5             # K: grad steps in diffusion
  boltz2_recycling_steps: 3              # trunk recycling iterations
  boltz2_num_sampling_steps: 200         # total diffusion steps T
```

Phase-2 config differences vs Phase-1:

| Field | Phase 1 | Phase 2 |
|---|---|---|
| `algorithm.iterations` | 100 | 500 |
| `algorithm.optimization.additive_learning_rate` | 0.05 | 0.0001 |
| `algorithm.optimization.l2_weight` | 1e-7 | 0.0 |
| `algorithm.optimization.smooth_stage_epochs` | 50 | null |
| `data.min_resolution` | 3.0 Å | null (all) |
| `execution.num_of_runs` | 3 | 1 |
| `paths.starting_bias` | — | `best_w_pair_*.pt` |
| `paths.starting_weights` | — | `best_b_pair_*.pt` |

---

## Python API

```python
import pickle
from rocket import run_boltz2_xray_refinement

feats = pickle.load(open("ROCKET_inputs/feats_boltz2.pkl", "rb"))
run_boltz2_xray_refinement("ROCKET_config_phase1_boltz2.yaml", feats)
```

`run_refinement` (used by `rk.refine`) also accepts the config path directly
and will auto-load feats from `config.boltz2.feats_path` if `feats=None`.
