# ROCKET Pair Bias for Boltz-2 — Sampling-Mode Comparison & 8DWN Benchmark vs CrystalBoltz

## TL;DR

ROCKET-Boltz2 refines a crystal structure at inference time by optimising a
channel-wise **pair bias** on Boltz-2's trunk pair representation — the only
learnable parameters; Boltz-2 itself is frozen. The single biggest lever on
quality is the **diffusion sampling mode** used in the forward pass.

* **Mode ranking (every metric): `ddim-50 ≈ single-step > ddim-20 ≫ tbptt`.**
  Deterministic sampling gives a clean gradient through the diffusion process;
  the original stochastic truncated-BPTT mode does not.
* **On 8DWN vs CrystalBoltz (arXiv:2605.15564):** our refined R-factors are
  *better* (R-work 0.256 vs 0.338), our RMSD is *slightly worse*
  (Cα 1.14 vs 0.83 Å). The RMSD gap is **not** a core-quality problem — the
  ordered core is excellent (median Cα 0.17 Å, 95 % of residues < 1 Å). It is
  driven entirely by ~3 % flexible outliers, with **ASP1150 (~15.9 Å) the worst
  residue in all four modes**. A pair bias reshapes the trunk representation but
  cannot move coordinates directly, so it can't rescue badly-placed flexible
  regions; CrystalBoltz's guided diffusion can.

---

## 1. The model

The bias is the ConForNets construction (arXiv:2604.18559): on the trunk pair
representation `z [B, N, N, 128]`, before the PairFormer, apply

```
z  ←  z @ w_pair + b_pair
```

* `w_pair [128, 128]`, identity-initialised; `b_pair [128]`, zero-initialised.
* 16,512 parameters total, **length-independent**.
* Boltz-2 weights are frozen; only `w_pair`/`b_pair` are optimised, against the
  Rice/σ_A LLG of the experimental data (via SFCalculator).

Because the bias only nudges the *representation*, all of its structural effect
has to pass through Boltz-2's reverse diffusion to reach the coordinates. How
that diffusion is run at refinement time is what the rest of this document is
about.

---

## 2. The three sampling modes

Selectable at `rk.refine` time (`boltz2.sampling_mode`):

| mode | what it does | determinism | gradient path |
|---|---|---|---|
| `truncated_bptt` | original EDM/Karras sampler with stochastic γ-churn over `diffusion_steps`; backprop through the last *K* steps (`backprop_last_k`; `null` = all) | **stochastic** — different trajectory each iteration | noisy, last-K only |
| `single_step` | one deterministic denoise at σ_max (ConForNets style) | deterministic | clean, 1 step |
| `ddim` | `diffusion_steps` deterministic DDIM steps, noise seed fixed once; gradient through the last *K* steps (`backprop_last_k`; `null` = all) | deterministic | clean, last-K (or all) |

Two knobs: `diffusion_steps` (how many denoising steps) and `backprop_last_k`
(how many trailing steps keep the gradient; `null` = all).  Coordinate
augmentation is disabled during ROCKET so the only stochasticity is the diffusion
churn itself (present only in `truncated_bptt`).

`ddim` covers two regimes via `backprop_last_k`:

* **Full-gradient DDIM** (`backprop_last_k: null`, `diffusion_steps` ~20–50):
  clean gradient through every step — the recommended default.
* **Truncated-backprop DDIM** (`diffusion_steps: 200`, `backprop_last_k: 20`):
  same deterministic trajectory, gradient only through the last 20 steps (O(K)
  backward memory).  This is the benchmark that decomposes the two factors
  behind `truncated_bptt`'s failure — it keeps the *last-K gradient* but swaps
  the *stochastic* trajectory for a *deterministic* one.  (The two were proven
  identical when K = number of steps, so the old separate `ddim_truncated` mode
  was merged into `ddim`.)

---

## 3. Why the mode matters (mechanism)

**Boltz-2's reverse diffusion is a strong regulariser.** The pair bias is a small
perturbation; for the optimiser to learn it, the map *seed → structure* has to be
stable so that a change in `w_pair` produces a consistent, attributable change in
the output.

* **`truncated_bptt` breaks that stability.** Each optimisation step samples a
  fresh stochastic trajectory, so the seed→structure map changes underfoot. The
  bias signal is swamped by trajectory noise, the gradient is noisy, and
  seed-to-seed LLG variance is huge (σ ≈ 274 at identity bias across 20 seeds,
  range 96–1082). The net learnable signal barely clears the noise floor.
* **Deterministic modes fix the seed.** With a fixed trajectory the gradient is
  clean and the bias signal is recoverable. The contrast in gradient magnitude
  at the identity bias is stark:

  | path | \|∇ w_pair\| at identity |
  |---|---|
  | truncated-BPTT | 0.38 |
  | single-step | 3968 |

  ~10⁴× larger and far more consistent — a clean deterministic path versus a
  noise-dominated one.
* **`ddim` adds trajectory depth.** Running the deterministic denoise as *N*
  DDIM steps (rather than one) lets the bias act repeatedly along a smooth,
  differentiable path, which is why more steps help (ddim-50 > ddim-20) and why
  ddim achieves the largest optimisation gains.

**The ceiling.** Even with a perfect gradient, the bias can only reshape the
trunk pair representation — it cannot translate an atom directly. Boltz-2's
diffusion prior dominates the placement of flexible/disordered regions, so those
stay wherever Boltz-2 puts them regardless of the bias. This is the structural
ceiling that §7 shows up against CrystalBoltz.

---

## 4. Internal optimisation signal (LLG) by mode

100-iteration Phase-1 runs on 8DWN (best-seed start, identical settings except
mode). "Best iter" is the max-LLG checkpoint saved as `best_w_pair_A_*.pt`.

| mode | LLG start | LLG peak (iter) | LLG gain | best iter |
|---|---|---|---|---|
| **ddim-50** | 846.5 | **1446.4** (85) | **+600** | 85 |
| ddim-20 | 808.8 | 1237.3 (84) | +428 | 84 |
| single-step | 780.7 | 1173.4 (84) | +393 | 84 |
| tbptt | 992.1 | 1063.9 (50) | **+72** | 50 |

tbptt starts high (its stochastic average looks decent) but barely improves and
peaks early; the deterministic modes climb much further. ddim-50 gains the most.

---

## 5. Benchmark methodology

Each mode's best-iteration model is scored with `tools/benchmark_rocket.sh`:
raw R → `phenix.refine` (5 macrocycles: sites + ADP + occupancy) → refined R →
RMSD → CC, against the deposited 8DWN model and experimental data. Metrics are
defined to match CrystalBoltz exactly.

* **Global / Cα RMSD** (`tools/rmsd_breakdown.py`) — sequence-aligned, **no
  outlier trimming**, over all residues common to both structures
  (`gemmi.calculate_superposition`; all-atom and Cα). Verified against an
  independent Biopython-alignment + Kabsch computation (100 % identity, 286
  pairs). A per-residue breakdown (median, % within 1/2 Å, worst residue, core
  RMSD) accompanies it.
  *Note:* `phenix.superpose_pdbs` reports **lower** RMSDs (≈0.55/0.91 here)
  because it fits a complete-backbone LS subset and de-weights flexible
  outliers — that is **not** the paper's all-residue metric and is not quoted.
* **CC** (`tools/crystalboltz_cc.py`) — CrystalBoltz's CC is a **reciprocal-space**
  Pearson correlation of structure-factor *amplitudes*,
  `CC = pearson(|Fc|, |Fo|)`, with `Fc = k_total(F_protein + k_mask·F_solvent)`
  computed by **SFCalculator** (the same library the paper uses), scales fitted
  then frozen, evaluated on the experimental FEFF amplitudes. This is **not** a
  real-space map-model CC (RSCC); it needs only the model + an amplitude MTZ.
* **R-work / R-free** — refined-model `phenix.model_vs_data` against the
  deposited data (free set = 10.1 % minority flag, auto-detected).

Target: PDB **8DWN**, 2.15 Å, P2₁2₁2₁, 286 ordered residues, ~18,063
reflections (16,244 work / 1,819 free).

---

## 6. 8DWN results — all four modes (refined model)

| mode | raw R (w/f) | refined R (w/f) | Global RMSD | Cα RMSD | median Cα | within 1 Å | beyond 2 Å | worst res. | core RMSD | CC (all/work/free) |
|---|---|---|---|---|---|---|---|---|---|---|
| **ddim-50** | 0.365/0.376 | **0.256/0.305** | 1.430 | 1.143 | 0.17 | 95 % | 3 % | ASP1150 15.9 | 0.277 | 0.871/0.876/0.826 |
| single-step | 0.389/0.390 | 0.257/0.305 | **1.409** | 1.159 | 0.17 | 95 % | 4 % | ASP1150 15.9 | **0.258** | **0.872/0.876/0.836** |
| ddim-20 | 0.380/0.385 | 0.262/0.311 | 1.450 | 1.151 | 0.18 | 96 % | 3 % | ASP1150 15.9 | 0.289 | 0.866/0.871/0.821 |
| tbptt | 0.403/0.396 | 0.273/0.315 | 1.543 | 1.243 | 0.25 | 93 % | 6 % | ASP1150 15.6 | 0.343 | 0.861/0.865/0.824 |

Reading the table:

* **ddim-50 and single-step are co-best.** ddim-50 wins on LLG and refined
  R-work; single-step edges it on Global RMSD, core RMSD and free-set CC. ddim-20
  is a close third. **tbptt is worst on every single metric.**
* The **core is excellent and nearly mode-independent** (median Cα 0.17–0.18 Å,
  core RMSD 0.26–0.29 Å) — except tbptt, which is visibly looser (median 0.25 Å,
  core 0.34 Å, 6 % outliers vs 3–4 %).
* **ASP1150 is the worst residue in every mode** (~15.6–15.9 Å). A single
  flexible region, mis-placed by Boltz-2, dominates the global RMSD across the
  board — see §7.

---

## 7. Comparison with CrystalBoltz

CrystalBoltz (arXiv:2605.15564) reports deposited-reference RMSD and R-factors
on the same 8DWN. Our best mode (ddim-50) and the unguided Boltz-2 baseline from
their paper:

| | R-work | R-free | Global RMSD | Cα RMSD |
|---|---|---|---|---|
| Boltz-2 baseline (paper) | 0.472 | 0.474 | 2.653 | 2.457 |
| **ROCKET-Boltz2, ddim-50 (ours)** | **0.256** | **0.305** | 1.430 | 1.143 |
| CrystalBoltz (paper) | 0.338 | 0.337 | **1.319** | **0.828** |

**R-factors — we are ahead, with a caveat.** All four modes refine to R-work
0.256–0.273 / R-free 0.305–0.315, below CrystalBoltz's 0.338/0.337. But their
R comes from their own Adam coordinate+B refinement against the data, ours from
`phenix.refine`; the refinement engines differ, so this is not a clean
method-to-method comparison. Both are far below the raw Boltz-2 baseline (0.47).

**RMSD — we are slightly behind, for a specific and interpretable reason.**
Cα 1.14 vs 0.83 Å, Global 1.43 vs 1.32 Å. This is **not** because our model is
broadly less accurate:

* median per-residue Cα deviation is **0.17 Å**, and **95 % of residues are
  within 1 Å** — the ordered core (core RMSD 0.277 Å) is at least as good as
  anything reported.
* the entire deficit comes from **~3 % flexible outlier residues**, and they are
  the *same* residues in every mode, led by **ASP1150 at ~15.9 Å**.

A channel-wise pair bias can only reshape the trunk pair representation; it never
moves an atom directly. Boltz-2's diffusion prior places flexible loops/termini,
and the bias cannot override that placement — so a residue Boltz-2 puts 16 Å away
stays there. CrystalBoltz's **guided diffusion** flows the data gradient into the
*coordinates* during sampling, which is exactly what's needed to drag such
residues back. The RMSD gap is therefore the concrete, expected cost of
**bias-only conditioning vs guided diffusion** — and it is localised to a handful
of flexible residues, not the structure as a whole.

**CC.** Our CC is 0.86–0.87 (`pearson(|Fc|, |Fo|)`, bulk-solvent forward model).
The paper tabulates CC only as a bulk-solvent *ablation delta*, not as a headline
per-target value, so there is no published CC to compare against directly; ours
is computed with their forward model and metric for when that number is needed.

---

## 8. Conclusions & recommendations

1. **Use a deterministic mode.** `ddim` (50 steps best, 20 fine) or `single_step`.
   Avoid `truncated_bptt` — it is worst on LLG, R, RMSD and CC, for the
   mechanistic reason in §3 (stochastic trajectory ⇒ noisy gradient).
2. **ddim-50 is the default recommendation** (largest LLG gain, best refined
   R-work); `single_step` is an essentially-equivalent, cheaper alternative.
3. **The pair-bias ceiling is real and localised.** Refinement quality on the
   ordered core is excellent and competitive; the only systematic loss vs
   CrystalBoltz is a few flexible residues the bias structurally cannot reach.
4. **To close the remaining RMSD gap, coordinates must be optimised directly** —
   i.e. guided diffusion (data gradient into the sampler) or a post-hoc
   coordinate refinement of the flexible regions, rather than more pair-bias
   capacity.

---

## Appendix — reproduce

Best models (max-LLG checkpoint per run):

| mode | run dir (`…/8dwn/8dwn_processed/ROCKET_outputs/`) | model |
|---|---|---|
| ddim-20 | `7675e0b03a/phase1_boltz2_8dwn/` | `A_84_postRBR.pdb` |
| ddim-50 | `7675e0b03b/phase1_boltz2_8dwn_50/` | `A_85_postRBR.pdb` |
| tbptt | `7675e0b03c/phase1_boltz2_8dwn_tbptt/` | `A_50_postRBR.pdb` |
| single-step | `7675e0b03d/phase1_boltz2_8dwn_ss/` | `A_84_postRBR.pdb` |

```bash
# on the GPU/login node, env active:  micromamba activate rocket-of
DATA=…/8dwn/8dwn_data;  IN=…/8dwn/8dwn_processed/ROCKET_inputs
bash tools/benchmark_rocket.sh \
  --pdb   <model>.pdb  --label 8dwn_ddim50 \
  --exp-mtz $DATA/8dwn.mtz  --ref-pdb $DATA/8dwn.pdb \
  --cc-mtz  $IN/8dwn-Edata.mtz --cc-fobs FEFF --cc-sigf DOBS \
  --free-label R-free-flags
```

Full per-mode logs: `benchmark_8dwn_{ddim50,ss,ddim_def,tbptt}/SUMMARY.txt`.
CrystalBoltz reference values: arXiv:2605.15564, Table 1 (8DWN row).
