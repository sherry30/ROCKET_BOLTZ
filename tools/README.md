# tools/

Standalone helper scripts for the ROCKET-Boltz2 workflow (not part of the
installed `rk.*` CLI). Run them directly.

| Script | Purpose |
|---|---|
| `download_pdb_data.sh` | Fetch a PDB entry's model + experimental structure factors from RCSB and convert to MTZ, laid out as `<id>/<id>_data/` + `<id>/<id>_fasta/` for `rk.preprocess`. `--with-pdb-redo` also pulls PDB-REDO for extended analysis. |
| `benchmark_rocket.sh` | Score a refined model against experimental data: raw R-factors → `phenix.refine` → refined R-factors → map-model CC, RMSD (+ per-residue breakdown), and the CrystalBoltz CC vs a ground-truth model. `--skip-refine` gives raw R + RMSD + CC only. |
| `rmsd_breakdown.py` | Paper-comparable Global (all-atom) + Cα RMSD via `gemmi`, plus a per-residue core-vs-outlier breakdown (median Cα deviation, % within 1/2 Å, worst residue, core RMSD) via Biopython alignment + Kabsch. Sequence-aligned, no trimming. |
| `crystalboltz_cc.py` | The CrystalBoltz (arXiv:2605.15564) "CC": Pearson(\|Fc\|, \|Fo\|) in reciprocal space with the bulk-solvent forward model (`SFC_Torch`, fitted scales). Auto-detects the amplitude column (prefers FEFF; French-Wilson converts intensities); reports overall / work / free. |
| `bfactor_reset_by_rscc.py` | Reset per-residue B-factors from a `phenix.map_correlations` per-residue CC log. |

All require the `rocket-of` env (`phenix` on PATH + `gemmi`/`biopython`/`SFC_Torch`),
e.g. `micromamba activate rocket-of`. `benchmark_rocket.sh` calls
`rmsd_breakdown.py` and `crystalboltz_cc.py` from this directory.
See `../docs/BOLTZ2_IMPLEMENTATION.md` for the full pipeline.

### CrystalBoltz CC vs map-model CC — two different "CC"s

`benchmark_rocket.sh` can report two correlation coefficients; they are not the
same thing:

* **CrystalBoltz CC** (`crystalboltz_cc.py`, step 7) — a *reciprocal-space*
  Pearson correlation between calculated and experimental structure-factor
  **amplitudes**, `CC = pearson(|Fc|, |Fo|)`, with `Fc = k_total(F_protein +
  k_mask·F_solvent)`. Needs only the model + an amplitude MTZ. This is the
  paper's metric.
* **Map-model CC / RSCC** (`phenix.map_correlations`, step 5) — a *real-space*
  correlation between the model-derived map and a deposited 2Fo−Fc map. Needs a
  `--ref-mtz` with map coefficients. The paper does **not** use this.
