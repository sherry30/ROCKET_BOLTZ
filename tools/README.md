# tools/

Standalone helper scripts for the ROCKET-Boltz2 workflow (not part of the
installed `rk.*` CLI). Run them directly.

| Script | Purpose |
|---|---|
| `download_pdb_data.sh` | Fetch a PDB entry's model + experimental structure factors from RCSB and convert to MTZ, laid out as `<id>/<id>_data/` + `<id>/<id>_fasta/` for `rk.preprocess`. `--with-pdb-redo` also pulls PDB-REDO for extended analysis. |
| `benchmark_rocket.sh` | Score a refined model against experimental data: raw R-factors → `phenix.refine` → refined R-factors → map-model CC and RMSD vs a ground-truth model. `--skip-refine` gives raw R + RMSD only. |
| `bfactor_reset_by_rscc.py` | Reset per-residue B-factors from a `phenix.map_correlations` per-residue CC log. |

All three require `phenix` on PATH (e.g. `micromamba activate rocket-of`).
See `../docs/BOLTZ2_IMPLEMENTATION.md` for the full pipeline.
