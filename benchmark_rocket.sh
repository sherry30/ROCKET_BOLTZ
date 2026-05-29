#!/bin/bash
# ============================================================================
# benchmark_rocket.sh  —  one-shot crystallographic benchmark of a model
#
# Generic across proteins: all data paths and parameters are passed as flags.
#
# Metric order (important — see notes at bottom):
#   1. RAW R-factors        — model as-is, native MR frame (direct ROCKET effect)
#   2. phenix.refine        — N macrocycles, native frame
#   3. REFINED R-factors    — practical end-quality
#   4. superpose -> ref_pdb + RMSD to ground truth        (skipped if no --ref-pdb)
#   5. map-model CC (RSCC) vs ref_mtz ground-truth map    (skipped if no --ref-mtz)
#
# R-factors are measured in the NATIVE frame (steps 1-3); superposition onto the
# deposited structure happens LAST and only for closeness-to-truth metrics.
#
# Requires phenix on PATH (e.g. `micromamba activate rocket-of` on the GPU node).
# ============================================================================
set -euo pipefail

usage() {
cat <<EOF
Usage: $0 --pdb <model.pdb> --label <name> --exp-mtz <data.mtz> [options]

Required:
  --pdb           PATH    input model to benchmark
  --label         NAME    short label (output goes to benchmark_<label>/)
  --exp-mtz       PATH    experimental data MTZ (for R-factors / refinement)

Ground-truth (optional — enables RMSD + RSCC):
  --ref-pdb       PATH    deposited/ground-truth PDB (for superpose + RMSD)
  --ref-mtz       PATH    ground-truth map-coeff MTZ (for map-model CC / RSCC)
  --ref-map-labels STR    map coeff labels in ref-mtz       [default: FWT,PHWT]
  --resolution    FLOAT   resolution for map-model CC (A)   [default: 2.0]

Refinement / data options:
  --free-label    STR     R-free flags label in exp-mtz     [default: R-free-flags]
  --macrocycles   INT     phenix.refine macrocycles         [default: 5]
  --strategy      STR     refine strategy
                          [default: individual_sites+individual_adp+occupancies]
  --outdir        PATH    output directory             [default: benchmark_<label>]
  --skip-refine           only compute RAW R-factors (steps 2-5 skipped)
  -h | --help             show this help

Example:
  $0 \\
    --pdb   /path/A_99_postRBR.pdb \\
    --label ddim10_best \\
    --exp-mtz /path/1lj5-tng_withrfree.mtz \\
    --ref-pdb /path/pdb_redo/1lj5_final.pdb \\
    --ref-mtz /path/pdb_redo/1lj5_final.mtz \\
    --resolution 1.8
EOF
exit 1
}

# ---- defaults -------------------------------------------------------------
PDB=""; LABEL=""; EXP_MTZ=""
REF_PDB=""; REF_MTZ=""; REF_MAP_LABELS="FWT,PHWT"; RESOLUTION="2.0"
FREE_LABEL="R-free-flags"; MACROCYCLES="5"
STRATEGY="individual_sites+individual_adp+occupancies"
OUTDIR=""; SKIP_REFINE=0

# ---- parse flags ----------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --pdb)            PDB="$2"; shift 2;;
        --label)          LABEL="$2"; shift 2;;
        --exp-mtz)        EXP_MTZ="$2"; shift 2;;
        --ref-pdb)        REF_PDB="$2"; shift 2;;
        --ref-mtz)        REF_MTZ="$2"; shift 2;;
        --ref-map-labels) REF_MAP_LABELS="$2"; shift 2;;
        --resolution)     RESOLUTION="$2"; shift 2;;
        --free-label)     FREE_LABEL="$2"; shift 2;;
        --macrocycles)    MACROCYCLES="$2"; shift 2;;
        --strategy)       STRATEGY="$2"; shift 2;;
        --outdir)         OUTDIR="$2"; shift 2;;
        --skip-refine)    SKIP_REFINE=1; shift;;
        -h|--help)        usage;;
        *) echo "Unknown option: $1"; usage;;
    esac
done

# ---- validate -------------------------------------------------------------
[ -z "$PDB" ]     && { echo "ERROR: --pdb required";     usage; }
[ -z "$LABEL" ]   && { echo "ERROR: --label required";   usage; }
[ -z "$EXP_MTZ" ] && { echo "ERROR: --exp-mtz required"; usage; }
[ -f "$PDB" ]     || { echo "ERROR: pdb not found: $PDB"; exit 1; }
[ -f "$EXP_MTZ" ] || { echo "ERROR: exp-mtz not found: $EXP_MTZ"; exit 1; }
command -v phenix.model_vs_data >/dev/null || {
    echo "ERROR: phenix not on PATH. Run 'micromamba activate rocket-of' first."; exit 1; }

PDB=$(readlink -f "$PDB")
EXP_MTZ=$(readlink -f "$EXP_MTZ")
[ -n "$REF_PDB" ] && REF_PDB=$(readlink -f "$REF_PDB")
[ -n "$REF_MTZ" ] && REF_MTZ=$(readlink -f "$REF_MTZ")
[ -z "$OUTDIR" ] && OUTDIR="benchmark_${LABEL}"

mkdir -p "$OUTDIR"; cd "$OUTDIR"
cp "$PDB" input.pdb
SUMMARY="SUMMARY.txt"; : > "$SUMMARY"
log()  { echo "$@" | tee -a "$SUMMARY"; }
grab() { grep -E "^  r_work|^  r_free" "$1" | head -2 | sed 's/^/    /'; }

log "============================================================"
log " BENCHMARK: $LABEL"
log " input:   $PDB"
log " exp-mtz: $EXP_MTZ"
log " ref-pdb: ${REF_PDB:-<none>}"
log " ref-mtz: ${REF_MTZ:-<none>}"
log " date:    $(date)"
log "============================================================"

# ---- 1. RAW R-factors -----------------------------------------------------
log ""
log "[1] RAW R-factors (model as-is, native frame)"
phenix.model_vs_data input.pdb "$EXP_MTZ" \
    r_free_flags_label="$FREE_LABEL" > 1_raw_mvd.log 2>&1 || true
grab 1_raw_mvd.log | tee -a "$SUMMARY"

if [ "$SKIP_REFINE" -eq 1 ]; then
    log ""; log "[2-5] skipped (--skip-refine)"; log "DONE. Logs in $(pwd)"; exit 0
fi

# ---- 2. phenix.refine -----------------------------------------------------
log ""
log "[2] phenix.refine — $MACROCYCLES macrocycles ($STRATEGY)"
phenix.refine input.pdb "$EXP_MTZ" \
    data_manager.miller_array.labels.name="$FREE_LABEL" \
    refinement.main.number_of_macro_cycles="$MACROCYCLES" \
    refinement.refine.strategy="$STRATEGY" \
    output.prefix=refined output.serial=1 \
    allow_polymer_cross_special_position=True \
    --overwrite > 2_refine.log 2>&1 || true
REFINED_PDB=$(ls -t refined_*.pdb 2>/dev/null | head -1 || true)
log "    refined model: ${REFINED_PDB:-NONE (see 2_refine.log)}"

# ---- 3. REFINED R-factors -------------------------------------------------
log ""
log "[3] REFINED R-factors"
if [ -n "${REFINED_PDB:-}" ]; then
    phenix.model_vs_data "$REFINED_PDB" "$EXP_MTZ" \
        r_free_flags_label="$FREE_LABEL" > 3_refined_mvd.log 2>&1 || true
    grab 3_refined_mvd.log | tee -a "$SUMMARY"
else
    log "    (refine failed)"
fi

# ---- 4. superpose onto ground truth + RMSD --------------------------------
log ""
if [ -n "$REF_PDB" ] && [ -n "${REFINED_PDB:-}" ]; then
    log "[4] Superpose refined model onto ground truth + RMSD"
    phenix.superpose_pdbs "$REF_PDB" "$REFINED_PDB" \
        file_name=superposed.pdb > 4_superpose.log 2>&1 || true
    grep -iE "rmsd|r.m.s" 4_superpose.log | head -4 | sed 's/^/    /' | tee -a "$SUMMARY"
else
    log "[4] skipped (no --ref-pdb or no refined model)"
fi

# ---- 5. map-model CC (RSCC) vs ground-truth map ---------------------------
log ""
if [ -n "$REF_MTZ" ] && [ -f superposed.pdb ]; then
    log "[5] Map-model CC (RSCC) vs ref map ($REF_MAP_LABELS @ ${RESOLUTION} A)"
    phenix.map_correlations superposed.pdb \
        input_files.map_coeffs_1="$REF_MTZ" \
        input_files.map_coeffs_labels_1="$REF_MAP_LABELS" \
        map_model_cc.resolution="$RESOLUTION" > 5_mapcc.log 2>&1 || true
    # Extract the whole CC block (overall + per-chain + main/side chain values).
    # The main/side-chain CC values sit on the line *after* their column header,
    # so a simple line-grep misses them — extract the contiguous block instead.
    awk '/Map-model CC \(overall\)/{f=1} /Per residue:/{exit} f' 5_mapcc.log \
        | sed 's/^/    /' | tee -a "$SUMMARY"
else
    log "[5] skipped (no --ref-mtz or no superposed model)"
fi

log ""
log "============================================================"
log " DONE. Full logs + SUMMARY.txt in $(pwd)"
log "============================================================"
