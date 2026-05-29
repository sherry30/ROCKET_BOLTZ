#!/bin/bash
# ============================================================================
# benchmark_rocket.sh  —  one-shot crystallographic benchmark of a model
#
# Generic across proteins: all data paths and parameters are passed as flags.
#
# Metric order:
#   1. RAW R-factors     — model as-is, NATIVE frame (direct ROCKET effect)
#   2. superpose RAW -> ref_pdb (BEFORE refine) + pre-refine RMSD   [needs --ref-pdb]
#   3. phenix.refine     — N macrocycles (in ref frame if superposed)
#   4. REFINED R-factors — practical end-quality
#   5. map-model CC (RSCC) vs ref_mtz map — refined model, in-frame [needs --ref-mtz]
#   6. refined-vs-truth RMSD + per-residue breakdown                [needs --ref-pdb]
#   7. CrystalBoltz CC   — Pearson(|Fc|,|Fo|), bulk-solvent (refined model)
#
# Why superpose BEFORE refine (when a ground-truth pdb is given):
#   map-model CC compares to a phased MAP, which is tied to the ground-truth
#   crystallographic frame.  Superposing first puts the model in that frame;
#   refinement against the same-crystal data then keeps it there and snaps atoms
#   into the density -> accurate in-frame CC.  Refining first (native frame) then
#   overlaying afterwards leaves a rigid-fit residual that under-measures the CC.
#   R-factors are frame-invariant, so raw R is still measured in the native frame.
#   Superposition is a pure rigid placement — it leaks no structural info, so the
#   comparison stays fair as long as the SAME pipeline is used for every model.
#
# Requires phenix on PATH (e.g. `micromamba activate rocket-of` on the GPU node).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

CrystalBoltz-style CC (Pearson |Fc| vs |Fo|, reciprocal space, bulk-solvent):
  --cc-mtz        PATH    amplitude MTZ for the CC      [default: --exp-mtz]
  --cc-fobs       STR     |Fo| amplitude label              [default: auto]
  --cc-sigf       STR     sigma label                       [default: auto]
                          (auto prefers FEFF; French-Wilson if only intensities)
  --no-cc                 skip the CrystalBoltz CC metric

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
CC_MTZ=""; CC_FOBS=""; CC_SIGF=""; NO_CC=0

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
        --cc-mtz)         CC_MTZ="$2"; shift 2;;
        --cc-fobs)        CC_FOBS="$2"; shift 2;;
        --cc-sigf)        CC_SIGF="$2"; shift 2;;
        --no-cc)          NO_CC=1; shift;;
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
[ -z "$CC_MTZ" ] && CC_MTZ="$EXP_MTZ"
[ -n "$CC_MTZ" ] && CC_MTZ=$(readlink -f "$CC_MTZ")
[ -z "$OUTDIR" ] && OUTDIR="benchmark_${LABEL}"

mkdir -p "$OUTDIR"; cd "$OUTDIR"
cp "$PDB" input.pdb
SUMMARY="SUMMARY.txt"; : > "$SUMMARY"
log()  { echo "$@" | tee -a "$SUMMARY"; }
grab() { grep -E "^  r_work|^  r_free" "$1" | head -2 | sed 's/^/    /'; }

# Paper-comparable RMSD (Global all-atom + Cα) + per-residue core-vs-outlier
# breakdown.  Sequence-aligned, NO outlier trimming — matches how papers report
# "Global RMSD" / "Cα RMSD".  NOTE: phenix.superpose_pdbs's "between fixed and
# moving" / "all matching atoms" numbers run LOWER because it fits on
# complete-backbone residues over an LS subset, de-weighting a few flexible
# outliers — those are NOT the paper's all-residue metric, so quote these.
paper_rmsd() {  # $1 model.pdb   $2 ref.pdb
    python3 "$SCRIPT_DIR/rmsd_breakdown.py" "$1" "$2" 2>/dev/null \
        || echo "    (RMSD breakdown failed — needs gemmi+biopython env)"
}

# CrystalBoltz CC: Pearson(|Fc|,|Fo|) with bulk-solvent forward model (SFC_Torch).
# Reciprocal-space amplitude correlation, NOT a real-space map CC.
paper_cc() {  # $1 model.pdb
    [ "$NO_CC" -eq 1 ] && return 0
    local args=(); [ -n "$CC_FOBS" ] && args+=(--fobs "$CC_FOBS")
    [ -n "$CC_SIGF" ] && args+=(--sigf "$CC_SIGF")
    [ -n "$FREE_LABEL" ] && args+=(--free "$FREE_LABEL")
    python3 "$SCRIPT_DIR/crystalboltz_cc.py" "$1" "$CC_MTZ" "${args[@]}" 2>/dev/null \
        || echo "    (CC failed — needs SFC_Torch env / amplitude MTZ)"
}

log "============================================================"
log " BENCHMARK: $LABEL"
log " input:   $PDB"
log " exp-mtz: $EXP_MTZ"
log " ref-pdb: ${REF_PDB:-<none>}"
log " ref-mtz: ${REF_MTZ:-<none>}"
[ "$NO_CC" -eq 0 ] && log " cc-mtz:  $CC_MTZ"
log " date:    $(date)"
log "============================================================"

# ---- 1. RAW R-factors -----------------------------------------------------
log ""
log "[1] RAW R-factors (model as-is, native frame)"
phenix.model_vs_data input.pdb "$EXP_MTZ" \
    r_free_flags_label="$FREE_LABEL" > 1_raw_mvd.log 2>&1 || true
grab 1_raw_mvd.log | tee -a "$SUMMARY"

if [ "$SKIP_REFINE" -eq 1 ]; then
    # Raw method output only: R-factors (above) + RMSD to the deposited model,
    # NO refinement, no map CC.
    # NOTE: this is the RAW prediction.  CrystalBoltz reports POST-refinement
    # numbers (it optimises coords+B against the deposited Fobs), so for an
    # apples-to-apples comparison run WITHOUT --skip-refine and compare the
    # REFINED R-factors.  Use --skip-refine only to inspect the raw output.
    log ""
    if [ -n "$REF_PDB" ]; then
        log "[RMSD] model vs reference (no refinement)"
        phenix.superpose_pdbs "$REF_PDB" input.pdb \
            file_name=superposed_norefine.pdb > skip_superpose.log 2>&1 || true
        grep -iE "rmsd|r.m.s" skip_superpose.log | head -4 | sed 's/^/    /' | tee -a "$SUMMARY"
        paper_rmsd input.pdb "$REF_PDB" | tee -a "$SUMMARY"
    else
        log "[RMSD] skipped (no --ref-pdb)"
    fi
    log ""
    if [ "$NO_CC" -eq 0 ]; then
        log "[CC] CrystalBoltz CC (raw model)"
        paper_cc input.pdb | tee -a "$SUMMARY"
    fi
    log ""; log "[refine/map-CC] skipped (--skip-refine)"
    log "DONE. Logs in $(pwd)"; exit 0
fi

# ---- 2. superpose RAW model onto ground truth (BEFORE refine) -------------
#
# Why superpose first: map-model CC (step 5) compares the model to the pdb_redo
# MAP, which carries phases tied to a specific crystallographic frame/origin.
# Superposing first puts the model in that frame; refinement against the
# (same-crystal) experimental data then keeps it there and snaps atoms precisely
# into the density -> accurate, in-frame map CC.  Refining first in the native MR
# frame and superposing afterwards leaves a rigid-overlay residual that
# artificially lowers the map CC.  R-factors are frame-invariant, so this does
# not affect them (raw R is measured above in the native frame).
log ""
REFINE_INPUT=input.pdb
if [ -n "$REF_PDB" ]; then
    log "[2] Superpose RAW model onto ground truth (-> ref frame) + pre-refine RMSD"
    phenix.superpose_pdbs "$REF_PDB" input.pdb \
        file_name=superposed_start.pdb > 2_superpose_start.log 2>&1 || true
    grep -iE "rmsd|r.m.s" 2_superpose_start.log | head -4 | sed 's/^/    /' | tee -a "$SUMMARY"
    [ -f superposed_start.pdb ] && REFINE_INPUT=superposed_start.pdb
else
    log "[2] (no --ref-pdb; refining in native frame, RSCC will be skipped)"
fi

# ---- 3. phenix.refine -----------------------------------------------------
log ""
log "[3] phenix.refine — $MACROCYCLES macrocycles ($STRATEGY)"
phenix.refine "$REFINE_INPUT" "$EXP_MTZ" \
    data_manager.miller_array.labels.name="$FREE_LABEL" \
    refinement.main.number_of_macro_cycles="$MACROCYCLES" \
    refinement.refine.strategy="$STRATEGY" \
    output.prefix=refined output.serial=1 \
    allow_polymer_cross_special_position=True \
    --overwrite > 3_refine.log 2>&1 || true
REFINED_PDB=$(ls -t refined_*.pdb 2>/dev/null | head -1 || true)
log "    refined model: ${REFINED_PDB:-NONE (see 3_refine.log)}"

# ---- 4. REFINED R-factors -------------------------------------------------
log ""
log "[4] REFINED R-factors"
if [ -n "${REFINED_PDB:-}" ]; then
    phenix.model_vs_data "$REFINED_PDB" "$EXP_MTZ" \
        r_free_flags_label="$FREE_LABEL" > 4_refined_mvd.log 2>&1 || true
    grab 4_refined_mvd.log | tee -a "$SUMMARY"
else
    log "    (refine failed)"
fi

# ---- 5. map-model CC (RSCC) vs ground-truth map ---------------------------
#
# The refined model is already in the ground-truth frame (superposed before
# refine), so map_correlations runs on it directly — no post-refine overlay.
log ""
if [ -n "$REF_MTZ" ] && [ -n "$REF_PDB" ] && [ -n "${REFINED_PDB:-}" ]; then
    log "[5] Map-model CC (RSCC) vs ref map ($REF_MAP_LABELS @ ${RESOLUTION} A)"
    phenix.map_correlations "$REFINED_PDB" \
        input_files.map_coeffs_1="$REF_MTZ" \
        input_files.map_coeffs_labels_1="$REF_MAP_LABELS" \
        map_model_cc.resolution="$RESOLUTION" > 5_mapcc.log 2>&1 || true
    awk '/Map-model CC \(overall\)/{f=1} /Per residue:/{exit} f' 5_mapcc.log \
        | sed 's/^/    /' | tee -a "$SUMMARY"
else
    log "[5] skipped (need both --ref-pdb and --ref-mtz, plus a refined model)"
fi

# ---- 6. refined-vs-truth RMSD (post-refine accuracy) ----------------------
# phenix.superpose_pdbs's reported RMSD runs LOW: it fits on complete-backbone
# residues over an LS subset, de-weighting a few flexible outliers.  paper_rmsd
# (no trimming, sequence-aligned) is the value to quote against papers: Global
# (all-atom) + Cα RMSD, plus a per-residue core-vs-outlier breakdown.
log ""
if [ -n "$REF_PDB" ] && [ -n "${REFINED_PDB:-}" ]; then
    log "[6] Refined model vs ground truth — RMSD (post-refine accuracy)"
    phenix.superpose_pdbs "$REF_PDB" "$REFINED_PDB" \
        file_name=refined_superposed.pdb > 6_superpose_refined.log 2>&1 || true
    grep -iE "rmsd|r.m.s" 6_superpose_refined.log | head -4 | sed 's/^/    /' | tee -a "$SUMMARY"
    paper_rmsd "$REFINED_PDB" "$REF_PDB" | tee -a "$SUMMARY"
else
    log "[6] skipped (no --ref-pdb or no refined model)"
fi

# ---- 7. CrystalBoltz CC (refined model) -----------------------------------
log ""
if [ "$NO_CC" -eq 0 ] && [ -n "${REFINED_PDB:-}" ]; then
    log "[7] CrystalBoltz CC — Pearson(|Fc|,|Fo|), bulk-solvent (refined model)"
    paper_cc "$REFINED_PDB" | tee -a "$SUMMARY"
elif [ "$NO_CC" -eq 0 ]; then
    log "[7] CC skipped (no refined model)"
fi

log ""
log "============================================================"
log " DONE. Full logs + SUMMARY.txt in $(pwd)"
log "============================================================"
