#!/bin/bash
# ============================================================================
# download_pdb_data.sh  —  fetch crystallographic data for a PDB entry, laid out
#                          for the ROCKET preprocessing pipeline.
#
# Usage:
#   ./download_pdb_data.sh --id <PDBID> --outdir <dir> [options]
#
# Output layout (matches what rk.preprocess expects):
#   <outdir>/<id>/
#     <id>_data/
#       <id>.mtz        EXPERIMENTAL data: FOBS, SIGFOBS, R-free-flags
#                       (converted from RCSB -sf.cif; this is the ONLY .mtz here
#                        so `<id>_data/*.mtz` globs cleanly)
#       <id>.pdb        deposited model (PDB format; absent for some huge entries)
#       <id>.cif        deposited model (mmCIF)
#       <id>-sf.cif     raw deposited structure factors (mmCIF)
#       *.log           conversion logs
#     <id>_fasta/
#       <id>.fasta      sequence(s) from RCSB
#     <id>_pdbredo/     (ONLY with --with-pdb-redo; kept out of <id>_data so it
#                        does not pollute the *.mtz glob)
#       <id>_final.pdb / .cif / .mtz
#
# This mirrors the CrystalBoltz data setup (arxiv 2605.15564): RCSB only,
# DEPOSITED experimental structure factors, DEPOSITED free-flag set, no PDB-REDO.
# The <id>.mtz is the original EXPERIMENTAL data (not refined) — exactly the kind
# of input rk.preprocess used for 1lj5.
#
# Requires: curl always; phenix on PATH for the sf.cif->mtz conversion
#   (e.g. `micromamba activate rocket-of` on the GPU node).
# ============================================================================
set -uo pipefail

usage() {
cat <<EOF
Usage: $0 --id <PDBID> --outdir <dir> [options]

Required:
  --id        STR    4-character PDB ID (e.g. 1l63)
  --outdir    PATH   parent dir; data goes to <outdir>/<id>/<id>_data etc.

Options:
  --with-pdb-redo      ALSO download PDB-REDO model+MTZ into <id>_pdbredo/
                       (extended analysis only; NOT used for CrystalBoltz comparison)
  --no-convert         download RCSB sf.cif but do NOT convert to MTZ
  --rfree-fraction F   if the deposited data has NO real free set, generate one
                       with this fraction (default 0.05; 0 = never generate)
  -h | --help          show this help

Example (CrystalBoltz / pipeline layout):
  $0 --id 1l63 --outdir ./data
  # -> ./data/1l63/1l63_data/1l63.mtz , ./data/1l63/1l63_fasta/1l63.fasta
EOF
exit 1
}

ID=""; OUTDIR=""; DO_REDO=0; DO_CONVERT=1; RFREE_FRAC="0.05"
while [ $# -gt 0 ]; do
    case "$1" in
        --id)             ID="$2"; shift 2;;
        --outdir)         OUTDIR="$2"; shift 2;;
        --with-pdb-redo)  DO_REDO=1; shift;;
        --no-convert)     DO_CONVERT=0; shift;;
        --rfree-fraction) RFREE_FRAC="$2"; shift 2;;
        -h|--help)        usage;;
        *) echo "Unknown option: $1"; usage;;
    esac
done
[ -z "$ID" ]     && { echo "ERROR: --id required"; usage; }
[ -z "$OUTDIR" ] && { echo "ERROR: --outdir required"; usage; }

ID_UP=$(echo "$ID" | tr '[:lower:]' '[:upper:]')
ID_LO=$(echo "$ID" | tr '[:upper:]' '[:lower:]')

ROOT="$OUTDIR/$ID_LO"
DATA_DIR="$ROOT/${ID_LO}_data"
FASTA_DIR="$ROOT/${ID_LO}_fasta"
REDO_DIR="$ROOT/${ID_LO}_pdbredo"
mkdir -p "$DATA_DIR" "$FASTA_DIR"

# fetch URL -> abs path; keep only on HTTP 200 + non-empty. ($3="nol" => no -L)
fetch() {  # $1 url   $2 outfile   [$3 nol]
    local url="$1" out="$2" code lflag="-L"
    [ "${3:-}" = "nol" ] && lflag=""
    code=$(curl -s $lflag -o "$out" -w "%{http_code}" "$url" 2>/dev/null)
    if [ "$code" = "200" ] && [ -s "$out" ]; then
        printf "  [ok ] %-28s (%s bytes)\n" "$(basename "$out")" "$(stat -c%s "$out")"
        return 0
    fi
    printf "  [--] %-28s (HTTP %s)\n" "$(basename "$out")" "$code"; rm -f "$out"; return 1
}

echo "============================================================"
echo " Downloading $ID_UP  ->  $ROOT"
echo "============================================================"

# ---- FASTA (RCSB) ----------------------------------------------------------
echo ""
echo "[FASTA] -> ${ID_LO}_fasta/${ID_LO}.fasta"
if fetch "https://www.rcsb.org/fasta/entry/${ID_UP}" "$FASTA_DIR/${ID_LO}.fasta" nol; then
    if ! head -c1 "$FASTA_DIR/${ID_LO}.fasta" | grep -q ">"; then
        echo "  WARNING: downloaded FASTA does not start with '>' (not FASTA?) — check it."
    fi
    nchains=$(grep -c "^>" "$FASTA_DIR/${ID_LO}.fasta" 2>/dev/null || echo 0)
    echo "  sequences/chains in FASTA: $nchains"
fi

# ---- model + structure factors (RCSB) -> <id>_data/ ------------------------
echo ""
echo "[RCSB] model + structure factors -> ${ID_LO}_data/"
fetch "https://files.rcsb.org/download/${ID_UP}.cif" "$DATA_DIR/${ID_LO}.cif" || true
fetch "https://files.rcsb.org/download/${ID_UP}.pdb" "$DATA_DIR/${ID_LO}.pdb" || \
    echo "       (no PDB-format model — large entry; use ${ID_LO}.cif)"
SF_CIF=""
if fetch "https://files.rcsb.org/download/${ID_UP}-sf.cif" "$DATA_DIR/${ID_LO}-sf.cif"; then
    SF_CIF="$DATA_DIR/${ID_LO}-sf.cif"
else
    echo "       WARNING: no structure factors on RCSB for $ID_UP"
fi

# ---- convert sf.cif -> single experimental MTZ in <id>_data/ ---------------
FINAL_MTZ="$DATA_DIR/${ID_LO}.mtz"
if [ "$DO_CONVERT" -eq 1 ] && [ -n "$SF_CIF" ]; then
    echo ""
    echo "[convert] ${ID_LO}-sf.cif -> ${ID_LO}_data/${ID_LO}.mtz"
    if command -v phenix.cif_as_mtz >/dev/null; then
        ( cd "$DATA_DIR"
          phenix.cif_as_mtz "${ID_LO}-sf.cif" \
              --output_file_name="${ID_LO}.mtz" \
              --merge --remove_systematic_absences --ignore_bad_sigmas \
              > convert_cif_as_mtz.log 2>&1
        )
        if [ -s "$FINAL_MTZ" ]; then
            echo "  [ok ] ${ID_LO}.mtz"
            ( cd "$DATA_DIR" && phenix.mtz.dump "${ID_LO}.mtz" > mtz_dump.log 2>&1 || true )
            # usable deposited free set => R-free column has a REAL split (min<max)
            read FMIN FMAX < <(awk 'tolower($1) ~ /free/ && $4 ~ /^[0-9.]+$/ {print $4,$5; exit}' "$DATA_DIR/mtz_dump.log")
            if [ -n "${FMIN:-}" ] && [ "$FMIN" != "$FMAX" ]; then
                echo "  R-free flags: DEPOSITED set (real split) — use as-is"
            else
                echo "  R-free flags: NO deposited free set (e.g. pre-1994 entry)"
                if [ "$RFREE_FRAC" != "0" ]; then
                    echo "                generating fresh set (fraction=$RFREE_FRAC) …"
                    ( cd "$DATA_DIR"
                      phenix.reflection_file_converter "${ID_LO}.mtz" \
                          --generate_r_free_flags --r_free_flags_fraction="$RFREE_FRAC" \
                          --non_anomalous --mtz="${ID_LO}_withfree.mtz" > generate_rfree.log 2>&1 \
                      && mv -f "${ID_LO}_withfree.mtz" "${ID_LO}.mtz"
                    )
                    [ -s "$FINAL_MTZ" ] && echo "                [ok ] ${ID_LO}.mtz now has a GENERATED free set"
                    echo "                NOTE: generated R-free is NOT comparable to CrystalBoltz"
                    echo "                for this entry — compare R-work only."
                fi
            fi
        else
            echo "  WARNING: conversion failed (see ${ID_LO}_data/convert_cif_as_mtz.log)"
        fi
    else
        echo "  SKIP: phenix not on PATH — activate it and run:"
        echo "    cd $DATA_DIR && phenix.cif_as_mtz ${ID_LO}-sf.cif --output_file_name=${ID_LO}.mtz --merge"
    fi
fi

# ---- PDB-REDO (opt-in) -> <id>_pdbredo/ ------------------------------------
if [ "$DO_REDO" -eq 1 ]; then
    echo ""
    echo "[PDB-REDO] -> ${ID_LO}_pdbredo/   (extended analysis only)"
    mkdir -p "$REDO_DIR"
    R="https://pdb-redo.eu/db/${ID_LO}"
    fetch "$R/${ID_LO}_final.pdb" "$REDO_DIR/${ID_LO}_final.pdb" || true
    fetch "$R/${ID_LO}_final.cif" "$REDO_DIR/${ID_LO}_final.cif" || true
    fetch "$R/${ID_LO}_final.mtz" "$REDO_DIR/${ID_LO}_final.mtz" || \
        echo "       (no PDB-REDO entry for $ID_LO)"
fi

# ---- summary ---------------------------------------------------------------
echo ""
echo "============================================================"
echo " DONE — $ROOT"
echo "============================================================"
echo " Pipeline inputs (rk.preprocess working dir = $ROOT):"
echo "   ${ID_LO}_fasta/${ID_LO}.fasta     protein sequence"
echo "   ${ID_LO}_data/${ID_LO}.mtz        EXPERIMENTAL data (FOBS/SIGFOBS/R-free-flags)"
echo ""
echo " The .mtz is the original experimental data (NOT refined) — same kind of"
echo " input as the 1lj5 run.  Run preprocessing from $ROOT, e.g.:"
echo "   cd $ROOT"
echo "   rk.preprocess --file_id ${ID_LO} --method xray --model boltz2 \\"
echo "       --output_dir ./${ID_LO}_processed --boltz2_cache_dir <cache> \\"
echo "       --precomputed_alignment_dir <alignments>   # contains <alignments>/${ID_LO}/${ID_LO}.a3m"
echo ""
echo " CrystalBoltz-style benchmark of your ROCKET output:"
echo "   ./benchmark_rocket.sh --pdb <out.pdb> --label ${ID_LO} \\"
echo "     --exp-mtz $DATA_DIR/${ID_LO}.mtz --free-label \"R-free-flags\" \\"
echo "     --ref-pdb $DATA_DIR/${ID_LO}.pdb"
echo "   (NO --skip-refine: CrystalBoltz reports POST-refinement R-factors, so"
echo "    compare against the REFINED R from the benchmark, not the raw one.)"
