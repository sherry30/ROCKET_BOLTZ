"""
rk.generate_msa — Generate an MSA a3m file for a protein FASTA using the
ColabFold MMseqs2 REST API (no local database required).

This wraps Boltz-2's own MMseqs2 client (``boltz.data.msa.mmseqs2.run_mmseqs2``)
so the MSA is produced exactly as the Boltz-2 pipeline expects.  It queries the
public ColabFold server (api.colabfold.com), running a fast MMseqs2 search over
UniRef + environmental databases (BFD/MGnify/etc).

Output:

    <output_dir>/<file_id>/<file_id>.a3m   — combined UniRef + environmental MSA;
                                             auto-detected by rk.preprocess via
                                             --precomputed_alignment_dir <output_dir>
    <output_dir>/<file_id>/_raw/           — raw per-database a3m (reference only)

Only the single combined <file_id>.a3m sits at the top level.  This is correct
for BOTH backends: Boltz-2's resolver picks <file_id>.a3m, and the AlphaFold2/
OpenFold pipeline (which concatenates every top-level .a3m in the dir) reads
exactly one MSA — no duplicated sequences.  The raw per-database files are kept
under _raw/ so they don't get double-counted.

Note: ColabFold MMseqs2 does not produce template hits (pdb70_hits.hhr), so an
AF2-ROCKET run from this directory runs without templates.

Usage
-----
    rk.generate_msa \\
        --fasta 1lj5_fasta/1lj5.fasta \\
        --file_id 1lj5 \\
        --output_dir alignments/

Chain handling
--------------
This script is correct for proteins with a SINGLE distinct sequence — i.e.
monomers and homo-oligomers (multiple identical chains).  In that case the one
merged a3m is the right MSA for every (identical) protein chain, and
``rk.preprocess --model boltz2`` assigns it accordingly.

It does NOT yet handle HETERO-multimers (two or more *different* protein
sequences) correctly: the per-chain alignments are concatenated into a single
file and that same file is assigned to all chains, which corrupts the per-chain
MSA.  Proper per-chain a3m output + paired-MSA support is not implemented yet.
For now use this only on single-sequence targets.

Notes
-----
- Internet access is required to reach api.colabfold.com.
- Typical turnaround: 1–5 min for a ~400-residue protein.
- Rate-limit: the public server allows a few concurrent jobs per IP.  For
  large-scale use, host a local ColabFold server with --host_url.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from loguru import logger

_API_URL = "https://api.colabfold.com"


# ---------------------------------------------------------------------------
# FASTA helpers
# ---------------------------------------------------------------------------

def _read_fasta(path: Path) -> list[tuple[str, str]]:
    """Return list of (header, sequence) pairs from a FASTA file."""
    records: list[tuple[str, str]] = []
    header, seq_parts = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header = line[1:].split()[0]
                seq_parts = []
            elif line:
                seq_parts.append(line.upper())
    if header is not None:
        records.append((header, "".join(seq_parts)))
    return records


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def generate_msa(
    fasta_path: Path,
    file_id: str,
    output_dir: Path,
    host: str = _API_URL,
) -> Path:
    """
    Generate an MSA for the sequence(s) in fasta_path via Boltz-2's own
    ColabFold MMseqs2 client and write the a3m used by rk.preprocess --model boltz2.

    Returns the path to the merged a3m file.
    """
    # Boltz's maintained ColabFold client — correct API form-encoding, headers,
    # tar member names (uniref.a3m / bfd.mgnify30.metaeuk30.smag30.a3m), and the
    # exact combined-MSA format the Boltz-2 featurizer expects.
    from boltz.data.msa.mmseqs2 import run_mmseqs2  # noqa: PLC0415

    records = _read_fasta(fasta_path)
    if not records:
        raise ValueError(f"No sequences found in {fasta_path}")

    seqs = [seq for _, seq in records]
    distinct = list(dict.fromkeys(seqs))
    if len(distinct) > 1:
        logger.warning(
            f"{len(distinct)} DISTINCT protein sequences detected — hetero-multimer "
            "MSA is not handled correctly yet (each chain needs its own / paired MSA). "
            "Writing the first sequence's MSA only; results for the other chains will "
            "be wrong.  Use single-sequence targets for now."
        )

    aln_dir = output_dir / file_id
    aln_dir.mkdir(parents=True, exist_ok=True)
    work_prefix = str(aln_dir / "_mmseqs2")

    logger.info(
        f"Submitting {len(seqs)} sequence(s) from {fasta_path.name} to {host} "
        "(ColabFold MMseqs2; UniRef + environmental) …"
    )
    a3m_lines = run_mmseqs2(
        seqs[0] if len(seqs) == 1 else seqs,
        prefix=work_prefix,
        use_env=True,
        use_filter=True,
        use_pairing=False,
        host_url=host,
    )

    # run_mmseqs2 returns one combined (UniRef + env) a3m string per input seq
    merged = a3m_lines[0]
    merged_path = aln_dir / f"{file_id}.a3m"
    merged_path.write_text(merged)
    logger.info(
        f"Wrote merged a3m → {merged_path}  ({merged.count(chr(10))} lines, "
        f"{merged.count('>')} sequences)"
    )

    # Keep the per-database a3m files for reference, but in a _raw/ SUBdir — NOT
    # alongside <file_id>.a3m.  This matters for the AlphaFold2/OpenFold pipeline:
    # OpenFold's DataPipeline does `for f in os.listdir(alignment_dir): if .a3m`,
    # i.e. it concatenates EVERY top-level .a3m.  If the merged file and its raw
    # parts both sat at top level, OpenFold would read the same sequences twice.
    # A subdir is ignored by os.listdir's extension check and by the Boltz-2
    # resolver's top-level glob, so both pipelines see exactly one MSA.
    raw_dir = aln_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)
    work_dir = Path(f"{work_prefix}_env")
    for raw in ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"):
        src = work_dir / raw
        if src.is_file():
            shutil.copy(src, raw_dir / raw)
    shutil.rmtree(work_dir, ignore_errors=True)

    return merged_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli_generate_msa() -> None:
    parser = argparse.ArgumentParser(
        prog="rk.generate_msa",
        description=(
            "Generate MSA a3m files from a protein FASTA via the "
            "ColabFold MMseqs2 API (no local database required)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Input FASTA file.  Use single-sequence targets (monomer or "
             "homo-oligomer); hetero-multimers are not handled correctly yet.",
    )
    parser.add_argument(
        "--file_id",
        required=True,
        help="Dataset identifier (e.g. '1lj5').  Determines the output sub-directory name.",
    )
    parser.add_argument(
        "--output_dir",
        default="alignments",
        help="Parent directory for alignment outputs.  A sub-directory <file_id>/ is created.",
    )
    parser.add_argument(
        "--host_url",
        default=_API_URL,
        help=(
            "ColabFold API server URL.  Override to use a local ColabFold "
            "server: e.g. http://localhost:8080"
        ),
    )
    args = parser.parse_args()

    fasta_path  = Path(args.fasta).resolve()
    output_dir  = Path(args.output_dir).resolve()

    if not fasta_path.exists():
        logger.error(f"FASTA file not found: {fasta_path}")
        sys.exit(1)

    merged_path = generate_msa(
        fasta_path=fasta_path,
        file_id=args.file_id,
        output_dir=output_dir,
        host=args.host_url,
    )

    print()
    print("=" * 60)
    print("MSA generation complete.")
    print()
    print(f"  Alignment directory:  {output_dir / args.file_id}/")
    print(f"  Merged a3m for Boltz-2 rk.preprocess:")
    print(f"    {merged_path}")
    print()
    print("Next steps (Boltz-2 ROCKET):")
    print()
    print(f"  rk.preprocess \\")
    print(f"    --file_id {args.file_id} \\")
    print(f"    --method xray \\")
    print(f"    --output_dir ./{args.file_id}_processed \\")
    print(f"    --model boltz2 \\")
    print(f"    --boltz2_cache_dir /path/to/boltz_cache \\")
    print(f"    --precomputed_alignment_dir {output_dir}")
    print()
    print(f"  (rk.preprocess auto-detects {output_dir}/{args.file_id}/{args.file_id}.a3m)")
    print("=" * 60)


if __name__ == "__main__":
    cli_generate_msa()
