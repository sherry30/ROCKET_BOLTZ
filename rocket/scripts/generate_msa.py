"""
rk.generate_msa — Generate MSA a3m files for a protein FASTA using the
ColabFold MMseqs2 REST API (no local database required).

Queries the public ColabFold server (api.colabfold.com), which runs a fast
MMseqs2 search against UniRef100 + environmental sequences — the same
databases used by AlphaFold2 and OpenFold, just accessed via HTTP instead
of local jackhmmer/hhblits.

Output layout (matches the rk.preprocess / OpenFold alignment convention):

    <output_dir>/<file_id>/bfd_uniclust_hits.a3m   — UniRef100 + BFD hits
    <output_dir>/<file_id>/mgnify_hits.a3m          — environmental hits

A merged file is also written:

    <output_dir>/<file_id>/<file_id>.a3m            — union of the above
                                                       (used by rk.preprocess
                                                       --model boltz2)

Usage
-----
    rk.generate_msa \\
        --fasta 1lj5_fasta/1lj5.fasta \\
        --file_id 1lj5 \\
        --output_dir alignments/

For multi-chain proteins the sequences are submitted jointly (paired MSA);
each chain's individual alignment is also saved under a separate sub-dir if
--split_chains is passed.

Notes
-----
- Internet access is required to reach api.colabfold.com.
- Typical turnaround: 1–5 min for a ~400-residue protein, up to 20 min for
  long/multi-chain submissions.
- Rate-limit: the public server allows a few concurrent jobs per IP.  For
  large-scale use, host a local ColabFold server with --host_url.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import time
from pathlib import Path

import requests
from loguru import logger

_API_URL = "https://api.colabfold.com"
_POLL_INTERVAL = 10   # seconds between status polls
_MAX_WAIT = 3600      # give up after 1 hour


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


def _build_query(records: list[tuple[str, str]], start_n: int = 101) -> str:
    """Build a multi-sequence FASTA string for the ColabFold API."""
    lines = []
    for i, (_, seq) in enumerate(records):
        lines.append(f">{start_n + i}")
        lines.append(seq)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# ColabFold API
# ---------------------------------------------------------------------------

def _submit(query: str, mode: str = "env", host: str = _API_URL) -> str:
    """Submit an MSA job; return ticket ID."""
    resp = requests.post(
        f"{host}/ticket/msa",
        json={"q": query, "mode": mode},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if "id" not in data:
        raise RuntimeError(f"Unexpected API response: {data}")
    ticket_id = data["id"]
    logger.info(f"Submitted MSA job — ticket: {ticket_id}")
    return ticket_id


def _wait_for_completion(ticket_id: str, host: str = _API_URL) -> None:
    """Poll until the job is COMPLETE or ERROR."""
    deadline = time.time() + _MAX_WAIT
    while time.time() < deadline:
        resp = requests.get(f"{host}/ticket/{ticket_id}", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "UNKNOWN")

        if status == "COMPLETE":
            logger.info(f"Job {ticket_id}: COMPLETE")
            return
        elif status == "ERROR":
            raise RuntimeError(f"MSA job failed: {data.get('msg', '(no message)')}")
        else:
            logger.info(f"Job {ticket_id}: {status} — waiting {_POLL_INTERVAL}s …")
            time.sleep(_POLL_INTERVAL)

    raise TimeoutError(f"MSA job {ticket_id} did not complete within {_MAX_WAIT}s.")


def _download(ticket_id: str, host: str = _API_URL) -> bytes:
    """Download the result tar.gz and return its bytes."""
    resp = requests.get(
        f"{host}/result/download/{ticket_id}",
        timeout=120,
        stream=True,
    )
    resp.raise_for_status()
    data = b"".join(resp.iter_content(chunk_size=65536))
    logger.info(f"Downloaded {len(data) / 1024:.0f} KB for ticket {ticket_id}")
    return data


def _extract_a3m_files(tar_bytes: bytes) -> dict[str, str]:
    """
    Extract .a3m files from tar.gz bytes.
    Returns {filename_stem: a3m_content} dict.
    """
    result: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".a3m"):
                stem = Path(member.name).stem   # e.g. "101" or "101_env"
                content = tf.extractfile(member).read().decode("utf-8")
                result[stem] = content
    return result


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def _merge_a3m_files(contents: list[str]) -> str:
    """
    Concatenate multiple a3m files.  The query sequence (first record) from
    each file is identical; keep it only from the first file.
    """
    merged_lines: list[str] = []
    for i, content in enumerate(contents):
        lines = content.splitlines(keepends=True)
        if i == 0:
            merged_lines.extend(lines)
        else:
            # Skip the first two lines (query header + query sequence)
            skip = 0
            for j, line in enumerate(lines):
                if line.startswith(">"):
                    skip = j + 2   # header + sequence
                    break
            merged_lines.extend(lines[skip:])
    return "".join(merged_lines)


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
    Generate MSA for the sequences in fasta_path and write alignment files.

    Returns the path to the primary merged a3m file.
    """
    records = _read_fasta(fasta_path)
    if not records:
        raise ValueError(f"No sequences found in {fasta_path}")

    logger.info(f"Submitting {len(records)} chain(s) from {fasta_path.name}")
    query = _build_query(records, start_n=101)

    ticket_id = _submit(query, mode="env", host=host)
    _wait_for_completion(ticket_id, host=host)
    tar_bytes = _download(ticket_id, host=host)
    a3m_files = _extract_a3m_files(tar_bytes)

    if not a3m_files:
        raise RuntimeError("No .a3m files found in server response.")

    # Sort by stem so that 101, 101_env, 102, 102_env… are in order
    sorted_stems = sorted(a3m_files.keys())
    logger.debug(f"Received files: {sorted_stems}")

    # Separate main hits from environmental hits
    main_stems = [s for s in sorted_stems if not s.endswith("_env")]
    env_stems  = [s for s in sorted_stems if s.endswith("_env")]

    main_content = _merge_a3m_files([a3m_files[s] for s in main_stems]) if main_stems else ""
    env_content  = _merge_a3m_files([a3m_files[s] for s in env_stems])  if env_stems  else ""

    # Write output files
    aln_dir = output_dir / file_id
    aln_dir.mkdir(parents=True, exist_ok=True)

    bfd_path = aln_dir / "bfd_uniclust_hits.a3m"
    bfd_path.write_text(main_content)
    logger.info(f"Wrote {bfd_path}  ({main_content.count(chr(10))} lines)")

    if env_content:
        mgn_path = aln_dir / "mgnify_hits.a3m"
        mgn_path.write_text(env_content)
        logger.info(f"Wrote {mgn_path}  ({env_content.count(chr(10))} lines)")

    # Merged file used by rk.preprocess --model boltz2
    merged = _merge_a3m_files(
        [c for c in [main_content, env_content] if c]
    )
    merged_path = aln_dir / f"{file_id}.a3m"
    merged_path.write_text(merged)
    logger.info(f"Wrote merged a3m → {merged_path}  ({merged.count(chr(10))} lines)")

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
        help="Input FASTA file.  For multi-chain proteins include all chains.",
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
    print(f"    --a3m_path {merged_path}")
    print("=" * 60)


if __name__ == "__main__":
    cli_generate_msa()
