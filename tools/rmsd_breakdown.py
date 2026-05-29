#!/usr/bin/env python3
"""
rmsd_breakdown.py  MODEL.pdb  REF.pdb

Paper-comparable RMSD against a deposited structure, PLUS a per-residue
breakdown that separates a well-fit core from a few flexible outliers — the
information a single RMSD number hides.

Headline (gemmi.calculate_superposition, sequence-aligned, no trimming):
  * Global (all-atom) RMSD  -> paper "Global RMSD"
  * Ca RMSD                 -> paper "Ca RMSD"

Breakdown (independent Biopython global alignment + Kabsch on Ca, no trimming;
cross-checks the gemmi Ca number):
  * median per-residue Ca deviation, % < 1 A, % < 2 A, % > 2 A
  * worst residue (id + deviation)
  * core Ca RMSD excluding > 2 A outliers (shows how good the ordered core is)

No outlier trimming is applied to the reported RMSDs; the "core" line is
diagnostic only.
"""
import sys

import gemmi
import numpy as np
from Bio.Align import PairwiseAligner


def one_letter(resname):
    info = gemmi.find_tabulated_residue(resname)
    if info and info.one_letter_code.isalpha():
        return info.one_letter_code.upper()
    return "X"


def get_ca(structure):
    """First chain's polymer: one-letter seq, Ca coords, residue ids."""
    pol = structure[0][0].get_polymer()
    seq, coords, ids = [], [], []
    for res in pol:
        ca = None
        for at in res:
            if at.name == "CA":
                ca = at
                break
        if ca is None:
            continue
        seq.append(one_letter(res.name))
        coords.append([ca.pos.x, ca.pos.y, ca.pos.z])
        ids.append(f"{res.name}{res.seqid.num}")
    return "".join(seq), np.array(coords), ids


def kabsch(P, Q):
    """Rotate/translate P onto Q (Kabsch). Returns transformed P."""
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (R @ Pc.T).T + Q.mean(0)


def main():
    mdl = gemmi.read_structure(sys.argv[1])
    ref = gemmi.read_structure(sys.argv[2])

    # ---- headline: gemmi superposition (standard, paper-comparable) --------
    pm = mdl[0][0].get_polymer()
    pr = ref[0][0].get_polymer()
    pt = pr.check_polymer_type()
    g_all = gemmi.calculate_superposition(pr, pm, pt, gemmi.SupSelect.All)
    g_ca = gemmi.calculate_superposition(pr, pm, pt, gemmi.SupSelect.CaP)
    print(f"    Global (all-atom) RMSD = {g_all.rmsd:.3f} A  ({g_all.count} atoms)   <- paper 'Global RMSD'")
    print(f"    Ca RMSD                = {g_ca.rmsd:.3f} A  ({g_ca.count} Ca)     <- paper 'Ca RMSD'")

    # ---- per-residue breakdown: Biopython alignment + Kabsch ---------------
    sr, Cr, Ir = get_ca(ref)
    sm, Cm, Im = get_ca(mdl)
    if len(sr) < 3 or len(sm) < 3:
        print("    (per-residue breakdown skipped: too few Ca)")
        return

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5
    aln = aligner.align(sr, sm)[0]

    pairs = []
    for (r0, r1), (m0, m1) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(r1 - r0):
            pairs.append((r0 + k, m0 + k))
    if not pairs:
        print("    (per-residue breakdown skipped: no aligned pairs)")
        return

    ident = 100.0 * sum(1 for ri, mi in pairs if sr[ri] == sm[mi]) / len(pairs)
    Pr = np.array([Cr[ri] for ri, _ in pairs])
    Pm = np.array([Cm[mi] for _, mi in pairs])
    Pm_fit = kabsch(Pm, Pr)
    d = np.linalg.norm(Pm_fit - Pr, axis=1)

    rmsd = float(np.sqrt((d ** 2).mean()))
    med = float(np.median(d))
    f1 = 100.0 * (d < 1).mean()
    f2 = 100.0 * (d < 2).mean()
    g2 = 100.0 * (d > 2).mean()
    w = int(np.argmax(d))
    core = d[d <= 2]
    core_rmsd = float(np.sqrt((core ** 2).mean())) if core.size else float("nan")

    print(f"    --- per-residue Ca breakdown ({len(pairs)} aligned pairs, {ident:.0f}% identity, no trim) ---")
    print(f"      Ca RMSD (cross-check) = {rmsd:.3f} A      median = {med:.2f} A")
    print(f"      within 1 A = {f1:.0f}%    within 2 A = {f2:.0f}%    beyond 2 A = {g2:.0f}%")
    print(f"      worst residue: {Ir[pairs[w][0]]} = {d[w]:.2f} A")
    print(f"      core RMSD (excl. >2 A outliers) = {core_rmsd:.3f} A   <- ordered-core quality (diagnostic)")


if __name__ == "__main__":
    main()
