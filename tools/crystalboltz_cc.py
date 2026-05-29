#!/usr/bin/env python3
"""
crystalboltz_cc.py  MODEL.pdb  DATA.mtz  [--fobs L] [--sigf L] [--free L] [--dmin F]

Reproduces the CrystalBoltz (arXiv:2605.15564) "CC" metric: the Pearson
correlation between calculated and experimental structure-factor *amplitudes*,
  CC = pearson(|F_calc|, |F_obs|),
where F_calc comes from the differentiable forward model WITH bulk solvent,
  F_calc = k_total * (F_protein + k_mask * F_solvent),
using SFCalculator (SFC_Torch) — the same library the paper uses. The
isotropic/anisotropic/solvent scales are fitted (get_scales_lbfgs) so the
calculated amplitudes best match the experimental ones, then frozen.

This is a RECIPROCAL-space amplitude correlation (over reflections), NOT a
real-space map-model CC. CC is reported overall and split into the working /
free reflection sets.

Experimental amplitudes:
  * If the MTZ has an amplitude column (type 'F'), it is used (prefers FEFF —
    the French-Wilson amplitudes ROCKET refines against — else the first F).
  * If the MTZ has only intensities (type 'J'), they are French-Wilson
    converted (reciprocalspaceship) to amplitudes on the fly.

Runs on CPU (no GPU / no model weights).
"""
import argparse
import os
import sys
import tempfile

import gemmi
import numpy as np
import torch

torch.set_grad_enabled(False)
DEVICE = torch.device("cpu")


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float((a @ b) / denom) if denom > 0 else float("nan")


def detect_columns(mtz_path, fobs, sigf, free):
    """Resolve (mtz_path, fobs, sigf, free) — French-Wilson converting if the
    file carries only intensities. Returns possibly a NEW temp mtz path."""
    m = gemmi.read_mtz_file(mtz_path)
    by_type = {}
    for c in m.columns:
        by_type.setdefault(c.type, []).append(c.label)
    labels = [c.label for c in m.columns]

    # free flag
    if free is None:
        cand = [l for l in by_type.get("I", []) if "free" in l.lower()]
        free = cand[0] if cand else (by_type.get("I", [None])[0])

    # amplitude already present?
    if fobs is None:
        if "FEFF" in labels:
            fobs = "FEFF"
        elif by_type.get("F"):
            fobs = by_type["F"][0]

    if fobs is not None:                       # amplitudes available
        if sigf is None:
            if fobs == "FEFF" and "DOBS" in labels:
                sigf = "DOBS"
            elif "SIG" + fobs in labels:
                sigf = "SIG" + fobs
            else:
                sigf = (by_type.get("Q") or by_type.get("R") or [fobs])[0]
        return mtz_path, fobs, sigf, free

    # only intensities -> French-Wilson convert
    icol = by_type.get("J", [None])[0]
    isig = by_type.get("Q", [None])[0]
    if icol is None:
        sys.exit("ERROR: MTZ has neither an amplitude (F) nor an intensity (J) column.")
    import reciprocalspaceship as rs
    ds = rs.read_mtz(mtz_path)
    ds = rs.algorithms.scale_merged_intensities(
        ds, icol, isig, output_columns=["FW-I", "FW-SIGI", "FW-F", "FW-SIGF"]
    )
    tmp = tempfile.NamedTemporaryFile(suffix="_fw.mtz", delete=False).name
    keep = ["FW-F", "FW-SIGF"] + ([free] if free in ds.columns else [])
    ds[keep].write_mtz(tmp)
    print(f"    (French-Wilson converted intensities '{icol}' -> amplitudes)")
    return tmp, "FW-F", "FW-SIGF", (free if free in ds.columns else None)


def minority_testset_value(mtz_path, free):
    if free is None:
        return 0
    import reciprocalspaceship as rs
    ds = rs.read_mtz(mtz_path)
    if free not in ds.columns:
        return 0
    vals, counts = np.unique(ds[free].to_numpy().round().astype(int), return_counts=True)
    return int(vals[np.argmin(counts)]) if len(vals) > 1 else int(vals[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("mtz")
    ap.add_argument("--fobs", default=None)
    ap.add_argument("--sigf", default=None)
    ap.add_argument("--free", default=None)
    ap.add_argument("--dmin", type=float, default=None)
    args = ap.parse_args()

    from SFC_Torch import SFcalculator

    mtz, fobs, sigf, free = detect_columns(args.mtz, args.fobs, args.sigf, args.free)
    tv = minority_testset_value(mtz, free)

    sfc = SFcalculator(
        args.model, mtz,
        expcolumns=[fobs, sigf],
        freeflag=(free or "R-free-flags"),
        set_experiment=True,
        testset_value=tv,
        dmin=args.dmin,
        device=DEVICE,
    )
    sfc.inspect_data(verbose=False)
    sfc.calc_fprotein()
    sfc.calc_fsolvent()
    sfc.init_scales(requires_grad=True)
    sfc.get_scales_lbfgs()                      # fit k_mask / k_iso / u_aniso, then freeze
    Fc = torch.abs(sfc.calc_ftotal()).cpu().numpy()
    Fo = sfc.Fo.cpu().numpy()

    keep = ~np.asarray(sfc.Outlier, bool)
    isfree = np.asarray(sfc.free_flag, bool)
    work = keep & ~isfree
    free_m = keep & isfree

    cc_all = pearson(Fc[keep], Fo[keep])
    cc_work = pearson(Fc[work], Fo[work]) if work.sum() else float("nan")
    cc_free = pearson(Fc[free_m], Fo[free_m]) if free_m.sum() else float("nan")

    print(f"    CC (|Fc| vs |Fo|, bulk-solvent fit)   <- paper 'CC'")
    print(f"      overall = {cc_all:.4f}   ({keep.sum()} refl, {fobs})")
    print(f"      work    = {cc_work:.4f}   ({work.sum()} refl)")
    print(f"      free    = {cc_free:.4f}   ({free_m.sum()} refl)")

    if mtz != args.mtz and os.path.exists(mtz):
        os.unlink(mtz)


if __name__ == "__main__":
    main()
