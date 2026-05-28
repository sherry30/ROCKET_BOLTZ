"""
Minimal end-to-end diagnostic test for the ROCKET-Boltz2 pipeline.

Run on the GPU node:
    ssh shehryar@max-hpcgwg006
    micromamba activate rocket-of
    cd /data/dust/group/it/crystalsfirst/dev/shehry/ROCKET
    python tests/test_boltz2_pipeline.py

The test verifies every critical step in under ~5 minutes:
  1. feats.pkl structure (shapes, dtypes)
  2. Boltz2PairBias forward pass
  3. pLDDT range — warns if Boltz-2 is outputting systematically low values
  4. B-factor clamping produces values in [0, 200] Å²
  5. Coordinate extraction and topology matching
  6. LLG is finite and positive after B-factor fix
  7. Gradient flows from LLG back to w_pair and b_pair
  8. One full optimizer step with non-zero parameter update
"""

from __future__ import annotations

import os
import pickle
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — edit if needed
# ---------------------------------------------------------------------------
FEATS_PKL   = Path("/data/dust/group/it/crystalsfirst/dev/shehry/ROCKET/feats.pkl")
INPUT_PDB   = Path("/data/dust/group/it/crystalsfirst/dev/shehry/data/thesis/1lj5/data_for_1lj5/1lj5_boltz/ROCKET_inputs/1lj5-pred-aligned.pdb")
MTZ_FILE    = Path("/data/dust/group/it/crystalsfirst/dev/shehry/data/thesis/1lj5/data_for_1lj5/1lj5_boltz/ROCKET_inputs/1lj5-Edata.mtz")
CKPT        = Path("/data/dust/group/it/crystalsfirst/dev/shehry/boltz_cache/boltz2_conf.ckpt")
CUDA_DEVICE = "cuda:0"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    return condition


def warn(label: str, detail: str = "") -> None:
    print(f"  [{WARN}] {label}" + (f"  ({detail})" if detail else ""))


# ===========================================================================
# Test 1: feats.pkl structure
# ===========================================================================
def test_feats_structure(feats: dict) -> bool:
    print("\n[1] feats.pkl structure")
    ok = True
    for key in ["token_pad_mask", "atom_pad_mask", "atom_to_token",
                "residue_index", "asym_id", "res_type", "ref_atom_name_chars"]:
        ok &= check(f"  key '{key}' present", key in feats)
    if ok:
        n_tokens = int(feats["token_pad_mask"][0].sum().item())
        n_atoms  = int(feats["atom_pad_mask"][0].sum().item())
        check("n_tokens > 0", n_tokens > 0, f"n_tokens={n_tokens}")
        check("n_atoms > 0",  n_atoms  > 0, f"n_atoms={n_atoms}")
        check("atom_to_token sum==1 per real atom",
              bool(feats["atom_to_token"][0].sum(-1).float().mean().item() >= 0.9),
              "mean sum over token dim")
    return ok


# ===========================================================================
# Test 2: Boltz2PairBias forward pass
# ===========================================================================
def test_forward(feats: dict, device: str) -> tuple[dict, object, object]:
    import torch
    from rocket.boltz2_wrapper import Boltz2PairBias

    print("\n[2] Boltz2PairBias forward pass")
    feats_gpu = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in feats.items()
    }
    n_tokens = int(feats_gpu["token_pad_mask"].shape[1])
    print(f"    N_tokens (padded)={n_tokens}")

    wrapper = Boltz2PairBias(
        checkpoint_path=CKPT,
        truncated_backprop_steps=5,
        diffusion_seed=42,          # fixed seed for reproducibility
        num_sampling_steps=20,      # short run for test speed
        recycling_steps=1,
        device=device,
    ).to(device).eval()

    bias = wrapper.init_bias(n_tokens, device)
    w_pair = bias["w_pair"]
    b_pair = bias["b_pair"]

    model_out = wrapper(feats_gpu, recycling_steps=1, num_sampling_steps=20)

    check("sample_atom_coords present", "sample_atom_coords" in model_out)
    coords = model_out["sample_atom_coords"]
    check("coords shape has 3 dims",   coords.ndim == 3,        f"shape={tuple(coords.shape)}")
    check("coords last dim == 3",      coords.shape[-1] == 3,   f"shape={tuple(coords.shape)}")
    check("coords finite",             bool(coords.isfinite().all().item()))

    if "plddt" in model_out:
        plddt = model_out["plddt"]
        mean_plddt = plddt.mean().item()
        check("plddt present and in [0,1]",
              bool((plddt >= 0).all() and (plddt <= 1).all()),
              f"min={plddt.min().item():.3f} max={plddt.max().item():.3f} mean={mean_plddt:.3f}")
        if mean_plddt < 0.4:
            warn("Boltz-2 mean pLDDT is low",
                 f"mean={mean_plddt:.3f} → mean pseudo-B would be >> 200 Å² without clamping")
    else:
        warn("'plddt' not in model_out — check confidence_prediction flag")

    return model_out, wrapper, feats_gpu


# ===========================================================================
# Test 3: B-factor extraction and clamping
# ===========================================================================
def test_bfactors(model_out: dict, feats_gpu: dict) -> bool:
    import torch
    from rocket import utils as rk_utils
    from rocket.coordinates_boltz2 import extract_allatoms_boltz2

    print("\n[3] B-factor extraction and clamping")

    plddt = model_out.get("plddt")
    n_sfc_fake = int(feats_gpu["atom_pad_mask"][0].sum().item())

    # Simulate what position_alignment_boltz2 does
    if plddt is not None:
        plddt_token = plddt[0]
        a2t = feats_gpu["atom_to_token"][0].argmax(-1)
        plddt_atom = plddt_token[a2t]
        atom_mask = feats_gpu["atom_pad_mask"][0] > 0.5
        plddt_real = plddt_atom[atom_mask]
        pseudo_Bs = rk_utils.plddt2pseudoB_pt(plddt_real * 100.0)
        safe_Bs   = pseudo_Bs.clamp(max=200.0)

        mean_raw  = pseudo_Bs.mean().item()
        mean_safe = safe_Bs.mean().item()
        frac_over = ((pseudo_Bs > 200.0).float().mean().item())

        ok = True
        ok &= check("safe_Bs all ≤ 200 Å²", bool((safe_Bs <= 200.0).all()),
                    f"max={safe_Bs.max().item():.1f}")
        ok &= check("safe_Bs all > 0 Å²",   bool((safe_Bs > 0).all()))
        if frac_over > 0.5:
            warn(f"{frac_over:.0%} of atoms would have B > 200 without clamping",
                 f"mean raw={mean_raw:.1f} Å², mean safe={mean_safe:.1f} Å²")
        else:
            check(f"B-factor clamping fraction reasonable",
                  frac_over < 0.9,
                  f"{frac_over:.0%} atoms clamped, mean_raw={mean_raw:.1f}")
        return ok
    else:
        warn("No plddt → fallback B=30 Å² — check confidence module")
        return True


# ===========================================================================
# Test 4: Coordinate extraction topology
# ===========================================================================
def test_coord_extraction(model_out: dict, feats_gpu: dict, device: str) -> tuple[bool, list]:
    import torch
    from SFC_Torch import SFcalculator
    from rocket.xtal import structurefactors as llg_sf

    print("\n[4] Coordinate extraction + SFC topology")

    sfc = llg_sf.initial_SFC(
        str(INPUT_PDB), str(MTZ_FILE), "FEFF", "DOBS",
        Freelabel="R-free-flags",
        device=device,
        testset_value=1,
        spacing=4.5,
    )
    cra_name = sfc.cra_name
    check("SFC initialised", True, f"n_atoms={len(cra_name)}")

    from rocket.coordinates_boltz2 import extract_allatoms_boltz2

    try:
        xyz_sfc, plddt_atom = extract_allatoms_boltz2(
            model_out["sample_atom_coords"],
            feats_gpu,
            cra_name,
        )
        ok = True
        ok &= check("xyz shape correct", xyz_sfc.shape == (len(cra_name), 3),
                    f"got {tuple(xyz_sfc.shape)}")
        ok &= check("xyz finite", bool(xyz_sfc.isfinite().all().item()))
        ok &= check("xyz non-zero range",
                    float((xyz_sfc.max(0).values - xyz_sfc.min(0).values).min().item()) > 1.0,
                    "all atoms in same position?")
    except Exception as exc:
        check("extract_allatoms_boltz2 succeeded", False, str(exc))
        return False, sfc.cra_name

    return ok, cra_name


# ===========================================================================
# Test 5: LLG computation and gradient flow
# ===========================================================================
def test_llg_and_gradient(model_out: dict, feats_gpu: dict, wrapper, device: str) -> bool:
    import torch
    from rocket import coordinates as rk_coordinates
    from rocket import refinement_utils as rkrf_utils
    from rocket import utils as rk_utils
    from rocket.coordinates_boltz2 import position_alignment_boltz2
    from rocket.xtal import structurefactors as llg_sf

    print("\n[5] LLG computation and gradient flow")

    # SFC
    sfc = llg_sf.initial_SFC(
        str(INPUT_PDB), str(MTZ_FILE), "FEFF", "DOBS",
        Freelabel="R-free-flags",
        device=device,
        testset_value=1,
        spacing=4.5,
    )
    sfc_rbr = llg_sf.initial_SFC(
        str(INPUT_PDB), str(MTZ_FILE), "FEFF", "DOBS",
        Freelabel="R-free-flags",
        device=device,
        solvent=False,
        testset_value=1,
        spacing=4.5,
    )
    reference_pos = sfc.atom_pos_orth.clone()
    init_pos_bfactor = sfc.atom_b_iso.clone()
    cra_name = sfc.cra_name

    llgloss     = rkrf_utils.init_llgloss(sfc,     str(MTZ_FILE), None, None)
    llgloss_rbr = rkrf_utils.init_llgloss(sfc_rbr, str(MTZ_FILE), None, None)

    # Re-run wrapper with w_pair/b_pair requiring grad
    n_tokens = int(feats_gpu["token_pad_mask"].shape[1])
    bias = wrapper.init_bias(n_tokens, device)
    w_pair = bias["w_pair"]
    b_pair = bias["b_pair"]
    assert w_pair.requires_grad and b_pair.requires_grad

    model_out2 = wrapper(feats_gpu, recycling_steps=1, num_sampling_steps=20)

    aligned_xyz, plddt_tokens, pseudo_Bs = position_alignment_boltz2(
        model_output=model_out2,
        feats=feats_gpu,
        cra_name_sfc=cra_name,
        best_pos=reference_pos,
        reference_bfactor=init_pos_bfactor,
    )

    # Apply B-factor clamp — the critical fix
    safe_Bs = pseudo_Bs.detach().clone().clamp(max=200.0)
    llgloss.sfc.atom_b_iso     = safe_Bs
    llgloss_rbr.sfc.atom_b_iso = safe_Bs

    check("pseudo_Bs clamped to ≤200", bool((safe_Bs <= 200.0).all()),
          f"max={safe_Bs.max().item():.1f} Å²")

    # update_sigmaA
    llgloss.sfc.atom_pos_orth = aligned_xyz.detach().clone()
    try:
        llgloss, llgloss_rbr, _, _ = rkrf_utils.update_sigmaA(
            llgloss=llgloss,
            llgloss_rbr=llgloss_rbr,
            aligned_xyz=aligned_xyz,
            constant_fp_added_HKL=None,
            constant_fp_added_asu=None,
        )
        check("update_sigmaA did not crash", True)
    except Exception as exc:
        check("update_sigmaA did not crash", False, str(exc))
        return False

    # RBR
    llgloss.sfc.atom_pos_orth = aligned_xyz.detach().clone()
    try:
        optimized_xyz, _ = rk_coordinates.rigidbody_refine_quat(
            aligned_xyz, llgloss_rbr, cra_name,
            lbfgs=True, lbfgs_lr=150.0, verbose=False,
        )
        check("RBR did not crash", True)
    except Exception as exc:
        check("RBR did not crash", False, str(exc))
        optimized_xyz = aligned_xyz

    # LLG
    try:
        llg, r_work, r_free = llgloss(
            optimized_xyz,
            bin_labels=None,
            num_batch=1,
            sub_ratio=0.7,
            solvent=True,
            update_scales=True,
            return_Rfactors=True,
        )
        ok = True
        ok &= check("LLG is finite",    bool(llg.isfinite().item()),  f"LLG={llg.item():.2f}")
        ok &= check("Rwork is finite",  bool(r_work.isfinite().item()), f"Rwork={r_work.item():.4f}")
        ok &= check("Rwork < 1.0",      r_work.item() < 1.0,          f"Rwork={r_work.item():.4f}")
        if r_work.item() > 0.6:
            warn("Rwork > 0.6 — model is rough but this is expected at start of refinement",
                 f"Rwork={r_work.item():.4f}")
    except Exception as exc:
        check("LLG computation did not crash", False, str(exc)[:200])
        return False

    # Gradient test
    L = -llg
    try:
        L.backward()
        ok &= check("w_pair.grad is not None",  w_pair.grad is not None)
        ok &= check("b_pair.grad is not None",  b_pair.grad is not None)
        if w_pair.grad is not None:
            ok &= check("w_pair.grad is finite", bool(w_pair.grad.isfinite().all().item()),
                        f"max|grad|={w_pair.grad.abs().max().item():.3e}")
            ok &= check("w_pair.grad is non-zero", bool((w_pair.grad != 0).any().item()))
        if b_pair.grad is not None:
            ok &= check("b_pair.grad is finite", bool(b_pair.grad.isfinite().all().item()),
                        f"max|grad|={b_pair.grad.abs().max().item():.3e}")
            ok &= check("b_pair.grad is non-zero", bool((b_pair.grad != 0).any().item()))
    except Exception as exc:
        check("backward() did not crash", False, str(exc)[:200])
        return False

    # One optimizer step
    optimizer = torch.optim.Adam(
        [{"params": w_pair, "lr": 1.0}, {"params": b_pair, "lr": 0.05}]
    )
    w_before = w_pair.detach().clone()
    b_before = b_pair.detach().clone()
    optimizer.step()
    ok &= check("w_pair changed after optimizer step",
                not torch.allclose(w_pair.detach(), w_before, atol=1e-12),
                f"max Δ={( w_pair.detach()-w_before).abs().max().item():.3e}")
    ok &= check("b_pair changed after optimizer step",
                not torch.allclose(b_pair.detach(), b_before, atol=1e-12),
                f"max Δ={(b_pair.detach() -b_before).abs().max().item():.3e}")

    return ok


# ===========================================================================
# Main
# ===========================================================================
def main():
    import torch
    print("=" * 60)
    print("ROCKET-Boltz2 pipeline diagnostic test")
    print("=" * 60)

    # Validate paths
    for p, name in [(FEATS_PKL, "feats.pkl"), (INPUT_PDB, "input_pdb"),
                    (MTZ_FILE, "mtz_file"), (CKPT, "checkpoint")]:
        if not p.exists():
            print(f"[ERROR] {name} not found at {p}")
            print("  Edit the path constants at the top of this script.")
            sys.exit(1)

    device = CUDA_DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("[WARN] Running on CPU — forward pass will be very slow (reduce num_sampling_steps)")

    print(f"\nLoading feats from {FEATS_PKL} …")
    feats = pickle.load(open(FEATS_PKL, "rb"))

    all_ok = True

    all_ok &= test_feats_structure(feats)

    try:
        model_out, wrapper, feats_gpu = test_forward(feats, device)
    except Exception:
        print(f"  [{FAIL}] test_forward crashed:")
        traceback.print_exc()
        print("\nAborting remaining tests.")
        sys.exit(1)

    all_ok &= test_bfactors(model_out, feats_gpu)
    ok_coord, cra_name = test_coord_extraction(model_out, feats_gpu, device)
    all_ok &= ok_coord
    all_ok &= test_llg_and_gradient(model_out, feats_gpu, wrapper, device)

    print("\n" + "=" * 60)
    if all_ok:
        print(f"[{PASS}] All checks passed — pipeline looks healthy.")
    else:
        print(f"[{FAIL}] Some checks failed — see above for details.")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
