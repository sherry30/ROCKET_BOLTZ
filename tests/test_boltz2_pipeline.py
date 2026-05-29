"""
End-to-end diagnostic for the ROCKET-Boltz2 pipeline.

This is a manual GPU integration check (it loads the Boltz-2 checkpoint and runs
a forward/backward pass), not a unit test — run it directly on the GPU node:

    ssh shehryar@max-hpcgwg001
    micromamba activate rocket-of
    cd /data/dust/group/it/crystalsfirst/dev/shehry/ROCKET
    python tests/test_boltz2_pipeline.py

Paths default to the 1lj5 test system and can be overridden with env vars
(ROCKET_TEST_FEATS, ROCKET_TEST_PDB, ROCKET_TEST_MTZ, ROCKET_TEST_CKPT).
If the data/checkpoint are absent the script skips (exit 0) rather than failing.

It verifies, in the DDIM sampling mode (the pipeline default), that:
  1. feats dict has the keys/shapes the featurizer produces
  2. Boltz2PairBias forward returns finite coordinates
  3. B-factors are usable after the 200 Å² clamp
  4. coordinates extract and match the SFC topology
  5. LLG is finite and gradients flow from LLG back to w_pair / b_pair
  6. one optimizer step actually moves the parameters
"""

from __future__ import annotations

import os
import pickle
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (override via env vars)
# ---------------------------------------------------------------------------
_DATA = "/data/dust/group/it/crystalsfirst/dev/shehry/data/thesis/1lj5/data_for_1lj5/1lj5_boltz_mod/ROCKET_inputs"
FEATS_PKL = Path(os.environ.get("ROCKET_TEST_FEATS", f"{_DATA}/feats_boltz2.pkl"))
INPUT_PDB = Path(os.environ.get("ROCKET_TEST_PDB",   f"{_DATA}/1lj5-pred-aligned.pdb"))
MTZ_FILE  = Path(os.environ.get("ROCKET_TEST_MTZ",   f"{_DATA}/1lj5-Edata.mtz"))
CKPT      = Path(os.environ.get("ROCKET_TEST_CKPT",
                 "/data/dust/group/it/crystalsfirst/dev/shehry/boltz_cache/boltz2_conf.ckpt"))
CUDA_DEVICE = os.environ.get("ROCKET_TEST_DEVICE", "cuda:0")

# Short, fast sampling settings for the smoke test.
SAMPLING_MODE = "ddim"
NUM_SAMPLING_STEPS = 5   # ddim runs this many steps (single step knob)
RECYCLING_STEPS = 1
DIFFUSION_SEED = 42

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  [{PASS if condition else FAIL}] {label}" + (f"  ({detail})" if detail else ""))
    return condition


def warn(label: str, detail: str = "") -> None:
    print(f"  [{WARN}] {label}" + (f"  ({detail})" if detail else ""))


def _make_wrapper(device: str):
    """Construct a Boltz2PairBias in the pipeline-default (DDIM) sampling mode."""
    from rocket.boltz2_wrapper import Boltz2PairBias

    wrapper = Boltz2PairBias(
        checkpoint_path=CKPT,
        diffusion_seed=DIFFUSION_SEED,
        num_sampling_steps=NUM_SAMPLING_STEPS,
        recycling_steps=RECYCLING_STEPS,
        sampling_mode=SAMPLING_MODE,
        device=device,
    ).to(device).eval()
    wrapper.init_bias(device)
    return wrapper


# ===========================================================================
# 1. feats structure
# ===========================================================================
def check_feats_structure(feats: dict) -> bool:
    print("\n[1] feats structure")
    ok = True
    for key in ["token_pad_mask", "atom_pad_mask", "atom_to_token",
                "residue_index", "asym_id", "res_type", "ref_atom_name_chars"]:
        ok &= check(f"  key '{key}' present", key in feats)
    if ok:
        n_tokens = int(feats["token_pad_mask"][0].sum().item())
        n_atoms  = int(feats["atom_pad_mask"][0].sum().item())
        check("n_tokens > 0", n_tokens > 0, f"n_tokens={n_tokens}")
        check("n_atoms > 0",  n_atoms  > 0, f"n_atoms={n_atoms}")
        check("atom_to_token one-hot per atom",
              bool(feats["atom_to_token"][0].sum(-1).float().mean().item() >= 0.9))
    return ok


# ===========================================================================
# 2. forward pass
# ===========================================================================
def check_forward(feats: dict, device: str):
    print(f"\n[2] Boltz2PairBias forward pass (sampling_mode={SAMPLING_MODE})")
    feats_gpu = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in feats.items()}

    wrapper = _make_wrapper(device)
    model_out = wrapper(feats_gpu)

    check("sample_atom_coords present", "sample_atom_coords" in model_out)
    coords = model_out["sample_atom_coords"]
    check("coords shape [1, N, 3]", coords.ndim == 3 and coords.shape[-1] == 3,
          f"shape={tuple(coords.shape)}")
    check("coords finite", bool(coords.isfinite().all().item()))

    if "plddt" in model_out:
        plddt = model_out["plddt"]
        check("plddt in [0,1]", bool((plddt >= 0).all() and (plddt <= 1).all()),
              f"mean={plddt.mean().item():.3f}")
    else:
        warn("'plddt' not in model_out — confidence_prediction disabled?")
    return model_out, wrapper, feats_gpu


# ===========================================================================
# 3. B-factor clamp
# ===========================================================================
def check_bfactors(feats_gpu: dict, device: str) -> bool:
    import torch
    from rocket.coordinates_boltz2 import position_alignment_boltz2
    from rocket.xtal import structurefactors as llg_sf

    print("\n[3] B-factor extraction + clamp")
    sfc = llg_sf.initial_SFC(str(INPUT_PDB), str(MTZ_FILE), "FEFF", "DOBS",
                             Freelabel="R-free-flags", device=device,
                             testset_value=1, spacing=4.5)
    wrapper = _make_wrapper(device)
    with torch.no_grad():
        out = wrapper(feats_gpu)
        _, _, pseudo_Bs = position_alignment_boltz2(
            out, feats_gpu, sfc.cra_name, sfc.atom_pos_orth.clone())
    safe_Bs = pseudo_Bs.detach().clamp(max=200.0)
    ok = check("B-factors in (0, 200] Å²",
               bool((safe_Bs > 0).all() and (safe_Bs <= 200.0).all()),
               f"mean={safe_Bs.mean().item():.1f} max={safe_Bs.max().item():.1f}")
    return ok


# ===========================================================================
# 4. coordinate extraction / topology
# ===========================================================================
def check_coord_extraction(model_out: dict, feats_gpu: dict, device: str) -> bool:
    from rocket.coordinates_boltz2 import extract_allatoms_boltz2
    from rocket.xtal import structurefactors as llg_sf

    print("\n[4] coordinate extraction + SFC topology")
    sfc = llg_sf.initial_SFC(str(INPUT_PDB), str(MTZ_FILE), "FEFF", "DOBS",
                             Freelabel="R-free-flags", device=device,
                             testset_value=1, spacing=4.5)
    cra_name = sfc.cra_name
    try:
        xyz_sfc, _, _ = extract_allatoms_boltz2(
            model_out["sample_atom_coords"], feats_gpu, cra_name)
    except Exception as exc:
        return check("extract_allatoms_boltz2 succeeded", False, str(exc))

    ok = True
    ok &= check("xyz shape == (n_sfc_atoms, 3)", xyz_sfc.shape == (len(cra_name), 3),
                f"got {tuple(xyz_sfc.shape)}")
    ok &= check("xyz finite", bool(xyz_sfc.isfinite().all().item()))
    ok &= check("xyz spans space (not collapsed)",
                float((xyz_sfc.max(0).values - xyz_sfc.min(0).values).min().item()) > 1.0)
    return ok


# ===========================================================================
# 5. LLG, gradient flow, optimizer step
# ===========================================================================
def check_llg_and_gradient(feats_gpu: dict, device: str) -> bool:
    import torch
    from rocket import coordinates as rk_coordinates
    from rocket import refinement_utils as rkrf_utils
    from rocket.coordinates_boltz2 import position_alignment_boltz2
    from rocket.xtal import structurefactors as llg_sf

    print("\n[5] LLG + gradient flow + optimizer step")
    sfc = llg_sf.initial_SFC(str(INPUT_PDB), str(MTZ_FILE), "FEFF", "DOBS",
                             Freelabel="R-free-flags", device=device,
                             testset_value=1, spacing=4.5)
    sfc_rbr = llg_sf.initial_SFC(str(INPUT_PDB), str(MTZ_FILE), "FEFF", "DOBS",
                                 Freelabel="R-free-flags", device=device,
                                 solvent=False, testset_value=1, spacing=4.5)
    reference_pos = sfc.atom_pos_orth.clone()
    init_bfactor = sfc.atom_b_iso.clone()
    cra_name = sfc.cra_name
    llgloss     = rkrf_utils.init_llgloss(sfc,     str(MTZ_FILE), None, None)
    llgloss_rbr = rkrf_utils.init_llgloss(sfc_rbr, str(MTZ_FILE), None, None)

    wrapper = _make_wrapper(device)
    w_pair, b_pair = wrapper._bias["w_pair"], wrapper._bias["b_pair"]
    if not check("w_pair / b_pair require grad", w_pair.requires_grad and b_pair.requires_grad):
        return False

    model_out = wrapper(feats_gpu)
    aligned_xyz, _, pseudo_Bs = position_alignment_boltz2(
        model_output=model_out, feats=feats_gpu, cra_name_sfc=cra_name,
        best_pos=reference_pos, reference_bfactor=init_bfactor)

    safe_Bs = pseudo_Bs.detach().clone().clamp(max=200.0)
    llgloss.sfc.atom_b_iso = safe_Bs
    llgloss_rbr.sfc.atom_b_iso = safe_Bs
    llgloss.sfc.atom_pos_orth = aligned_xyz.detach().clone()

    try:
        llgloss, llgloss_rbr, _, _ = rkrf_utils.update_sigmaA(
            llgloss=llgloss, llgloss_rbr=llgloss_rbr, aligned_xyz=aligned_xyz,
            constant_fp_added_HKL=None, constant_fp_added_asu=None)
        check("update_sigmaA ok", True)
    except Exception as exc:
        return check("update_sigmaA ok", False, str(exc)[:200])

    try:
        optimized_xyz, _ = rk_coordinates.rigidbody_refine_quat(
            aligned_xyz, llgloss_rbr, cra_name, lbfgs=True, lbfgs_lr=150.0, verbose=False)
        check("rigid-body refine ok", True)
    except Exception as exc:
        check("rigid-body refine ok", False, str(exc)[:200])
        optimized_xyz = aligned_xyz

    try:
        llg, r_work, _ = llgloss(optimized_xyz, bin_labels=None, num_batch=1,
                                 sub_ratio=1.0, solvent=True, update_scales=True,
                                 return_Rfactors=True)
    except Exception as exc:
        return check("LLG computation ok", False, str(exc)[:200])

    ok = True
    ok &= check("LLG finite",   bool(llg.isfinite().item()),    f"LLG={llg.item():.1f}")
    ok &= check("Rwork finite", bool(r_work.isfinite().item()), f"Rwork={r_work.item():.4f}")

    try:
        (-llg).backward()
    except Exception as exc:
        return check("backward() ok", False, str(exc)[:200])

    for name, p in [("w_pair", w_pair), ("b_pair", b_pair)]:
        ok &= check(f"{name}.grad finite & non-zero",
                    p.grad is not None and bool(p.grad.isfinite().all()) and bool((p.grad != 0).any()),
                    f"max|grad|={p.grad.abs().max().item():.2e}" if p.grad is not None else "None")

    optimizer = torch.optim.Adam([{"params": w_pair, "lr": 1e-3},
                                  {"params": b_pair, "lr": 1e-3}])
    w_before = w_pair.detach().clone()
    optimizer.step()
    ok &= check("optimizer step moves w_pair",
                not torch.allclose(w_pair.detach(), w_before),
                f"max Δ={(w_pair.detach() - w_before).abs().max().item():.2e}")
    return ok


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    import torch

    print("=" * 60)
    print("ROCKET-Boltz2 pipeline diagnostic")
    print("=" * 60)

    missing = [str(p) for p in (FEATS_PKL, INPUT_PDB, MTZ_FILE, CKPT) if not p.exists()]
    if missing:
        print("[SKIP] required inputs not found:")
        for m in missing:
            print(f"        {m}")
        print("  Set ROCKET_TEST_FEATS / _PDB / _MTZ / _CKPT to override.")
        return 0

    device = CUDA_DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        warn("running on CPU — the forward pass will be slow")

    feats = pickle.load(open(FEATS_PKL, "rb"))

    all_ok = True
    all_ok &= check_feats_structure(feats)
    try:
        model_out, _wrapper, feats_gpu = check_forward(feats, device)
    except Exception:
        print(f"  [{FAIL}] forward pass crashed:")
        traceback.print_exc()
        return 1
    all_ok &= check_bfactors(feats_gpu, device)
    all_ok &= check_coord_extraction(model_out, feats_gpu, device)
    all_ok &= check_llg_and_gradient(feats_gpu, device)

    print("\n" + "=" * 60)
    print(f"[{PASS}] all checks passed" if all_ok else f"[{FAIL}] some checks failed")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
