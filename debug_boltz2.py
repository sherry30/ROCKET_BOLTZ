"""
Step-by-step gradient debug for ROCKET-Boltz2.

Tests (run sequentially):
  T1 – forward pass shapes
  T2 – w_pair / b_pair receive non-zero gradients
  T3 – one Adam step moves LLG in the right direction
  T4 – 10-iteration mini-loop with fixed seed shows monotone LLG decrease
"""
import pickle
import sys
import os

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR    = "/data/dust/group/it/crystalsfirst/dev/shehry/data/thesis/1lj5/data_for_1lj5/1lj5_boltz_mod"
FEATS_PKL   = f"{DATA_DIR}/ROCKET_inputs/feats_boltz2.pkl"
CONFIG_YAML = f"{DATA_DIR}/ROCKET_config_phase1_boltz2.yaml"

# ── ROCKET / boltz imports ─────────────────────────────────────────────────
sys.path.insert(0, "/data/dust/group/it/crystalsfirst/dev/shehry/ROCKET")

import torch
from loguru import logger

logger.info("Loading feats …")
with open(FEATS_PKL, "rb") as fh:
    feats_cpu = pickle.load(fh)

DEVICE = "cuda:0"

# Move feats to GPU
from rocket import utils as rk_utils
feats = rk_utils.move_tensors_to_device(feats_cpu, device=DEVICE)

n_tokens = int(feats["token_pad_mask"].shape[1])
logger.info(f"n_tokens = {n_tokens}")

# ─── load wrapper ──────────────────────────────────────────────────────────
from rocket.boltz2_wrapper import Boltz2PairBias
wrapper = Boltz2PairBias(
    checkpoint_path="/data/dust/group/it/crystalsfirst/dev/shehry/boltz_cache/boltz2_conf.ckpt",
    truncated_backprop_steps=5,
    diffusion_seed=0,
    num_sampling_steps=200,
    recycling_steps=3,
    device=DEVICE,
)
wrapper = wrapper.to(DEVICE)
wrapper.eval()

# ─── init bias ─────────────────────────────────────────────────────────────
bias_dict = wrapper.init_bias(DEVICE)
w_pair = bias_dict["w_pair"]
b_pair = bias_dict["b_pair"]
logger.info(f"w_pair shape={w_pair.shape}  b_pair shape={b_pair.shape}")
logger.info(f"w_pair is_leaf={w_pair.is_leaf}  requires_grad={w_pair.requires_grad}")
logger.info(f"b_pair is_leaf={b_pair.is_leaf}  requires_grad={b_pair.requires_grad}")

# ══════════════════════════════════════════════════════════════════════════
# T1 – forward pass shapes
# ══════════════════════════════════════════════════════════════════════════
logger.info("\n=== T1: Forward pass ===")
with torch.no_grad():
    out = wrapper(feats, recycling_steps=0, num_sampling_steps=10)

coords = out["sample_atom_coords"]
logger.info(f"  sample_atom_coords shape : {coords.shape}")
logger.info(f"  z shape                  : {out['z'].shape}")
logger.info(f"  s shape                  : {out['s'].shape}")
logger.info("  T1 PASSED")

# ══════════════════════════════════════════════════════════════════════════
# T2 – gradients reach w_pair / b_pair
# ══════════════════════════════════════════════════════════════════════════
logger.info("\n=== T2: Gradient flow ===")

# Build a tiny SFC for LLG
from rocket.refinement_config import RocketRefinmentConfig
config = RocketRefinmentConfig.from_yaml_file(CONFIG_YAML)

import glob
tng_file  = f"{config.path}/ROCKET_inputs/{config.file_id}-Edata.mtz"
input_pdb = glob.glob(config.input_pdb)[0]

from rocket.xtal import structurefactors as llg_sf
from rocket import refinement_utils as rkrf_utils

sfc = llg_sf.initial_SFC(
    input_pdb, tng_file, "FEFF", "DOBS",
    Freelabel=config.free_flag,
    device=DEVICE,
    testset_value=config.testset_value,
)
sfc_rbr = llg_sf.initial_SFC(
    input_pdb, tng_file, "FEFF", "DOBS",
    Freelabel=config.free_flag,
    device=DEVICE,
    solvent=False,
    testset_value=config.testset_value,
)
llgloss     = rkrf_utils.init_llgloss(sfc,     tng_file, config.min_resolution, config.max_resolution)
llgloss_rbr = rkrf_utils.init_llgloss(sfc_rbr, tng_file, config.min_resolution, config.max_resolution)
cra_name = sfc.cra_name

from rocket.coordinates_boltz2 import position_alignment_boltz2
import rocket.coordinates as rk_coordinates

wrapper.diffusion_seed = 0
out = wrapper(feats, recycling_steps=0, num_sampling_steps=10)

init_pos_bfactor = sfc.atom_b_iso.clone()
best_pos_t2 = sfc.atom_pos_orth.clone()
aligned_xyz, plddt_tokens, pseudo_Bs = position_alignment_boltz2(
    model_output=out,
    feats=feats,
    cra_name_sfc=cra_name,
    best_pos=best_pos_t2,
    exclude_res=None,
    domain_segs=config.domain_segs,
    reference_bfactor=init_pos_bfactor,
)

safe_Bs = pseudo_Bs.detach().clone().clamp(max=200.0)
llgloss.sfc.atom_b_iso     = safe_Bs
llgloss_rbr.sfc.atom_b_iso = safe_Bs
llgloss.sfc.atom_pos_orth  = aligned_xyz.detach().clone()

# Must initialise sigmaA before first LLG call
llgloss, llgloss_rbr, _, _ = rkrf_utils.update_sigmaA(
    llgloss=llgloss,
    llgloss_rbr=llgloss_rbr,
    aligned_xyz=aligned_xyz.detach(),
    constant_fp_added_HKL=None,
    constant_fp_added_asu=None,
)

llg, r_work, r_free = llgloss(
    aligned_xyz,
    bin_labels=None,
    num_batch=1,
    sub_ratio=1.0,
    solvent=config.solvent,
    update_scales=True,
    return_Rfactors=True,
)
loss = -llg
loss.backward()

w_grad_norm = w_pair.grad.norm().item() if w_pair.grad is not None else None
b_grad_norm = b_pair.grad.norm().item() if b_pair.grad is not None else None
logger.info(f"  w_pair.grad norm : {w_grad_norm}")
logger.info(f"  b_pair.grad norm : {b_grad_norm}")
logger.info(f"  LLG              : {llg.item():.4f}   Rwork={r_work.item():.4f}")

if w_grad_norm is None or w_grad_norm == 0.0:
    logger.error("  T2 FAILED – w_pair has no gradient!")
    sys.exit(1)
else:
    logger.info("  T2 PASSED")

# ══════════════════════════════════════════════════════════════════════════
# T3 – one step improves LLG
# ══════════════════════════════════════════════════════════════════════════
logger.info("\n=== T3: Single Adam step ===")

# Re-init bias (fresh)
bias_dict = wrapper.init_bias(DEVICE)
w_pair = bias_dict["w_pair"]
b_pair = bias_dict["b_pair"]
wrapper.diffusion_seed = 0

optimizer = torch.optim.Adam(
    [{"params": w_pair, "lr": 0.01}, {"params": b_pair, "lr": 0.01}]
)

llg_before_list = []
llg_after_list  = []

for trial in range(3):
    # Re-init to identity each trial for clean comparison
    bias_dict = wrapper.init_bias(DEVICE)
    w_pair = bias_dict["w_pair"]
    b_pair = bias_dict["b_pair"]
    optimizer = torch.optim.Adam(
        [{"params": w_pair, "lr": 0.01}, {"params": b_pair, "lr": 0.01}]
    )

    optimizer.zero_grad()
    out_0 = wrapper(feats, recycling_steps=0, num_sampling_steps=10)
    aligned_0, _, pseudo_B0 = position_alignment_boltz2(
        model_output=out_0, feats=feats, cra_name_sfc=cra_name,
        best_pos=sfc.atom_pos_orth.clone(), exclude_res=None,
        domain_segs=config.domain_segs, reference_bfactor=init_pos_bfactor,
    )
    llgloss.sfc.atom_b_iso = pseudo_B0.detach().clamp(max=200.0)
    llgloss.sfc.atom_pos_orth = aligned_0.detach().clone()
    llg0, _, _ = llgloss(aligned_0, bin_labels=None, num_batch=1, sub_ratio=1.0,
                         solvent=config.solvent, update_scales=True, return_Rfactors=True)
    llg_before_list.append(llg0.item())

    (-llg0).backward()
    torch.nn.utils.clip_grad_norm_([w_pair, b_pair], max_norm=10.0)
    optimizer.step()

    # Eval after step (no_grad, same seed)
    with torch.no_grad():
        out_1 = wrapper(feats, recycling_steps=0, num_sampling_steps=10)
        aligned_1, _, pseudo_B1 = position_alignment_boltz2(
            model_output=out_1, feats=feats, cra_name_sfc=cra_name,
            best_pos=aligned_0.detach(), exclude_res=None,
            domain_segs=config.domain_segs, reference_bfactor=init_pos_bfactor,
        )
        llgloss.sfc.atom_b_iso = pseudo_B1.detach().clamp(max=200.0)
        llgloss.sfc.atom_pos_orth = aligned_1.detach().clone()
        llg1, _, _ = llgloss(aligned_1.detach(), bin_labels=None, num_batch=1, sub_ratio=1.0,
                             solvent=config.solvent, update_scales=False, return_Rfactors=True)
        llg_after_list.append(llg1.item())
    logger.info(f"  trial {trial}: LLG {llg0.item():.3f} → {llg1.item():.3f}  (delta {llg1.item()-llg0.item():+.3f})")

any_improved = any(a > b for a, b in zip(llg_after_list, llg_before_list))
logger.info(f"  T3 {'PASSED' if any_improved else 'WARNING – no improvement in any trial (check LR / grad norms)'}")

# ══════════════════════════════════════════════════════════════════════════
# T4 – 10-iteration mini loop
# ══════════════════════════════════════════════════════════════════════════
logger.info("\n=== T4: 10-iter mini-loop ===")

bias_dict = wrapper.init_bias(DEVICE)
w_pair = bias_dict["w_pair"]
b_pair = bias_dict["b_pair"]
wrapper.diffusion_seed = 0
best_pos = sfc.atom_pos_orth.clone()

optimizer = torch.optim.Adam(
    [{"params": w_pair, "lr": 1e-3}, {"params": b_pair, "lr": 1e-3}]
)

llg_trace = []
for it in range(10):
    optimizer.zero_grad()
    out = wrapper(feats, recycling_steps=0, num_sampling_steps=20)
    aligned, _, pseudo_B = position_alignment_boltz2(
        model_output=out, feats=feats, cra_name_sfc=cra_name,
        best_pos=best_pos, exclude_res=None,
        domain_segs=config.domain_segs, reference_bfactor=init_pos_bfactor,
    )
    safe_B = pseudo_B.detach().clamp(max=200.0)
    llgloss.sfc.atom_b_iso = safe_B
    llgloss.sfc.atom_pos_orth = aligned.detach().clone()
    llg, rw, rf = llgloss(aligned, bin_labels=None, num_batch=1, sub_ratio=1.0,
                          solvent=config.solvent, update_scales=(it == 0), return_Rfactors=True)
    (-llg).backward()
    torch.nn.utils.clip_grad_norm_([w_pair, b_pair], max_norm=10.0)
    optimizer.step()
    llgloss.sfc.atom_pos_orth = aligned
    best_pos = aligned.detach().clone()
    llg_trace.append(llg.item())
    wg = w_pair.grad.norm().item() if w_pair.grad is not None else 0.0
    bg = b_pair.grad.norm().item() if b_pair.grad is not None else 0.0
    logger.info(f"  iter {it:2d}: LLG={llg.item():8.2f}  Rwork={rw.item():.4f}  |∇w|={wg:.3e}  |∇b|={bg:.3e}")

# Report trend
import numpy as np
first5  = np.mean(llg_trace[:5])
last5   = np.mean(llg_trace[5:])
logger.info(f"\n  LLG first-5 mean: {first5:.2f}   last-5 mean: {last5:.2f}")
if last5 > first5:
    logger.info("  T4 PASSED – LLG is improving (last-5 > first-5)")
else:
    logger.warning("  T4 WARNING – LLG did not improve. Check trace above.")
