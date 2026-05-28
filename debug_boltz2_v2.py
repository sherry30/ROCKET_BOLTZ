"""
Targeted root-cause debugging for ROCKET-Boltz2 LLG collapse.

Hypothesis A: diffusion seed only covers initial noise, not per-step eps
              → each iter uses a different 200-step noise trajectory
              → gradient dominated by noise, not by bias signal

Hypothesis B: lr_m=1.0 is catastrophically large for [128,128] identity matrix
              → first Adam step moves each element by ±1.0, destroying identity

Test plan:
  T1: B-factor inspection (what pseudo_Bs are actually used?)
  T2: Seed coverage test (does fixing full trajectory stabilise LLG?)
  T3: Gradient direction test (tiny lr=1e-6 — does LLG improve or worsen?)
  T4: LR sweep (1e-4, 1e-3, 1e-2 with fixed full-trajectory seed)
"""
import pickle, sys, os
import numpy as np

DATA_DIR    = "/data/dust/group/it/crystalsfirst/dev/shehry/data/thesis/1lj5/data_for_1lj5/1lj5_boltz_mod"
FEATS_PKL   = f"{DATA_DIR}/ROCKET_inputs/feats_boltz2.pkl"
CONFIG_YAML = f"{DATA_DIR}/ROCKET_config_phase1_boltz2.yaml"
CKPT        = "/data/dust/group/it/crystalsfirst/dev/shehry/boltz_cache/boltz2_conf.ckpt"

sys.path.insert(0, "/data/dust/group/it/crystalsfirst/dev/shehry/ROCKET")
import torch
from loguru import logger

# ── load feats ──────────────────────────────────────────────────────────────
with open(FEATS_PKL, "rb") as fh:
    feats_cpu = pickle.load(fh)

DEVICE = "cuda:0"
from rocket import utils as rk_utils
feats = rk_utils.move_tensors_to_device(feats_cpu, device=DEVICE)

# ── load model ──────────────────────────────────────────────────────────────
from rocket.boltz2_wrapper import Boltz2PairBias
wrapper = Boltz2PairBias(
    checkpoint_path=CKPT,
    truncated_backprop_steps=5,
    diffusion_seed=0,
    num_sampling_steps=200,
    recycling_steps=3,
    device=DEVICE,
)
wrapper = wrapper.to(DEVICE)
wrapper.eval()

# ── load SFC/LLG infrastructure ─────────────────────────────────────────────
import glob
from rocket.refinement_config import RocketRefinmentConfig
from rocket.xtal import structurefactors as llg_sf
from rocket import refinement_utils as rkrf_utils
from rocket.coordinates_boltz2 import position_alignment_boltz2
import rocket.coordinates as rk_coordinates

config   = RocketRefinmentConfig.from_yaml_file(CONFIG_YAML)
tng_file = f"{config.path}/ROCKET_inputs/{config.file_id}-Edata.mtz"
input_pdb = glob.glob(config.input_pdb)[0]

sfc = llg_sf.initial_SFC(input_pdb, tng_file, "FEFF", "DOBS",
    Freelabel=config.free_flag, device=DEVICE, testset_value=config.testset_value)
sfc_rbr = llg_sf.initial_SFC(input_pdb, tng_file, "FEFF", "DOBS",
    Freelabel=config.free_flag, device=DEVICE, solvent=False,
    testset_value=config.testset_value)
llgloss     = rkrf_utils.init_llgloss(sfc,     tng_file, config.min_resolution, config.max_resolution)
llgloss_rbr = rkrf_utils.init_llgloss(sfc_rbr, tng_file, config.min_resolution, config.max_resolution)
cra_name         = sfc.cra_name
init_pos_bfactor = sfc.atom_b_iso.clone()
reference_pos    = sfc.atom_pos_orth.clone()

def run_sigma_update(aligned_xyz_detached):
    global llgloss, llgloss_rbr
    llgloss, llgloss_rbr, _, _ = rkrf_utils.update_sigmaA(
        llgloss=llgloss, llgloss_rbr=llgloss_rbr,
        aligned_xyz=aligned_xyz_detached,
        constant_fp_added_HKL=None, constant_fp_added_asu=None,
    )

# ════════════════════════════════════════════════════════════════════════════
# T1 — B-factor inspection
# ════════════════════════════════════════════════════════════════════════════
logger.info("\n=== T1: B-factor inspection ===")
bias_dict = wrapper.init_bias(DEVICE)
wrapper.diffusion_seed = 0

with torch.no_grad():
    out_t1 = wrapper(feats, recycling_steps=3, num_sampling_steps=200)

plddt_tokens = out_t1.get("plddt")
logger.info(f"  confidence_prediction running: {plddt_tokens is not None}")

_, _, pseudo_Bs_t1 = position_alignment_boltz2(
    model_output=out_t1, feats=feats, cra_name_sfc=cra_name,
    best_pos=reference_pos.clone(), exclude_res=None,
    domain_segs=config.domain_segs, reference_bfactor=init_pos_bfactor,
)

if plddt_tokens is not None:
    logger.info(f"  pLDDT  mean={plddt_tokens.mean().item():.3f}  "
                f"min={plddt_tokens.min().item():.3f}  max={plddt_tokens.max().item():.3f}")

logger.info(f"  raw pseudo_Bs  mean={pseudo_Bs_t1.mean():.1f}  "
            f"min={pseudo_Bs_t1.min():.1f}  max={pseudo_Bs_t1.max():.1f}  Å²")
clamped = pseudo_Bs_t1.clamp(max=200.0)
logger.info(f"  clamped (≤200) mean={clamped.mean():.1f}  "
            f"fraction_clamped={((pseudo_Bs_t1>200).float().mean()):.2%}")

# DWF at 3Å for clamped B
import math
s_3A = 1.0/(2*3.0)
mean_B  = clamped.mean().item()
dwf_3A  = math.exp(-mean_B * s_3A**2)
logger.info(f"  DWF at 3Å with mean B={mean_B:.0f} Å²: {dwf_3A:.5f}  "
            f"(F_calc attenuated to {dwf_3A*100:.2f}% of B=0 value)")

# ════════════════════════════════════════════════════════════════════════════
# T2 — Seed coverage: current (seed=initial only) vs fixed (seed=full trajectory)
# ════════════════════════════════════════════════════════════════════════════
logger.info("\n=== T2: Seed coverage — LLG variance across calls ===")

bias_dict = wrapper.init_bias(DEVICE)
wrapper.diffusion_seed = 0

# Run 5 forward passes with diffusion_seed=0 (current behaviour: initial noise fixed, eps random)
llg_current = []
with torch.no_grad():
    for _ in range(5):
        out = wrapper(feats, recycling_steps=0, num_sampling_steps=200)
        # just measure spread in atom_coords norm as a proxy
        c = out["sample_atom_coords"][0]
        llg_current.append(c.norm().item())

logger.info(f"  Current seed (initial only): coord-norm over 5 calls = "
            f"{[f'{v:.1f}' for v in llg_current]}  std={np.std(llg_current):.2f}")

# ── Now test: if we set the seed BEFORE every call (fixes full trajectory) ──
# We'll temporarily patch _sample_truncated to NOT restore rng_state
import rocket.boltz2_wrapper as bw_mod
orig_sample = wrapper._sample_truncated.__func__

def _sample_full_seed(self, s_trunk, s_inputs, feats, diffusion_conditioning, num_sampling_steps):
    """Patched version: seed covers the entire trajectory (no rng_state restore)."""
    from boltz.model.modules.diffusionv2 import AtomDiffusion
    from math import sqrt as _sqrt
    import torch.nn.functional as F

    diff_module = self._boltz.structure_module
    atom_mask   = feats["atom_pad_mask"].float()

    sigmas = diff_module.sample_schedule(num_sampling_steps)
    gammas = torch.where(sigmas > diff_module.gamma_min, diff_module.gamma_0, 0.0)
    sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

    # Seed EVERYTHING — do NOT restore rng_state
    if self.diffusion_seed is not None:
        torch.manual_seed(self.diffusion_seed)

    shape = (*atom_mask.shape, 3)
    init_sigma = sigmas[0]
    atom_coords = init_sigma * torch.randn(shape, device=atom_mask.device)

    detach_boundary = num_sampling_steps - self.K
    network_condition_kwargs = dict(
        s_trunk=s_trunk, s_inputs=s_inputs, feats=feats,
        diffusion_conditioning=diffusion_conditioning, multiplicity=1,
    )
    orig_aug = diff_module.coordinate_augmentation_inference
    diff_module.coordinate_augmentation_inference = False
    step_scale = diff_module.step_scale

    for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
        sigma_tm_f = sigma_tm.item()
        sigma_t_f  = sigma_t.item()
        gamma_f    = gamma.item()
        t_hat      = sigma_tm_f * (1 + gamma_f)
        noise_var  = diff_module.noise_scale**2 * (t_hat**2 - sigma_tm_f**2)
        eps = _sqrt(noise_var) * torch.randn_like(atom_coords)
        atom_coords_noisy = atom_coords.detach() + eps

        use_grad = step_idx >= detach_boundary
        ctx = torch.enable_grad() if use_grad else torch.no_grad()
        with ctx:
            atom_coords_denoised = diff_module.preconditioned_network_forward(
                atom_coords_noisy, t_hat,
                network_condition_kwargs=network_condition_kwargs,
            )

        if diff_module.alignment_reverse_diff:
            from boltz.model.modules.diffusionv2 import weighted_rigid_align
            with torch.no_grad():
                atom_coords_noisy_aligned = weighted_rigid_align(
                    atom_coords_noisy.float(), atom_coords_denoised.detach().float(),
                    atom_mask.float(), atom_mask.float(),
                ).to(atom_coords_denoised)
        else:
            atom_coords_noisy_aligned = atom_coords_noisy

        denoised_over_sigma = (atom_coords_noisy_aligned - atom_coords_denoised) / t_hat
        atom_coords_next = atom_coords_noisy_aligned + step_scale * (sigma_t_f - t_hat) * denoised_over_sigma

        if use_grad:
            atom_coords = atom_coords_next
        else:
            atom_coords = atom_coords_next.detach()

    diff_module.coordinate_augmentation_inference = orig_aug
    return {"sample_atom_coords": atom_coords}

# Monkey-patch
import types
wrapper._sample_truncated = types.MethodType(_sample_full_seed, wrapper)

llg_fixed = []
with torch.no_grad():
    for _ in range(5):
        out = wrapper(feats, recycling_steps=0, num_sampling_steps=200)
        c = out["sample_atom_coords"][0]
        llg_fixed.append(c.norm().item())

logger.info(f"  Fixed seed  (full traj):    coord-norm over 5 calls = "
            f"{[f'{v:.1f}' for v in llg_fixed]}  std={np.std(llg_fixed):.2f}")
logger.info(f"  → std ratio fixed/current: {np.std(llg_fixed)/max(np.std(llg_current),1e-9):.4f} "
            f"(0 = perfectly deterministic, 1 = same as before)")

# ════════════════════════════════════════════════════════════════════════════
# T3 — Gradient direction test with production settings + fixed full seed
# ════════════════════════════════════════════════════════════════════════════
logger.info("\n=== T3: Gradient direction (fixed full seed, production settings) ===")

bias_dict = wrapper.init_bias(DEVICE)
w = bias_dict["w_pair"]
b = bias_dict["b_pair"]
wrapper.diffusion_seed = 0

# Initialise sigmaA from identity-bias prediction
out0 = wrapper(feats, recycling_steps=3, num_sampling_steps=200)
aligned0, _, pBs0 = position_alignment_boltz2(
    out0, feats, cra_name, reference_pos.clone(), None,
    config.domain_segs, init_pos_bfactor,
)
safe_B0 = pBs0.detach().clamp(max=200.0)
llgloss.sfc.atom_b_iso     = safe_B0
llgloss_rbr.sfc.atom_b_iso = safe_B0
llgloss, llgloss_rbr, _, _ = rkrf_utils.update_sigmaA(
    llgloss=llgloss, llgloss_rbr=llgloss_rbr,
    aligned_xyz=aligned0.detach(),
    constant_fp_added_HKL=None, constant_fp_added_asu=None,
)

llgloss.sfc.atom_pos_orth = aligned0.detach().clone()
optim0, _ = rk_coordinates.rigidbody_refine_quat(
    aligned0, llgloss_rbr, cra_name,
    domain_segs=config.domain_segs, lbfgs=True,
    added_chain_HKL=None, added_chain_asu=None,
    lbfgs_lr=config.rbr_lbfgs_learning_rate, verbose=False,
)
llgloss.sfc.atom_pos_orth = optim0

llg0, rw0, _ = llgloss(
    optim0, bin_labels=None, num_batch=1, sub_ratio=1.0,
    solvent=config.solvent, update_scales=True, return_Rfactors=True,
)
(-llg0).backward()

w_grad = w.grad.clone() if w.grad is not None else None
b_grad = b.grad.clone() if b.grad is not None else None
logger.info(f"  LLG0 = {llg0.item():.2f}  Rwork0 = {rw0.item():.4f}")
logger.info(f"  |∇w| = {w_grad.norm().item():.5e}" if w_grad is not None else "  ∇w = None")
logger.info(f"  |∇b| = {b_grad.norm().item():.5e}" if b_grad is not None else "  ∇b = None")

if w_grad is not None:
    # Test: move a TINY step in -gradient direction, measure LLG change
    best_pos_t3 = optim0.detach().clone()
    for eps_lr in [1e-6, 1e-5, 1e-4, 1e-3]:
        with torch.no_grad():
            w_test = w.data - eps_lr * w_grad
            b_test = b.data - eps_lr * b_grad
            wrapper._bias = {"w_pair": w_test.requires_grad_(False),
                             "b_pair": b_test.requires_grad_(False)}
            out_t = wrapper(feats, recycling_steps=3, num_sampling_steps=200)
        aligned_t, _, pBs_t = position_alignment_boltz2(
            out_t, feats, cra_name, best_pos_t3, None,
            config.domain_segs, init_pos_bfactor,
        )
        safe_Bt = pBs_t.detach().clamp(max=200.0)
        llgloss.sfc.atom_b_iso     = safe_Bt
        llgloss_rbr.sfc.atom_b_iso = safe_Bt
        llgloss.sfc.atom_pos_orth  = aligned_t.detach().clone()
        optim_t, _ = rk_coordinates.rigidbody_refine_quat(
            aligned_t.detach(), llgloss_rbr, cra_name,
            domain_segs=config.domain_segs, lbfgs=True,
            added_chain_HKL=None, added_chain_asu=None,
            lbfgs_lr=config.rbr_lbfgs_learning_rate, verbose=False,
        )
        llgloss.sfc.atom_pos_orth = optim_t
        with torch.no_grad():
            llg_t, rw_t, _ = llgloss(
                optim_t, bin_labels=None, num_batch=1, sub_ratio=1.0,
                solvent=config.solvent, update_scales=False, return_Rfactors=True,
            )
        delta = llg_t.item() - llg0.item()
        logger.info(f"  lr={eps_lr:.0e}: LLG {llg0.item():.2f} → {llg_t.item():.2f}  "
                    f"delta={delta:+.3f}  Rwork={rw_t.item():.4f}")

# ════════════════════════════════════════════════════════════════════════════
# T4 — 10-iter loop with FIXED FULL SEED, lr=1e-3
# ════════════════════════════════════════════════════════════════════════════
logger.info("\n=== T4: 10-iter loop (fixed full seed, lr=1e-3) ===")

bias_dict = wrapper.init_bias(DEVICE)
w = bias_dict["w_pair"]
b = bias_dict["b_pair"]
wrapper.diffusion_seed = 0
best_pos = reference_pos.clone()

optimizer = torch.optim.Adam(
    [{"params": w, "lr": 1e-3}, {"params": b, "lr": 1e-3}]
)

# Init sigmaA once
out_init = wrapper(feats, recycling_steps=3, num_sampling_steps=200)
aligned_init, _, pBs_init = position_alignment_boltz2(
    out_init, feats, cra_name, best_pos, None, config.domain_segs, init_pos_bfactor,
)
safe_B_init = pBs_init.detach().clamp(max=200.0)
llgloss.sfc.atom_b_iso = safe_B_init
llgloss_rbr.sfc.atom_b_iso = safe_B_init
llgloss, llgloss_rbr, _, _ = rkrf_utils.update_sigmaA(
    llgloss=llgloss, llgloss_rbr=llgloss_rbr,
    aligned_xyz=aligned_init.detach(),
    constant_fp_added_HKL=None, constant_fp_added_asu=None,
)

llg_trace = []
for it in range(10):
    optimizer.zero_grad()
    out = wrapper(feats, recycling_steps=3, num_sampling_steps=200)
    aligned, _, pBs = position_alignment_boltz2(
        out, feats, cra_name, best_pos, None, config.domain_segs, init_pos_bfactor,
    )
    safe_B = pBs.detach().clamp(max=200.0)
    llgloss.sfc.atom_b_iso     = safe_B
    llgloss_rbr.sfc.atom_b_iso = safe_B
    llgloss.sfc.atom_pos_orth  = aligned.detach().clone()

    optim_xyz, _ = rk_coordinates.rigidbody_refine_quat(
        aligned, llgloss_rbr, cra_name,
        domain_segs=config.domain_segs, lbfgs=True,
        added_chain_HKL=None, added_chain_asu=None,
        lbfgs_lr=config.rbr_lbfgs_learning_rate, verbose=False,
    )
    llgloss.sfc.atom_pos_orth = optim_xyz

    llg, rw, rf = llgloss(
        optim_xyz, bin_labels=None, num_batch=1, sub_ratio=1.0,
        solvent=config.solvent, update_scales=(it == 0), return_Rfactors=True,
    )
    (-llg).backward()
    torch.nn.utils.clip_grad_norm_([w, b], max_norm=10.0)
    optimizer.step()
    llgloss.sfc.atom_pos_orth = optim_xyz
    best_pos = optim_xyz.detach().clone()
    llg_trace.append(llg.item())
    wg = w.grad.norm().item() if w.grad is not None else 0.0
    logger.info(f"  iter {it:2d}: LLG={llg.item():8.2f}  Rwork={rw.item():.4f}  |∇w|={wg:.3e}")

first5 = np.mean(llg_trace[:5])
last5  = np.mean(llg_trace[5:])
logger.info(f"\n  T4 first-5={first5:.2f}  last-5={last5:.2f}  "
            f"→ {'IMPROVING' if last5>first5 else 'NOT IMPROVING'}")
