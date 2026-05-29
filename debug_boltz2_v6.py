"""
Debug v6: Single-step deterministic denoising — the ConForNets approach.

ConForNets (arxiv 2604.18559) uses a SINGLE deterministic denoising step
x̂₀ = D_θ(σ_max·ε, σ_max, z_biased) for gradient computation, NOT TBPTT
through stochastic sampling steps.

Gradient path (single-step):
  w_pair → z @ w_pair + b → PairFormer → DiffusionConditioning → (q, c)
    → preconditioned_network_forward(σ_max·ε, σ_max) → x̂₀ → LLG

vs our TBPTT path (200 stochastic steps, last K with grad):
  noisy, long, non-deterministic

Tests:
  T1: Structural sensitivity — single-step RMSD vs w_dev  (compare to TBPTT T4)
  T2: LLG variance at identity (single-step x̂₀) — noise floor
  T3: 30-iter optimisation loop using single-step gradients
      Does this actually improve LLG and Rwork?
"""
import sys, pickle, torch, numpy as np
sys.path.insert(0, '/data/dust/group/it/crystalsfirst/dev/shehry/ROCKET')

from rocket.boltz2_wrapper import Boltz2PairBias
from rocket.xtal import structurefactors as llg_sf
from rocket import refinement_utils as rkrf_utils
from rocket import utils as rk_utils
from rocket import coordinates as rk_coordinates
from rocket.coordinates_boltz2 import position_alignment_boltz2

DEVICE = 'cuda:0'
CKPT   = '/data/dust/group/it/crystalsfirst/dev/shehry/boltz_cache/boltz2_conf.ckpt'
TNG    = '/data/dust/group/it/crystalsfirst/dev/shehry/data/thesis/1lj5/data_for_1lj5/1lj5_boltz_mod/ROCKET_inputs/1lj5-Edata.mtz'
PDB    = '/data/dust/group/it/crystalsfirst/dev/shehry/data/thesis/1lj5/data_for_1lj5/1lj5_boltz_mod/ROCKET_inputs/1lj5-pred-aligned.pdb'
FEATS  = '/data/dust/group/it/crystalsfirst/dev/shehry/data/thesis/1lj5/data_for_1lj5/1lj5_boltz_mod/ROCKET_inputs/feats_boltz2.pkl'

tng2  = rk_utils.apply_resolution_cutoff(TNG, min_resolution=3.0)
feats = pickle.load(open(FEATS, 'rb'))
feats = rk_utils.move_tensors_to_device(feats, device=DEVICE)

wrapper = Boltz2PairBias(CKPT, truncated_backprop_steps=20, num_sampling_steps=200,
                         recycling_steps=3, device=DEVICE).to(DEVICE).eval()

sfc = llg_sf.initial_SFC(PDB, tng2, 'FEFF', 'DOBS', Freelabel='R-free-flags',
                          device=DEVICE, testset_value=1, spacing=4.5)
sfc_rbr = llg_sf.initial_SFC(PDB, tng2, 'FEFF', 'DOBS', Freelabel='R-free-flags',
                              device=DEVICE, testset_value=1, solvent=False, spacing=4.5)
llgloss     = rkrf_utils.init_llgloss(sfc,     tng2, 3.0, None)
llgloss_rbr = rkrf_utils.init_llgloss(sfc_rbr, tng2, 3.0, None)
ref_pos  = sfc.atom_pos_orth.clone()
cra_name = sfc.cra_name


def forward_one_step(wrapper, feats, w_pair, b_pair, seed=None):
    """
    Single deterministic denoising step — ConForNets style.

    Trunk forward is identical to the standard path.
    Replaces the 200-step Euler loop with a single call to
    preconditioned_network_forward at σ_max.

    Returns x̂₀ = D_θ(σ_max·ε, σ_max, z_biased) — grad-carrying.
    """
    model = wrapper._boltz

    # 1. Trunk with pair bias (same as standard forward)
    wrapper._bias['w_pair'].data.copy_(w_pair)
    wrapper._bias['b_pair'].data.copy_(b_pair)
    s, z = wrapper._run_trunk(feats, wrapper.recycling_steps, w_pair, b_pair)

    # 2. Diffusion conditioning (grad flows through z → w_pair)
    relative_pos = model.rel_pos(feats)
    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = model.diffusion_conditioning(
        s_trunk=s, z_trunk=z, relative_position_encoding=relative_pos, feats=feats)
    diffusion_conditioning = dict(q=q, c=c, to_keys=to_keys,
                                  atom_enc_bias=atom_enc_bias,
                                  atom_dec_bias=atom_dec_bias,
                                  token_trans_bias=token_trans_bias)
    s_inputs = model.input_embedder(feats)

    # 3. Single-step denoising: x̂₀ = D_θ(σ_max·ε, σ_max)
    diff_module = model.structure_module
    atom_mask   = feats["atom_pad_mask"].float()
    sigmas = diff_module.sample_schedule(wrapper.num_sampling_steps)
    sigma_max = sigmas[0].item()

    # Fix seed for reproducibility
    if seed is not None:
        rng = torch.get_rng_state()
        torch.manual_seed(seed)
    shape = (*atom_mask.shape, 3)
    atom_coords_noisy = sigma_max * torch.randn(shape, device=DEVICE)
    if seed is not None:
        torch.set_rng_state(rng)

    orig_aug = diff_module.coordinate_augmentation_inference
    diff_module.coordinate_augmentation_inference = False

    with torch.autocast("cuda", enabled=False):
        x_hat = diff_module.preconditioned_network_forward(
            atom_coords_noisy.float(),
            sigma_max,
            network_condition_kwargs=dict(
                s_trunk=s.float(), s_inputs=s_inputs.float(),
                feats=feats, diffusion_conditioning=diffusion_conditioning,
                multiplicity=1),
        )

    diff_module.coordinate_augmentation_inference = orig_aug
    return {"sample_atom_coords": x_hat, "s": s, "z": z}


def get_coords_one_step(w, b, seed):
    wrapper.init_bias(DEVICE)
    out = forward_one_step(wrapper, feats, w, b, seed=seed)
    with torch.no_grad():
        xyz, _, Bs = position_alignment_boltz2(out, feats, cra_name, ref_pos)
    return xyz.detach(), Bs.detach().clamp(max=200.0)


def get_coords_one_step_grad(w, b, seed):
    """Same but keeps grad on xyz for backprop."""
    wrapper.init_bias(DEVICE)
    out = forward_one_step(wrapper, feats, w, b, seed=seed)
    xyz, _, Bs = position_alignment_boltz2(out, feats, cra_name, ref_pos)
    return xyz, Bs.detach().clamp(max=200.0)


def sfc_pass(xyz, Bs, sub_ratio=1.0):
    llgloss.sfc.atom_b_iso     = Bs
    llgloss_rbr.sfc.atom_b_iso = Bs
    rkrf_utils.update_sigmaA(llgloss, llgloss_rbr, xyz.detach() if xyz.requires_grad else xyz)
    opt, _ = rk_coordinates.rigidbody_refine_quat(
        xyz, llgloss_rbr, cra_name, lbfgs=True, lbfgs_lr=150.0, verbose=False)
    llg, rw, _ = llgloss(opt, sub_ratio=sub_ratio, solvent=True, update_scales=True, return_Rfactors=True)
    return llg, rw, opt


w_id = torch.eye(128, device=DEVICE)
b_id = torch.zeros(128, device=DEVICE)


# ─── T1: Structural sensitivity — single-step vs TBPTT ───────────────────────
print("=" * 65)
print("T1: Structural sensitivity — single-step denoising (seed=4)")
print("    Compare to TBPTT results from debug_v4:")
print("    TBPTT: w_dev=0.3→0.108Å, w_dev=2.0→0.191Å")
print("=" * 65)
torch.manual_seed(42)
xyz_base, _ = get_coords_one_step(w_id, b_id, seed=4)
devs = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 2.00]
t1_results = []
for dev in devs:
    delta = torch.randn(128, 128, device=DEVICE)
    delta = delta / delta.norm() * dev
    w_r   = w_id + delta
    xyz_r, _ = get_coords_one_step(w_r, b_id, seed=4)
    rmsd  = torch.sqrt(((xyz_base - xyz_r)**2).sum(-1).mean()).item()
    sens  = rmsd / dev
    print(f"  w_dev={dev:.2f}: RMSD={rmsd:.4f} Å  (sensitivity: {sens:.4f} Å/unit)")
    t1_results.append((dev, rmsd))
np.save('debug_v6_t1.npy', np.array(t1_results))
print()


# ─── T2: LLG noise floor — single-step, 20 seeds ────────────────────────────
print("=" * 65)
print("T2: LLG at identity — single-step x̂₀, 20 seeds")
print("    (noise floor — compare to TBPTT std=273)")
print("=" * 65)
llgs_t2, rworks_t2 = [], []
for seed in range(20):
    xyz, Bs = get_coords_one_step(w_id, b_id, seed=seed)
    llg, rw, _ = sfc_pass(xyz, Bs)
    llgs_t2.append(llg.item())
    rworks_t2.append(rw.item())
    print(f"  seed {seed:2d}: LLG={llg.item():8.1f}  Rwork={rw.item():.4f}")
print(f"\n  mean={np.mean(llgs_t2):.1f}  std={np.std(llgs_t2):.1f}  "
      f"min={np.min(llgs_t2):.1f}  max={np.max(llgs_t2):.1f}")
print(f"  Range = {np.max(llgs_t2)-np.min(llgs_t2):.1f}  "
      f"(TBPTT was 987 — does single-step reduce variance?)")
np.save('debug_v6_t2.npy', np.array(list(zip(range(20), llgs_t2, rworks_t2))))
print()


# ─── T3: 30-iter optimisation — single-step gradients ───────────────────────
print("=" * 65)
print("T3: 30-iter optimisation — single-step gradients, lr=1e-4, seed=4")
print("=" * 65)
sfc3    = llg_sf.initial_SFC(PDB, tng2, 'FEFF', 'DOBS', Freelabel='R-free-flags',
                              device=DEVICE, testset_value=1, spacing=4.5)
sfc3_r  = llg_sf.initial_SFC(PDB, tng2, 'FEFF', 'DOBS', Freelabel='R-free-flags',
                               device=DEVICE, testset_value=1, solvent=False, spacing=4.5)
llg3    = rkrf_utils.init_llgloss(sfc3,  tng2, 3.0, None)
llg3_r  = rkrf_utils.init_llgloss(sfc3_r, tng2, 3.0, None)
ref3    = sfc3.atom_pos_orth.clone()

wrapper.init_bias(DEVICE)
w3 = wrapper._bias['w_pair']
b3 = wrapper._bias['b_pair']
opt3 = torch.optim.Adam([{'params': w3, 'lr': 1e-4}, {'params': b3, 'lr': 1e-4}])

t3_results = []
for it in range(30):
    opt3.zero_grad()
    # Forward with gradient
    out3 = forward_one_step(wrapper, feats, w3, b3, seed=4)
    xyz3, _, Bs3 = position_alignment_boltz2(out3, feats, cra_name, ref3)
    safe3 = Bs3.detach().clamp(max=200.0)
    llg3.sfc.atom_b_iso  = safe3
    llg3_r.sfc.atom_b_iso = safe3
    rkrf_utils.update_sigmaA(llg3, llg3_r, xyz3.detach())
    opt3_xyz, _ = rk_coordinates.rigidbody_refine_quat(
        xyz3, llg3_r, cra_name, lbfgs=True, lbfgs_lr=150.0, verbose=False)
    llg_it, rw_it, _ = llg3(opt3_xyz, sub_ratio=1.0, solvent=True, update_scales=True, return_Rfactors=True)
    (-llg_it).backward()
    gn   = torch.nn.utils.clip_grad_norm_([w3, b3], max_norm=10.0)
    opt3.step()
    wdev = torch.norm(w3 - torch.eye(128, device=DEVICE)).item()
    print(f"  iter {it:3d}: LLG={llg_it.item():8.1f}  Rwork={rw_it.item():.4f}  "
          f"grad_norm={gn:.4f}  w_dev={wdev:.4f}")
    t3_results.append((it, llg_it.item(), rw_it.item(), wdev))

np.save('debug_v6_t3.npy', np.array(t3_results))
best_it = max(range(len(t3_results)), key=lambda i: t3_results[i][1])
print(f"\n  Best LLG = {t3_results[best_it][1]:.1f} at iter {best_it}  "
      f"Rwork = {t3_results[best_it][2]:.4f}")
print(f"  Identity baseline (seed=4) ≈ 1086  →  gain = "
      f"{t3_results[best_it][1]-1086:.1f} LLG units")
print()


# ─── T4: Gradient norm comparison ────────────────────────────────────────────
print("=" * 65)
print("T4: Gradient quality — single-step vs TBPTT (seed=4, identity)")
print("=" * 65)
wrapper.init_bias(DEVICE)
w_p = wrapper._bias['w_pair']
b_p = wrapper._bias['b_pair']
out = forward_one_step(wrapper, feats, w_p, b_p, seed=4)
xyz, _, Bs = position_alignment_boltz2(out, feats, cra_name, ref_pos)
llgloss.sfc.atom_b_iso     = Bs.detach().clamp(max=200.0)
llgloss_rbr.sfc.atom_b_iso = Bs.detach().clamp(max=200.0)
rkrf_utils.update_sigmaA(llgloss, llgloss_rbr, xyz.detach())
opt_xyz, _ = rk_coordinates.rigidbody_refine_quat(
    xyz, llgloss_rbr, cra_name, lbfgs=True, lbfgs_lr=150.0, verbose=False)
llg, rw, _ = llgloss(opt_xyz, sub_ratio=1.0, solvent=True, update_scales=True, return_Rfactors=True)
(-llg).backward()
print(f"  Single-step: LLG={llg.item():.1f}  Rwork={rw.item():.4f}")
print(f"  |∇w_pair|={w_p.grad.norm().item():.4f}  (TBPTT was 0.376)")
print(f"  max ∇w element={w_p.grad.abs().max().item():.4f}")
