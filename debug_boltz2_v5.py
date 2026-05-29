"""
Debug v5: All-recycling-pass bias injection.

Current design injects the pair bias only on the FINAL recycling pass.
This experiment asks: what if we inject at EVERY recycling pass?

Tests:
  T1: Structural sensitivity (RMSD vs w_dev) for final-pass-only vs all-passes injection
  T2: LLG and gradient norms at identity (all-passes)
  T3: Short 30-iter optimisation loop with all-passes injection
      (does more structural signal translate to actual LLG improvement?)
"""
import sys, pickle, torch, numpy as np, types
from torch.utils import checkpoint as grad_checkpoint
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


def _run_trunk_all_passes(self, feats, recycling_steps, w_pair, b_pair):
    """Modified trunk: inject pair bias at EVERY recycling pass, not just last."""
    model = self._boltz
    s_inputs = model.input_embedder(feats)
    s_init   = model.s_init(s_inputs)
    z_init   = (model.z_init_1(s_inputs)[:, :, None]
                + model.z_init_2(s_inputs)[:, None, :])
    z_init   = z_init + model.rel_pos(feats)
    z_init   = z_init + model.token_bonds(feats["token_bonds"].float())
    z_init   = z_init + model.contact_conditioning(feats)

    mask      = feats["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]
    s = torch.zeros_like(s_init)
    z = torch.zeros_like(z_init)

    for i in range(recycling_steps + 1):
        is_last = i == recycling_steps
        ctx = torch.enable_grad() if is_last else torch.no_grad()
        with ctx:
            s_in = s.detach() if not is_last else s
            z_in = z.detach() if not is_last else z
            s = s_init + model.s_recycle(model.s_norm(s_in))
            z = z_init + model.z_recycle(model.z_norm(z_in))
            if model.use_templates:
                z = z + model.template_module(z, feats, pair_mask, use_kernels=False)
            z = z + model.msa_module(z, s_inputs, feats, use_kernels=False)

            # MODIFICATION: inject at EVERY pass (not just is_last)
            z = torch.matmul(z, w_pair) + b_pair

            if is_last:
                chunk_size_tri_attn = 128 if z.shape[1] > 512 else 512
                for layer in model.pairformer_module.layers:
                    def run_block(s_t, z_t, _l=layer):
                        return _l(s_t, z_t, mask, pair_mask, chunk_size_tri_attn, False)
                    s, z = grad_checkpoint.checkpoint(run_block, s, z, use_reentrant=False)
            else:
                s, z = model.pairformer_module(s, z, mask=mask, pair_mask=pair_mask, use_kernels=False)
    return s, z


# ─── Helpers ─────────────────────────────────────────────────────────────────
def get_coords(w, b, seed, all_passes=False):
    wrapper.init_bias(DEVICE)
    wrapper._bias['w_pair'].data.copy_(w)
    wrapper._bias['b_pair'].data.copy_(b)
    if all_passes:
        wrapper._run_trunk = types.MethodType(_run_trunk_all_passes, wrapper)
    else:
        # restore original (reload the bound method from the class)
        wrapper._run_trunk = types.MethodType(Boltz2PairBias._run_trunk, wrapper)
    wrapper.diffusion_seed = seed
    with torch.no_grad():
        out  = wrapper(feats)
        xyz, _, Bs = position_alignment_boltz2(out, feats, cra_name, ref_pos)
    return xyz.detach(), Bs.detach().clamp(max=200.0)

def get_llg_no_grad(w, b, seed, all_passes=False):
    xyz, Bs = get_coords(w, b, seed, all_passes)
    llgloss.sfc.atom_b_iso     = Bs
    llgloss_rbr.sfc.atom_b_iso = Bs
    rkrf_utils.update_sigmaA(llgloss, llgloss_rbr, xyz)
    opt, _ = rk_coordinates.rigidbody_refine_quat(xyz, llgloss_rbr, cra_name, lbfgs=True, lbfgs_lr=150.0, verbose=False)
    llg, rw, _ = llgloss(opt, sub_ratio=1.0, solvent=True, update_scales=True, return_Rfactors=True)
    return llg.item(), rw.item()

w_id = torch.eye(128, device=DEVICE)
b_id = torch.zeros(128, device=DEVICE)


# ─── T1: Structural sensitivity — final-pass-only vs all-passes ───────────────
print("=" * 65)
print("T1: Structural sensitivity: final-pass-only  vs  all-passes")
print("    (RMSD vs w_dev, seed=4)")
print("=" * 65)
torch.manual_seed(42)
devs = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
print(f"{'w_dev':>6}  {'RMSD_final':>12}  {'RMSD_all':>12}  {'ratio':>8}")
results_t1 = []
xyz_base_final, _ = get_coords(w_id, b_id, 4, all_passes=False)
xyz_base_all,   _ = get_coords(w_id, b_id, 4, all_passes=True)
for dev in devs:
    delta = torch.randn(128, 128, device=DEVICE)
    delta = delta / delta.norm() * dev
    w_r = w_id + delta
    xyz_f, _ = get_coords(w_r, b_id, 4, all_passes=False)
    xyz_a, _ = get_coords(w_r, b_id, 4, all_passes=True)
    rmsd_f = torch.sqrt(((xyz_base_final - xyz_f)**2).sum(-1).mean()).item()
    rmsd_a = torch.sqrt(((xyz_base_all   - xyz_a)**2).sum(-1).mean()).item()
    ratio  = rmsd_a / rmsd_f if rmsd_f > 0 else float('nan')
    print(f"{dev:>6.2f}  {rmsd_f:>12.4f}  {rmsd_a:>12.4f}  {ratio:>8.2f}x")
    results_t1.append((dev, rmsd_f, rmsd_a))
np.save('debug_v5_t1.npy', np.array(results_t1))
print()


# ─── T2: Gradient norm at identity — all-passes ───────────────────────────────
print("=" * 65)
print("T2: Gradient norms at identity — all-passes injection (seed=4)")
print("=" * 65)
wrapper.init_bias(DEVICE)
wrapper._run_trunk = types.MethodType(_run_trunk_all_passes, wrapper)
w_p = wrapper._bias['w_pair']
b_p = wrapper._bias['b_pair']
wrapper.diffusion_seed = 4
out  = wrapper(feats)
xyz, _, Bs = position_alignment_boltz2(out, feats, cra_name, ref_pos)
safe_Bs = Bs.detach().clamp(max=200.0)
llgloss.sfc.atom_b_iso     = safe_Bs
llgloss_rbr.sfc.atom_b_iso = safe_Bs
rkrf_utils.update_sigmaA(llgloss, llgloss_rbr, xyz.detach())
opt_xyz, _ = rk_coordinates.rigidbody_refine_quat(xyz, llgloss_rbr, cra_name, lbfgs=True, lbfgs_lr=150.0, verbose=False)
llg, rw, _ = llgloss(opt_xyz, sub_ratio=1.0, solvent=True, update_scales=True, return_Rfactors=True)
(-llg).backward()
print(f"  LLG={llg.item():.1f}  Rwork={rw.item():.4f}")
print(f"  |∇w_pair|={w_p.grad.norm().item():.4f}  (final-only was 0.3756)")
print(f"  |∇b_pair|={b_p.grad.norm().item():.6f}")
print()


# ─── T3: 30-iter optimisation — all-passes, lr=1e-4 ──────────────────────────
print("=" * 65)
print("T3: 30-iter optimisation — all-passes injection, lr=1e-4, seed=4")
print("=" * 65)
sfc3    = llg_sf.initial_SFC(PDB, tng2, 'FEFF', 'DOBS', Freelabel='R-free-flags',
                              device=DEVICE, testset_value=1, spacing=4.5)
sfc3_rbr = llg_sf.initial_SFC(PDB, tng2, 'FEFF', 'DOBS', Freelabel='R-free-flags',
                               device=DEVICE, testset_value=1, solvent=False, spacing=4.5)
llg3     = rkrf_utils.init_llgloss(sfc3,     tng2, 3.0, None)
llg3_rbr = rkrf_utils.init_llgloss(sfc3_rbr, tng2, 3.0, None)
ref3     = sfc3.atom_pos_orth.clone()

wrapper.init_bias(DEVICE)
wrapper._run_trunk = types.MethodType(_run_trunk_all_passes, wrapper)
w3 = wrapper._bias['w_pair']
b3 = wrapper._bias['b_pair']
opt3 = torch.optim.Adam([{'params': w3, 'lr': 1e-4}, {'params': b3, 'lr': 1e-4}])
wrapper.diffusion_seed = 4

t3_results = []
for it in range(30):
    opt3.zero_grad()
    out3  = wrapper(feats)
    xyz3, _, Bs3 = position_alignment_boltz2(out3, feats, cra_name, ref3)
    safe3 = Bs3.detach().clamp(max=200.0)
    llg3.sfc.atom_b_iso     = safe3
    llg3_rbr.sfc.atom_b_iso = safe3
    rkrf_utils.update_sigmaA(llg3, llg3_rbr, xyz3.detach())
    opt3_xyz, _ = rk_coordinates.rigidbody_refine_quat(xyz3, llg3_rbr, cra_name, lbfgs=True, lbfgs_lr=150.0, verbose=False)
    llg_it, rw_it, _ = llg3(opt3_xyz, sub_ratio=1.0, solvent=True, update_scales=True, return_Rfactors=True)
    (-llg_it).backward()
    gn = torch.nn.utils.clip_grad_norm_([w3, b3], max_norm=10.0)
    opt3.step()
    wdev = torch.norm(w3 - torch.eye(128, device=DEVICE)).item()
    print(f"  iter {it:3d}: LLG={llg_it.item():8.1f}  Rwork={rw_it.item():.4f}  "
          f"grad_norm={gn:.4f}  w_dev={wdev:.4f}")
    t3_results.append((it, llg_it.item(), rw_it.item(), wdev))

np.save('debug_v5_t3.npy', np.array(t3_results))
print()
print(f"  Best LLG = {max(r[1] for r in t3_results):.1f} "
      f"at iter {max(range(len(t3_results)), key=lambda i: t3_results[i][1])}")
