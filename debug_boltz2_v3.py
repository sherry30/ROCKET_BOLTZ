"""
Debug v3: diagnose gradient noise and collapse mechanism.

Tests:
  T1: LLG variance from subset sampling (sub_ratio=0.7) at identity bias
  T2: Gradient direction quality — does a step in grad direction improve FULL LLG?
  T3: Step-by-step optimization, logging grad norms and full-reflection LLG each iter
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

tng2   = rk_utils.apply_resolution_cutoff(TNG, min_resolution=3.0)
feats  = pickle.load(open(FEATS, 'rb'))
feats  = rk_utils.move_tensors_to_device(feats, device=DEVICE)

wrapper = Boltz2PairBias(CKPT, truncated_backprop_steps=5, num_sampling_steps=200,
                         recycling_steps=3, device=DEVICE).to(DEVICE).eval()
wrapper.diffusion_seed = 4   # best seed from scan

def make_sfc():
    sfc = llg_sf.initial_SFC(PDB, tng2, 'FEFF', 'DOBS', Freelabel='R-free-flags',
                              device=DEVICE, testset_value=1, spacing=4.5)
    sfc_rbr = llg_sf.initial_SFC(PDB, tng2, 'FEFF', 'DOBS', Freelabel='R-free-flags',
                                  device=DEVICE, testset_value=1, solvent=False, spacing=4.5)
    llgloss = rkrf_utils.init_llgloss(sfc, tng2, 3.0, None)
    llgloss_rbr = rkrf_utils.init_llgloss(sfc_rbr, tng2, 3.0, None)
    return sfc, llgloss, llgloss_rbr

def model_forward(ref_pos, cra_name):
    """Model forward only — wrapped in no_grad. Returns detached xyz and pseudo_Bs."""
    with torch.no_grad():
        model_out = wrapper(feats)
        aligned_xyz, _, pseudo_Bs = position_alignment_boltz2(
            model_out, feats, cra_name, ref_pos)
    return aligned_xyz.detach(), pseudo_Bs.detach().clamp(max=200.0)


def model_forward_grad(ref_pos, cra_name):
    """Model forward WITH gradient — for backprop through w_pair/b_pair."""
    model_out = wrapper(feats)
    aligned_xyz, _, pseudo_Bs = position_alignment_boltz2(
        model_out, feats, cra_name, ref_pos)
    return aligned_xyz, pseudo_Bs.detach().clamp(max=200.0)


def sfc_pass(llgloss, llgloss_rbr, aligned_xyz, pseudo_Bs, sub_ratio=1.0):
    """SFC ops — update_sigmaA uses internal backward(), must NOT be inside no_grad."""
    llgloss.sfc.atom_b_iso = pseudo_Bs
    llgloss_rbr.sfc.atom_b_iso = pseudo_Bs
    rkrf_utils.update_sigmaA(llgloss, llgloss_rbr, aligned_xyz)
    opt_xyz, _ = rk_coordinates.rigidbody_refine_quat(
        aligned_xyz, llgloss_rbr, llgloss.sfc.cra_name, lbfgs=True, lbfgs_lr=150.0, verbose=False)
    llg, rw, _ = llgloss(opt_xyz, sub_ratio=sub_ratio, solvent=True,
                         update_scales=True, return_Rfactors=True)
    return llg, rw, opt_xyz


sfc, llgloss, llgloss_rbr = make_sfc()
ref_pos = sfc.atom_pos_orth.clone()
cra_name = sfc.cra_name

print("=" * 60)
print("T1: LLG variance from subset sampling at identity bias")
print("=" * 60)
wrapper.init_bias(DEVICE)   # identity w_pair, zero b_pair
full_llgs = []
sub_llgs = []
for i in range(10):
    xyz_d, Bs_d = model_forward(ref_pos, cra_name)
    full_llg, rw, _ = sfc_pass(llgloss, llgloss_rbr, xyz_d, Bs_d, sub_ratio=1.0)
    full_llgs.append(full_llg.item())
    xyz_d2, Bs_d2 = model_forward(ref_pos, cra_name)
    sub_llg, _, _ = sfc_pass(llgloss, llgloss_rbr, xyz_d2, Bs_d2, sub_ratio=0.7)
    sub_llgs.append(sub_llg.item() / 0.7)
print(f"Full (sub_ratio=1.0): mean={np.mean(full_llgs):.1f}  std={np.std(full_llgs):.1f}")
print(f"Sub  (sub_ratio=0.7, scaled): mean={np.mean(sub_llgs):.1f}  std={np.std(sub_llgs):.1f}")

print()
print("=" * 60)
print("T2: Gradient direction — does a tiny step improve FULL LLG?")
print("=" * 60)
bias2 = wrapper.init_bias(DEVICE)
w2, b2 = bias2['w_pair'], bias2['b_pair']
opt2 = torch.optim.Adam([{'params': w2, 'lr': 1e-3}, {'params': b2, 'lr': 1e-3}])

xyz_d0, Bs_d0 = model_forward(ref_pos, cra_name)
llg0, rw0, _ = sfc_pass(llgloss, llgloss_rbr, xyz_d0, Bs_d0, sub_ratio=1.0)
print(f"Before step:  LLG={llg0.item():.1f}  Rwork={rw0.item():.4f}")

opt2.zero_grad()
xyz_g, Bs_g = model_forward_grad(ref_pos, cra_name)
llg_g, _, _ = sfc_pass(llgloss, llgloss_rbr, xyz_g, Bs_g, sub_ratio=1.0)
(-llg_g).backward()
gn = torch.nn.utils.clip_grad_norm_([w2, b2], max_norm=10.0)
print(f"  w2 grad: mean={w2.grad.abs().mean():.6f}  max={w2.grad.abs().max():.6f}  clipped_norm={gn:.4f}")
opt2.step()

xyz_d1, Bs_d1 = model_forward(ref_pos, cra_name)
llg1, rw1, _ = sfc_pass(llgloss, llgloss_rbr, xyz_d1, Bs_d1, sub_ratio=1.0)
print(f"After step (lr=1e-3):  LLG={llg1.item():.1f}  Rwork={rw1.item():.4f}  Δ={llg1.item()-llg0.item():.1f}")

print()
print("=" * 60)
print("T3: 30-iter loop with sub_ratio=1.0, lr=1e-4 (low noise)")
print("=" * 60)
sfc3, llg3, llg3_rbr = make_sfc()
ref3 = sfc3.atom_pos_orth.clone()
bias3 = wrapper.init_bias(DEVICE)
w3, b3 = bias3['w_pair'], bias3['b_pair']
opt3 = torch.optim.Adam([{'params': w3, 'lr': 1e-4}, {'params': b3, 'lr': 1e-4}])

for it in range(30):
    opt3.zero_grad()
    xyz_it, Bs_it = model_forward_grad(ref3, sfc3.cra_name)
    llg_it, rw_it, _ = sfc_pass(llg3, llg3_rbr, xyz_it, Bs_it, sub_ratio=1.0)
    (-llg_it).backward()
    gn = torch.nn.utils.clip_grad_norm_([w3, b3], max_norm=10.0)
    opt3.step()
    print(f"  iter {it:3d}: LLG={llg_it.item():8.1f}  Rwork={rw_it.item():.4f}  grad_norm={gn:.4f}  w_dev={torch.norm(w3 - torch.eye(128,device=DEVICE)):.4f}")

