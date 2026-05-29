"""
Debug v4: Does the pair bias actually change the predicted structure?

Tests:
  T1: RMSD between identity-bias prediction and best PDB (iter 19)
      → If RMSD is tiny, the bias has negligible structural effect.

  T2: Gradient magnitude at identity (seed=4, full reflections)
      → Is the gradient signal meaningful?

  T3: Noise-averaged LLG over 20 seeds at identity
      → Baseline LLG distribution and variance.

  T4: Structural sensitivity — random w_pair perturbations of increasing size
      → How much does the structure actually move per unit of w_pair change?
"""
import sys, pickle, torch, numpy as np
sys.path.insert(0, '/data/dust/group/it/crystalsfirst/dev/shehry/ROCKET')

from rocket.boltz2_wrapper import Boltz2PairBias
from rocket.xtal import structurefactors as llg_sf
from rocket import refinement_utils as rkrf_utils
from rocket import utils as rk_utils
from rocket import coordinates as rk_coordinates
from rocket.coordinates_boltz2 import position_alignment_boltz2
from SFC_Torch import SFcalculator

DEVICE   = 'cuda:0'
CKPT     = '/data/dust/group/it/crystalsfirst/dev/shehry/boltz_cache/boltz2_conf.ckpt'
TNG      = '/data/dust/group/it/crystalsfirst/dev/shehry/data/thesis/1lj5/data_for_1lj5/1lj5_boltz_mod/ROCKET_inputs/1lj5-Edata.mtz'
PDB      = '/data/dust/group/it/crystalsfirst/dev/shehry/data/thesis/1lj5/data_for_1lj5/1lj5_boltz_mod/ROCKET_inputs/1lj5-pred-aligned.pdb'
BEST_PDB = '/data/dust/group/it/crystalsfirst/dev/shehry/data/thesis/1lj5/data_for_1lj5/1lj5_boltz_mod/ROCKET_outputs/a4766be58a/phase1_boltz2_1lj5/A_19_postRBR.pdb'
FEATS    = '/data/dust/group/it/crystalsfirst/dev/shehry/data/thesis/1lj5/data_for_1lj5/1lj5_boltz_mod/ROCKET_inputs/feats_boltz2.pkl'

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

def get_coords(w_pair, b_pair, seed):
    wrapper.init_bias(DEVICE)
    wrapper._bias['w_pair'].data.copy_(w_pair)
    wrapper._bias['b_pair'].data.copy_(b_pair)
    wrapper.diffusion_seed = seed
    with torch.no_grad():
        out = wrapper(feats)
        xyz, _, Bs = position_alignment_boltz2(out, feats, cra_name, ref_pos)
    return xyz.detach(), Bs.detach().clamp(max=200.0)

def get_llg(w_pair, b_pair, seed):
    xyz, Bs = get_coords(w_pair, b_pair, seed)
    llgloss.sfc.atom_b_iso     = Bs
    llgloss_rbr.sfc.atom_b_iso = Bs
    rkrf_utils.update_sigmaA(llgloss, llgloss_rbr, xyz)
    opt_xyz, _ = rk_coordinates.rigidbody_refine_quat(
        xyz, llgloss_rbr, cra_name, lbfgs=True, lbfgs_lr=150.0, verbose=False)
    llg, rw, _ = llgloss(opt_xyz, sub_ratio=1.0, solvent=True, update_scales=True, return_Rfactors=True)
    return llg.item(), rw.item(), opt_xyz

w_id = torch.eye(128, device=DEVICE)
b_id = torch.zeros(128, device=DEVICE)

# ─── T1: RMSD between identity-prediction and best saved PDB ─────────────────
print("=" * 60)
print("T1: RMSD(Boltz2 identity-bias coords, best saved PDB at iter 19)")
print("=" * 60)
# Load best PDB coordinates via a fresh SFC
sfc_best = llg_sf.initial_SFC(BEST_PDB, tng2, 'FEFF', 'DOBS', Freelabel='R-free-flags',
                                device=DEVICE, testset_value=1, spacing=4.5)
best_pos_from_pdb = sfc_best.atom_pos_orth.clone()

rmsds = []
for seed in [4, 0, 3]:
    xyz_id, _ = get_coords(w_id, b_id, seed)
    diff = xyz_id - best_pos_from_pdb
    rmsd = torch.sqrt((diff**2).sum(-1).mean()).item()
    rmsds.append(rmsd)
    print(f"  seed={seed}: RMSD(Boltz2 identity vs best PDB) = {rmsd:.4f} Å")
# Also compare initial PDB to best PDB
diff_init = ref_pos - best_pos_from_pdb
rmsd_init = torch.sqrt((diff_init**2).sum(-1).mean()).item()
print(f"  RMSD(initial PDB vs best PDB) = {rmsd_init:.4f} Å  ← how much RBR moved it")
print()

# ─── T2: Gradient at identity ────────────────────────────────────────────────
print("=" * 60)
print("T2: Gradient at identity (seed=4, full reflections)")
print("=" * 60)
wrapper.init_bias(DEVICE)
w_p = wrapper._bias['w_pair']
b_p = wrapper._bias['b_pair']
wrapper.diffusion_seed = 4
out = wrapper(feats)
xyz, _, Bs = position_alignment_boltz2(out, feats, cra_name, ref_pos)
llgloss.sfc.atom_b_iso     = Bs.detach().clamp(max=200.0)
llgloss_rbr.sfc.atom_b_iso = Bs.detach().clamp(max=200.0)
rkrf_utils.update_sigmaA(llgloss, llgloss_rbr, xyz.detach())
opt_xyz, _ = rk_coordinates.rigidbody_refine_quat(
    xyz, llgloss_rbr, cra_name, lbfgs=True, lbfgs_lr=150.0, verbose=False)
llg, rw, _ = llgloss(opt_xyz, sub_ratio=1.0, solvent=True, update_scales=True, return_Rfactors=True)
(-llg).backward()
gn_w = w_p.grad.norm().item()
gn_b = b_p.grad.norm().item()
print(f"  LLG={llg.item():.1f}  Rwork={rw.item():.4f}")
print(f"  |∇w_pair|={gn_w:.6f}  |∇b_pair|={gn_b:.6f}")
print(f"  ∇w max element={w_p.grad.abs().max().item():.6f}")
print(f"  Adam step ≈ lr × sign(grad) = 1e-4 per element")
print(f"  → total w_pair movement after 20 steps ≈ {20*1e-4:.4f} Frobenius")
print()

# ─── T3: Noise-averaged LLG at identity over 20 seeds ────────────────────────
print("=" * 60)
print("T3: LLG at identity over 20 seeds (noise floor)")
print("=" * 60)
llgs = []
for seed in range(20):
    llg_i, rw_i, _ = get_llg(w_id, b_id, seed)
    llgs.append(llg_i)
    print(f"  seed {seed:2d}: LLG={llg_i:8.1f}  Rwork={rw_i:.4f}")
print(f"\n  mean={np.mean(llgs):.1f}  std={np.std(llgs):.1f}  min={np.min(llgs):.1f}  max={np.max(llgs):.1f}")
print(f"  Range = {np.max(llgs)-np.min(llgs):.1f}  ← seed variance (signal must exceed this)")
print()

# ─── T4: Structural sensitivity ──────────────────────────────────────────────
print("=" * 60)
print("T4: How much does the structure change per unit of w_pair perturbation?")
print("=" * 60)
torch.manual_seed(42)
xyz_base, _ = get_coords(w_id, b_id, 4)
for dev in [0.02, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]:
    delta = torch.randn(128, 128, device=DEVICE)
    delta = delta / delta.norm() * dev
    w_rand = w_id + delta
    xyz_rand, _ = get_coords(w_rand, b_id, 4)
    rmsd = torch.sqrt(((xyz_base - xyz_rand)**2).sum(-1).mean()).item()
    print(f"  w_dev={dev:.2f}: RMSD={rmsd:.4f} Å  (sensitivity: {rmsd/dev:.4f} Å per unit w_dev)")
