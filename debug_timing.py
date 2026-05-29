"""
Per-component timing breakdown of one ROCKET-Boltz2 DDIM iteration.

Identifies which step dominates the ~15.6s/iter wall time, to explain why
toggling ddim_checkpoint shows no net speed change.

Run on a FREE GPU (e.g. cuda:3):
    python debug_timing.py 3
"""
import sys, time, pickle, torch, numpy as np
sys.path.insert(0, '/data/dust/group/it/crystalsfirst/dev/shehry/ROCKET')

DEV = f"cuda:{sys.argv[1] if len(sys.argv) > 1 else 0}"

from rocket.boltz2_wrapper import Boltz2PairBias
from rocket.xtal import structurefactors as llg_sf
from rocket import refinement_utils as rkrf_utils
from rocket import utils as rk_utils
from rocket import coordinates as rk_coordinates
from rocket.coordinates_boltz2 import position_alignment_boltz2

CKPT  = '/data/dust/group/it/crystalsfirst/dev/shehry/boltz_cache/boltz2_conf.ckpt'
TNG   = '/data/dust/group/it/crystalsfirst/dev/shehry/data/thesis/1lj5/data_for_1lj5/1lj5_boltz_mod/ROCKET_inputs/1lj5-Edata.mtz'
PDB   = '/data/dust/group/it/crystalsfirst/dev/shehry/data/thesis/1lj5/data_for_1lj5/1lj5_boltz_mod/ROCKET_inputs/1lj5-pred-aligned.pdb'
FEATS = '/data/dust/group/it/crystalsfirst/dev/shehry/data/thesis/1lj5/data_for_1lj5/1lj5_boltz_mod/ROCKET_inputs/feats_boltz2.pkl'

tng2  = rk_utils.apply_resolution_cutoff(TNG, min_resolution=3.0)
feats = rk_utils.move_tensors_to_device(pickle.load(open(FEATS, 'rb')), device=DEV)


class T:
    """cuda-synchronised timer accumulator."""
    def __init__(self): self.acc = {}
    def __call__(self, label):
        self.label = label; return self
    def __enter__(self):
        torch.cuda.synchronize(DEV); self.t0 = time.time()
    def __exit__(self, *a):
        torch.cuda.synchronize(DEV)
        self.acc.setdefault(self.label, []).append(time.time() - self.t0)


def run(ddim_checkpoint, n_iters=5):
    print(f"\n{'='*60}\nddim_checkpoint={ddim_checkpoint}\n{'='*60}")
    wrapper = Boltz2PairBias(
        CKPT, truncated_backprop_steps=20, num_sampling_steps=200,
        recycling_steps=3, sampling_mode="ddim", ddim_steps=20,
        ddim_checkpoint=ddim_checkpoint, device=DEV,
    ).to(DEV).eval()
    wrapper.init_bias(DEV)
    wrapper.diffusion_seed = 4
    w = wrapper._bias['w_pair']; b = wrapper._bias['b_pair']
    opt = torch.optim.Adam([{'params': w, 'lr': 3e-4}, {'params': b, 'lr': 3e-4}])

    sfc = llg_sf.initial_SFC(PDB, tng2, 'FEFF', 'DOBS', Freelabel='R-free-flags',
                              device=DEV, testset_value=1, spacing=4.5)
    sfc_rbr = llg_sf.initial_SFC(PDB, tng2, 'FEFF', 'DOBS', Freelabel='R-free-flags',
                                  device=DEV, testset_value=1, solvent=False, spacing=4.5)
    llgloss     = rkrf_utils.init_llgloss(sfc,     tng2, 3.0, None)
    llgloss_rbr = rkrf_utils.init_llgloss(sfc_rbr, tng2, 3.0, None)
    ref = sfc.atom_pos_orth.clone()
    cra = sfc.cra_name
    t = T()

    for it in range(n_iters):
        opt.zero_grad()
        with t("1_wrapper_fwd"):
            out = wrapper(feats)
        with t("2_kabsch"):
            xyz, _, Bs = position_alignment_boltz2(out, feats, cra, ref,
                                                   reference_bfactor=sfc.atom_b_iso.clone())
        safe = Bs.detach().clamp(max=200.0)
        llgloss.sfc.atom_b_iso = safe; llgloss_rbr.sfc.atom_b_iso = safe
        with t("3_update_sigmaA"):
            rkrf_utils.update_sigmaA(llgloss, llgloss_rbr, xyz)
        llgloss.sfc.atom_pos_orth = xyz.detach().clone()
        with t("4_rbr_lbfgs"):
            opt_xyz, _ = rk_coordinates.rigidbody_refine_quat(
                xyz, llgloss_rbr, cra, lbfgs=True, lbfgs_lr=150.0, verbose=False)
        with t("5_llg_fwd"):
            llg, rw, rf = llgloss(opt_xyz, sub_ratio=1.0, solvent=True,
                                  update_scales=True, return_Rfactors=True)
        with t("6_backward"):
            (-llg).backward()
        with t("7_opt_step"):
            torch.nn.utils.clip_grad_norm_([w, b], 1.0); opt.step()

    print(f"  (warmup iter 0 dropped; mean over {n_iters-1} iters)")
    total = 0.0
    for label in sorted(t.acc):
        vals = t.acc[label][1:]            # drop warmup
        m = np.mean(vals); total += m
        print(f"  {label:18s}: {m:6.2f}s")
    print(f"  {'TOTAL':18s}: {total:6.2f}s")
    return t.acc


a = run(ddim_checkpoint=True)
b = run(ddim_checkpoint=False)

print(f"\n{'='*60}\nDIFFERENCE (backward step only)\n{'='*60}")
bw_true  = np.mean(a['6_backward'][1:])
bw_false = np.mean(b['6_backward'][1:])
print(f"  backward WITH checkpoint:    {bw_true:.2f}s")
print(f"  backward WITHOUT checkpoint: {bw_false:.2f}s")
print(f"  savings from disabling:      {bw_true - bw_false:.2f}s/iter")
print(f"  → over 300 iters: {(bw_true-bw_false)*300/60:.1f} min saved")
