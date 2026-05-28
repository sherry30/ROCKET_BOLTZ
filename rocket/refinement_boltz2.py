"""
ROCKET-Boltz2: Inference-time X-ray crystallographic refinement using Boltz-2.

Implements the naive pair-embedding-bias baseline described in
BOLTZ2_INTEGRATION.md.  Learnable parameters: w_pair and b_pair acting on
Boltz-2's trunk pair representation z.  No guided diffusion.

Usage
-----
Prepare Boltz-2 feats (batched, device=cpu) beforehand via
``prepare_boltz2_feats`` (see helper below), then call::

    config = RocketRefinmentConfig.from_yaml_file("ROCKET_config_phase1.yaml")
    feats  = prepare_boltz2_feats(pdb_path, cache_dir=cache_dir)
    config = run_boltz2_xray_refinement(config, feats)
"""

from __future__ import annotations

import os
import pickle
import tarfile
import time
import uuid
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger
from SFC_Torch import SFcalculator
from tqdm import tqdm

from rocket import coordinates as rk_coordinates
from rocket import refinement_utils as rkrf_utils
from rocket import utils as rk_utils
from rocket.boltz2_wrapper import Boltz2PairBias
from rocket.coordinates_boltz2 import position_alignment_boltz2
from rocket.refinement_config import RocketRefinmentConfig
from rocket.wandb_logger import WandbLogger
from rocket.xtal import structurefactors as llg_sf


def precompute_boltz2_seeds(
    config: RocketRefinmentConfig | str,
    feats: dict,
    output_path: str,
    n_seeds: int = 9,
) -> np.ndarray:
    """
    Evaluate n_seeds diffusion seeds with an identity pair bias and save
    sorted (LLG, seed) pairs to output_path as a .npy file.

    Call this from rk.prep_boltz2 / rk.preprocess so the seed ranking is
    computed once.  rk.refine then loads the file instead of re-scanning at
    the beginning of every run.

    Parameters
    ----------
    config : RocketRefinmentConfig or str
        Phase-1 config (used for SFC setup — PDB, MTZ, resolution, device …).
    feats : dict
        Boltz-2 featurizer output (CPU tensors, batch dim = 1).
    output_path : str
        Where to write the .npy file (e.g. ROCKET_inputs/seed_scan.npy).
    n_seeds : int
        Number of diffusion seeds to evaluate (default 9 → enough for 3 runs).

    Returns
    -------
    np.ndarray  shape (n_seeds, 2) — columns (LLG, seed_index), best first.
    """
    import glob as _glob

    if isinstance(config, str):
        config = RocketRefinmentConfig.from_yaml_file(config)

    device    = f"cuda:{config.cuda_device}"
    tng_file  = f"{config.path}/ROCKET_inputs/{config.file_id}-Edata.mtz"
    input_pdb = _glob.glob(config.input_pdb)[0]

    if config.min_resolution is not None or config.max_resolution is not None:
        tng_file = rk_utils.apply_resolution_cutoff(
            tng_file,
            min_resolution=config.min_resolution,
            max_resolution=config.max_resolution,
        )

    sfc = llg_sf.initial_SFC(
        input_pdb, tng_file, "FEFF", "DOBS",
        Freelabel=config.free_flag,
        device=device,
        testset_value=config.testset_value,
        spacing=config.voxel_spacing,
    )
    sfc_rbr = llg_sf.initial_SFC(
        input_pdb, tng_file, "FEFF", "DOBS",
        Freelabel=config.free_flag,
        device=device,
        solvent=False,
        testset_value=config.testset_value,
        spacing=config.voxel_spacing,
    )
    llgloss     = rkrf_utils.init_llgloss(sfc,     tng_file, config.min_resolution, config.max_resolution)
    llgloss_rbr = rkrf_utils.init_llgloss(sfc_rbr, tng_file, config.min_resolution, config.max_resolution)

    reference_pos = sfc.atom_pos_orth.clone()
    cra_name      = sfc.cra_name
    RBR_LBFGS     = config.rbr_opt_algorithm == "lbfgs"

    K           = getattr(config, "truncated_backprop_steps", 5)
    n_recycling = getattr(config, "boltz2_recycling_steps", config.init_recycling)
    n_sampling  = getattr(config, "boltz2_num_sampling_steps", 200)
    boltz2_ckpt = getattr(config, "boltz2_checkpoint_path", None)

    wrapper = Boltz2PairBias(
        checkpoint_path=boltz2_ckpt,
        truncated_backprop_steps=K,
        diffusion_seed=0,
        num_sampling_steps=n_sampling,
        recycling_steps=n_recycling,
        device=device,
    ).to(device).eval()
    wrapper.init_bias(device)

    feats_gpu = rk_utils.move_tensors_to_device(feats, device=device)

    logger.info(f"Seed pre-scan: evaluating {n_seeds} seeds (saving to {output_path}) …")
    scan_results: list[tuple[float, int]] = []

    for seed in range(n_seeds):
        wrapper.diffusion_seed = seed
        with torch.no_grad():
            model_out = wrapper(feats_gpu, recycling_steps=n_recycling,
                                num_sampling_steps=n_sampling)
            xyz_scan, _, pseudo_Bs = position_alignment_boltz2(
                model_output=model_out, feats=feats_gpu,
                cra_name_sfc=cra_name, best_pos=reference_pos,
            )
        xyz_d   = xyz_scan.detach()
        safe_Bs = pseudo_Bs.detach().clamp(max=200.0)
        llgloss.sfc.atom_b_iso     = safe_Bs
        llgloss_rbr.sfc.atom_b_iso = safe_Bs
        rkrf_utils.update_sigmaA(llgloss, llgloss_rbr, xyz_d)
        opt_xyz, _ = rk_coordinates.rigidbody_refine_quat(
            xyz_d, llgloss_rbr, cra_name,
            domain_segs=config.domain_segs,
            lbfgs=RBR_LBFGS,
            lbfgs_lr=config.rbr_lbfgs_learning_rate,
            verbose=False,
        )
        llg_val, _, _ = llgloss(
            opt_xyz, sub_ratio=1.0, solvent=config.solvent,
            update_scales=config.sfc_scale, return_Rfactors=True,
        )
        scan_results.append((llg_val.item(), seed))
        logger.info(f"  seed {seed}: LLG={llg_val.item():.1f}")

    scan_results.sort(key=lambda x: x[0], reverse=True)
    arr = np.array(scan_results)
    np.save(output_path, arr)
    logger.info(
        f"Seed scan saved → {output_path}  "
        f"best: seed {int(arr[0,1])} LLG={arr[0,0]:.1f}"
    )

    del wrapper
    torch.cuda.empty_cache()
    return arr


def run_boltz2_xray_refinement(
    config: RocketRefinmentConfig | str,
    feats: dict,
) -> RocketRefinmentConfig:
    """
    Run ROCKET X-ray crystallographic refinement with Boltz-2.

    Parameters
    ----------
    config : RocketRefinmentConfig or str
        Run configuration.  If a str, loaded from that YAML path.
    feats : dict
        Boltz-2 featurizer-v2 output dict **with a leading batch dimension of 1**
        (i.e. tensors have shape ``[1, ...]``).  All tensors on CPU; will be
        moved to the GPU inside this function.  Required keys:
        ``atom_pad_mask``, ``atom_to_token``, ``residue_index``, ``asym_id``,
        ``res_type``, ``ref_atom_name_chars``, ``token_pad_mask``,
        ``token_bonds``, ``rel_pos`` (or whatever the model's rel_pos expects).

    Returns
    -------
    RocketRefinmentConfig
        The (possibly updated) config after refinement.
    """
    if isinstance(config, str):
        config = RocketRefinmentConfig.from_yaml_file(config)
    assert config.datamode == "xray", "datamode must be 'xray' for Boltz-2 refinement"

    # ------------------------------------------------------------------
    # 1. Global settings
    # ------------------------------------------------------------------
    device = f"cuda:{config.cuda_device}"
    RBR_LBFGS = config.rbr_opt_algorithm == "lbfgs"

    # Paths
    import glob
    tng_file = f"{config.path}/ROCKET_inputs/{config.file_id}-Edata.mtz"
    try:
        input_pdb = glob.glob(config.input_pdb)[0]
    except Exception as err:
        raise ValueError(f"input_pdb path is not valid: {config.input_pdb}") from err

    # Output directory
    if config.uuid_hex:
        run_uuid = config.uuid_hex
    else:
        config.paths.uuid_hex = uuid.uuid4().hex[:10]
        run_uuid = config.uuid_hex
    out_dir = f"{config.path}/ROCKET_outputs/{run_uuid}/{config.note}"
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"ROCKET-Boltz2 | system: {config.file_id} | run: {run_uuid} | note: {config.note}")

    wandb_logger = WandbLogger(
        enabled=config.use_wandb,
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_name,
        tags=config.wandb_tags,
        notes=config.wandb_notes,
        config=config.model_dump(),
    )

    if not config.verbose:
        warnings.filterwarnings("ignore")

    # ------------------------------------------------------------------
    # 2. Initializations
    # ------------------------------------------------------------------
    # Resolution cutoff on reflection file
    if config.min_resolution is not None or config.max_resolution is not None:
        tng_file = rk_utils.apply_resolution_cutoff(
            tng_file,
            min_resolution=config.min_resolution,
            max_resolution=config.max_resolution,
        )

    # Additional chain (fixed partial model)
    constant_fp_added_HKL = None
    constant_fp_added_asu = None
    if config.additional_chain:
        added_chain_pdb = f"{config.path}/ROCKET_inputs/{config.file_id}_added_chain.pdb"
        sfc_added = SFcalculator(
            added_chain_pdb, tng_file,
            expcolumns=["FEFF", "DOBS"],
            freeflag=config.free_flag,
            set_experiment=True,
            testset_value=config.testset_value,
            device=device,
        )
        sfc_added.calc_fprotein()
        constant_fp_added_HKL = sfc_added.Fprotein_HKL.clone().detach()
        constant_fp_added_asu = sfc_added.Fprotein_asu.clone().detach()
        del sfc_added

    # SFC initialisation
    sfc = llg_sf.initial_SFC(
        input_pdb, tng_file, "FEFF", "DOBS",
        Freelabel=config.free_flag,
        device=device,
        testset_value=config.testset_value,
        added_chain_HKL=constant_fp_added_HKL,
        added_chain_asu=constant_fp_added_asu,
        total_chain_copy=config.total_chain_copy,
        spacing=config.voxel_spacing,
    )
    reference_pos  = sfc.atom_pos_orth.clone()
    init_pos_bfactor = sfc.atom_b_iso.clone()
    bfactor_weights  = rk_utils.weighting_torch(init_pos_bfactor, cutoff2=20.0)

    sfc_rbr = llg_sf.initial_SFC(
        input_pdb, tng_file, "FEFF", "DOBS",
        Freelabel=config.free_flag,
        device=device,
        solvent=False,
        testset_value=config.testset_value,
        added_chain_HKL=constant_fp_added_HKL,
        added_chain_asu=constant_fp_added_asu,
        total_chain_copy=config.total_chain_copy,
        spacing=config.voxel_spacing,
    )

    llgloss     = rkrf_utils.init_llgloss(sfc,     tng_file, config.min_resolution, config.max_resolution)
    llgloss_rbr = rkrf_utils.init_llgloss(sfc_rbr, tng_file, config.min_resolution, config.max_resolution)

    # sigmaA ground truth (optional)
    SIGMA_TRUE = False
    phitrue = Etrue = None
    for suffix in ["-phitrue", "-Etrue"]:
        path = f"{config.path}/ROCKET_inputs/{config.file_id}{suffix}-solvent{config.solvent}.npy"
        if not os.path.exists(path):
            break
    else:
        SIGMA_TRUE = True
        phitrue = np.load(f"{config.path}/ROCKET_inputs/{config.file_id}-phitrue-solvent{config.solvent}.npy")
        Etrue   = np.load(f"{config.path}/ROCKET_inputs/{config.file_id}-Etrue-solvent{config.solvent}.npy")

    # ------------------------------------------------------------------
    # 3. Boltz-2 model and feats
    # ------------------------------------------------------------------
    # Resolve checkpoint path
    boltz2_ckpt = getattr(config, "boltz2_checkpoint_path", None)
    K = getattr(config, "truncated_backprop_steps", 5)
    n_recycling = getattr(config, "boltz2_recycling_steps", config.init_recycling)
    n_sampling  = getattr(config, "boltz2_num_sampling_steps", 200)

    wrapper = Boltz2PairBias(
        checkpoint_path=boltz2_ckpt,
        truncated_backprop_steps=K,
        diffusion_seed=0,             # overridden per run below
        num_sampling_steps=n_sampling,
        recycling_steps=n_recycling,
        device=device,
    )
    wrapper = wrapper.to(device)
    wrapper.eval()

    # Move feats to GPU and ensure float where appropriate
    feats_gpu = rk_utils.move_tensors_to_device(feats, device=device)

    # Number of Boltz-2 tokens (sequence length)
    n_tokens = int(feats_gpu["token_pad_mask"].shape[1])

    # ------------------------------------------------------------------
    # 4. Learning-rate schedule
    # ------------------------------------------------------------------
    if "phase1" in config.note:
        lr_a = config.additive_learning_rate
        lr_m = config.multiplicative_learning_rate
        w_L2 = config.l2_weight
    elif "phase2" in config.note:
        lr_a = config.phase2_final_lr
        lr_m = config.phase2_final_lr
        w_L2 = 0.0
    else:
        # Default to phase-1 settings
        lr_a = config.additive_learning_rate
        lr_m = config.multiplicative_learning_rate
        w_L2 = config.l2_weight

    # ------------------------------------------------------------------
    # 5. Save config and run refinement loops
    # ------------------------------------------------------------------
    config.to_yaml_file(f"{out_dir}/config.yaml")

    best_llg       = float("inf")
    best_w_pair    = None
    best_b_pair    = None
    best_run       = None
    best_iter      = None
    best_pos       = reference_pos.clone()

    cra_name = sfc.cra_name  # list of "chain-resid-resname-atomname"

    # ------------------------------------------------------------------
    # Seed selection: use precomputed scan if available, otherwise scan inline.
    #
    # Different seeds produce very different initial structures (LLG variance
    # ~300-400 units even with an identity bias), which far exceeds the
    # optimization signal.  Starting from the best seeds is critical.
    #
    # rk.prep_boltz2 / rk.preprocess now precompute the scan once and write
    # the path to config.boltz2.precomputed_seed_scan.  When that file exists
    # we load it directly (saves ~9 × model-forward time at the start of refine).
    # ------------------------------------------------------------------
    precomputed_scan_path = getattr(config, "precomputed_seed_scan", None)
    if precomputed_scan_path and os.path.exists(str(precomputed_scan_path)):
        logger.info(f"Loading precomputed seed scan from {precomputed_scan_path}")
        scan_arr = np.load(str(precomputed_scan_path))
        scan_llg_by_seed = [(float(r[0]), int(r[1])) for r in scan_arr]
        scan_llg_by_seed.sort(key=lambda x: x[0], reverse=True)
        np.save(f"{out_dir}/seed_scan.npy", scan_arr)
        wrapper.init_bias(device)   # identity init for first run
    else:
        n_scan = max(config.num_of_runs * 3, 6)
        logger.info(f"Seed pre-scan: evaluating {n_scan} seeds with identity bias …")
        wrapper.init_bias(device)   # identity init — not used for grad
        scan_llg_by_seed = []

        for seed in range(n_scan):
            wrapper.diffusion_seed = seed
            # Only the model forward is in no_grad; SFC ops (update_sigmaA,
            # get_scales_adam) call their own internal backward() and must be
            # outside no_grad.
            with torch.no_grad():
                model_scan = wrapper(feats_gpu, recycling_steps=n_recycling,
                                      num_sampling_steps=n_sampling)
                xyz_scan, _, pseudo_Bs_scan = position_alignment_boltz2(
                    model_output=model_scan, feats=feats_gpu,
                    cra_name_sfc=cra_name, best_pos=reference_pos,
                )
            xyz_scan_d  = xyz_scan.detach()
            safe_b_scan = pseudo_Bs_scan.detach().clamp(max=200.0)
            llgloss.sfc.atom_b_iso     = safe_b_scan
            llgloss_rbr.sfc.atom_b_iso = safe_b_scan
            rkrf_utils.update_sigmaA(llgloss, llgloss_rbr, xyz_scan_d)
            opt_xyz_scan, _ = rk_coordinates.rigidbody_refine_quat(
                xyz_scan_d, llgloss_rbr, cra_name,
                domain_segs=config.domain_segs,
                lbfgs=RBR_LBFGS,
                added_chain_HKL=constant_fp_added_HKL,
                added_chain_asu=constant_fp_added_asu,
                lbfgs_lr=config.rbr_lbfgs_learning_rate,
                verbose=False,
            )
            llg_scan, _, _ = llgloss(
                opt_xyz_scan, sub_ratio=1.0, solvent=config.solvent,
                update_scales=config.sfc_scale,
                added_chain_HKL=constant_fp_added_HKL,
                added_chain_asu=constant_fp_added_asu,
                return_Rfactors=True,
            )
            scan_llg_by_seed.append((llg_scan.item(), seed))
            logger.info(f"  seed {seed}: LLG={llg_scan.item():.1f}")

        scan_llg_by_seed.sort(key=lambda x: x[0], reverse=True)
        np.save(f"{out_dir}/seed_scan.npy", np.array(scan_llg_by_seed))

    selected_seeds = [seed for _, seed in scan_llg_by_seed[: config.num_of_runs]]
    logger.info(f"Selected seeds: {selected_seeds}  (scan: {scan_llg_by_seed})")

    for n, seed in enumerate(selected_seeds):
        run_id = rkrf_utils.number_to_letter(n)

        # Use the pre-selected best seed for this run
        wrapper.diffusion_seed = seed

        # -- bias initialisation (phase2: warm-start from phase1 best bias) --
        # bias_dict = wrapper.init_bias(n_tokens, device)
        bias_dict = wrapper.init_bias(device)
        w_pair = bias_dict["w_pair"]
        b_pair = bias_dict["b_pair"]

        if "phase2" in config.note:
            import glob as _glob
            w_paths = _glob.glob(config.starting_bias)   if config.starting_bias   else []
            b_paths = _glob.glob(config.starting_weights) if config.starting_weights else []
            if w_paths and b_paths:
                w_pair.data.copy_(torch.load(w_paths[0], map_location=device))
                b_pair.data.copy_(torch.load(b_paths[0], map_location=device))
                logger.info(f"Phase-2 warm-start: loaded {w_paths[0]} and {b_paths[0]}")

        optimizer = torch.optim.Adam(
            [
                {"params": w_pair, "lr": lr_m},
                {"params": b_pair, "lr": lr_a},
            ],
            weight_decay=config.weight_decay if config.weight_decay else 0.0,
        )

        # Smooth LR decay over the final smooth_stage_epochs iterations of phase 1
        # (mirrors the AF2 refinement schedule; prevents Adam divergence after the optimum)
        lr_a_current = lr_a
        lr_m_current = lr_m
        w_L2_current = w_L2
        smooth_stage_epochs = config.smooth_stage_epochs
        if ("phase1" in config.note) and (smooth_stage_epochs is not None):
            lr_stage1_final = config.phase2_final_lr
            decay_rate_a = (lr_stage1_final / lr_a) ** (1.0 / smooth_stage_epochs)
            decay_rate_m = (lr_stage1_final / lr_m) ** (1.0 / smooth_stage_epochs)
        else:
            decay_rate_a = decay_rate_m = 1.0

        # per-run tracking
        llg_losses     = []
        rwork_by_epoch = []
        rfree_by_epoch = []
        time_by_epoch  = []
        memory_by_epoch = []
        rbr_loss_by_epoch = []

        early_stopper = rkrf_utils.EarlyStopper(patience=200, min_delta=10.0)

        progress_bar = tqdm(
            range(config.iterations),
            desc=f"{config.file_id}, uuid: {run_uuid[:4]}, run: {run_id}",
        )

        # ------------------------------------------------------------------
        # 5a. Refinement loop
        # ------------------------------------------------------------------
        for iteration in progress_bar:
            start_time = time.time()
            optimizer.zero_grad()

            # Boltz-2 forward pass (trunk → biased-z → diffusion → confidence)
            model_out = wrapper(feats_gpu, recycling_steps=n_recycling,
                                num_sampling_steps=n_sampling)

            # pLDDT loss (per-token plddt in [0,1]; maximise mean)
            if "plddt" in model_out:
                L_plddt = -torch.mean(model_out["plddt"])
            else:
                L_plddt = torch.tensor(0.0, device=device)

            # Kabsch alignment + pseudo-B extraction
            aligned_xyz, plddt_tokens, pseudo_Bs = position_alignment_boltz2(
                model_output=model_out,
                feats=feats_gpu,
                cra_name_sfc=cra_name,
                best_pos=best_pos,
                exclude_res=None,
                domain_segs=config.domain_segs,
                reference_bfactor=init_pos_bfactor,
            )

            # Boltz-2 single-sequence pLDDT is typically 0.3–0.4, mapping through
            # plddt2pseudoB to B-factors > 600 Å².  B >> 200 attenuates F_calc to ~0 for
            # all reflections, making sigmaA ≈ 0, LLG ≈ 0, and gradient ≈ 0.
            # Clamp at 200 Å² so structure factors remain informative throughout.
            safe_Bs = pseudo_Bs.detach().clone().clamp(max=200.0)
            if iteration == 0 and n == 0:
                logger.info(
                    f"B-factor check: Boltz-2 pseudo_B mean={pseudo_Bs.mean().item():.1f} Å², "
                    f"clamped mean={safe_Bs.mean().item():.1f} Å², "
                    f"fraction clamped={((pseudo_Bs > 200.0).float().mean().item()):.2%}"
                )
            llgloss.sfc.atom_b_iso     = safe_Bs
            llgloss_rbr.sfc.atom_b_iso = safe_Bs

            # Update/refine sigmaA
            if config.refine_sigmaA:
                llgloss, llgloss_rbr, _, _ = rkrf_utils.update_sigmaA(
                    llgloss=llgloss,
                    llgloss_rbr=llgloss_rbr,
                    aligned_xyz=aligned_xyz,
                    constant_fp_added_HKL=constant_fp_added_HKL,
                    constant_fp_added_asu=constant_fp_added_asu,
                )
            elif SIGMA_TRUE:
                llgloss, llgloss_rbr = rkrf_utils.sigmaA_from_true(
                    llgloss=llgloss,
                    llgloss_rbr=llgloss_rbr,
                    aligned_xyz=aligned_xyz,
                    Etrue=Etrue,
                    phitrue=phitrue,
                    constant_fp_added_HKL=constant_fp_added_HKL,
                    constant_fp_added_asu=constant_fp_added_asu,
                )

            llgloss.sfc.atom_pos_orth = aligned_xyz.detach().clone()

            # Rigid-body refinement (LBFGS on detached coords, returns torch coords)
            optimized_xyz, rbr_track = rk_coordinates.rigidbody_refine_quat(
                aligned_xyz,
                llgloss_rbr,
                cra_name,
                domain_segs=config.domain_segs,
                lbfgs=RBR_LBFGS,
                added_chain_HKL=constant_fp_added_HKL,
                added_chain_asu=constant_fp_added_asu,
                lbfgs_lr=config.rbr_lbfgs_learning_rate,
                verbose=config.verbose,
            )
            rbr_loss_by_epoch.append(rbr_track)

            # LLG loss (crystallographic objective)
            llg, r_work, r_free = llgloss(
                optimized_xyz,
                bin_labels=None,
                num_batch=config.number_of_batches,
                sub_ratio=config.batch_sub_ratio,
                solvent=config.solvent,
                update_scales=config.sfc_scale,
                added_chain_HKL=constant_fp_added_HKL,
                added_chain_asu=constant_fp_added_asu,
                return_Rfactors=True,
            )
            L_llg = -llg

            llg_estimate = L_llg.item() / (config.batch_sub_ratio * config.number_of_batches)
            llg_losses.append(llg_estimate)
            rwork_by_epoch.append(r_work.item())
            rfree_by_epoch.append(r_free.item())

            # Track best run
            if llg_losses[-1] < best_llg:
                best_llg    = llg_losses[-1]
                best_w_pair = w_pair.detach().cpu().clone()
                best_b_pair = b_pair.detach().cpu().clone()
                best_run    = run_id
                best_iter   = iteration
                best_pos    = optimized_xyz.detach().clone()

            llgloss.sfc.atom_pos_orth = optimized_xyz

            # Save current frame
            pdb_path = f"{out_dir}/{run_id}_{iteration}_postRBR.pdb"
            llgloss.sfc.savePDB(pdb_path)

            progress_bar.set_postfix(
                NEG_LLG=f"{llg_estimate:.2f}",
                Rwork=f"{r_work.item():.3f}",
                Rfree=f"{r_free.item():.3f}",
                mem=f"{torch.cuda.max_memory_allocated() / 1024**3:.1f}G",
            )

            wandb_logger.log(
                {
                    f"{run_id}/neg_llg":     llg_estimate,
                    f"{run_id}/rwork":       r_work.item(),
                    f"{run_id}/rfree":       r_free.item(),
                    f"{run_id}/gpu_mem_gb":  torch.cuda.max_memory_allocated() / 1024**3,
                    f"{run_id}/iter_sec":    time.time() - start_time,
                },
                step=iteration,
            )

            # Total loss and backward (use w_L2_current so L2 also decays in smooth stage)
            if w_L2_current > 0.0:
                L2_loss = torch.sum(
                    bfactor_weights.unsqueeze(-1) * (optimized_xyz - reference_pos) ** 2
                )
                loss = L_llg + w_L2_current * L2_loss + config.w_plddt * L_plddt
            else:
                loss = L_llg + config.w_plddt * L_plddt
                if early_stopper.early_stop(loss.item()):
                    logger.info(f"Early stopping at iteration {iteration}")
                    break

            loss.backward()
            torch.nn.utils.clip_grad_norm_([w_pair, b_pair], max_norm=10.0)

            # Apply smooth LR decay in the final smooth_stage_epochs iterations
            if ("phase1" in config.note) and (smooth_stage_epochs is not None):
                stage_start = config.iterations - smooth_stage_epochs
                if iteration > stage_start:
                    stage_step = iteration - stage_start  # 1 … smooth_stage_epochs
                    lr_a_current = lr_a * (decay_rate_a ** stage_step)
                    lr_m_current = lr_m * (decay_rate_m ** stage_step)
                    w_L2_current = w_L2 * max(0.0, 1.0 - stage_step / smooth_stage_epochs)
                    optimizer.param_groups[0]["lr"] = lr_m_current
                    optimizer.param_groups[1]["lr"] = lr_a_current

            optimizer.step()

            time_by_epoch.append(time.time() - start_time)
            memory_by_epoch.append(torch.cuda.max_memory_allocated() / 1024**3)

        # ------------------------------------------------------------------
        # 5b. Save per-run data
        # ------------------------------------------------------------------
        np.save(f"{out_dir}/NEG_LLG_it_{run_id}.npy",    np.array(llg_losses))
        np.save(f"{out_dir}/rwork_it_{run_id}.npy",       np.array(rwork_by_epoch))
        np.save(f"{out_dir}/rfree_it_{run_id}.npy",       np.array(rfree_by_epoch))
        np.save(f"{out_dir}/time_it_{run_id}.npy",        np.array(time_by_epoch))
        np.save(f"{out_dir}/memory_it_{run_id}.npy",      np.array(memory_by_epoch))

    # ------------------------------------------------------------------
    # 6. Save best bias parameters and final model
    # ------------------------------------------------------------------
    torch.save(best_w_pair, f"{out_dir}/best_w_pair_{best_run}_{best_iter}.pt")
    torch.save(best_b_pair, f"{out_dir}/best_b_pair_{best_run}_{best_iter}.pt")

    if best_pos is not None:
        try:
            llgloss.sfc.atom_pos_orth = best_pos
            best_pdb = f"{out_dir}/best_model_{best_run}_{best_iter}.pdb"
            llgloss.sfc.savePDB(best_pdb)
            wandb_logger.log_artifact(best_pdb, name="best_model", artifact_type="model")
            wandb_logger.log_molecule_3d(best_pdb, name="best_model_3d")
        except Exception as exc:
            logger.warning(f"Could not save best model PDB: {exc}")

    wandb_logger.log({
        "summary/best_neg_llg": best_llg,
        "summary/best_run":     best_run,
        "summary/best_iter":    best_iter,
    })
    wandb_logger.log_artifact(f"{out_dir}/config.yaml", name="config", artifact_type="config")
    wandb_logger.finish()

    return config


# ---------------------------------------------------------------------------
# Feats preparation helper
# ---------------------------------------------------------------------------

def _load_mols(mol_dir: Path, names) -> dict:
    """Load named molecules, with fallback to mols.tar in the parent directory.

    The current boltz API expects ``{name}.pkl`` files in mol_dir.  Older
    caches store molecules as numbered shards (000.pkl, 001.pkl, …) instead;
    in that case the named files live in ``mols.tar`` one level up.
    """
    named_probe = mol_dir / "ALA.pkl"
    if named_probe.exists():
        from boltz.data.mol import load_molecules
        return load_molecules(mol_dir, list(names))

    tar_path = mol_dir.parent / "mols.tar"
    if not tar_path.exists():
        raise FileNotFoundError(
            f"Neither named mol files (e.g. ALA.pkl) in {mol_dir} "
            f"nor mols.tar in {mol_dir.parent} were found."
        )

    result = {}
    with tarfile.open(tar_path, "r") as tf:
        for name in names:
            try:
                member = tf.getmember(f"mols/{name}.pkl")
                f = tf.extractfile(member)
                if f is not None:
                    result[name] = pickle.load(f)  # noqa: S301
            except KeyError:
                pass  # molecule not in tar — caller decides if that's fatal
    return result


def _load_canonicals(mol_dir: Path) -> dict:
    """Load canonical molecules, with mols.tar fallback (see _load_mols)."""
    named_probe = mol_dir / "ALA.pkl"
    if named_probe.exists():
        from boltz.data.mol import load_canonicals
        return load_canonicals(mol_dir)

    from boltz.data import const
    return _load_mols(mol_dir, const.canonical_tokens)


def prepare_boltz2_feats(
    pdb_path: str | Path,
    cache_dir: Optional[str | Path] = None,
    a3m_path: Optional[str | Path] = None,
    method_override: str = "x-ray diffraction",
    device: str = "cpu",
) -> dict:
    """
    Prepare Boltz-2 featurizer-v2 features from a PDB file.

    Extracts amino acid sequences from the PDB, then runs Boltz-2's
    schema → tokenize → featurize pipeline and returns a feats dict with a
    leading batch dimension of 1, suitable for passing to
    ``run_boltz2_xray_refinement``.

    Parameters
    ----------
    pdb_path : str or Path
        Path to the input PDB file.
    cache_dir : str or Path or None
        Boltz cache directory containing the ``mols/`` sub-directory with
        canonical molecule definitions.  Uses ``~/.boltz`` if None.
    a3m_path : str or Path or None
        Optional path to a pre-computed MSA a3m file (e.g. from
        ``rk.generate_msa``).  When provided the MSA is loaded and passed to
        all protein chains, enabling the Boltz-2 MSA module during refinement.
        Without an MSA the model runs in single-sequence mode.
    method_override : str
        Experimental method string passed to the featurizer
        (default ``"x-ray diffraction"``).
    device : str
        Device for the returned tensors.  Usually ``"cpu"`` — move to GPU
        inside the refinement loop.

    Returns
    -------
    dict
        Boltz-2 feats dict with all tensors unsqueezed to [1, ...] (batch=1).
    """
    try:
        import gemmi
        from boltz.data import const
        from boltz.data.feature.featurizerv2 import Boltz2Featurizer
        from boltz.data.parse.schema import parse_boltz_schema
        from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
        from boltz.data.types import Input
    except ImportError as exc:
        raise ImportError(
            "boltz package not found.  Add the boltz/src directory to PYTHONPATH."
        ) from exc

    pdb_path  = Path(pdb_path)
    cache_dir = Path(cache_dir or (Path.home() / ".boltz"))
    mol_dir   = cache_dir / "mols"

    # Extract chain sequences directly from the PDB, bypassing parse_pdb which
    # uses gemmi.align_sequence_to_polymer and fails on prediction PDB files
    # (AF2 outputs lack SEQRES records, so inferred full_sequence is misaligned).
    logger.info(f"Extracting sequences from {pdb_path} …")
    gemmi_struct = gemmi.read_structure(str(pdb_path))
    sequences = []
    for chain in gemmi_struct[0]:
        polymer = chain.get_polymer()
        ptype = polymer.check_polymer_type()
        if ptype in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD):
            entity_type = "protein"
        elif ptype == gemmi.PolymerType.Rna:
            entity_type = "rna"
        elif ptype in (gemmi.PolymerType.Dna, gemmi.PolymerType.DnaRnaHybrid):
            entity_type = "dna"
        else:
            continue  # skip non-polymer or unsupported types

        seq = "".join(
            (gemmi.find_tabulated_residue(res.name).one_letter_code or "X")
            for res in polymer
        )
        if seq:
            sequences.append({entity_type: {"id": chain.name, "sequence": seq}})

    if not sequences:
        raise ValueError(f"No polymer chains found in {pdb_path}")

    schema = {"version": 1, "sequences": sequences}

    # Load canonical molecules (needed by parse_boltz_schema for non-std residues)
    logger.info("Loading canonical molecules …")
    ccd = _load_canonicals(mol_dir)

    # parse_boltz_schema is the same code path used by `boltz predict` for FASTA/YAML
    # inputs; with boltz_2=True it returns a Target whose .structure is StructureV2.
    logger.info("Building Boltz-2 feats schema …")
    target = parse_boltz_schema(
        name=pdb_path.stem,
        schema=schema,
        ccd=ccd,
        mol_dir=mol_dir,
        boltz_2=True,
    )

    # Build MSA dict: {chain_asym_id (int) → MSA object}
    # The featurizerv2 looks up by integer asym_id; chains without an entry
    # automatically fall back to dummy_msa (single-sequence mode).
    msa_dict: dict = {}
    if a3m_path is not None:
        from boltz.data.parse.a3m import parse_a3m

        logger.info(f"Loading MSA from {a3m_path} …")
        msa_obj = parse_a3m(Path(a3m_path), taxonomy=None)
        prot_type = const.chain_type_ids["PROTEIN"]
        for chain in target.structure.chains:
            if int(chain["mol_type"]) == prot_type:
                msa_dict[int(chain["asym_id"])] = msa_obj
        logger.info(
            f"MSA loaded: {len(msa_obj.sequences)} sequences → "
            f"assigned to {len(msa_dict)} protein chain(s)"
        )
    else:
        logger.info("No MSA provided — running in single-sequence mode.")

    input_data = Input(structure=target.structure, msa=msa_dict, record=target.record)

    logger.info("Tokenizing …")
    tokenized = Boltz2Tokenizer().tokenize(input_data)

    # Load canonical molecules and any ligand-specific conformers needed
    logger.info("Loading molecules …")
    mol_names = set(tokenized.tokens["res_name"].tolist()) - set(ccd.keys())
    molecules = {**ccd, **_load_mols(mol_dir, mol_names)}

    logger.info("Featurizing …")
    rng = np.random.default_rng(42)
    raw_feats = Boltz2Featurizer().process(
        tokenized,
        molecules=molecules,
        random=rng,
        training=False,
        max_atoms=None,
        max_tokens=None,
        max_seqs=const.max_msa_seqs,
        pad_to_max_seqs=False,
        compute_frames=True,
        override_method=method_override,
    )

    # Add leading batch dimension and move to device
    batched: dict = {}
    for k, v in raw_feats.items():
        if isinstance(v, torch.Tensor):
            batched[k] = v.unsqueeze(0).to(device)
        else:
            batched[k] = v

    return batched
