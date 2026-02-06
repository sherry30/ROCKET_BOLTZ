import shutil
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from loguru import logger
from openfold.config import model_config
from openfold.data import data_pipeline, feature_pipeline
from tqdm import tqdm

import rocket
from rocket import coordinates as rk_coordinates
from rocket import refinement_utils as rkrf_utils
from rocket import utils as rk_utils


def run_msa_score(
    path: str | Path,
    system: str,
    msa_input_dir: str | Path,
    output_dir: str | Path,
    datamode: Literal["xray", "cryoem"],
    domain_segs: list[int] | None = None,
    additional_chain: bool = False,
    init_recycling: int = 4,
    free_flag: str = "R-free-flags",
    testset_value: int = 1,
    voxel_spacing: float = 4.5,
    min_resolution: float = 3.0,
    chimera_profile: bool = False,
    score_fullmsa: bool = False,
    full_msa_dir: str | None = None,
) -> str:
    """
    Run LLG scoring for system with different MSAs.

    Args:
        path: Path to parent folder
        system: file_id for the dataset
        msa_input_dir: dir with msa's to use
        output_dir: output directory to write prediction and scoring to
        datamode: Choose between "xray" or "cryoem" mode
        domain_segs: A list of resid as domain boundaries
        additional_chain: Additional Chain in ASU
        init_recycling: Number of initial recycling iterations
        free_flag: Column name of free flag
        testset_value: Value for test set
        voxel_spacing: Voxel spacing for solvent percentage estimation
        min_resolution: Min resolution cut
        chimera_profile: Use chimera profile
        score_fullmsa: Also score the full msa
        full_msa_dir: Path to full MSA directory (default: path/alignments)

    Returns:
        Path to the output directory
    """
    device = rk_utils.try_gpu()
    RBR_LBFGS = True

    output_directory_path = Path(output_dir)
    output_directory_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Working with system {system}", flush=True)

    # Input paths
    tng_file = Path(path) / "ROCKET_inputs" / f"{system}-Edata.mtz"
    input_pdb = Path(path) / "ROCKET_inputs" / f"{system}-pred-aligned.pdb"

    # Handle additional chain
    constant_fp_added_HKL = None
    constant_fp_added_asu = None
    if additional_chain:
        constant_fp_added_HKL = torch.load(
            str(Path(path) / "ROCKET_inputs" / f"{system}_added_chain_atoms_HKL.pt")
        ).to(device=device)
        constant_fp_added_asu = torch.load(
            str(Path(path) / "ROCKET_inputs" / f"{system}_added_chain_atoms_asu.pt")
        ).to(device=device)

    # --- Data mode specific imports and objects ---
    if datamode == "xray":
        from rocket.xtal import structurefactors as sf_module

        def llgloss_init(sfc, mtz, minres, maxres):
            return rkrf_utils.init_llgloss(sfc, mtz, minres, maxres)

        initial_SFC = sf_module.initial_SFC
        SFC_kwargs = {
            "Freelabel": free_flag,
            "device": device,
            "testset_value": testset_value,
            "added_chain_HKL": constant_fp_added_HKL,
            "added_chain_asu": constant_fp_added_asu,
            "spacing": voxel_spacing,
        }
        LBFGS_LR = 150

    elif datamode == "cryoem":
        from rocket.cryo import structurefactors as sf_module
        from rocket.cryo import targets as cryo_targets

        def llgloss_init(sfc, mtz, *_):
            return cryo_targets.LLGloss(sfc, mtz)

        def initial_SFC(
            pdb,
            mtz,
            FP,
            SIGFP,
            Freelabel,
            device,
            testset_value,
            added_chain_HKL,
            added_chain_asu,
            spacing,
        ):
            return sf_module.initial_cryoSFC(pdb, mtz, "Emean", "PHIEmean", device, 20)

        SFC_kwargs = {
            "Freelabel": free_flag,
            "device": device,
            "testset_value": testset_value,
            "added_chain_HKL": constant_fp_added_HKL,
            "added_chain_asu": constant_fp_added_asu,
            "spacing": voxel_spacing,
        }
        LBFGS_LR = 0.1

    else:
        raise ValueError(f"Unknown datamode: {datamode}")

    # --- SFC and LLGloss Initialization ---
    sfc = initial_SFC(str(input_pdb), str(tng_file), "FEFF", "DOBS", **SFC_kwargs)
    reference_pos = sfc.atom_pos_orth.clone()
    init_pos_bfactor = sfc.atom_b_iso.clone()

    sfc_rbr = initial_SFC(str(input_pdb), str(tng_file), "FEFF", "DOBS", **SFC_kwargs)

    llgloss = llgloss_init(sfc, str(tng_file), min_resolution, None)
    llgloss_rbr = llgloss_init(sfc_rbr, str(tng_file), min_resolution, None)

    # AF2 model initialization
    af_bias = rocket.MSABiasAFv3(model_config(PRESET, train=True), PRESET).to(device)
    af_bias.freeze()

    fasta_candidates = [
        str(p)
        for ext in ("*.fa", "*.fasta")
        for p in Path(path).glob(ext)
    ]
    fasta_path = fasta_candidates[0]

    # Use provided msa_dir or fall back to default
    if full_msa_dir is None:
        full_msa_dir = str(Path(path) / "alignments")
    data_processor = data_pipeline.DataPipeline(template_featurizer=None)
    fullmsa_feature_dict = rkrf_utils.generate_feature_dict(
        fasta_path,
        full_msa_dir,
        data_processor,
    )

    afconfig = model_config(PRESET)
    afconfig.data.common.max_recycling_iters = init_recycling
    del afconfig.data.common.masked_msa
    afconfig.data.common.resample_msa_in_recycling = False
    feature_processor = feature_pipeline.FeaturePipeline(afconfig.data)
    fullmsa_processed_feature_dict = feature_processor.process_features(
        fullmsa_feature_dict, mode="predict"
    )
    full_profile = fullmsa_processed_feature_dict["msa_feat"][:, :, 25:48].clone()

    df = pd.DataFrame(
        columns=[
            "msa_name",
            "depth",
            "mean_plddt",
            "llg",
            #  "rfree",
            #  "rwork"
        ]
    )
    df.to_csv(output_directory_path / "msa_scoring.csv", index=False)

    # --- (Optional) Score the full MSA ---
    if score_fullmsa:
        msa_name = "fullmsa"
        device_processed_features = rk_utils.move_tensors_to_device(
            fullmsa_processed_feature_dict, device=device
        )
        af2_output, prevs = af_bias(
            device_processed_features,
            [None, None, None],
            num_iters=init_recycling,
            bias=False,
        )
        prevs = [tensor.detach() for tensor in prevs]
        deep_copied_prevs = [tensor.clone().detach() for tensor in prevs]
        af2_output, __ = af_bias(
            device_processed_features, deep_copied_prevs, num_iters=1, bias=False
        )
        plddt = torch.mean(af2_output["plddt"])
        aligned_xyz, plddts_res, pseudo_Bs = rkrf_utils.position_alignment(
            af2_output=af2_output,
            device_processed_features=device_processed_features,
            cra_name=sfc.cra_name,
            best_pos=reference_pos,
            exclude_res=None,
            domain_segs=domain_segs,
            reference_bfactor=init_pos_bfactor,
        )
        llgloss.sfc.atom_b_iso = pseudo_Bs.detach()
        llgloss_rbr.sfc.atom_b_iso = pseudo_Bs.detach()

        if datamode == "xray":
            llgloss, llgloss_rbr, Ecalc, Fc = rkrf_utils.update_sigmaA(
                llgloss=llgloss,
                llgloss_rbr=llgloss_rbr,
                aligned_xyz=aligned_xyz,
                constant_fp_added_HKL=constant_fp_added_HKL,
                constant_fp_added_asu=constant_fp_added_asu,
            )
        optimized_xyz, loss_track_pose = rk_coordinates.rigidbody_refine_quat(
            aligned_xyz,
            llgloss_rbr,
            sfc.cra_name,
            domain_segs=domain_segs,
            lbfgs=RBR_LBFGS,
            added_chain_HKL=constant_fp_added_HKL,
            added_chain_asu=constant_fp_added_asu,
            lbfgs_lr=LBFGS_LR,
            verbose=False,
        )
        llg = llgloss(
            optimized_xyz,
            bin_labels=None,
            num_batch=1,
            sub_ratio=1.0,
            solvent=True,
            update_scales=True,
            added_chain_HKL=constant_fp_added_HKL,
            added_chain_asu=constant_fp_added_asu,
        )
        llgloss.sfc.atom_pos_orth = optimized_xyz
        llgloss.sfc.savePDB(str(output_directory_path / f"{msa_name}_postRBR.pdb"))
        (
            plddt_i,
            llg_i,
        ) = (
            plddt.item(),
            llg.item(),
            # llgloss.sfc.r_free.item(),
            # llgloss.sfc.r_work.item(),
        )
        df_tmp = pd.DataFrame({
            "msa_name": [msa_name],
            "depth": [fullmsa_feature_dict["msa"].shape[0]],
            "mean_plddt": [plddt_i],
            "llg": [llg_i],
            # "rfree": [rfree_i],
            # "rwork": [rwork_i],
        })
        df_tmp.to_csv(
            output_directory_path / "msa_scoring.csv",
            mode="a",
            header=False,
            index=False,
        )

    # --- Score all MSAs ---
    msa_dir = Path(msa_input_dir)
    a3m_paths = sorted([str(p) for p in msa_dir.glob("*.a3m")]) if msa_dir.is_dir() else []
    print(f"{len(a3m_paths)} msa files available...", flush=True)

    for a3m_path in tqdm(a3m_paths):
        p = Path(a3m_path)
        msa_name = p.stem
        data_processor = data_pipeline.DataPipeline(template_featurizer=None)
        temp_alignment_dir = p.parent / "tmp_align"
        temp_alignment_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(p), str(temp_alignment_dir / (msa_name + ".a3m")))
        feature_dict = rkrf_utils.generate_feature_dict(
            fasta_path,
            str(temp_alignment_dir),
            data_processor,
        )
        afconfig = model_config(PRESET)
        afconfig.data.common.max_recycling_iters = init_recycling
        del afconfig.data.common.masked_msa
        afconfig.data.common.resample_msa_in_recycling = False
        feature_processor = feature_pipeline.FeaturePipeline(afconfig.data)
        processed_feature_dict = feature_processor.process_features(
            feature_dict, mode="predict"
        )

        if chimera_profile:
            sub_profile = processed_feature_dict["msa_feat"][:, :, 25:48].clone()
            processed_feature_dict["msa_feat"][:, :, 25:48] = torch.where(
                sub_profile == 0.0, full_profile.clone(), sub_profile.clone()
            )

        device_processed_features = rk_utils.move_tensors_to_device(
            processed_feature_dict, device=device
        )

        af2_output, prevs = af_bias(
            device_processed_features,
            [None, None, None],
            num_iters=init_recycling,
            bias=False,
        )
        prevs = [tensor.detach() for tensor in prevs]
        deep_copied_prevs = [tensor.clone().detach() for tensor in prevs]
        af2_output, __ = af_bias(
            device_processed_features, deep_copied_prevs, num_iters=1, bias=False
        )
        plddt = torch.mean(af2_output["plddt"])
        aligned_xyz, plddts_res, pseudo_Bs = rkrf_utils.position_alignment(
            af2_output=af2_output,
            device_processed_features=device_processed_features,
            cra_name=sfc.cra_name,
            best_pos=reference_pos,
            exclude_res=None,
            domain_segs=domain_segs,
            reference_bfactor=init_pos_bfactor,
        )
        llgloss.sfc.atom_b_iso = pseudo_Bs.detach()
        llgloss_rbr.sfc.atom_b_iso = pseudo_Bs.detach()
        if datamode == "xray":
            llgloss, llgloss_rbr, Ecalc, Fc = rkrf_utils.update_sigmaA(
                llgloss=llgloss,
                llgloss_rbr=llgloss_rbr,
                aligned_xyz=aligned_xyz,
                constant_fp_added_HKL=constant_fp_added_HKL,
                constant_fp_added_asu=constant_fp_added_asu,
            )
        optimized_xyz, loss_track_pose = rk_coordinates.rigidbody_refine_quat(
            aligned_xyz,
            llgloss_rbr,
            sfc.cra_name,
            domain_segs=domain_segs,
            lbfgs=RBR_LBFGS,
            added_chain_HKL=constant_fp_added_HKL,
            added_chain_asu=constant_fp_added_asu,
            lbfgs_lr=LBFGS_LR,
            verbose=False,
        )
        llg = llgloss(
            optimized_xyz,
            bin_labels=None,
            num_batch=1,
            sub_ratio=1.0,
            solvent=True,
            update_scales=True,
            added_chain_HKL=constant_fp_added_HKL,
            added_chain_asu=constant_fp_added_asu,
        )
        llgloss.sfc.atom_pos_orth = optimized_xyz
        llgloss.sfc.savePDB(str(output_directory_path / f"{msa_name}_postRBR.pdb"))
        (
            plddt_i,
            llg_i,
        ) = (
            plddt.item(),
            llg.item(),
            # llgloss.sfc.r_free.item(),
            # llgloss.sfc.r_work.item(),
        )
        df_tmp = pd.DataFrame({
            "msa_name": [msa_name],
            "depth": [feature_dict["msa"].shape[0]],
            "mean_plddt": [plddt_i],
            "llg": [llg_i],
            # "rfree": [rfree_i],
            # "rwork": [rwork_i],
        })
        df_tmp.to_csv(
            str(output_directory_path / "msa_scoring.csv"),
            mode="a",
            header=False,
            index=False,
        )
        shutil.rmtree(str(temp_alignment_dir))

    return str(output_directory_path)

