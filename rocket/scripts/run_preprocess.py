import argparse
import glob
import os
import pickle
import shutil
import subprocess
import uuid
from pathlib import Path

from loguru import logger
from SFC_Torch import PDBParser

from ..refinement_config import gen_config_phase1, gen_config_phase2
from ..utils import plddt2pseudoB_np

### Phenix variables
phenix_directory = os.environ["PHENIX_ROOT"]
phenix_source = os.path.join(phenix_directory, "phenix_env.sh")


def get_script_path(import_stmt: str) -> str:
    """Source Phenix and run phenix.python to get script path."""
    module_name = import_stmt.split("import")[-1].strip()
    python_code = f"{import_stmt}; print({module_name}.__file__)"

    bash_cmd = f'source {phenix_source} && phenix.python -c "{python_code}"'

    try:
        result = subprocess.run(
            ["bash", "-c", bash_cmd], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error resolving script path for '{import_stmt}':\n{e.stderr}")
        return None


# Phenix scripts paths
em_nodockedmodel_script = get_script_path(
    "from New_Voyager.scripts import emplace_simple"
)
em_dockedmodel_script = get_script_path(
    "from cctbx.maptbx import prepare_map_for_refinement"
)
xtal_edata_script = get_script_path("from phasertng.scripts import mtz_generator")


def run_command(command, env_source=None):
    """Runs a shell command with optional Phenix environment sourcing."""
    cmd_str = (
        f"bash -c 'source {env_source} && {' '.join(command)}'"
        if env_source
        else " ".join(command)
    )

    logger.info(f"Executing: {cmd_str}")

    subprocess.run(cmd_str, shell=True, check=True, executable="/bin/bash")


def run_openfold(
    file_id,
    output_dir,
    precomputed_alignment_dir,
    jax_param_path,
    max_recycling_iters,
    use_deepspeed_evoformer_attention,
):
    """Runs OpenFold inference using the specified parameters."""
    fasta_dir = f"{file_id}_fasta"
    predicted_model = os.path.join(
        output_dir, "predictions", f"{file_id}_model_1_ptm_unrelaxed.pdb"
    )

    if os.path.exists(predicted_model):
        logger.info(f"Skipping OpenFold: output {predicted_model} already exists.")
        return predicted_model

    openfold_cmd = [
        "rk.predict",
        fasta_dir,
        "--output_dir",
        f"{output_dir}",
        "--config_preset",
        "model_1_ptm",
        "--model_device",
        "cuda:0",
        "--save_output",
        "--data_random_seed",
        "42",
        "--skip_relaxation",
        "--max_recycling_iters",
        f"{max_recycling_iters}",
        "--use_precomputed_alignments",
        f"{precomputed_alignment_dir}",
    ]
    if use_deepspeed_evoformer_attention:
        openfold_cmd.extend(["--use_deepspeed_evoformer_attention"])

    if jax_param_path:
        openfold_cmd.extend(["--jax_params_path", jax_param_path])

    run_command(openfold_cmd)

    if not os.path.exists(predicted_model):
        raise FileNotFoundError(f"Expected output model {predicted_model} not found.")

    return predicted_model


def generate_seg_id_file(file_id, output_dir):
    """Generates seg_id.txt using chain changes and >20-residue continuous stretches.
    Skips first seg_id, outputs None if only one domain."""
    seg_id_path = os.path.join(output_dir, "ROCKET_inputs", "seg_id.txt")
    aligned_pdb_path = os.path.join(output_dir, "ROCKET_inputs", f"{file_id}-MRed.pdb")

    if not os.path.exists(aligned_pdb_path):
        raise FileNotFoundError(f"Aligned PDB file not found at {aligned_pdb_path}")

    # Collect residues per chain in order of appearance
    chain_residues = {}
    chain_order = []
    with open(aligned_pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                try:
                    chain_id = line[21].strip()
                    res_num = int(line[22:26].strip())
                    if chain_id not in chain_residues:
                        chain_residues[chain_id] = set()
                        chain_order.append(chain_id)
                    chain_residues[chain_id].add(res_num)
                except ValueError:
                    continue

    domain_ranges = []
    seg_start_residues = []
    previous_chain = None

    for chain_id in chain_order:
        if chain_id == previous_chain:
            continue  # Only one domain per unique chain

        residues = sorted(chain_residues[chain_id])
        if not residues:
            continue

        # Find first continuous stretch >20
        current_stretch = [residues[0]]
        for i in range(1, len(residues)):
            if residues[i] == residues[i - 1] + 1:
                current_stretch.append(residues[i])
            else:
                if len(current_stretch) > 20:
                    domain_ranges.append((current_stretch[0], current_stretch[-1]))
                    seg_start_residues.append(current_stretch[0])
                    break
                current_stretch = [residues[i]]

        # Handle final stretch
        if len(current_stretch) > 20 and (
            not domain_ranges
            or domain_ranges[-1] != (current_stretch[0], current_stretch[-1])
        ):
            domain_ranges.append((current_stretch[0], current_stretch[-1]))
            seg_start_residues.append(current_stretch[0])

        previous_chain = chain_id

    # Write seg_id.txt
    with open(seg_id_path, "w") as out_f:
        for i, (start, end) in enumerate(domain_ranges, 1):
            out_f.write(f"domain{i}: {start}-{end}\n")

        if len(seg_start_residues) > 1:
            seg_ids = ",".join(str(r) for r in seg_start_residues[1:])  # Skip first
            out_f.write(f'seg_id: "{seg_ids}"\n')
            logger.info(f"Segment ID file written to {seg_id_path}")
            return seg_start_residues[1:]
        else:
            out_f.write("seg_id: None\n")
            logger.info("No segment, only one domain found.")
            return None


def run_process_predicted_model(file_id, input_dir, predicted_model):
    """Processes the predicted model using Phenix."""
    logger.info("Looking for", predicted_model)

    process_cmd = [
        "phenix.process_predicted_model",
        "output_files.mark_atoms_to_keep_with_occ_one=True",
        f"{predicted_model}",
        "minimum_domain_length=20",
        "b_value_field_is=plddt",
        "minimum_sequential_residues=10",
        f"pae_file={os.path.join(input_dir, f'{file_id}_pae.json')}",
        "pae_power=2",
        "pae_cutoff=4",
        "pae_graph_resolution=0.5",
    ]

    run_command(process_cmd, env_source=phenix_source)


def move_processed_predicted_files(output_dir):
    """Moves processed files into a 'processed_predicted_files' directory."""
    processed_dir = os.path.join(output_dir, "processed_predicted_files")
    os.makedirs(processed_dir, exist_ok=True)

    processed_files = glob.glob("*processed*") + glob.glob("*.seq")

    if not processed_files:
        logger.info("No processed files found to move.")
        return

    for file_path in processed_files:
        if os.path.isfile(file_path):
            shutil.move(
                file_path, os.path.join(processed_dir, os.path.basename(file_path))
            )


def dock_into_data(
    file_id,
    method,
    resolution,
    output_dir,
    predicted_model,
    predocked_model,
    map_file,
    map1,
    map2,
    fixed_model=None,
    fasta_composition=None,
):
    """Handles molecular docking for Xray or CryoEM data."""
    docking_output_dir = os.path.join(output_dir, "docking_outputs")
    os.makedirs(docking_output_dir, exist_ok=True)

    if method == "xray":
        mtz_files = glob.glob(os.path.join(f"{file_id}_data", "*.mtz"))

        for mtz_file in mtz_files:
            if os.path.isfile(mtz_file):
                shutil.copy2(
                    mtz_file,
                    os.path.join(
                        output_dir,
                        "processed_predicted_files",
                        os.path.basename(mtz_file),
                    ),
                )

                # Always run Edata generation
                edata_cmd = ["phenix.python", xtal_edata_script, "-i", mtz_file]
                run_command(edata_cmd, env_source=phenix_source)

        # If predocked_model is provided, skip MR and copy the model directly
        if predocked_model:
            print("Predocked model provided for Xray: skipping MR step.")
            rocket_dir = os.path.join(output_dir, "ROCKET_inputs")
            os.makedirs(rocket_dir, exist_ok=True)

            aligned_pdb_path = os.path.join(rocket_dir, f"{file_id}-MRed.pdb")
            shutil.copy2(predocked_model, aligned_pdb_path)
        else:
            # Proceed with MR step
            mr_cmd = [
                "phasertng.picard",
                f"directory={os.path.join(output_dir, 'processed_predicted_files')}",
                f"database={os.path.join(output_dir, 'phaser_files')}",
            ]
            run_command(mr_cmd, env_source=phenix_source)

    elif method == "cryoem":
        docking_script = (
            em_dockedmodel_script if predocked_model else em_nodockedmodel_script
        )
        docking_cmd = ["phenix.python", docking_script]

        if map_file is not None:
            map_args = [f"--map={map_file}"]
        else:
            map_args = [f"--map1={map1}", f"--map2={map2}"]

        if predocked_model:
            docking_cmd += [
                f"--d_min={resolution}",
                f"--working_model={predocked_model}",
            ]
            docking_cmd += map_args
            if fixed_model:
                docking_cmd.append(f"--fixed_model={fixed_model}")
        else:
            docking_cmd += [
                f"--d_min={resolution}",
                f"--output_folder={docking_output_dir}",
                f"--model_file={predicted_model}",
                f"--sequence_composition={fasta_composition}",
                "--level=logfile",
            ]
            docking_cmd += map_args
            if fixed_model:
                docking_cmd.append(f"--fixed_model={fixed_model}")

        run_command(docking_cmd, env_source=phenix_source)

        if predocked_model:
            for file in ["weighted_map_data.mtz", "likelihood_weighted.map"]:
                src_path = os.path.join(".", file)
                dest_path = os.path.join(docking_output_dir, file)
                if os.path.exists(src_path):
                    shutil.move(src_path, dest_path)

            # Move the predocked model
            model_filename = os.path.basename(predocked_model)
            model_dest_path = os.path.join(docking_output_dir, model_filename)
            if os.path.exists(predocked_model):
                shutil.copy(predocked_model, model_dest_path)


def prepare_rk_inputs(file_id, output_dir, method):
    """Creates ROCKET_inputs directory and moves necessary files."""
    rocket_dir = os.path.join(output_dir, "ROCKET_inputs")
    os.makedirs(rocket_dir, exist_ok=True)

    if method == "xray":
        best_pdb_src = os.path.join(
            output_dir, "phaser_files", "best.1.coordinates.pdb"
        )
        mtz_files = glob.glob("./*feff/*.data.mtz")
    elif method == "cryoem":
        best_pdb_src = next(
            iter(glob.glob(os.path.join(output_dir, "docking_outputs", "*.pdb"))), None
        )
        mtz_files = glob.glob(f"{output_dir}/docking_outputs/weighted_map_data.mtz")
    else:
        raise ValueError("Invalid method. Choose either 'xray' or 'cryoem'.")

    best_pdb_dst = os.path.join(rocket_dir, f"{file_id}-MRed.pdb")
    if best_pdb_src and os.path.exists(best_pdb_src):
        shutil.copy2(best_pdb_src, best_pdb_dst)

    for mtz_src in mtz_files:
        mtz_dst = os.path.join(rocket_dir, f"{file_id}-Edata.mtz")
        shutil.copy2(mtz_src, mtz_dst)


def prepare_pred_aligned(output_dir, file_id, pred_model_path=None):
    mr_model_path = os.path.join(output_dir, "ROCKET_inputs", f"{file_id}-MRed.pdb")
    assert os.path.exists(mr_model_path), f"MR model not found: {mr_model_path}"
    if pred_model_path is None:
        pred_model_path = os.path.join(
            output_dir, "predictions", f"{file_id}_model_1_ptm_unrelaxed.pdb"
        )
    assert os.path.exists(pred_model_path), (
        f"Predicted model not found: {pred_model_path}"
    )
    superpose_command = [
        "phenix.superpose_pdbs",
        f"{mr_model_path}",
        f"{pred_model_path}",
        f"output.file_name={os.path.join(output_dir, 'ROCKET_inputs', f'{file_id}-pred-aligned_unprocessed.pdb')}",  # noqa: E501
    ]
    run_command(superpose_command, env_source=phenix_source)
    aligned_model_path = os.path.join(
        output_dir, "ROCKET_inputs", f"{file_id}-pred-aligned_unprocessed.pdb"
    )
    assert os.path.exists(aligned_model_path), (
        f"Failed to superpose models: {aligned_model_path}"
    )

    mr_model = PDBParser(mr_model_path)
    align_model = PDBParser(aligned_model_path)
    align_model.set_spacegroup(mr_model.spacegroup)
    align_model.set_unitcell(mr_model.cell)
    align_model.set_biso(plddt2pseudoB_np(align_model.atom_b_iso))
    align_model.savePDB(
        os.path.join(output_dir, "ROCKET_inputs", f"{file_id}-pred-aligned.pdb")
    )


def symlink_input_files(file_id, output_dir, precomputed_alignment_dir=None):
    """Symlinks the sequence FASTA and (optionally) alignment directory to the output folder."""
    fasta_src = os.path.join(f"{file_id}_fasta", f"{file_id}.fasta")
    fasta_dst = os.path.join(output_dir, f"{file_id}.fasta")
    if os.path.exists(fasta_src):
        if not os.path.exists(fasta_dst):
            os.symlink(os.path.abspath(fasta_src), fasta_dst)
    else:
        raise FileNotFoundError(f"FASTA file not found: {fasta_src}")

    if precomputed_alignment_dir is not None:
        alignments_dst = os.path.join(output_dir, "alignments")
        if not os.path.exists(alignments_dst):
            os.symlink(
                os.path.join(os.path.abspath(precomputed_alignment_dir), file_id),
                alignments_dst,
            )


def _read_fasta_sequences(fasta_path: str) -> list:
    """Parse a FASTA file and return list of {id, sequence} dicts."""
    sequences = []
    current_id = None
    current_seq = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences.append({"id": current_id, "sequence": "".join(current_seq)})
                current_id = line[1:].split()[0]
                current_seq = []
            elif line:
                current_seq.append(line)
    if current_id is not None:
        sequences.append({"id": current_id, "sequence": "".join(current_seq)})
    return sequences


def _write_boltz_yaml(
    sequences: list, output_dir: str, file_id: str, a3m_path: str = None
) -> str:
    """Write a boltz YAML input file from sequence list; returns path."""
    yaml_path = os.path.join(output_dir, f"{file_id}_boltz_input.yaml")
    if os.path.exists(yaml_path):
        return yaml_path
    lines = ["version: 1", "sequences:"]
    for i, s in enumerate(sequences):
        chain_id = s["id"] if len(s["id"]) == 1 else chr(65 + i)
        lines.append("  - protein:")
        lines.append(f"      id: {chain_id}")
        lines.append(f"      sequence: {s['sequence']}")
        if a3m_path:
            lines.append(f"      msa: {os.path.abspath(a3m_path)}")
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return yaml_path


def _find_a3m_from_alignment_dir(alignment_dir: str, file_id: str) -> str | None:
    """Return the best available a3m file from a precomputed alignment directory.

    Looks inside ``{alignment_dir}/{file_id}/`` for a3m files, preferring
    bfd_uniclust_hits.a3m > uniref90_hits.a3m > mgnify_hits.a3m > any *.a3m.
    Returns None if the directory or no a3m files are found.
    """
    if alignment_dir is None:
        return None
    search_dir = os.path.join(os.path.abspath(alignment_dir), file_id)
    if not os.path.isdir(search_dir):
        return None
    for name in ("bfd_uniclust_hits.a3m", "uniref90_hits.a3m", "mgnify_hits.a3m"):
        candidate = os.path.join(search_dir, name)
        if os.path.exists(candidate):
            logger.info(f"Found MSA for Boltz-2: {candidate}")
            return candidate
    hits = glob.glob(os.path.join(search_dir, "*.a3m"))
    if hits:
        logger.info(f"Found MSA for Boltz-2: {hits[0]}")
        return hits[0]
    return None


def run_boltz2_predict(
    fasta_path: str, output_dir: str, cache_dir: str = None, a3m_path: str = None
) -> str:
    """Run boltz predict on the given FASTA and return path to the output PDB."""
    file_id = os.path.splitext(os.path.basename(fasta_path))[0]
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    pdb_path = os.path.join(predictions_dir, f"{file_id}_boltz2_unrelaxed.pdb")
    if os.path.exists(pdb_path):
        logger.info(f"Skipping boltz predict: output {pdb_path} already exists.")
        return pdb_path

    sequences = _read_fasta_sequences(fasta_path)
    boltz_yaml = _write_boltz_yaml(sequences, output_dir, file_id, a3m_path=a3m_path)
    yaml_stem = os.path.splitext(os.path.basename(boltz_yaml))[0]

    boltz_out_dir = os.path.join(output_dir, "boltz_predictions")
    cmd = [
        "boltz", "predict", boltz_yaml,
        "--out_dir", boltz_out_dir,
        "--model", "boltz2",
        "--diffusion_samples", "1",
        "--output_format", "pdb",
        "--no_kernels",
        "--override",
    ]
    if cache_dir:
        cmd.extend(["--cache", cache_dir])
    run_command(cmd)

    # boltz writes: <out_dir>/boltz_results_<stem>/predictions/<stem>/<stem>_model_0.pdb
    pdb_files = glob.glob(
        os.path.join(boltz_out_dir, "**", "*.pdb"), recursive=True
    )
    if not pdb_files:
        raise FileNotFoundError(f"No boltz predict PDB output found in {boltz_out_dir}")
    boltz_pdb = sorted(pdb_files)[0]
    logger.info(f"Boltz-2 prediction: {boltz_pdb}")

    shutil.copy2(boltz_pdb, pdb_path)
    logger.info(f"Copied to: {pdb_path}")
    return pdb_path


def _generate_boltz2_outputs(args, seg_id: list | None, a3m_path: str = None) -> None:
    """Generate feats_boltz2.pkl and Boltz-2 ROCKET config YAMLs."""
    from ..refinement_boltz2 import prepare_boltz2_feats, precompute_boltz2_seeds
    from ..refinement_config import (
        AlgorithmConfig,
        AlphaFoldConfig,
        Boltz2Config,
        DataConfig,
        ExecutionConfig,
        OptimizationParams,
        PathConfig,
        RocketRefinmentConfig,
    )

    output_dir = os.path.abspath(args.output_dir)
    rocket_inputs = os.path.join(output_dir, "ROCKET_inputs")
    pdb_path = os.path.join(rocket_inputs, f"{args.file_id}-pred-aligned.pdb")

    cache_dir = args.boltz2_cache_dir or os.environ.get(
        "BOLTZ_CACHE", str(Path.home() / ".boltz")
    )
    checkpoint = args.boltz2_checkpoint or os.path.join(cache_dir, "boltz2_conf.ckpt")

    feats = prepare_boltz2_feats(
        pdb_path=pdb_path, cache_dir=cache_dir, a3m_path=a3m_path, device="cpu"
    )
    feats_path = os.path.join(rocket_inputs, "feats_boltz2.pkl")
    with open(feats_path, "wb") as fh:
        pickle.dump(feats, fh)
    logger.info(f"Saved feats → {feats_path}")

    run_uuid = uuid.uuid4().hex[:10]
    phase1_config = RocketRefinmentConfig(
        note=f"phase1_boltz2_{args.file_id}",
        paths=PathConfig(
            path=output_dir,
            file_id=args.file_id,
            input_pdb=pdb_path,
            uuid_hex=run_uuid,
        ),
        execution=ExecutionConfig(
            cuda_device=0,
            num_of_runs=3,
            verbose=False,
            model="boltz2",
        ),
        algorithm=AlgorithmConfig(
            iterations=100,
            init_recycling=3,
            optimization=OptimizationParams(
                # lr=1e-3 overshoots even on the first Adam step (T2 debug result:
                # single step at lr=1e-3 yields Δ=-5 LLG even on full reflections).
                # lr=1e-4 is stable; smooth_stage_epochs=80 starts decay at iter 20,
                # decaying from 1e-4 → 1e-5 to prevent post-peak drift.
                additive_learning_rate=1e-4,
                multiplicative_learning_rate=1e-4,
                l2_weight=1e-7,
                phase2_final_lr=1e-5,
                smooth_stage_epochs=80,
            ),
        ),
        data=DataConfig(datamode=args.method, min_resolution=3.0),
        alphafold=AlphaFoldConfig(use_deepspeed_evo_attention=False),
        boltz2=Boltz2Config(
            boltz2_checkpoint_path=checkpoint,
            truncated_backprop_steps=20,
            boltz2_recycling_steps=3,
            boltz2_num_sampling_steps=200,
            feats_path=feats_path,
        ),
    )
    if seg_id:
        phase1_config.algorithm.domain_segs = seg_id

    # Seed scan — evaluate n_seeds_to_scan diffusion seeds with identity bias
    # so rk.refine can skip the scan and start immediately from the best seeds.
    seed_scan_path = os.path.join(rocket_inputs, "seed_scan.npy")
    n_seeds = getattr(args, "n_seeds_to_scan", 9)
    try:
        precompute_boltz2_seeds(
            config=phase1_config,
            feats=feats,
            output_path=seed_scan_path,
            n_seeds=n_seeds,
        )
        phase1_config.boltz2.precomputed_seed_scan = seed_scan_path
        logger.info(f"Saved seed scan → {seed_scan_path}")
    except Exception as exc:
        logger.warning(f"Seed scan failed ({exc}); rk.refine will scan at runtime")
        seed_scan_path = None

    phase1_yaml = os.path.join(output_dir, "ROCKET_config_phase1_boltz2.yaml")
    phase1_config.to_yaml_file(phase1_yaml)
    logger.info(f"Saved Phase-1 config → {phase1_yaml}")

    phase2_config = gen_config_phase2(phase1_config)
    phase2_config.note = f"phase2_boltz2_{args.file_id}"
    phase2_config.algorithm.iterations = 500
    phase2_config.boltz2.feats_path = feats_path
    if seed_scan_path:
        phase2_config.boltz2.precomputed_seed_scan = seed_scan_path

    phase2_yaml = os.path.join(output_dir, "ROCKET_config_phase2_boltz2.yaml")
    phase2_config.to_yaml_file(phase2_yaml)
    logger.info(f"Saved Phase-2 config → {phase2_yaml}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OpenFold inference and dock into data"
    )

    parser.add_argument("--file_id", required=True)
    parser.add_argument("--method", choices=["xray", "cryoem"], required=True)
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--output_dir", default="preprocessing_output")
    parser.add_argument("--model", choices=["alphafold", "boltz2"], default="alphafold",
                        help="Structure predictor to use (default: alphafold).")
    parser.add_argument("--boltz2_cache_dir", default=None,
                        help="Boltz cache dir (mols/ + boltz2_conf.ckpt). "
                             "Defaults to $BOLTZ_CACHE or ~/.boltz.")
    parser.add_argument("--boltz2_checkpoint", default=None,
                        help="Path to boltz2_conf.ckpt. "
                             "Defaults to <boltz2_cache_dir>/boltz2_conf.ckpt.")
    parser.add_argument("--precomputed_alignment_dir", default="alignments/")
    parser.add_argument("--max_recycling_iters", type=int, default=4)
    parser.add_argument(
        "--use_deepspeed_evoformer_attention",
        action="store_true",
        default=False,
        help="Whether to use the DeepSpeed evoformer attention layer. "
        "Must have deepspeed installed in the environment.",
    )
    parser.add_argument("--jax_params_path", default=None)
    parser.add_argument("--predocked_model", default=None)
    parser.add_argument("--fixed_model", default=None)
    parser.add_argument("--map", default=None)
    parser.add_argument("--map1", default=None)
    parser.add_argument("--map2", default=None)
    parser.add_argument("--full_composition", default=None)
    parser.add_argument(
        "--n_seeds_to_scan", type=int, default=9,
        help="Number of diffusion seeds to pre-evaluate during Boltz-2 preprocessing (default 9).",
    )

    args = parser.parse_args()

    if args.method == "cryoem":
        # Validate required arguments specific to cryo-EM runs.
        missing = [arg for arg in ["resolution"] if getattr(args, arg) is None]
        if missing:
            parser.error(
                f"The following arguments are required for 'cryoem' method: {', '.join(missing)}"  # noqa: E501
            )

        # Reject empty-string values for map arguments early, as they are
        # almost certainly misconfigurations and can hide errors.
        for map_arg in ("map", "map1", "map2"):
            value = getattr(args, map_arg)
            if value == "":
                parser.error(f"--{map_arg} must not be an empty string.")

        # Enforce mutual exclusivity between --map and --map1/--map2.
        if args.map is not None and (args.map1 is not None or args.map2 is not None):
            parser.error(
                "For 'cryoem', use either --map, or the pair --map1 and --map2, but not both."  # noqa: E501
            )

        # Require either a single map or a pair of maps.
        if args.map is None and (args.map1 is None or args.map2 is None):
            parser.error(
                "For 'cryoem', provide either --map, or both --map1 and --map2."
            )

        # Require full_composition only if predocked_model is not provided
        if not args.predocked_model and args.full_composition is None:
            parser.error(
                "--full_composition is required for cryoem when --predocked_model is not provided."  # noqa: E501
            )

    return args


def cli_runpreprocess():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model == "boltz2":
        # --- Boltz-2 path: boltz predict replaces OpenFold + Phenix processing ---
        symlink_input_files(args.file_id, args.output_dir)

        # Auto-detect MSA from precomputed_alignment_dir if available
        a3m_path = _find_a3m_from_alignment_dir(
            args.precomputed_alignment_dir, args.file_id
        )

        fasta_path = os.path.join(args.output_dir, f"{args.file_id}.fasta")
        predicted_model = run_boltz2_predict(
            fasta_path=fasta_path,
            output_dir=args.output_dir,
            cache_dir=args.boltz2_cache_dir,
            a3m_path=a3m_path,
        )

        # Place boltz2 PDB in processed_predicted_files so Phaser MR can find it
        processed_dir = os.path.join(args.output_dir, "processed_predicted_files")
        os.makedirs(processed_dir, exist_ok=True)
        shutil.copy2(
            predicted_model,
            os.path.join(processed_dir, os.path.basename(predicted_model)),
        )

        dock_into_data(
            args.file_id,
            args.method,
            args.resolution,
            args.output_dir,
            predicted_model,
            args.predocked_model,
            args.map,
            args.map1,
            args.map2,
            args.fixed_model,
            args.full_composition,
        )
        prepare_rk_inputs(args.file_id, args.output_dir, args.method)
        prepare_pred_aligned(
            args.output_dir, args.file_id, pred_model_path=predicted_model
        )
        seg_id = generate_seg_id_file(args.file_id, args.output_dir)
        _generate_boltz2_outputs(args, seg_id, a3m_path=a3m_path)

    else:
        # --- AlphaFold / OpenFold path (original) ---
        symlink_input_files(
            args.file_id, args.output_dir, args.precomputed_alignment_dir
        )

        predicted_model = run_openfold(
            args.file_id,
            args.output_dir,
            args.precomputed_alignment_dir,
            args.jax_params_path,
            args.max_recycling_iters,
            args.use_deepspeed_evoformer_attention,
        )
        run_process_predicted_model(args.file_id, args.output_dir, predicted_model)
        move_processed_predicted_files(args.output_dir)

        dock_into_data(
            args.file_id,
            args.method,
            args.resolution,
            args.output_dir,
            predicted_model,
            args.predocked_model,
            args.map,
            args.map1,
            args.map2,
            args.fixed_model,
            args.full_composition,
        )
        prepare_rk_inputs(args.file_id, args.output_dir, args.method)
        prepare_pred_aligned(args.output_dir, args.file_id)
        seg_id = generate_seg_id_file(args.file_id, args.output_dir)

        # Generate AF2 ROCKET configuration yaml files
        phase1_config = gen_config_phase1(
            datamode=args.method,
            file_id=args.file_id,
            working_dir=os.path.abspath(args.output_dir),
            use_deepspeed_evo_attention=args.use_deepspeed_evoformer_attention,
        )
        phase1_config.algorithm.init_recycling = args.max_recycling_iters

        # Use smaller learning rates for low-resolution cryoEM data (> 5 Å)
        resolution_value = None
        if args.resolution is not None:
            try:
                resolution_value = float(args.resolution)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid resolution value '{}' provided; skipping resolution-based "
                    "learning rate adjustment.",
                    args.resolution,
                )
        is_low_res_cryoem = (
            args.method == "cryoem"
            and resolution_value is not None
            and resolution_value > 5.0
        )
        if is_low_res_cryoem:
            logger.info(
                f"Low-resolution cryoEM ({args.resolution} Å): setting lr_a=1e-4, lr_m=1e-3"
            )
            phase1_config.algorithm.optimization.additive_learning_rate = 1e-4
            phase1_config.algorithm.optimization.multiplicative_learning_rate = 1e-3

        if seg_id:
            phase1_config.algorithm.domain_segs = seg_id
        phase1_config.to_yaml_file(
            os.path.join(args.output_dir, "ROCKET_config_phase1.yaml")
        )
        phase2_config = gen_config_phase2(phase1_config)
        phase2_config.to_yaml_file(
            os.path.join(args.output_dir, "ROCKET_config_phase2.yaml")
        )


if __name__ == "__main__":
    cli_runpreprocess()
