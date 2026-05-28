"""
rk.prep_boltz2 — Prepare Boltz-2 features and ROCKET configs for crystallographic
refinement.

This command bridges rk.preprocess (which produces ROCKET_inputs/) and rk.refine
by generating:
  1. feats_boltz2.pkl   — Boltz-2 featurizer output (sequence → tokenized → batched)
  2. ROCKET_config_phase1_boltz2.yaml
  3. ROCKET_config_phase2_boltz2.yaml

Usage
-----
    rk.prep_boltz2 \\
        --output_dir ./1lj5_processed \\
        --file_id 1lj5 \\
        --cache_dir /path/to/boltz_cache

Then on the GPU node:
    rk.refine ./1lj5_processed/ROCKET_config_phase1_boltz2.yaml \\
              --feats ./1lj5_processed/ROCKET_inputs/feats_boltz2.pkl
    rk.refine ./1lj5_processed/ROCKET_config_phase2_boltz2.yaml \\
              --feats ./1lj5_processed/ROCKET_inputs/feats_boltz2.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import uuid
from pathlib import Path


def cli_run_prep_boltz2() -> None:
    parser = argparse.ArgumentParser(
        prog="rk.prep_boltz2",
        description="Prepare Boltz-2 feats and ROCKET config files for refinement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Working directory produced by rk.preprocess (must contain ROCKET_inputs/).",
    )
    parser.add_argument(
        "--file_id",
        required=True,
        help="Dataset identifier used by rk.preprocess (e.g. '1lj5').",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help=(
            "Boltz cache directory (contains mols/ sub-dir and boltz2_conf.ckpt). "
            "Defaults to $BOLTZ_CACHE env var, then ~/.boltz."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit path to boltz2_conf.ckpt. Defaults to <cache_dir>/boltz2_conf.ckpt.",
    )
    parser.add_argument(
        "--a3m_path",
        default=None,
        help=(
            "Path to a pre-computed MSA a3m file (recommended). "
            "Generate with: rk.generate_msa. "
            "Activates Boltz-2's MSA module during refinement for better gradient signal."
        ),
    )
    parser.add_argument(
        "--truncated_backprop_steps",
        type=int,
        default=20,
        help="Number of reverse-diffusion steps to retain in the autograd graph (K). K=5 gives too noisy gradients; K=20 is the validated default.",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=200,
        help="Total reverse-diffusion steps per forward pass.",
    )
    parser.add_argument(
        "--recycling_steps",
        type=int,
        default=3,
        help="Boltz-2 trunk recycling iterations.",
    )
    parser.add_argument(
        "--num_of_runs",
        type=int,
        default=3,
        help="Number of independent Phase-1 traces (diversity via different diffusion seeds).",
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="CUDA device index for refinement.",
    )
    parser.add_argument(
        "--phase1_iterations",
        type=int,
        default=100,
        help="Gradient steps per Phase-1 trace.",
    )
    parser.add_argument(
        "--phase2_iterations",
        type=int,
        default=500,
        help="Gradient steps for Phase-2 refinement.",
    )
    parser.add_argument(
        "--n_seeds_to_scan",
        type=int,
        default=9,
        help="Number of diffusion seeds to evaluate during seed pre-scan (default 9).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    rocket_inputs = output_dir / "ROCKET_inputs"
    pdb_path = rocket_inputs / f"{args.file_id}-pred-aligned.pdb"

    if not pdb_path.exists():
        raise FileNotFoundError(
            f"Expected AlphaFold-aligned PDB at:\n  {pdb_path}\n"
            "Run 'rk.preprocess' first to generate ROCKET_inputs/."
        )

    cache_dir = Path(
        args.cache_dir
        if args.cache_dir
        else os.environ.get("BOLTZ_CACHE", Path.home() / ".boltz")
    )
    checkpoint = Path(args.checkpoint) if args.checkpoint else cache_dir / "boltz2_conf.ckpt"

    # ------------------------------------------------------------------
    # 1. Prepare feats (CPU — no GPU needed)
    # ------------------------------------------------------------------
    from rocket.refinement_boltz2 import prepare_boltz2_feats, precompute_boltz2_seeds

    feats = prepare_boltz2_feats(
        pdb_path=pdb_path,
        cache_dir=cache_dir,
        a3m_path=args.a3m_path if args.a3m_path else None,
        device="cpu",
    )
    feats_path = rocket_inputs / "feats_boltz2.pkl"
    with open(feats_path, "wb") as fh:
        pickle.dump(feats, fh)
    print(f"[rk.prep_boltz2] Saved feats → {feats_path}")

    # ------------------------------------------------------------------
    # 2. Generate Phase-1 config
    # ------------------------------------------------------------------
    from rocket.refinement_config import (
        AlgorithmConfig,
        AlphaFoldConfig,
        Boltz2Config,
        DataConfig,
        ExecutionConfig,
        OptimizationParams,
        PathConfig,
        RocketRefinmentConfig,
        gen_config_phase2,
    )

    run_uuid = uuid.uuid4().hex[:10]

    phase1_config = RocketRefinmentConfig(
        note=f"phase1_boltz2_{args.file_id}",
        paths=PathConfig(
            path=str(output_dir),
            file_id=args.file_id,
            input_pdb=str(pdb_path),
            uuid_hex=run_uuid,
        ),
        execution=ExecutionConfig(
            cuda_device=args.cuda_device,
            num_of_runs=args.num_of_runs,
            verbose=False,
            model="boltz2",
        ),
        algorithm=AlgorithmConfig(
            iterations=args.phase1_iterations,
            init_recycling=args.recycling_steps,
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
        data=DataConfig(
            datamode="xray",
            min_resolution=3.0,
        ),
        alphafold=AlphaFoldConfig(use_deepspeed_evo_attention=True),
        boltz2=Boltz2Config(
            boltz2_checkpoint_path=str(checkpoint),
            truncated_backprop_steps=args.truncated_backprop_steps,
            boltz2_recycling_steps=args.recycling_steps,
            boltz2_num_sampling_steps=args.sampling_steps,
            feats_path=str(feats_path),
        ),
    )

    # ------------------------------------------------------------------
    # 2b. Precompute diffusion seed scan (GPU required)
    #
    # Evaluate args.n_seeds diffusion seeds with identity bias once; the best
    # seeds are then selected by rk.refine without re-scanning.
    # ------------------------------------------------------------------
    seed_scan_path = rocket_inputs / "seed_scan.npy"
    try:
        precompute_boltz2_seeds(
            config=phase1_config,
            feats=feats,
            output_path=str(seed_scan_path),
            n_seeds=args.n_seeds_to_scan,
        )
        phase1_config.boltz2.precomputed_seed_scan = str(seed_scan_path)
        print(f"[rk.prep_boltz2] Saved seed scan → {seed_scan_path}")
    except Exception as exc:
        print(f"[rk.prep_boltz2] WARNING: seed scan failed ({exc}); rk.refine will scan at runtime")
        seed_scan_path = None

    phase1_yaml = output_dir / "ROCKET_config_phase1_boltz2.yaml"
    phase1_config.to_yaml_file(str(phase1_yaml))
    print(f"[rk.prep_boltz2] Saved Phase-1 config → {phase1_yaml}")

    # ------------------------------------------------------------------
    # 3. Generate Phase-2 config (warm-starts from Phase-1 best bias)
    # ------------------------------------------------------------------
    phase2_config = gen_config_phase2(phase1_config)
    phase2_config.note = f"phase2_boltz2_{args.file_id}"
    phase2_config.algorithm.iterations = args.phase2_iterations
    phase2_config.boltz2.feats_path = str(feats_path)
    if seed_scan_path:
        phase2_config.boltz2.precomputed_seed_scan = str(seed_scan_path)

    phase2_yaml = output_dir / "ROCKET_config_phase2_boltz2.yaml"
    phase2_config.to_yaml_file(str(phase2_yaml))
    print(f"[rk.prep_boltz2] Saved Phase-2 config → {phase2_yaml}")

    # ------------------------------------------------------------------
    # 4. Print next-step instructions
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Preparation complete.  Run refinement on the GPU node:")
    print()
    print("  ssh shehryar@max-hpcgwg006")
    print("  micromamba activate rocket-of")
    print()
    print("  # Phase 1 (3 independent traces, selects best)")
    print(f"  rk.refine {phase1_yaml} \\")
    print(f"            --feats {feats_path}")
    print()
    print("  # Phase 2 (warm-start from Phase-1 best bias)")
    print(f"  rk.refine {phase2_yaml} \\")
    print(f"            --feats {feats_path}")
    print("=" * 70)
