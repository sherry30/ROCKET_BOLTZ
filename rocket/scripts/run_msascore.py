#!/usr/bin/env python3
"""
Run LLG scoring for system with different MSAs, supporting both xray and cryoem modes.
"""

import argparse
from pathlib import Path

from rocket.msa_score import run_msa_score

PRESET = "model_1_ptm"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run LLG scoring for system with different msas, "
            "supporting both xray and cryoem modes"
        )
    )
    parser.add_argument("path", action="store", help="Path to parent folder")
    parser.add_argument("system", action="store", help="file_id for the dataset")
    parser.add_argument(
        "-i", action="store", help="prefix for msas to use, path will prepend"
    )
    parser.add_argument(
        "-o",
        action="store",
        help=(
            "name of output directory to write prediction and scoring to, "
            "path will prepend"
        ),
    )
    parser.add_argument(
        "--datamode",
        choices=["xray", "cryoem"],
        required=True,
        help="Choose between xray or cryoem mode",
    )
    parser.add_argument(
        "--domain_segs",
        type=int,
        nargs="*",
        default=None,
        help="A list of resid as domain boundaries",
    )
    parser.add_argument(
        "--additional_chain", action="store_true", help="Additional Chain in ASU"
    )
    parser.add_argument(
        "--init_recycling", default=4, type=int, help="number of initial recycling"
    )
    parser.add_argument(
        "--free_flag", default="R-free-flags", type=str, help="Column name of free flag"
    )
    parser.add_argument(
        "--testset_value", default=1, type=int, help="Value for test set"
    )
    parser.add_argument(
        "--voxel_spacing",
        default=4.5,
        type=float,
        help="Voxel spacing for solvent percentage estimation",
    )
    parser.add_argument(
        "--min_resolution", default=3.0, type=float, help="min resolution cut"
    )
    parser.add_argument(
        "--chimera_profile", action="store_true", help="Use chimera profile"
    )
    parser.add_argument(
        "--score_fullmsa", action="store_true", help="Also score the full msa"
    )
    parser.add_argument(
        "--full_msa_dir",
        default=None,
        type=str,
        help="Path to full MSA directory (default: path/alignments)",
    )
    config = parser.parse_args()


    msa_input_prefix=config.i
    output_dir_name=config.o

    msa_input_dir = Path(config.path) / msa_input_prefix
    output_dir = Path(config.path) /output_dir_name

    run_msa_score(
        path=config.path,
        system=config.system,
        msa_input_dir=msa_input_dir,
        output_dir=output_dir,
        datamode=config.datamode,
        domain_segs=config.domain_segs,
        additional_chain=config.additional_chain,
        init_recycling=config.init_recycling,
        free_flag=config.free_flag,
        testset_value=config.testset_value,
        voxel_spacing=config.voxel_spacing,
        min_resolution=config.min_resolution,
        chimera_profile=config.chimera_profile,
        score_fullmsa=config.score_fullmsa,
        full_msa_dir=config.full_msa_dir,
    )


if __name__ == "__main__":
    main()
