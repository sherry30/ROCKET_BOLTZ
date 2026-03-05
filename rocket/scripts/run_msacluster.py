"""
Cluster sequences in a MSA using DBSCAN algorithm and write .a3m file for each cluster.
Assumes first sequence in fasta is the query sequence.

rk.msacluster v1 -i alignments/ -o msaclusters --run_TSNE

This script is modified from AF_cluster repo:
https://github.com/HWaymentSteele/AF_Cluster/blob/main/scripts/ClusterMSA.py
"""

import argparse

from rocket import run_msa_cluster


def main():
    p = argparse.ArgumentParser(
        description="""
    Cluster sequences in a MSA using DBSCAN algorithm and write .a3m file for each cluster.
    Assumes first sequence in fasta is the query sequence.

    H Wayment-Steele, 2022

    Modified by Minhuan Li for ROCKET, 2024
    """  # noqa: E501
    )

    p.add_argument(
        "keyword", action="store", help="Keyword to call all generated MSAs."
    )
    p.add_argument(
        "-i",
        action="store",
        help="fasta/a3m file of original alignment, or path containing fasta/a3m files",
    )
    p.add_argument(
        "-o", action="store", help="name of output directory to write MSAs to."
    )
    p.add_argument(
        "--n_controls",
        action="store",
        default=10,
        type=int,
        help="Number of control msas to generate (Default 10)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print cluster info as they are generated.",
    )

    p.add_argument(
        "--scan", action="store_true", help="Select eps value on 1/4 of data, shuffled."
    )
    p.add_argument(
        "--eps_val",
        action="store",
        type=float,
        help="Use single value for eps instead of scanning.",
    )
    p.add_argument(
        "--resample",
        action="store_true",
        help="If included, will resample the original MSA with replacement before writing.",  # noqa: E501
    )
    p.add_argument(
        "--gap_cutoff",
        action="store",
        type=float,
        default=0.25,
        help="Remove sequences with gaps representing more than this frac of seq.",
    )
    p.add_argument(
        "--min_eps",
        action="store",
        type=float,
        default=3.0,
        help="Min epsilon value to scan for DBSCAN (Default 3).",
    )
    p.add_argument(
        "--max_eps",
        action="store",
        type=float,
        default=20.0,
        help="Max epsilon value to scan for DBSCAN (Default 20).",
    )
    p.add_argument(
        "--eps_step",
        action="store",
        type=float,
        default=0.5,
        help="step for epsilon scan for DBSCAN (Default 0.5).",
    )
    p.add_argument(
        "--min_samples",
        action="store",
        type=int,
        default=10,
        help="Default min_samples for DBSCAN (Default 3, recommended no lower than that).",  # noqa: E501
    )

    p.add_argument(
        "--run_PCA",
        action="store_true",
        help="Run PCA on one-hot embedding of sequences and store in output_cluster_metadata.tsv",  # noqa: E501
    )
    p.add_argument(
        "--run_TSNE",
        action="store_true",
        help="Run TSNE on one-hot embedding of sequences and store in output_cluster_metadata.tsv",  # noqa: E501
    )

    args = p.parse_args()

    run_msa_cluster(
        keyword=args.keyword,
        input_path=args.i,
        output_dir=args.o,
        n_controls=args.n_controls,
        verbose=args.verbose,
        eps_val=args.eps_val,
        resample=args.resample,
        gap_cutoff=args.gap_cutoff,
        min_eps=args.min_eps,
        max_eps=args.max_eps,
        eps_step=args.eps_step,
        min_samples=args.min_samples,
        run_pca=args.run_PCA,
        run_tsne=args.run_TSNE,
    )


if __name__ == "__main__":
    main()
