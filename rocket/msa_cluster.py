from pathlib import Path

import numpy as np
import pandas as pd
from polyleven import levenshtein
from sklearn.cluster import DBSCAN

from rocket.msacluster_utils import (
    consensusVoting,
    encode_seqs,
    load_fasta,
    lprint,
    plot_landscape,
    write_fasta,
)

def run_msa_cluster(
    keyword: str,
    input_path: str,
    output_dir: str,
    n_controls: int = 10,
    verbose: bool = False,
    eps_val: float | None = None,
    resample: bool = False,
    gap_cutoff: float = 0.25,
    min_eps: float = 3.0,
    max_eps: float = 20.0,
    eps_step: float = 0.5,
    min_samples: int = 10,
    run_pca: bool = False,
    run_tsne: bool = False,
) -> str:
    """
    Cluster sequences in a MSA using DBSCAN algorithm and write .a3m file for each cluster.

    Args:
        keyword: Keyword to call all generated MSAs
        input_path: fasta/a3m file of original alignment, or path containing fasta/a3m files
        output_dir: name of output directory to write MSAs to
        n_controls: Number of control msas to generate (default 10)
        verbose: Print cluster info as they are generated
        eps_val: Use single value for eps instead of scanning (default None = scan)
        resample: If True, will resample the original MSA with replacement before writing
        gap_cutoff: Remove sequences with gaps representing more than this frac of seq
        min_eps: Min epsilon value to scan for DBSCAN
        max_eps: Max epsilon value to scan for DBSCAN
        eps_step: Step for epsilon scan for DBSCAN
        min_samples: min_samples parameter for DBSCAN
        run_pca: Run PCA on one-hot embedding of sequences and store in metadata
        run_tsne: Run TSNE on one-hot embedding of sequences and store in metadata

    Returns:
        Path to the output directory
    """
    if run_pca:
        from sklearn.decomposition import PCA

    if run_tsne:
        from sklearn.manifold import TSNE

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    f = open(f"msacluster_{keyword}.log", "w")  # noqa: SIM115
    IDs, seqs = load_fasta(input_path)

    seqs = [
        "".join([x for x in s if x.isupper() or x == "-"]) for s in seqs
    ]  # remove lowercase letters in alignment

    df = pd.DataFrame({"SequenceName": IDs, "sequence": seqs})

    query_ = df.iloc[:1]
    df = df.iloc[1:]

    if resample:
        df = df.sample(frac=1)

    L = len(df.sequence.iloc[0])
    N = len(df)  # noqa: F841

    df["frac_gaps"] = [x.count("-") / L for x in df["sequence"]]

    former_len = len(df)
    df = df.loc[df.frac_gaps < gap_cutoff]

    new_len = len(df)
    lprint(keyword, f)
    lprint(
        f"{former_len - new_len} seqs removed for containing more than {int(gap_cutoff * 100)}% gaps, {new_len} remaining",  # noqa: E501
        f,
    )

    ohe_seqs = encode_seqs(df.sequence.tolist(), max_len=L)

    n_clusters_list = []
    eps_test_vals = np.arange(min_eps, max_eps + eps_step, eps_step)

    if eps_val is None:  # performing scan
        lprint("eps\tn_clusters\tn_not_clustered", f)

        for eps in eps_test_vals:
            testset = encode_seqs(df.sample(frac=0.25).sequence.tolist(), max_len=L)
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(testset)
            n_clust = len(set(clustering.labels_))
            n_not_clustered = len(
                clustering.labels_[np.where(clustering.labels_ == -1)]
            )
            lprint("%.2f\t%d\t%d" % (eps, n_clust, n_not_clustered), f)  # noqa: UP031
            n_clusters_list.append(n_clust)
            if eps > 10 and n_clust == 1:
                break

        eps_to_select = eps_test_vals[np.argmax(n_clusters_list)]
    else:
        eps_to_select = eps_val

    # perform actual clustering

    clustering = DBSCAN(eps=eps_to_select, min_samples=min_samples).fit(ohe_seqs)

    lprint("Selected eps={:.2f}".format(eps_to_select), f)  # noqa: UP032

    lprint("%d total seqs" % len(df), f)  # noqa: UP031

    df["dbscan_label"] = clustering.labels_

    clusters = [x for x in df.dbscan_label.unique() if x >= 0]
    unclustered = len(df.loc[df.dbscan_label == -1])

    lprint(
        "%d clusters, %d of %d not clustered (%.2f)"  # noqa: UP031
        % (len(clusters), unclustered, len(df), unclustered / len(df)),
        f,
    )

    avg_dist_to_query = np.mean([
        1 - levenshtein(x, query_["sequence"].iloc[0]) / L
        for x in df.loc[df.dbscan_label == -1]["sequence"].tolist()
    ])
    lprint("avg identity to query of unclustered: %.2f" % avg_dist_to_query, f)  # noqa: UP031

    avg_dist_to_query = np.mean([
        1 - levenshtein(x, query_["sequence"].iloc[0]) / L
        for x in df.loc[df.dbscan_label != -1]["sequence"].tolist()
    ])
    lprint("avg identity to query of clustered: %.2f" % avg_dist_to_query, f)  # noqa: UP031

    cluster_metadata = []
    for clust in clusters:
        tmp = df.loc[df.dbscan_label == clust]

        cs = consensusVoting(tmp.sequence.tolist())

        avg_dist_to_cs = np.mean([
            1 - levenshtein(x, cs) / L for x in tmp.sequence.tolist()
        ])
        avg_dist_to_query = np.mean([
            1 - levenshtein(x, query_["sequence"].iloc[0]) / L
            for x in tmp.sequence.tolist()
        ])

        if verbose:
            print("Cluster %d consensus seq, %d seqs:" % (clust, len(tmp)))  # noqa: UP031
            print(cs)
            print("#########################################")
            for _, row in tmp.iterrows():
                print(row["SequenceName"], row["sequence"])
            print("#########################################")

        tmp = pd.concat([query_, tmp], axis=0)

        cluster_metadata.append({
            "cluster_ind": clust,
            "consensusSeq": cs,
            "avg_lev_dist": "%.3f" % avg_dist_to_cs,  # noqa: UP031
            "avg_dist_to_query": "%.3f" % avg_dist_to_query,  # noqa: UP031
            "size": len(tmp),
        })

        write_fasta(
            tmp.SequenceName.tolist(),
            tmp.sequence.tolist(),
            outfile=str(Path(output_dir) / f"{keyword}_{clust:03d}.a3m"),
        )

    print(f"writing {n_controls} size-10 uniformly sampled clusters", flush=True)
    for i in range(n_controls):
        tmp = df.sample(n=10)
        tmp = pd.concat([query_, tmp], axis=0)
        write_fasta(
            tmp.SequenceName.tolist(),
            tmp.sequence.tolist(),
            outfile=str(Path(output_dir) / f"U10-{keyword}_{i:03d}.a3m"),
        )
    if len(df) > 100:
        print(
            f"writing {n_controls} size-100 uniformly sampled clusters", flush=True
        )
        for i in range(n_controls):
            tmp = df.sample(n=100)
            tmp = pd.concat([query_, tmp], axis=0)
            write_fasta(
                tmp.SequenceName.tolist(),
                tmp.sequence.tolist(),
                outfile=str(Path(output_dir) / f"U100-{keyword}_{i:03d}.a3m"),
            )

    if run_pca:
        lprint("Running PCA ...", f)
        ohe_vecs = encode_seqs(df.sequence.tolist(), max_len=L)
        mdl = PCA()
        embedding = mdl.fit_transform(ohe_vecs)

        query_embedding = mdl.transform(
            encode_seqs(query_.sequence.tolist(), max_len=L)
        )

        df["PC 1"] = embedding[:, 0]
        df["PC 2"] = embedding[:, 1]

        query_["PC 1"] = query_embedding[:, 0]
        query_["PC 2"] = query_embedding[:, 1]

        # Create a simple namespace for plot_landscape compatibility
        class Args:
            pass
        args = Args()
        args.keyword = keyword
        args.o = output_dir

        plot_landscape("PC 1", "PC 2", df, query_, "PCA", output_dir=output_dir, keyword=keyword)

        lprint("Saved PCA plot to " + str(Path(output_dir) / f"{keyword}_PCA.pdf"), f)

    if run_tsne:
        lprint("Running TSNE ...", f)
        ohe_vecs = encode_seqs(
            df.sequence.tolist() + [query_.sequence.tolist()], max_len=L
        )
        # different than PCA because tSNE doesn't have .transform attribute

        mdl = TSNE()
        embedding = mdl.fit_transform(ohe_vecs)

        df["TSNE 1"] = embedding[:-1, 0]
        df["TSNE 2"] = embedding[:-1, 1]

        query_["TSNE 1"] = embedding[-1:, 0]
        query_["TSNE 2"] = embedding[-1:, 1]

        plot_landscape("TSNE 1", "TSNE 2", df, query_, "TSNE", output_dir=output_dir, keyword=keyword)

        lprint("Saved TSNE plot to " + str(Path(output_dir) / f"{keyword}_TSNE.pdf"), f)

    outfile = str(Path(output_dir) / f"{keyword}_clustering_assignments.tsv")
    lprint("wrote clustering data to %s" % outfile, f)  # noqa: UP031
    df.to_csv(outfile, index=False, sep="\t")

    metad_outfile = str(Path(output_dir) / f"{keyword}_cluster_metadata.tsv")
    lprint("wrote cluster metadata to %s" % metad_outfile, f)  # noqa: UP031
    metad_df = pd.DataFrame.from_records(cluster_metadata)
    metad_df.to_csv(metad_outfile, index=False, sep="\t")

    print("Saved this output to msacluster_%s.log" % keyword)  # noqa: UP031
    f.close()

    return output_dir
