import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
from libraries import plt, CategorySNP, my_dpi, sfs_weight, theta, write_sfs

weak = ["A", "T"]
strong = ["G", "C"]
SNP = namedtuple('SNP', ['chr', 'pos', 's_sift', 's_mutsel', 's_omega', 's_omega_0', 'derived_count', 'frequency'])


def daf_to_sfs(daf_list, min_n):
    array_daf = np.array(daf_list, dtype=np.int64).T
    return np.array([np.bincount(d, minlength=min_n) for d in array_daf])


def normalize_sfs(sfs):
    return (sfs.T / np.sum(sfs, axis=1)).T


def plot_sfs(cat_snps, snp_sfs, normed_sfs_syn_mean, max_daf, daf_axis, cat_dico_count, output, scaled):
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
    for cat in snp_sfs:
        sfs = snp_sfs[cat][:, 1:].copy()
        if scaled == "neutral":
            sfs *= np.array([i for i in range(1, max_daf)])
        elif scaled == "normalize":
            sfs = normalize_sfs(sfs)
        elif scaled == "synonymous":
            if cat == "syn":
                continue
            sfs = normalize_sfs(sfs)
            sfs /= normed_sfs_syn_mean

        mean_sfs = np.mean(sfs, axis=0)
        std_sfs = np.std(sfs, axis=0)
        label = cat_snps.label(cat) + f" $({int(cat_dico_count[cat])}~mutations)$"
        plt.scatter(daf_axis, mean_sfs, color=cat_snps.color(cat))
        plt.plot(daf_axis, mean_sfs, label=label, color=cat_snps.color(cat), linewidth=1.0)
        plt.fill_between(daf_axis, mean_sfs - std_sfs, mean_sfs + std_sfs, linewidth=1.0,
                         color=cat_snps.color(cat), alpha=0.2)
    if max_daf < 32:
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.xlabel("Derived allele count")
    if scaled == "neutral":
        plt.ylabel('Proportion of mutations (scaled by neutral expectation)')
    elif scaled == "normalize":
        plt.ylabel('Proportion of mutations')
        plt.yscale("log")
    elif scaled == "synonymous":
        plt.axhline(1.0, color="black")
        plt.ylabel('Proportion of mutations (relative to synonymous)')
    else:
        plt.ylabel("Proportion of mutations (scaled by opportunities)")
        plt.yscale("log")
    if len(snp_sfs) < 8:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output, format="pdf")
    plt.clf()
    plt.close("all")


def main(args):
    os.makedirs(os.path.dirname(args.output_tsv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_pdf), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    assert args.method in ["MutSel", "Omega", "SIFT"]

    cat_snps = CategorySNP(args.method, args.bounds, bins=args.bins, windows=args.windows)
    opp_results = pd.read_csv(args.opportunities, sep='\t')
    opp_dico = {k: opp_results[k].values[0] for k in opp_results}
    ldn, lds = opp_dico["Ldn"], opp_dico["Lds"]
    print(f'ratio Ldn/(Lds + Ldn)={ldn / (lds + ldn)}')

    df_snps = pd.read_csv(args.tsv, sep='\t')
    assert len(set(df_snps["sample_size"])) == 1
    sample_size = max(df_snps["sample_size"])
    max_daf = min(sample_size, args.subsample)
    snps_daf = defaultdict(list)

    for cat in cat_snps:
        df = df_snps[df_snps[f"cat_{args.method}"].str.contains(f"|{cat}|", regex=False)]
        for k in df["count"]:
            if max_daf < sample_size:
                count = [np.random.hypergeometric(k, sample_size - k, max_daf) for _ in range(args.nbr_replicates)]
                count = [i if i != max_daf else 0 for i in count]
            else:
                count = [k]
            snps_daf[cat].append(count)

    cat_dico_count = {cat: len(daf) for cat, daf in snps_daf.items()}
    snp_sfs = {cat: daf_to_sfs(daf, max_daf) for cat, daf in snps_daf.items()}
    snp_sfs_mean = {cat: np.mean(sfs, axis=0) for cat, sfs in snp_sfs.items()}

    sfs_nonsyn_mean = np.sum([sfs for cat, sfs in snp_sfs_mean.items() if cat != "syn"], axis=0)
    output_all = os.path.join(args.output_dir, 'all')
    write_sfs(snp_sfs_mean["syn"], sfs_nonsyn_mean, ldn, lds, max_daf, output_all, args.pop, "")

    daf_axis = range(1, max_daf)
    theta_dict = defaultdict(list)
    for cat, mean_sfs in snp_sfs_mean.items():
        if cat == "syn":
            ld_cat = lds
        else:
            ld_cat = ldn * opp_dico[cat]
            output_cat = os.path.join(args.output_dir, cat)
            write_sfs(snp_sfs_mean["syn"], mean_sfs, ld_cat, lds, max_daf, output_cat, args.pop, "")

        theta_dict["category"].append(cat)
        for theta_method in sfs_weight:
            theta_dict[theta_method].append(theta(mean_sfs[1:] / ld_cat, max_daf, theta_method))
        snp_sfs[cat] = snp_sfs[cat] / ld_cat

    df = pd.DataFrame(theta_dict)
    df.to_csv(args.output_tsv, sep="\t", index=False)

    normed_sfs_syn_mean = snp_sfs_mean["syn"][1:] / np.sum(snp_sfs_mean["syn"][1:])
    for scaled in ["", "neutral", "normalize", "synonymous"]:
        output = args.output_pdf.replace('.pdf', f'.{scaled}.pdf') if scaled != "" else args.output_pdf
        plot_sfs(cat_snps, snp_sfs, normed_sfs_syn_mean, max_daf, daf_axis, cat_dico_count, output, scaled)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', required=False, type=str, dest="tsv", help="Input tsv file")
    parser.add_argument('--method', required=False, type=str, dest="method", help="Sel coeff parameter")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    parser.add_argument('--windows', required=False, default=0, type=int, dest="windows", help="Number of windows")
    parser.add_argument('--bounds', required=False, default="", type=str, dest="bounds", help="Input bound file path")
    parser.add_argument('--output_tsv', required=False, type=str, dest="output_tsv", help="Output tsv file")
    parser.add_argument('--output_pdf', required=False, type=str, dest="output_pdf", help="Output pdf file")
    parser.add_argument('--output_dir', required=False, type=str, dest="output_dir", help="Output directory for sfs")
    parser.add_argument('--opportunities', required=False, type=str, dest="opportunities",
                        help="Input opportunities results tsv file")
    parser.add_argument('--pop', required=False, type=str, dest="pop", help="Focal population")
    parser.add_argument('--subsample', required=False, type=int, default=16, dest="subsample")
    parser.add_argument('--nbr_replicates', required=False, type=int, default=1, dest="nbr_replicates")
    main(parser.parse_args())
