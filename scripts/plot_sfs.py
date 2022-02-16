import argparse
import gzip
import numpy as np
import pandas as pd
import scipy.stats as sps
from libraries_plot import plt, CategorySNP, my_dpi, multiline
from libraries import sfs_weight, theta, write_sfs, write_dofe
from collections import defaultdict

weak = ["A", "T"]
strong = ["G", "C"]


def daf_to_sfs(daf_list, min_n):
    array_daf = np.array(daf_list, dtype=np.int64).T
    return np.array([np.bincount(d, minlength=min_n) for d in array_daf])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vcf', required=False, type=str, dest="vcf", help="Input vcf file")
    parser.add_argument('--method', required=False, type=str, dest="method", help="Sel coeff parameter")
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output pdf file")
    parser.add_argument('--genome_results', required=False, type=str, dest="genome",
                        help="Input genome results tsv file")
    parser.add_argument('--pop', required=False, type=str, dest="pop", help="Focal population")
    parser.add_argument('--subsample', required=False, type=int, default=16, dest="subsample")
    parser.add_argument('--nbr_replicates', required=False, type=int, default=1, dest="nbr_replicates")
    args = parser.parse_args()

    assert args.method in ["MutSel", "Omega", "Omega_0", "WS", "SW", "SIFT"]

    genome_results = pd.read_csv(args.genome, sep='\t')
    row = genome_results[genome_results["pop"] == args.pop.replace("_", " ")]
    assert len(row) > 0
    Ldn, dn, Lds, ds = row["Ldn"].values[0], row["dn"].values[0], row["Lds"].values[0], row["ds"].values[0]
    print(f'ratio Ldn/(Lds + Ldn)={Ldn / (Lds + Ldn)}')
    vcf_file = gzip.open(args.vcf, 'rt')
    list_non_syn = []
    max_daf_set = set()
    snp_dico = defaultdict(list)
    header = {}
    for vcf_line in vcf_file:
        if vcf_line[0] == '#':
            if vcf_line[1] != '#':
                line_strip = vcf_line.strip()
                if args.method == "SIFT":
                    line_strip = line_strip.replace("\t", "\tSIFT_")
                    line_strip += "\t#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"
                header = {k: i for i, k in enumerate(line_strip.split("\t"))}
            continue

        split_line = vcf_line.strip().split("\t")
        info = split_line[header["INFO"]]
        dico_info = {k: v for k, v in [s.split('=') for s in info.split(';') if '=' in s]}

        if args.method == "WS" and not ((dico_info["NUC_ANC"] in weak) and (dico_info["NUC_DER"] in strong)):
            continue

        if args.method == "SW" and not ((dico_info["NUC_ANC"] in strong) and (dico_info["NUC_DER"] in weak)):
            continue

        polarized = dico_info["POLARIZED"] == "True"
        if not polarized and args.method == "SIFT" and dico_info["SNP_TYPE"] == "NonSyn":
            continue

        sample_size = int(dico_info["SAMPLE_SIZE"])
        k = int(dico_info["COUNT_POLARIZED"])
        max_daf = min(sample_size, args.subsample)
        max_daf_set.add(max_daf)
        if sample_size > args.subsample:
            count = [np.random.hypergeometric(k, sample_size - k, args.subsample) for i in range(args.nbr_replicates)]
            count = [i if i != max_daf else 0 for i in count]
        else:
            count = [k]

        if np.all([i == 0 for i in count]):
            continue

        if dico_info["SNP_TYPE"] == "Syn":
            snp_dico["syn"].append(count)
        elif dico_info["SNP_TYPE"] == "NonSyn":
            if args.method in ["MutSel", "WS", "SW", "SIFT"]:
                S = float(dico_info["SEL_COEFF"])
            elif args.method == "Omega":
                S = float(dico_info["SITE_OMEGA_SEL_COEFF"])
            else:
                assert args.method == "Omega_0"
                S = float(dico_info["SITE_OMEGA_0_SEL_COEFF"])

            if args.method == "SIFT":
                sif_info = split_line[header["SIFT_INFO"]]
                sift_find = sif_info.find('Sift=0')
                if sift_find == -1:
                    continue
                sift_score = float(sif_info[sift_find:].split("|")[2])
                list_non_syn.append((sift_score, S, count))
            elif np.isfinite(S):
                list_non_syn.append((S, count))
    vcf_file.close()

    assert len(max_daf_set) == 1
    max_daf = max(max_daf_set)
    cat_snps = CategorySNP(args.method)
    if args.method == "SIFT":
        for sift_score, s, c in list_non_syn:
            assert (0 <= sift_score <= 1.0)
            snp_dico[cat_snps.selcoeff2cat(sift_score)].append(c)

        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
        x = [sift_score for sift_score, s, c in list_non_syn if abs(s) < 20]
        y = [s for sift_score, s, c in list_non_syn if abs(s) < 20]
        plt.scatter(x, y, alpha=0.4, s=5.0)
        plt.xlabel("SIFT score")
        plt.ylabel("S given by Mutation-Selection")
        plt.tight_layout()
        plt.savefig(args.output.replace('.SIFT.pdf', '.SIFT_vs_MutSel.pdf'), format="pdf")
        plt.clf()
        plt.close("all")
    else:
        for s, c in list_non_syn:
            snp_dico[cat_snps.selcoeff2cat(s)].append(c)

        fig, ax = plt.subplots(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
        n_lines = max_daf
        x = np.linspace(-10, 5, 101)
        ys = list()
        f_cutoff_list = np.linspace(1 / max_daf, 1.0, n_lines)[::-1]
        for f in f_cutoff_list:
            s_list = [s for s, c in list_non_syn if abs(s) < 10 and np.mean(c) / max_daf <= f]
            kde = sps.gaussian_kde(s_list)
            ys.append(kde.pdf(x))
        lc = multiline(np.array([x] * n_lines), np.array(ys), f_cutoff_list, ax=ax, cmap='magma', lw=1)
        colors = np.arange(n_lines)
        axcb = fig.colorbar(lc)
        axcb.set_label('Frequency cut-off')
        plt.xlabel("Scaled selection coefficient (S)")
        plt.ylabel("Density")
        plt.axvline(-1, color="grey", lw=1, ls='--')
        plt.axvline(1, color="grey", lw=1, ls='--')
        plt.axvline(0, color="black", lw=2, ls='--')
        plt.tight_layout()
        plt.savefig(args.output.replace('.pdf', '.histogram.pdf'), format="pdf")
        plt.clf()
        plt.close("all")

    daf_axis = range(1, max_daf)
    theta_dict = defaultdict(list)
    snp_sfs = {cat: daf_to_sfs(daf, max_daf) for cat, daf in snp_dico.items()}

    sfs_syn_mean = np.mean(snp_sfs["syn"], axis=0)
    sfs_nonsyn_mean = np.mean(daf_to_sfs([c for s, c in list_non_syn], max_daf), axis=0)
    out = args.output.replace('.pdf', '.all')
    write_sfs(sfs_syn_mean, sfs_nonsyn_mean, Ldn, dn, Lds, ds, max_daf, out, args.pop, "", div=False)
    write_dofe(sfs_syn_mean, sfs_nonsyn_mean, Ldn, dn, Lds, ds, max_daf, out, args.pop, "", True, Ldn + Lds)

    for cat in cat_snps:
        if cat == "syn":
            sfs_cat_mean = sfs_syn_mean / Lds
        else:
            sfs_cat_mean = np.mean(snp_sfs[cat], axis=0)
            out = args.output.replace('.pdf', f'.{cat}')
            catLdn = (Ldn * len(snp_dico[cat]) / len(list_non_syn))
            catdn = (dn * len(snp_dico[cat]) / len(list_non_syn))
            write_sfs(sfs_syn_mean, sfs_cat_mean, catLdn, catdn, Lds, ds, max_daf, out, args.pop, "", div=False)
            write_dofe(sfs_syn_mean, sfs_cat_mean, catLdn, catdn, Lds, ds, max_daf, out, args.pop, "", True, Ldn + Lds)
            sfs_cat_mean = sfs_cat_mean / catLdn

        theta_dict["category"].append(cat)
        for theta_method in sfs_weight:
            theta_dict[theta_method].append(theta(sfs_cat_mean[1:], max_daf, theta_method))

    normed_sfs_syn_mean = sfs_syn_mean / np.sum(sfs_syn_mean)
    for scaled in [True, False]:
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
        for cat in cat_snps:
            if len(snp_dico[cat]) == 0 or (scaled and cat == "syn"):
                continue

            sfs = snp_sfs[cat]
            if scaled:
                sfs = (sfs.T / np.sum(sfs, axis=1)).T
                sfs /= normed_sfs_syn_mean
            else:
                sfs = sfs / (Ldn * len(snp_dico[cat]) / len(list_non_syn))

            mean_sfs = np.mean(sfs, axis=0)[1:]
            std_sfs = np.std(sfs, axis=0)[1:]
            nb_snp = int(np.sum([1 for i in np.array(snp_dico[cat]).flatten() if i != 0]) / args.nbr_replicates)
            label = cat_snps.label(cat) + f" ({nb_snp} SNPs)"
            plt.scatter(daf_axis, mean_sfs, color=cat_snps.color(cat))
            plt.plot(daf_axis, mean_sfs, label=label, color=cat_snps.color(cat), linewidth=1.0)
            plt.fill_between(daf_axis, mean_sfs - std_sfs, mean_sfs + std_sfs, linewidth=1.0,
                             color=cat_snps.color(cat), alpha=0.2)

        plt.legend()
        plt.xlabel("Derived allele count")
        if scaled:
            plt.axhline(1.0, color="black")
            plt.ylabel('Frequency of SNPs (scaled by synonymous)')
        else:
            plt.ylabel("Frequency of SNPs")
            plt.yscale("log")
        plt.tight_layout()
        output = args.output.replace('.pdf', '.normalize.pdf') if scaled else args.output
        plt.savefig(output, format="pdf")
        plt.clf()
        plt.close("all")

    df = pd.DataFrame(theta_dict)
    df.to_csv(args.output.replace('.pdf', '.tsv'), sep="\t", index=False)
