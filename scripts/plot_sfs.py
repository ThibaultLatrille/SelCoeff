import argparse
import gzip
import numpy as np
import pandas as pd
import scipy.stats as sps
from collections import defaultdict, namedtuple
from matplotlib.patches import Rectangle
from libraries import plt, CategorySNP, my_dpi, multiline, sfs_weight, theta, write_sfs, write_dofe

weak = ["A", "T"]
strong = ["G", "C"]
SNP = namedtuple('SNP', ['id', 's_sift', 's_mutsel', 's_omega', 's_omega_0', 'derived_count', 'frequency'])


def daf_to_sfs(daf_list, min_n):
    array_daf = np.array(daf_list, dtype=np.int64).T
    return np.array([np.bincount(d, minlength=min_n) for d in array_daf])


def read_vcf(vcf, method, subsample, nbr_replicates):
    vcf_file = gzip.open(vcf, 'rt')
    list_non_syn, list_syn = [], []
    max_daf_set = set()
    header = {}
    for vcf_line in vcf_file:
        if vcf_line[0] == '#':
            if vcf_line[1] != '#':
                line_strip = vcf_line.strip()
                if method == "SIFT":
                    line_strip = line_strip.replace("\t", "\tSIFT_")
                    line_strip += "\t#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"
                header = {k: i for i, k in enumerate(line_strip.split("\t"))}
            continue

        split_line = vcf_line.strip().split("\t")
        snp_id = str(split_line[header["ID"]])
        info = str(split_line[header["INFO"]])
        dico_info = {k: v for k, v in [s.split('=') for s in info.split(';') if '=' in s]}

        if method == "WS" and not ((dico_info["NUC_ANC"] in weak) and (dico_info["NUC_DER"] in strong)):
            continue

        if method == "SW" and not ((dico_info["NUC_ANC"] in strong) and (dico_info["NUC_DER"] in weak)):
            continue

        polarized = dico_info["POLARIZED"] == "True"
        if not polarized and method == "SIFT":
            continue

        sample_size = int(dico_info["SAMPLE_SIZE"])
        k = int(dico_info["COUNT_POLARIZED"])
        if k == 0:
            continue
        max_daf = min(sample_size, subsample)
        max_daf_set.add(max_daf)
        if sample_size > subsample:
            count = [np.random.hypergeometric(k, sample_size - k, subsample) for _ in range(nbr_replicates)]
            count = [i if i != max_daf else 0 for i in count]
        else:
            count = [k]

        if dico_info["SNP_TYPE"] == "Syn":
            list_syn.append(count)
        elif dico_info["SNP_TYPE"] == "NonSyn":
            s_sift = np.nan
            if "SIFT_INFO" in header:
                sif_info = str(split_line[header["SIFT_INFO"]])
                sift_find = sif_info.find('Sift=0')
                if sift_find != -1:
                    s_sift = float(sif_info[sift_find:].split("|")[2])

            s_mutsel = float(dico_info["SEL_COEFF"])
            s_omega = float(dico_info["SITE_OMEGA_SEL_COEFF"])
            s_omega_0 = float(dico_info["SITE_OMEGA_0_SEL_COEFF"])
            list_non_syn.append(SNP(snp_id, s_sift, s_mutsel, s_omega, s_omega_0, count, k / sample_size))
    vcf_file.close()

    assert len(max_daf_set) == 1
    max_daf = max(max_daf_set)
    return list_non_syn, list_syn, max_daf


def classify_snps(list_non_syn, s_method, cat_snps):
    snp_dico = defaultdict(list)
    if cat_snps.bins:
        sorted_non_syn = sorted(list_non_syn, key=lambda x: getattr(x, s_method))
        for i, snp in enumerate(sorted_non_syn):
            snp_dico[f"bin{1 + int((i * cat_snps.nbr_non_syn()) / len(sorted_non_syn))}"].append(snp)
    else:
        for snp in list_non_syn:
            snp_dico[cat_snps.selcoeff2cat(getattr(snp, s_method))].append(snp)
    return snp_dico


def plot_sift(list_non_syn, file):
    plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    x = [snp.s_sift for snp in list_non_syn if snp.s_mutsel < 20]
    y = [snp.s_mutsel for snp in list_non_syn if snp.s_mutsel < 20]
    plt.scatter(x, y, alpha=0.4, s=5.0)
    plt.xlabel("SIFT score")
    plt.ylabel("S given by Mutation-Selection")
    plt.tight_layout()
    plt.savefig(file, format="pdf")
    plt.clf()
    plt.close("all")


def plot_density(list_non_syn, s_method, file):
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
    f_min = np.min([snp.frequency for snp in list_non_syn])
    assert f_min != 0
    n_lines = min(int(1 / f_min), 500)
    s_list = [getattr(snp, s_method) for snp in list_non_syn if np.isfinite(getattr(snp, s_method))]
    xmin, xmax = (-10, 5) if min(s_list) < 0.0 else (0, 1)
    x = np.linspace(xmin, xmax, 101)
    ys = list()
    f_cutoff_list = np.linspace(f_min, 1.0, n_lines)[::-1]
    for f_cut in f_cutoff_list:
        s_list = [getattr(snp, s_method) for snp in list_non_syn if
                  snp.frequency <= f_cut and np.isfinite(getattr(snp, s_method))]
        kde = sps.gaussian_kde(s_list)
        ys.append(kde.pdf(x))
    lc = multiline(np.array([x] * n_lines), np.array(ys), f_cutoff_list, ax=ax, cmap='magma', lw=1)
    axcb = fig.colorbar(lc)
    axcb.set_label('Frequency cut-off')
    plt.xlabel("Scaled selection coefficient (S)")
    plt.ylabel("Density")
    if xmin < -1.0 and xmax > 1.0:
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        plt.axvline(-1, color="grey", lw=1, ls='--')
        plt.axvline(1, color="grey", lw=1, ls='--')
        plt.axvline(0, color="black", lw=2, ls='--')
    plt.xlim((xmin, xmax))
    plt.tight_layout()
    plt.savefig(file, format="pdf")
    plt.clf()
    plt.close("all")


def plot_histogram(list_non_syn, cat_snps, s_method, file):
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
    s_list = [getattr(snp, s_method) for snp in list_non_syn if np.isfinite(getattr(snp, s_method))]
    xmin, xmax = (-10, 5) if min(s_list) < 0.0 else (0, 1)
    n, bins, patches = plt.hist(s_list, bins=np.linspace(xmin, xmax, 61), range=(xmin, xmax))
    n_cat = defaultdict(int)
    for i, b in enumerate(bins[1:]):
        cat = cat_snps.selcoeff2cat(b)
        patches[i].set_facecolor(cat_snps.color(cat))
        n_cat[cat] += n[i]
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in [cat_snps.color(cat) for cat in cat_snps.non_syn()]]
    labels = [cat_snps.label(cat) + f" $({int(n_cat[cat])}~mutations)$" for cat in cat_snps.non_syn()]
    plt.legend(handles, labels, loc="upper left")
    plt.xlabel("Scaled selection coefficient (S)")
    plt.ylabel("Density")
    for x in cat_snps.inner_bound:
        plt.axvline(x, color="grey", lw=1, ls='--')
    if xmin < -1.0 and xmax > 1.0:
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.axvline(0, color="black", lw=2)
    plt.xlim((xmin, xmax))
    plt.tight_layout()
    plt.savefig(file, format="pdf")
    plt.clf()
    plt.close("all")
    return n_cat


def plot_sfs(cat_snps, snp_sfs, max_daf, daf_axis, cat_dico_count, output, scaled):
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    for cat in cat_snps:
        if scaled:
            snp_sfs[cat] *= np.array([i for i in range(max_daf)])

        mean_sfs = np.mean(snp_sfs[cat], axis=0)[1:]
        std_sfs = np.std(snp_sfs[cat], axis=0)[1:]
        label = cat_snps.label(cat) + f" $({int(cat_dico_count[cat])}~mutations)$"
        plt.scatter(daf_axis, mean_sfs, color=cat_snps.color(cat))
        plt.plot(daf_axis, mean_sfs, label=label, color=cat_snps.color(cat), linewidth=1.0)
        plt.fill_between(daf_axis, mean_sfs - std_sfs, mean_sfs + std_sfs, linewidth=1.0,
                         color=cat_snps.color(cat), alpha=0.2)
    if max_daf < 32:
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.xlabel("Derived allele count")
    if scaled:
        plt.ylabel('Proportion of mutations (scaled)')
    else:
        plt.ylabel("Proportion of mutations")
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, format="pdf")
    plt.clf()
    plt.close("all")


def plot_venn(list_non_syn, method, cat_snps, snps_classified, file):
    from venn import venn
    cat_snps_mutsel = CategorySNP("MutSel", bins=cat_snps.bins)
    snps_classified_mutsel = classify_snps(list_non_syn, "s_mutsel", cat_snps_mutsel)

    n_i, n_j = cat_snps_mutsel.nbr_non_syn(), cat_snps.nbr_non_syn()
    fig, axs = plt.subplots(n_i, n_j, figsize=(1280 * n_i / my_dpi, 1280 * n_j / my_dpi), dpi=my_dpi)
    for cat_i, cat_mutsel in enumerate(cat_snps_mutsel.non_syn()):
        for cat_j, cat in enumerate(cat_snps.non_syn()):
            dico_venn = {f"MutSel {cat_mutsel}": set([snp.id for snp in snps_classified_mutsel[cat_mutsel]]),
                         f"{method} {cat}": set([snp.id for snp in snps_classified[cat]])}
            venn(dico_venn, ax=axs[cat_i, cat_j])
    plt.tight_layout()
    plt.savefig(file, format="pdf")


def main(args):
    assert args.method in ["MutSel", "Omega", "Omega_0", "WS", "SW", "SIFT"]
    s_method = "s_mutsel" if args.method in ["SW", "WS"] else "s_" + args.method.lower()

    genome_results = pd.read_csv(args.genome, sep='\t')
    row = genome_results[genome_results["pop"] == args.pop.replace("_", " ")]
    assert len(row) > 0
    ldn, dn, lds, ds = row["Ldn"].values[0], row["dn"].values[0], row["Lds"].values[0], row["ds"].values[0]
    print(f'ratio Ldn/(Lds + Ldn)={ldn / (lds + ldn)}')

    cat_snps = CategorySNP(args.method, bins=args.bins)
    opp_results = pd.read_csv(args.opportunities, sep='\t')
    row_opp = opp_results[opp_results["species"] == args.species]
    opp_dico = {cat: row_opp[cat].values[0] for cat in cat_snps.non_syn()}

    list_non_syn, list_syn, max_daf = read_vcf(args.vcf, args.method, args.subsample, args.nbr_replicates)
    print(f'ratio pn/(ps + pn)={len(list_non_syn) / (len(list_syn) + len(list_syn))}')

    snps_classified = classify_snps(list_non_syn, s_method, cat_snps)

    plot_density(list_non_syn, s_method, args.output.replace('.pdf', '.distribution.pdf'))

    cat_dico_count = plot_histogram(list_non_syn, cat_snps, s_method, args.output.replace('.pdf', '.histogram.pdf'))
    cat_dico_count["syn"] = len(list_syn)

    if args.method != "MutSel":
        plot_venn(list_non_syn, args.method, cat_snps, snps_classified, args.output.replace('.pdf', '.venn.pdf'))
    if args.method == "SIFT":
        plot_sift(list_non_syn, args.output.replace('.SIFT.pdf', '.SIFT_vs_MutSel.pdf'))

    snps_daf = {cat: [snp.derived_count for snp in snp_list] for cat, snp_list in snps_classified.items()}
    snps_daf["syn"] = list_syn
    snp_sfs = {cat: daf_to_sfs(daf, max_daf) for cat, daf in snps_daf.items()}
    snp_sfs_mean = {cat: np.mean(sfs, axis=0) for cat, sfs in snp_sfs.items()}

    sfs_nonsyn_mean = np.mean(daf_to_sfs([snp.derived_count for snp in list_non_syn], max_daf), axis=0)
    out = args.output.replace('.pdf', '.all')
    write_sfs(snp_sfs_mean["syn"], sfs_nonsyn_mean, ldn, dn, lds, ds, max_daf, out, args.pop, "", div=False)
    write_dofe(snp_sfs_mean["syn"], sfs_nonsyn_mean, ldn, dn, lds, ds, max_daf, out, args.pop, "", True, ldn + lds)

    daf_axis = range(1, max_daf)
    theta_dict = defaultdict(list)
    for cat in cat_snps:
        if cat == "syn":
            ld_cat = lds
        else:
            out = args.output.replace('.pdf', f'.{cat}')
            ld_cat = ldn * opp_dico[cat]
            write_sfs(snp_sfs_mean["syn"], snp_sfs_mean[cat], ld_cat, dn * opp_dico[cat], lds, ds, max_daf,
                      out, args.pop, "", div=False)
            write_dofe(snp_sfs_mean["syn"], snp_sfs_mean[cat], ld_cat, dn * opp_dico[cat], lds, ds, max_daf,
                       out, args.pop, "", True, ld_cat + lds)

        theta_dict["category"].append(cat)
        for theta_method in sfs_weight:
            theta_dict[theta_method].append(theta(snp_sfs_mean[cat][1:] / ld_cat, max_daf, theta_method))
        snp_sfs[cat] = snp_sfs[cat] / ld_cat

    df = pd.DataFrame(theta_dict)
    df.to_csv(args.output.replace('.pdf', '.tsv'), sep="\t", index=False)

    for scaled in [False, True]:
        output = args.output.replace('.pdf', '.normalize.pdf') if scaled else args.output
        plot_sfs(cat_snps, snp_sfs, max_daf, daf_axis, cat_dico_count, output, scaled)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vcf', required=False, type=str, dest="vcf", help="Input vcf file")
    parser.add_argument('--method', required=False, type=str, dest="method", help="Sel coeff parameter")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output pdf file")
    parser.add_argument('--genome_results', required=False, type=str, dest="genome",
                        help="Input genome results tsv file")
    parser.add_argument('--opportunities', required=False, type=str, dest="opportunities",
                        help="Input opportunities results tsv file")
    parser.add_argument('--pop', required=False, type=str, dest="pop", help="Focal population")
    parser.add_argument('--species', required=False, type=str, dest="species", help="Focal species")
    parser.add_argument('--subsample', required=False, type=int, default=16, dest="subsample")
    parser.add_argument('--nbr_replicates', required=False, type=int, default=1, dest="nbr_replicates")
    main(parser.parse_args())
