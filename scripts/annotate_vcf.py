import argparse
import scipy.stats as sps
from matplotlib.patches import Rectangle
from libraries import *


def open_sift(sift_file):
    output = defaultdict(dict)
    df = pd.read_csv(sift_file, sep="\t", dtype={"chr": 'string', "pos": int, "sift_score": float})
    for row_id, row in df.iterrows():
        output[row["chr"]][row["pos"]] = row["sift_score"]
    return output


def read_vcf(vcf, sift_file, mask_grouped, subsample, anc_proba, mask_CpG, seqs):
    dict_sift = open_sift(sift_file)
    vcf_file = gzip.open(vcf, 'rt')
    dico_snp = defaultdict(list)
    discarded, filtered = 0, 0
    header = {}
    for vcf_line in vcf_file:
        if vcf_line[0] == '#':
            if vcf_line[1] != '#':
                line_strip = vcf_line.strip()
                header = {k: i for i, k in enumerate(line_strip.split("\t"))}
            continue
        assert len(header) > 0

        split_line = vcf_line.strip().split("\t")
        snp_chr = str(split_line[header["#CHROM"]])
        snp_pos = int(split_line[header["POS"]])
        info = str(split_line[header["INFO"]])
        dico_info = {k: v for k, v in [s.split('=') for s in info.split(';') if '=' in s]}
        ensg = dico_info["ENSG"]
        c_site = int(dico_info["ENSG_POS"]) // 3

        if ensg in mask_grouped and c_site in mask_grouped[ensg]:
            discarded += 1
            continue

        if float(dico_info["ANC_PROBA"]) < anc_proba:
            filtered += 1
            continue

        sample_size = int(dico_info["SAMPLE_SIZE"])
        k = int(dico_info["COUNT_POLARIZED"])
        if k == 0 or k == sample_size:
            continue

        if mask_CpG:
            der_nuc = dico_info["NUC_DER"]
            anc_nuc = dico_info["NUC_ANC"]
            pos = int(dico_info["ENSG_POS"])
            seq = seqs[ensg]
            if pos == 0 or pos == len(seq) - 1:
                continue
            anc_context = seq[pos - 1:pos + 2]
            der_context = anc_context[0] + der_nuc + anc_context[-1]
            assert anc_context[1] == anc_nuc
            assert der_context[1] == der_nuc
            assert len(anc_context) == 3
            assert len(der_context) == 3
            if "CG" in anc_context or "CG" in der_context:
                discarded += 1
                continue

        s_sift, s_mutsel, omega = np.nan, np.nan, np.nan
        if dico_info["SNP_TYPE"] == "NonSyn":
            if dict_sift is not None:
                s_sift = dict_sift[snp_chr][snp_pos]
            s_mutsel = float(dico_info["SEL_COEFF"])
            omega = float(dico_info["SITE_OMEGA"])

        dico_snp["snp_type"].append(dico_info["SNP_TYPE"])
        dico_snp["sample_size"].append(sample_size)
        dico_snp["count"].append(k)
        max_daf = min(sample_size, subsample)
        sampled_k = np.random.hypergeometric(k, sample_size - k, max_daf)
        dico_snp["sampled"].append(sampled_k != 0 and sampled_k != max_daf)
        dico_snp["freq"].append(k / sample_size)
        dico_snp["SIFT"].append(s_sift)
        dico_snp["MutSel"].append(s_mutsel)
        dico_snp["Omega"].append(omega)
    vcf_file.close()

    return dico_snp, discarded, filtered


def classify_snps(dico_snp, method, cat_snps):
    s_list = dico_snp[method]
    type_list = dico_snp["snp_type"]
    sampled = dico_snp["sampled"]
    cat_list, intervals, dico_cat = list(), list(), defaultdict(list)

    if cat_snps.bins > 10:
        s_ns_list = [s for s, snp_type, sample in zip(s_list, type_list, sampled) if
                     sample and snp_type == "NonSyn" and np.isfinite(s)]
        if cat_snps.windows == 0:
            array_split = np.array_split(sorted(s_ns_list), cat_snps.bins)
            assert len(array_split) == cat_snps.bins
            for i, current_bin in enumerate(array_split):
                lower = current_bin[0]
                upper = current_bin[-1] if i == (len(array_split) - 1) else array_split[i + 1][0]
                if lower == upper:
                    continue
                intervals.append(BOUND(f"b_{i + 1}", lower, upper))
        else:
            sorted_list = sorted(s_ns_list)
            assert cat_snps.windows < len(s_ns_list)
            dico_bounds = {"MutSel": [-0.5, 0.5], "Omega": [0.2, 2.0], "SIFT": [0.2, 1.0]}
            start, end = np.searchsorted(sorted_list, dico_bounds[method])
            start = start - cat_snps.windows // 2
            end = end - cat_snps.windows // 2
            assert start < end
            linspace = np.linspace(start, end, cat_snps.bins, dtype=int)

            for i, start_chunk in enumerate(linspace):
                end_chunk = min(start_chunk + cat_snps.windows, len(s_ns_list) - 1)
                assert start_chunk != end_chunk
                intervals.append(BOUND(f"w_{i + 1}", sorted_list[start_chunk], sorted_list[end_chunk]))
        cat_snps.add_intervals(intervals)
    else:
        for cat in cat_snps.non_syn_list:
            intervals.append(BOUND(cat, cat_snps.dico[cat].lower, cat_snps.dico[cat].upper))

    dico_sampled = defaultdict(float)
    for s, snp_type, sample in zip(s_list, type_list, sampled):
        if snp_type == "Syn":
            cat_list.append("|syn|")
        elif np.isfinite(s):
            cats = cat_snps.rate2cats(s)
            if cat_snps.windows == 0:
                if len(cats) != 1:
                    cats = [cats[0]]
                assert len(cats) == 1
            cat_list.append("|" + "|".join(cats) + "|")
            for cat in cats:
                dico_cat[cat].append(s)

            if sample:
                dico_sampled['tot'] += 1
                for cat in cats:
                    dico_sampled[cat] += 1
        else:
            cat_list.append("None")

    mean_list = [np.mean(dico_cat[b.cat]) for b in intervals]
    count_list = [len(dico_cat[b.cat]) for b in intervals]
    prop_sampled_list = [dico_sampled[b.cat] / dico_sampled['tot'] for b in intervals]
    return cat_list, intervals, mean_list, count_list, prop_sampled_list


def plot_sift(x_list, y_list, file):
    plt.figure(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
    x = [x for x, y in zip(x_list, y_list) if np.isfinite(x) and np.isfinite(y) and abs(x) < 20]
    y = [y for x, y in zip(x_list, y_list) if np.isfinite(x) and np.isfinite(y) and abs(x) < 20]
    plt.scatter(x, y, alpha=0.4, s=5.0)
    plt.xlabel(rate_dico["SIFT"])
    plt.ylabel(rate_dico["MutSel"])
    plt.tight_layout()
    plt.savefig(file, format="pdf")
    plt.clf()
    plt.close("all")


def plot_density(score_list, list_freqs, method, file):
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
    f_min = np.min(list_freqs)
    assert f_min != 0
    n_lines = min(int(1 / f_min), 500)
    xmin, xmax = xlim_dico[method][0], xlim_dico[method][1]
    x = np.linspace(xmin, xmax, 101)
    ys = list()
    f_cutoff_list = np.linspace(f_min, 1.0, n_lines)[::-1]
    for f_cut in f_cutoff_list:
        cut_list = [s for s, f in zip(score_list, list_freqs) if f <= f_cut and np.isfinite(s)]
        kde = sps.gaussian_kde(cut_list)
        ys.append(kde.pdf(x))
    lc = multiline(np.array([x] * n_lines), np.array(ys), f_cutoff_list, ax=ax, cmap='magma', lw=1)
    axcb = fig.colorbar(lc)
    axcb.set_label('Frequency cut-off')
    plt.xlabel(rate_dico[method])
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


def plot_histogram(score_list, cat_snps, method, file):
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
    xmin, xmax = xlim_dico[method][0], xlim_dico[method][1]
    n, bins, patches = plt.hist([s for s in score_list if np.isfinite(s)], bins=np.linspace(xmin, xmax, 61),
                                range=(xmin, xmax), edgecolor="black", linewidth=1.0)
    total_n = sum(n)
    if cat_snps.bins <= 10:
        n_cat = defaultdict(float)
        for i, b in enumerate(bins[1:]):
            cats = cat_snps.rate2cats(b)
            assert len(cats) >= 1
            cat = cats[0]
            patches[i].set_facecolor(cat_snps.color(cat))
            n_cat[cat] += n[i] / total_n
        handles = [Rectangle((0, 0), 1, 1, color=c) for c in [cat_snps.color(cat) for cat in cat_snps.non_syn_list]]
        labels = [cat_snps.label(cat) + f" ({n_cat[cat] * 100:.2f}% of total)" for cat in cat_snps.non_syn_list]
        plt.legend(handles, labels)
    plt.xlabel(rate_dico[method])
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


def main(args):
    assert 0 <= args.anc_proba <= 1
    os.makedirs(os.path.dirname(args.output_tsv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_bounds), exist_ok=True)

    mask_grouped = merge_mask_list(args.mask)
    seqs = open_fasta(args.fasta_pop)
    dico_snp, discarded, filtered = read_vcf(args.vcf, args.sift_file, mask_grouped, args.subsample,
                                             args.anc_proba, args.mask_CpG, seqs)
    total = len(dico_snp["snp_type"]) + discarded + filtered
    print(f'{discarded * 100 / total:.2f}% of SNPs are discarded because their are masked.')
    print(f'{filtered * 100 / total:.2f}% of SNPs are discarded because they are filtered.')
    print(f'{len(dico_snp["snp_type"])} SNPs are kept ({len(dico_snp["snp_type"]) * 100 / total:.2f}%).')
    dico_bounds = defaultdict(list)

    for method in ["MutSel", "Omega", "SIFT"]:
        cat_snps = CategorySNP(method, bins=args.bins, windows=args.windows)
        cat_list, bounds, mean_list, count_list, prop_list = classify_snps(dico_snp, method, cat_snps)
        dico_snp["cat_" + method] = cat_list
        dico_bounds["method"].extend([method] * len(bounds))
        dico_bounds["cat"].extend([b.cat for b in bounds])
        dico_bounds["lower"].extend([b.lower for b in bounds])
        dico_bounds["upper"].extend([b.upper for b in bounds])
        dico_bounds["mean"].extend(mean_list)
        dico_bounds["count"].extend(count_list)
        dico_bounds["sampled_fraction"].extend(prop_list)

        plot_density(dico_snp[method], dico_snp["freq"], method,
                     args.output_tsv.replace('.tsv.gz', f'.{method}.density.pdf'))
        plot_histogram(dico_snp[method], cat_snps, method,
                       args.output_tsv.replace('.tsv.gz', f'.{method}.histogram.full.pdf'))
        sampled = [s for s, sample in zip(dico_snp[method], dico_snp["sampled"]) if sample]
        plot_histogram(sampled, cat_snps, method, args.output_tsv.replace('.tsv.gz', f'.{method}.histogram.pdf'))

    plot_sift(dico_snp["SIFT"], dico_snp["MutSel"], args.output_tsv.replace('.tsv.gz', '.SIFT_vs_MutSel.pdf'))

    df = pd.DataFrame(dico_snp)
    df.to_csv(args.output_tsv, sep="\t", index=False)

    df_bounds = pd.DataFrame(dico_bounds)
    df_bounds.to_csv(args.output_bounds, sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vcf', required=False, type=str, dest="vcf", help="Input vcf file")
    parser.add_argument('--fasta_pop', required=True, type=str, dest="fasta_pop", help="The fasta path")
    parser.add_argument('--anc_proba', required=True, type=float, dest="anc_proba", default=0.5,
                        help="Mask the substitutions with reconstruction probability lower than this threshold")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    parser.add_argument('--windows', required=False, default=0, type=int, dest="windows", help="Number of windows")
    parser.add_argument('--sift_file', required=False, type=str, default="", dest="sift_file", help="The SIFT file")
    parser.add_argument('--mask', required=False, default="", nargs="+", type=str, dest="mask",
                        help="List of input mask file path")
    parser.add_argument('--mask_CpG', required=False, default=False, action="store_true", dest="mask_CpG",
                        help="Mask CpG opportunities")
    parser.add_argument('--output_tsv', required=False, type=str, dest="output_tsv", help="Output tsv file")
    parser.add_argument('--output_bounds', required=False, type=str, dest="output_bounds", help="Output bounds file")
    parser.add_argument('--subsample', required=False, type=int, default=16, dest="subsample")
    main(parser.parse_args())
