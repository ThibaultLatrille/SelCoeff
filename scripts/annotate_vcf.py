import os
import gzip
import argparse
import numpy as np
import pandas as pd
import scipy.stats as sps
from collections import defaultdict
from matplotlib.patches import Rectangle
from libraries import plt, CategorySNP, my_dpi, multiline, xlim_dico, rate_dico, BOUND


def open_sift(sift_file):
    output = defaultdict(dict)
    df = pd.read_csv(sift_file, sep="\t", dtype={"chr": 'string', "pos": int, "sift_score": float})
    for row_id, row in df.iterrows():
        output[row["chr"]][row["pos"]] = row["sift_score"]
    return output


def read_vcf(vcf, sift_file):
    dict_sift = open_sift(sift_file)
    vcf_file = gzip.open(vcf, 'rt')
    dico_snp = defaultdict(list)
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

        sample_size = int(dico_info["SAMPLE_SIZE"])
        k = int(dico_info["COUNT_POLARIZED"])
        if k == 0 or k == sample_size:
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
        dico_snp["freq"].append(k / sample_size)
        dico_snp["SIFT"].append(s_sift)
        dico_snp["MutSel"].append(s_mutsel)
        dico_snp["Omega"].append(omega)
    vcf_file.close()

    return dico_snp


def classify_snps(s_list, type_list, cat_snps):
    cat_list, intervals, dico_cat = list(), list(), defaultdict(list)

    if cat_snps.bins != 0:
        s_ns_list = [s for s, snp_type in zip(s_list, type_list) if snp_type == "NonSyn" and np.isfinite(s)]
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
            chunk = int((len(s_ns_list) - cat_snps.windows) / cat_snps.bins)
            start = 0
            for i in range(cat_snps.bins):
                end = min(start + cat_snps.windows, len(s_ns_list) - 1)
                assert start != end
                intervals.append(BOUND(f"w_{i + 1}", sorted_list[start], sorted_list[end]))
                start += chunk
    else:
        for cat in range(cat_snps.non_syn_list):
            bounds = cat_snps.dico[cat].bounds
            intervals.append(BOUND(cat, bounds[0], bounds[1]))

    cat_snps.add_intervals(intervals)
    for s, snp_type in zip(s_list, type_list):
        if snp_type == "Syn":
            cat_list.append("-syn-")
        elif np.isfinite(s):
            cats = cat_snps.rate2cats(s)
            if cat_snps.bins == 0 or (cat_snps.bins != 0 and cat_snps.windows == 0):
                if len(cats) != 1:
                    cats = [cats[0]]
                assert len(cats) == 1
            cat_list.append("-" + "-".join(cats) + "-")
            for cat in cats:
                dico_cat[cat].append(s)
        else:
            cat_list.append("None")

    return cat_list, intervals, [np.mean(dico_cat[b.cat]) for b in intervals], [len(dico_cat[b.cat]) for b in intervals]


def plot_sift(x_list, y_list, file):
    plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
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
                                range=(xmin, xmax))
    if cat_snps.bins == 0:
        n_cat = defaultdict(int)
        for i, b in enumerate(bins[1:]):
            cats = cat_snps.rate2cats(b)
            assert len(cats) == 1
            cat = cats[0]
            patches[i].set_facecolor(cat_snps.color(cat))
            n_cat[cat] += n[i]
        handles = [Rectangle((0, 0), 1, 1, color=c) for c in [cat_snps.color(cat) for cat in cat_snps.non_syn_list]]
        labels = [cat_snps.label(cat) + f" $({int(n_cat[cat])}~mutations)$" for cat in cat_snps.non_syn_list]
        plt.legend(handles, labels, loc="upper left")
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
    os.makedirs(os.path.dirname(args.output_tsv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_bounds), exist_ok=True)
    dico_snp = read_vcf(args.vcf, args.sift_file)
    dico_bounds = defaultdict(list)

    for method in ["MutSel", "Omega", "SIFT"]:
        cat_snps = CategorySNP(method, bins=args.bins, windows=args.windows)
        cat_list, bounds, mean_list, count_list = classify_snps(dico_snp[method], dico_snp["snp_type"], cat_snps)
        dico_snp["cat_" + method] = cat_list
        dico_bounds["method"].extend([method] * len(bounds))
        dico_bounds["cat"].extend([b.cat for b in bounds])
        dico_bounds["lower"].extend([b.lower for b in bounds])
        dico_bounds["upper"].extend([b.upper for b in bounds])
        dico_bounds["mean"].extend(mean_list)
        dico_bounds["count"].extend(count_list)

        plot_density(dico_snp[method], dico_snp["freq"], method,
                     args.output_tsv.replace('.tsv.gz', f'.{method}.density.pdf'))
        plot_histogram(dico_snp[method], cat_snps, method,
                       args.output_tsv.replace('.tsv.gz', f'.{method}.histogram.pdf'))

    plot_sift(dico_snp["SIFT"], dico_snp["MutSel"], args.output_tsv.replace('.tsv.gz', '.SIFT_vs_MutSel.pdf'))

    df = pd.DataFrame(dico_snp)
    df.to_csv(args.output_tsv, sep="\t", index=False)

    df_bounds = pd.DataFrame(dico_bounds)
    df_bounds.to_csv(args.output_bounds, sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vcf', required=False, type=str, dest="vcf", help="Input vcf file")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    parser.add_argument('--windows', required=False, default=0, type=int, dest="windows", help="Number of windows")
    parser.add_argument('--sift_file', required=False, type=str, default="", dest="sift_file",
                        help="The SIFT file path")
    parser.add_argument('--output_tsv', required=False, type=str, dest="output_tsv", help="Output tsv file")
    parser.add_argument('--output_bounds', required=False, type=str, dest="output_bounds", help="Output bounds file")
    main(parser.parse_args())
