import argparse
from matplotlib import cm
from libraries import *

theta_dict = {"watterson": "Watterson $\\theta_W$", "tajima": "Tajima $\\theta_{\\pi}$ ",
              "fay_wu": "Fay and Wu $\\theta_{H}$"}


def open_tsv(filepath):
    ddf = pd.read_csv(filepath, sep="\t")
    ddf["species"], pop, ddf["method"] = os.path.basename(filepath).replace(".tsv", "").split(".")[:3]
    ddf["pop"] = format_pop(pop.replace("_", " "))
    return ddf


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_merge = pd.concat([open_tsv(filepath) for filepath in sorted(args.tsv)])
    if ("tajima" in df_merge) and ("watterson" in df_merge) and ("fay_wu" in df_merge):
        df_merge["D_tajima"] = df_merge["tajima"] - df_merge["watterson"]
        df_merge["H_fay_wu"] = df_merge["tajima"] - df_merge["fay_wu"]
        d_dict = {"D_tajima": "Tajima's $D$", "H_fay_wu": "Fay and Wu's $H$"}
        d_dict.update(theta_dict)
    else:
        d_dict = polydfe_cat_dico
    df_merge.to_csv(args.output, sep="\t", index=False)
    pop2sp = {pop: sp for (pop, sp), d in df_merge.groupby(["pop", "species"])}

    for method, df in df_merge.groupby(["method"]):
        cat_snps = CategorySNP(method, bins=args.bins)
        for d, d_label in d_dict.items():
            matrix = df[df["category"] != "syn"].pivot(index="pop", columns="category", values=d)
            matrix = matrix.iloc[matrix.apply(lambda row: sp_sorted(row.name, pop2sp[row.name]), axis=1).argsort()]
            matrix = matrix.reindex(cat_snps.non_syn_list, axis=1)
            if abs(np.max(matrix.values)) < 1e-2:
                matrix *= 1e4
                d_label += ' ($\\times 10^4$)'
            _, ax = plt.subplots(figsize=(1920 / my_dpi, 880 / my_dpi), dpi=my_dpi)
            cat_labels = [cat_snps.label(cat) for cat in cat_snps.non_syn_list]
            start, end = np.nanmin(matrix), np.nanmax(matrix)
            rd_bu = cm.get_cmap('RdBu_r')
            if np.sign(start) != np.sign(end):
                midpoint = - start / (np.nanmax(matrix) - start)
                rd_bu = shiftedColorMap(rd_bu, midpoint=midpoint, name='shifted')

            im, _ = heatmap(matrix.T, cat_labels, matrix.index, ax=ax, cmap=rd_bu, cbarlabel=d_label)
            annotate_heatmap(im, valfmt=lambda p: "{0:.2f}".format(p), div=True, fontsize=5)
            plt.tight_layout()
            plt.savefig(args.output.replace('results.tsv', f'{method}.{d}.heatmap.pdf'), format="pdf")
            plt.clf()
            plt.close("all")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', required=False, type=str, nargs="+", dest="tsv", help="Input tsv file")
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output tsv file")
    parser.add_argument('--sample_list', required=False, type=str, dest="sample_list", help="Sample list file")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    main(parser.parse_args())
