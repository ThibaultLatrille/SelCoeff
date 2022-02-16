import argparse
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from libraries_plot import plt, my_dpi, format_pop, sp_sorted, CategorySNP, annotate_heatmap, heatmap, cm, \
    shiftedColorMap


def open_tsv(filepath):
    ddf = pd.read_csv(filepath, sep="\t")
    ddf["species"], pop, ddf["method"] = os.path.basename(filepath).replace(".tsv", "").split(".")
    ddf["pop"] = format_pop(pop.replace("_", " "))
    return ddf


theta_dict = {"watterson": "$\\theta_W$ (Watterson)", "tajima": "$\\theta_{\\pi}$ (Tajima)",
              "fay_wu": "$\\theta_{H}$ (Fay and Wu)"}
d_dict = {"D_tajima": "Tajima's $D$", "H_fay_wu": "Fay and Wu's $H$"}
d_dict.update(theta_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--sfs', required=False, type=str, nargs="+", dest="tsv", help="Input pdf file")
    parser.add_argument('-o', '--output', required=False, type=str, dest="output", help="Output tex file")
    parser.add_argument('-l', '--sample_list', required=False, type=str, dest="sample_list", help="Sample list file")

    args = parser.parse_args()
    df_merge = pd.concat([open_tsv(filepath.replace(".pdf", ".tsv")) for filepath in args.tsv])
    df_merge["D_tajima"] = df_merge["tajima"] - df_merge["watterson"]
    df_merge["H_fay_wu"] = df_merge["tajima"] - df_merge["fay_wu"]
    df_merge.to_csv(args.output, sep="\t", index=False)
    pop2sp = {pop: sp for (pop, sp), d in df_merge.groupby(["pop", "species"])}

    theta = "tajima"
    for method, df in df_merge.groupby(["method"]):
        cat_snps = CategorySNP(method)
        for d, d_label in d_dict.items():
            _, ax = plt.subplots(figsize=(1920 / my_dpi, 880 / my_dpi), dpi=my_dpi)
            for cat in cat_snps.non_syn():
                x = df[df["category"] == "syn"][theta].values
                y = df[df["category"] == cat][d].values
                plt.scatter(x, y, color=cat_snps.color(cat))
                model = sm.OLS(y, sm.add_constant(x))
                r = model.fit()
                b, a = r.params[0:2]
                idf = np.linspace(min(x), max(x), 100)
                sign = '+' if float(b) > 0 else '-'
                label = f"{cat_snps.label(cat)}: $y={a:.2g}x {sign} {abs(b):.2g}$ ($r^2={r.rsquared:.2g})$"
                plt.plot(idf, a * idf + b, '-', color=cat_snps.color(cat), label=label)
            plt.xlabel(theta_dict[theta])
            plt.ylabel(f"{d_dict[d]}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{args.output.replace('.tsv', '')}/{method}.{d}.scatter.pdf", format="pdf")
            plt.clf()
            plt.close("all")

            matrix = df[df["category"] != "syn"].pivot(index="pop", columns="category", values=d)
            matrix = matrix.iloc[matrix.apply(lambda row: sp_sorted(row.name, pop2sp[row.name]), axis=1).argsort()]
            matrix = matrix.reindex(cat_snps.non_syn(), axis=1)
            matrix *= 10**4

            _, ax = plt.subplots(figsize=(1920 / my_dpi, 880 / my_dpi), dpi=my_dpi)
            cat_labels = [cat_snps.label(cat) for cat in cat_snps.non_syn()]
            start, end = np.nanmin(matrix), np.nanmax(matrix)
            RdBu = cm.get_cmap('RdBu_r')
            if np.sign(start) != np.sign(end):
                midpoint = - start / (np.nanmax(matrix) - start)
                RdBu = shiftedColorMap(RdBu, midpoint=midpoint, name='shifted')

            im, _ = heatmap(matrix.T, cat_labels, matrix.index, ax=ax, cmap=RdBu, cbarlabel=d_label)
            annotate_heatmap(im, valfmt=lambda p: "{0:.2f}".format(p), div=True, fontsize=5)
            plt.tight_layout()
            plt.savefig(f"{args.output.replace('.tsv', '')}/{method}.{d}.heatmap.pdf", format="pdf")
            plt.clf()
            plt.close("all")
