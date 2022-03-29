import argparse
from matplotlib import cm
import seaborn as sns
from libraries import *

theta_dict = {"watterson": "Watterson $\\theta_W$"}


# theta_dict = {"watterson": "Watterson $\\theta_W$", "tajima": "Tajima $\\theta_{\\pi}$ ",
#               "fay_wu": "Fay and Wu $\\theta_{H}$", "D_tajima": "Tajima's $D$", "H_fay_wu": "Fay and Wu's $H$"}


def open_tsv(filepath):
    ddf = pd.read_csv(filepath, sep="\t")
    ddf["species"], pop, ddf["method"] = os.path.basename(filepath).replace("-", ".").split(".")[:3]
    ddf["pop"] = format_pop(pop.replace("_", " "))
    return ddf


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_merge = pd.concat([open_tsv(filepath) for filepath in sorted(args.tsv)])
    if ("tajima" in df_merge) and ("watterson" in df_merge) and ("fay_wu" in df_merge):
        df_merge["D_tajima"] = df_merge["tajima"] - df_merge["watterson"]
        df_merge["H_fay_wu"] = df_merge["tajima"] - df_merge["fay_wu"]
        d_dict = theta_dict
    else:
        d_dict = polydfe_cat_dico
    df_merge.to_csv(args.output, sep="\t", index=False)
    pop2sp = {pop: sp for (pop, sp), d in df_merge.groupby(["pop", "species"])}

    for method, df in df_merge.groupby(["method"]):
        cat_snps = CategorySNP(method, bins=args.bins, windows=args.windows)
        for d, d_label in d_dict.items():
            df = df[(df["category"] != "all") & (df["category"] != "syn")]
            cats_df = set(df["category"])
            cats = [cat for cat in cat_snps.non_syn_list if cat in cats_df]
            assert len(cats) == len(cats_df)
            matrix = df.pivot(index="pop", columns="category", values=d)
            matrix = matrix.iloc[matrix.apply(lambda row: sp_sorted(row.name, pop2sp[row.name]), axis=1).argsort()]
            sp_list = [pop2sp[pop] for pop in matrix.index]
            vlist = [i + 1 for i in range(len(sp_list) - 1) if sp_list[i] != sp_list[i + 1]]
            matrix = matrix.reindex(cats, axis=1)
            matrix = matrix.rename(columns={cat: cat_snps.label(cat) for cat in cats})
            if abs(np.max(matrix.values)) < 1e-2:
                matrix *= 1e4
                d_label += ' ($\\times 10^4$)'
            _, ax = plt.subplots(figsize=(1920 / my_dpi, 880 / my_dpi), dpi=my_dpi)
            start, end = np.nanmin(matrix), np.nanmax(matrix)
            rd_bu = cm.get_cmap('RdBu_r')
            if start != 0.0 and np.sign(start) != np.sign(end):
                midpoint = - start / (np.nanmax(matrix) - start)
                rd_bu = shiftedColorMap(rd_bu, midpoint=midpoint, name='shifted')

            cbar_ws = {'label': d_label}
            if args.bins < 10:
                cbar_ws.update(dict(shrink=0.3, fraction=0.1))
            ax = sns.heatmap(matrix.T, linewidths=0.05, linecolor="black", ax=ax, cmap=rd_bu,
                             cbar_kws=cbar_ws, square=(args.bins < 10))
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.vlines(vlist, color="black", linewidths=2.0, *ax.get_ylim())
            ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
            plt.setp(ax.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")
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
    parser.add_argument('--windows', required=False, default=0, type=int, dest="windows", help="Number of windows")
    main(parser.parse_args())
