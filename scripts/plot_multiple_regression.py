import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import statsmodels.api as sm
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from libraries import my_dpi, format_pop, plt, sp_sorted


def open_tsv(filepath):
    ddf = pd.read_csv(filepath, sep="\t")
    ddf["species"], pop, ddf["method"] = os.path.basename(filepath).replace("-", ".").split(".")[:3]
    ddf["pop"] = format_pop(pop.replace("_", " "))
    return ddf


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_merge = pd.concat([open_tsv(filepath.replace(".tsv", ".scatter.tsv")) for filepath in sorted(args.tsv)])
    df_merge = df_merge.iloc[df_merge.apply(lambda r: sp_sorted(format_pop(r["pop"]), r["species"]), axis=1).argsort()]
    sp2pop = {sp: list(set(df["pop"])) for sp, df in df_merge.groupby(["species"], sort=False)}
    df_merge = df_merge.iloc[::-1]
    merge_out = []
    cm = get_cmap('Set2')
    colors = {sp: cm((i + 1) / len(sp2pop)) for i, sp in enumerate(sp2pop)}
    for method, df_filter in df_merge.groupby(["method"]):
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
        plt.xlim((-0.5, 0.5))
        idf = np.linspace(min(df_filter["S_phy"]), max(df_filter["S_phy"]), 30)
        dico_out = defaultdict(list)
        for (sp, pop), df in df_filter.groupby(["species", "pop"], sort=False):
            zorder = 0
            if len(sp2pop[sp]) <= 2:
                zorder += 150
            df = df.iloc[df.apply(lambda r: r["S_phy"], axis=1).argsort()]
            c = colors[sp]
            x = df["S_phy"]
            y = df["S_pop"]

            coeffs = np.polyfit(x, y, 2)
            ffit = np.poly1d(coeffs)
            pred = [ffit(i) for i in idf]
            plt.plot(idf, pred, '-', linewidth=2.0, color="silver", zorder=zorder)
            plt.plot(idf, pred, '-', linewidth=2.0, color=c, alpha=0.8, zorder=zorder)
            plt.plot(x, y, '-', linewidth=0.25, color="dimgrey", alpha=0.1, zorder=zorder + 50)
            plt.plot(x, y, '-', linewidth=0.25, color=c, alpha=0.3, zorder=zorder + 50)
            plt.scatter(x, y, s=8.0, color=c, edgecolors="dimgrey", alpha=0.5, linewidths=0.05, zorder=zorder + 100)
            dico_out["pop"].append(pop)
            dico_out["species"].append(sp)
            dico_out["a"].append(ffit.deriv()(0))

        df_out = pd.DataFrame(dico_out)
        df_out = df_out.iloc[df_out.apply(lambda r: sp_sorted(format_pop(r["pop"]), r["species"]), axis=1).argsort()]
        df_out["method"] = method
        merge_out.append(df_out)

        sp_slopes = {
            sp: (f"[{min(df['a']):.2f}, {max(df['a']):.2f}]" if len(df) > 1 else f"{df['a'].values[0]:.2f}") for
            sp, df in df_out.groupby(["species"])}
        legend_elements = [Line2D([0], [0], color=colors[sp],
                                  label=f'{sp.replace("_", " ")}: f\'(0)={sp_slopes[sp]}') for sp in sp2pop]
        plt.legend(handles=legend_elements)
        plt.xlabel("$S$ at the phylogenetic scale (Mutation-selection)")
        plt.ylabel("$S^{pop}$ at the population scale (polyDFE)")
        plt.tight_layout()
        plt.savefig(args.output.replace('results.tsv', f'{method}.SelCoeff.scatter.pdf'), format="pdf")
        plt.clf()
        plt.close("all")
    pd.concat(merge_out).to_csv(args.output, sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', required=False, type=str, nargs="+", dest="tsv", help="Input tsv file")
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output tsv file")
    parser.add_argument('--sample_list', required=False, type=str, dest="sample_list", help="Sample list file")
    main(parser.parse_args())
