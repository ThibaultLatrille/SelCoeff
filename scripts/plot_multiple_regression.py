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

    species = {k: None for k in df_merge["species"]}
    cm = get_cmap('Set2')
    colors = {sp: cm(i / len(species)) for i, sp in enumerate(species)}
    for method, df_method in df_merge.groupby(["method"]):
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
        if method == "MutSel":
            df_filter = df_method[np.abs(df_method["S_phy"]) < 0.5]
        else:
            df_filter = df_method[np.abs(df_method["S_pop"]) < 2.0]
        idf = np.linspace(min(df_filter["S_phy"]), max(df_filter["S_phy"]), 30)
        plt.xlim((min(df_filter["S_phy"]), max(df_filter["S_phy"])))
        dico_out = defaultdict(list)
        for (pop, sp), df in df_filter.groupby(["pop", "species"]):
            dico_out["pop"].append(pop)
            dico_out["species"].append(sp)
            c = colors[sp]
            x = df["S_phy"]
            y = df["S_pop"]
            plt.scatter(x, y, s=1.0, color=c)
            results = sm.OLS(y, sm.add_constant(x)).fit()
            b, a = results.params[0:2]
            linear = a * idf + b
            dico_out["a"].append(a)
            dico_out["b"].append(b)
            dico_out["rsquared"].append(results.rsquared)
            plt.plot(idf, linear, '-', linewidth=0.5, color=c)

        df_out = pd.DataFrame(dico_out)
        df_out = df_out.iloc[df_out.apply(lambda r: sp_sorted(format_pop(r["pop"]), r["species"]), axis=1).argsort()]
        df_out.to_csv(args.output, sep="\t", index=False)
        sp_slopes = {
            sp: (f"a=[{min(df['a']):.2f}, {max(df['a']):.2f}]" if len(df) > 1 else f"a={df['a'].values[0]:.2f}") for
            sp, df in df_out.groupby(["species"])}
        legend_elements = [Line2D([0], [0], color=colors[sp],
                                  label=f'{sp.replace("_", " ")}: {sp_slopes[sp]}') for sp in species]
        plt.legend(handles=legend_elements)
        plt.xlabel("Phylogenetic scale")
        plt.ylabel("Population scale")
        plt.tight_layout()
        plt.savefig(args.output.replace('results.tsv', f'{method}.SelCoeff.scatter.pdf'), format="pdf")
        plt.clf()
        plt.close("all")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', required=False, type=str, nargs="+", dest="tsv", help="Input tsv file")
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output tsv file")
    parser.add_argument('--sample_list', required=False, type=str, dest="sample_list", help="Sample list file")
    main(parser.parse_args())
