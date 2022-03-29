import os
import argparse
import numpy as np
import pandas as pd
from functools import reduce
from collections import defaultdict
import statsmodels.api as sm
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from libraries import my_dpi, format_pop, plt, sp_sorted


def open_tsv(filepath):
    ddf = pd.read_csv(filepath, sep="\t")
    if os.path.basename(filepath) == "Theta.results.tsv":
        ddf = ddf[ddf["category"] == "syn"]
    return ddf


def discard_col(col, df):
    return (col not in df) or (df.dtypes[col] == np.object) or (not np.all(np.isfinite(df[col])))


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_list = [open_tsv(filepath) for filepath in sorted(args.tsv)]
    df_merged = reduce(lambda left, right: pd.merge(left, right, how="inner", on=["pop"]), df_list)
    df_merged = df_merged.iloc[
        df_merged.apply(lambda r: sp_sorted(format_pop(r["pop"]), r["species"]), axis=1).argsort()]

    species = {k: None for k in df_merged["species"]}
    cm = get_cmap('Set2')
    color_dict = {sp: cm(i / len(species)) for i, sp in enumerate(species)}
    color_list = [color_dict[sp] for sp in df_merged["species"]]

    out_dict = defaultdict(list)
    for col_1 in ["watterson"]:
        if discard_col(col_1, df_merged):
            continue
        for col_2 in ["flow_pos", "flow_neg", "S_mean", "log_fitness", "a", "b"]:
            if discard_col(col_2, df_merged):
                continue
            plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
            x = df_merged[col_1]
            idf = np.linspace(min(x), max(x), 30)
            plt.xlim(min(x), max(x), 30)
            y = df_merged[col_2]
            plt.scatter(x, y, s=25.0, color=color_list)
            results = sm.OLS(y, sm.add_constant(x)).fit()
            b, a = results.params[0:2]
            out_dict['x'].append(col_1)
            out_dict['y'].append(col_2)
            out_dict['a'].append(a)
            out_dict['b'].append(b)
            out_dict['rsquared'].append(results.rsquared)
            linear = a * idf + b
            plt.plot(idf, linear, '-', linewidth=2)
            plt.xlabel(col_1)
            plt.ylabel(col_2)
            legend_elements = [Line2D([0], [0], label=f'Slope of {a:.2g} ($r^2={results.rsquared:.2g}$)')]
            legend_elements += [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[sp],
                                       label=f'{sp.replace("_", " ")}') for sp in species]
            plt.legend(handles=legend_elements)
            plt.tight_layout()
            plt.savefig(args.output.replace('.tsv', f'.{col_1}.{col_2}.scatter.pdf'), format="pdf")
            plt.clf()
            plt.close("all")

    df_out = pd.DataFrame()
    df_out.to_csv(args.output, sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', required=False, type=str, nargs="+", dest="tsv", help="Input tsv file")
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output tsv file")
    parser.add_argument('--sample_list', required=False, type=str, dest="sample_list", help="Sample list file")
    main(parser.parse_args())
