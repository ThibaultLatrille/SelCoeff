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


param_dict = {"watterson": "Watterson $\\theta_W$", "tajima": "Tajima $\\theta_{\\pi}$ ",
              "flow_pos": "$\\Psi_{+}$", "flow_neg": "$\\Psi_{-}$", "flow_r": "$\\Psi_{+} / \\Psi_{-}$",
              "log_fitness": "Mean log-fitness", "a": "Slope of $S^{pop}/S$", "S_mean": "$S$",
              "fay_wu": "Fay and Wu $\\theta_{H}$", "D_tajima": "Tajima's $D$", "H_fay_wu": "Fay and Wu's $H$"}


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_list = [open_tsv(filepath) for filepath in sorted(args.tsv)]
    df = reduce(lambda left, right: pd.merge(left, right, how="inner", on=["pop"]), df_list)
    if "species" not in df:
        df["species"] = df["species_x"]
    df = df.iloc[df.apply(lambda r: sp_sorted(format_pop(r["pop"]), r["species"]), axis=1).argsort()]
    df["flow_r"] = df["flow_pos"] / df["flow_neg"]

    species = {k: None for k in df["species"]}
    cm = get_cmap('tab10')
    color_dict = {sp: cm((i + 1) / len(species)) for i, sp in enumerate(species)}
    color_list = [color_dict[sp] for sp in df["species"]]

    out_dict = defaultdict(list)
    for col_1 in ["watterson", "tajima", "fay_wu"]:
        if discard_col(col_1, df):
            continue
        for col_2 in ["flow_pos", "flow_neg", "flow_r", "S_mean", "log_fitness", "a", "D_tajima", "H_fay_wu"]:
            if discard_col(col_2, df):
                continue
            plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
            x = df[col_1]
            idf = np.linspace(0, max(x) * 1.05, 30)
            plt.xlim((0, max(x) * 1.05))
            y = df[col_2]

            results = sm.OLS(y, sm.add_constant(x)).fit()
            b, a = results.params[0:2]
            pred = a * idf + b

            plt.plot(idf, pred, '-', color='black', linewidth=2)
            plt.scatter(x, y, s=80.0, edgecolors="black", linewidths=0.5, color=color_list, zorder=100)
            plt.xlabel(param_dict[col_1] if col_1 in param_dict else col_1)
            plt.ylabel(param_dict[col_2] if col_2 in param_dict else col_2)
            legend_elements = [Line2D([0], [0], color='black',
                                      label=f'Slope of ${a:.2f}$ ($r^2={results.rsquared:.2g}$)')]
            legend_elements += [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[sp],
                                       label=f'{sp.replace("_", " ")}') for sp in species]
            plt.legend(handles=legend_elements)
            plt.tight_layout()
            plt.savefig(args.output.replace('.tsv', f'.{col_1}.{col_2}.scatter.pdf'), format="pdf")
            plt.clf()
            plt.close("all")
            out_dict['x'].append(col_1)
            out_dict['y'].append(col_2)
            out_dict['a'].append(a)
            out_dict['b'].append(b)
            out_dict['rsquared'].append(results.rsquared)

    df_out = pd.DataFrame(out_dict)
    df_out.to_csv(args.output, sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', required=False, type=str, nargs="+", dest="tsv", help="Input tsv file")
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output tsv file")
    parser.add_argument('--sample_list', required=False, type=str, dest="sample_list", help="Sample list file")
    main(parser.parse_args())
