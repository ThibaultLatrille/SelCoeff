import os
import argparse
import numpy as np
import pandas as pd
from functools import reduce
from itertools import product
from collections import defaultdict
import statsmodels.api as sm
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from libraries import my_dpi, plt, polydfe_cat_dico, tex_f, sort_df, sp_sorted, format_pop, CategorySNP

markers = ["o", "d", "s", '.']


def open_tsv(filepath, cat_snps, method):
    ddf = pd.read_csv(filepath, sep="\t")
    if os.path.basename(filepath) == "Theta.results.tsv":
        df_theta = ddf[(ddf["method"] == method) & (ddf["category"] == "syn")]
        df_theta = df_theta.dropna(axis='columns')
        df_theta = df_theta.drop(["method", "species", "category"], axis=1)
        df_dfe = ddf[(ddf["method"] == method) & (ddf["category"] != "syn")]
        cols = list(polydfe_cat_dico) + [f"P-{cat}" for cat in cat_snps.non_syn_list]
        cols += ['p_b', 'S_b', 'S_d', 'b', 'logL', 'S', 'S+', 'S-', 'fay_wu', 'tajima', 'watterson', 'div']
        cols += ["omega_Adiv", "omega_div"]
        m_list = []
        for col in cols:
            if col in df_dfe:
                matrix = df_dfe.pivot(index="pop", columns="category", values=col).add_suffix(f'_{col}')
                m_list.append(matrix)
        df_pivot = pd.concat(m_list, axis=1)
        ddf = pd.merge(df_pivot, df_theta, on=["pop"])
    elif "bounds" in os.path.basename(filepath):
        ddf = ddf[ddf["method"] == method]
        ddf = ddf.pivot(index="pop", columns="cat", values="sampled_fraction").add_suffix('_snps')
    return ddf


def discard_col(col, df):
    return (col not in df) or (df.dtypes[col] == object) or (not np.all(np.isfinite(df[col])))


def generate_xy_plot(cat_snps):
    pr = "\\mathbb{P}"
    xy_dico = defaultdict(list)
    dico_label = {'pop': "Population", "species": "Species", "watterson": "Watterson $\\theta_S$",
                  "proba": "$\\mathbb{P}$", "logL": "logL"}

    y_dico = {"tajima": "Tajima $\\theta_{\\pi}$ ",
              "flowPos": "$\\Psi_{+}$", "flowNeg": "$\\Psi_{-}$", "flowRatio": "$\\Psi_{+} / \\Psi_{-}$",
              "logFitness": "Mean log-fitness", "SMean": "$\\overline{S_0}$",
              "betaS_ratio": "Slope of $\\overline{S}(\\overline{S_0})$ at $\\overline{S_0}=0$",
              "fay_wu": "Fay and Wu $\\theta_{H}$", "D_tajima": "Tajima's $D$", "H_fay_wu": "Fay and Wu's $H$"}
    y_dico.update({f'all_{cat_poly}': v for cat_poly, v in polydfe_cat_dico.items()})
    y_dico.update({f'mut_sum_{cat_poly}': v for cat_poly, v in polydfe_cat_dico.items()})

    for cat_S0 in cat_snps.non_syn_list + ['all']:
        s0 = ""
        if cat_S0 != "all":
            s0 = cat_snps.label(cat_S0).replace("$", "")

        dico_label[cat_S0] = f"${pr}[{s0}]$"
        dico_label[f"{cat_S0}_snps"] = f"${pr}_{{poly}}[{s0}]$"
        dico_label[f"proba_{cat_S0}_div"] = f"${pr}_{{div}}[{s0}]$"

        for cat_S, s_tex in polydfe_cat_dico.items():
            s = s_tex[s_tex.find("[") + 1:s_tex.rfind("]")]
            dico_label[f'bayes_{cat_S}_P-{cat_S0}'] = f"${pr}[{s0}|{s}]$"
            dico_label[f'mut_sum_{cat_S}'] = f"${pr}[{s}]$"
            dico_label[f'frac_{cat_S0}_{cat_S}'] = f"$\\frac{{{pr}[{s0}]}}{{{pr}[{s}]}}$"
        dico_label[f'bayes_p_b_P-{cat_S0}'] = f"${pr}[{s0}|S>0]$"
        dico_label[f'frac_{cat_S0}_p_b'] = f"$\\frac{{{pr}[{s0}]}}{{{pr}[S>0]}}$"

        bracket_s0 = f"({s0})" if cat_S0 != "all" else s0
        dico_label[f'{cat_S0}_omega_div'] = f"$\\omega {bracket_s0}$"
        dico_label[f'{cat_S0}_omega_Adiv'] = f"$\\omega_{{A}} {bracket_s0}$"
        given_s0 = f" | {s0}" if cat_S0 != "all" else s0
        for p, param in [('S_b', "\\beta_b"), ('S_d', "\\beta_d"), ('b', "b"), ('p_b', "p_b"), ('logL', "LnL"),
                         ('S', "S_0"), ('S+', "S^+_0"), ('S-', "S^-_0"), ('pnpsT', "\\pi_T"),
                         ('pnpsF', "\\pi_F"), ('pnpsW', "\\pi_W"), ('div', "d")]:
            dico_label[f'{cat_S0}_{p}'] = f"${param} {given_s0}$"
        dico_label[f'{cat_S0}_p_b'] = f"${pr} [S > 0 {given_s0}]$"
        dico_label[f'sensitivity_{cat_S0}'] = f"TPR ${bracket_s0}$"
        dico_label[f'precision_{cat_S0}'] = f"PPV ${bracket_s0}$"
        y_dico.update(
            {f'{cat_S0}_{cat_poly}': v.replace(']', f'{given_s0}]') for cat_poly, v in polydfe_cat_dico.items()})
        for cat_beta in cat_snps.non_syn_list + ['all']:
            s = cat_snps.label(cat_beta).replace('S_0', 'S').replace("$", "")
            y_dico[f'bayes_{cat_S0}_P-{cat_beta}'] = f"{pr}[{s} {given_s0}]$"
    for y, y_label in y_dico.items():
        xy_dico["x"].append("watterson")
        xy_dico["y"].append(y)

    xy_list = [("pos_snps", "all_P-Spos"), ("pos_snps", "pos_P-Spos"), ("pos_snps", "pos")]
    xy_list += [("watterson", "all_p_b"), ("watterson", "ratio_neg_omega_div")]
    xy_list += [("watterson", "ratio_neg_P-Spos"), ("watterson", "bayes_P-Spos_P-pos")]
    xy_list += [("pos", "all_P-Spos"), ("pos", "pos_P-Spos")]
    xy_list += [("all_P-Sweak", "weak_P-Sweak"), ("all_P-Sweak", "pos-weak_P-Sweak"),
                ("all_P-Sweak", "neg-weak_P-Sweak")]
    xy_list += [("all_P-neg-weak", "neg-weak_P-neg-weak"), ("all_P-pos-weak", "pos-weak_P-pos-weak")]
    for x, y in xy_list:
        xy_dico["x"].append(x)
        xy_dico["y"].append(y)

    df_xy = pd.DataFrame(xy_dico)
    df_xy["group"] = df_xy["x"] + "." + df_xy["y"]
    df_xy["y_group"] = df_xy["y"]

    dico_label |= y_dico
    return df_xy, dico_label


def column_format(size):
    return "|" + "|".join(["l"] * 2 + ["r"] * (size - 2)) + "|"


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cat_snps = CategorySNP(args.method, bins=args.bins, windows=args.windows)
    df_list = [open_tsv(filepath, cat_snps, args.method) for filepath in sorted(args.tsv)]
    df = reduce(lambda left, right: pd.merge(left, right, how="inner", on=["pop"]), df_list)
    if "species" not in df:
        df["species"] = df["species_x"]
    df = df.iloc[df.apply(lambda r: sp_sorted(format_pop(r["pop"]), r["species"]), axis=1).argsort()]
    df["flowRatio"] = df["flowPos"] / df["flowNeg"]

    if args.bins <= 10:
        assert (np.abs(np.sum([df[cat] for cat in cat_snps.non_syn_list], axis=0) - 1.0) < 1e-6).all()
        assert (np.abs(np.sum([df[f"{cat}_snps"] for cat in cat_snps.non_syn_list], axis=0) - 1.0) < 1e-6).all()
        for cat_S0 in cat_snps.non_syn_list + ["all"]:
            s_list = np.abs(np.sum([df[f'{cat_S0}_{cat_poly}'] for cat_poly in polydfe_cat_dico], axis=0) - 1.0) < 1e-6
            assert s_list.all()

        for cat_S in polydfe_cat_dico:
            df[f'mut_sum_{cat_S}'] = np.sum([df[f"{cat}"] * df[f"{cat}_{cat_S}"] for cat in cat_snps.non_syn_list],
                                            axis=0)
            for cat_S0 in cat_snps.non_syn_list:
                assert f'{cat_S}_P-{cat_S0}' not in df
                df[f'frac_{cat_S0}_{cat_S}'] = df[cat_S0] / df[f'mut_sum_{cat_S}']
                df[f'bayes_{cat_S}_P-{cat_S0}'] = df[f'{cat_S0}_{cat_S}'] * df[f'frac_{cat_S0}_{cat_S}']
                # print(np.sum([df[f'{given_cat}_P-{cat}'] for cat in cat_snps.non_syn_list], axis=0))

                if cat_S == f"P-S{cat_S0}":
                    df[f'sensitivity_{cat_S0}'] = df[f'bayes_{cat_S}_P-{cat_S0}']
                    df[f'precision_{cat_S0}'] = df[f'{cat_S0}_{cat_S}']

    species = {k: None for k in df["species"]}
    cm = get_cmap('tab10')
    color_dict = {sp: cm((i + 1) / len(species)) for i, sp in enumerate(species)}
    color_list = [color_dict[sp] for sp in df["species"]]

    out_dict = defaultdict(list)
    df_xy, dico_label = generate_xy_plot(cat_snps)

    cat_list = ['all'] + cat_snps.non_syn_list
    for cat_S0 in cat_list:
        if f"{cat_S0}_tajima" in df and "tajima" in df:
            df[f"{cat_S0}_pnpsT"] = df[f"{cat_S0}_tajima"] / df["tajima"]
            df[f"{cat_S0}_pnpsW"] = df[f"{cat_S0}_watterson"] / df["watterson"]
            df[f"{cat_S0}_pnpsF"] = df[f"{cat_S0}_fay_wu"] / df["fay_wu"]

        if f"{cat_S0}_div" in df and "all_div" in df:
            df[f"proba_{cat_S0}_div"] = df[f"{cat_S0}_div"] / df["all_div"]

        if f"{cat_S0}_omega_div" in df and "all_omega_div" in df:
            df[f"ratio_{cat_S0}_omega_div"] = 100 * (df["all_omega_div"] - df[f"{cat_S0}_omega_div"]) / df[
                "all_omega_div"]
            dico_label[f"ratio_{cat_S0}_omega_div"] = f"R({dico_label['all_omega_div']})"
        df = df.copy()  # To obtain a dataframe not fragmented

    for (group, col_x, y_group), df_group in df_xy.groupby(["group", "x", "y_group"]):
        if discard_col(col_x, df):
            continue
        plt.figure(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
        x = df[col_x]
        min_x = 0 if col_x == "tajima" else min(x) * 0.95
        max_x = max(x) * 1.05
        idf = np.linspace(min_x, max_x, 30)
        plt.xlim((min_x, max_x))
        legend_elements = []
        for row_id, (_, row) in enumerate(df_group.iterrows()):
            col_y = row["y"]
            if discard_col(col_y, df):
                continue
            x, y = df[col_x], df[col_y]

            results = sm.OLS(y, sm.add_constant(x)).fit()
            b, a = results.params[0:2]
            pred = a * idf + b
            label = f'$r^2={results.rsquared:.2g}$' if len(
                df_group) == 1 else f'{dico_label[col_y]}: slope of ${a:.2f}$ ($r^2={results.rsquared:.2g}$)'
            marker = markers[row_id % len(markers)]
            legend_elements += [Line2D([0], [0], color='black', label=label)]
            plt.plot(idf, pred, '-', color='black', linewidth=2)
            plt.scatter(x, y, s=80.0, edgecolors="black", linewidths=0.5, marker=marker, color=color_list,
                        zorder=5, label=label)

            out_dict['x'].append(col_x)
            out_dict['y'].append(col_y)
            out_dict['a'].append(a)
            out_dict['b'].append(b)
            out_dict['rsquared'].append(results.rsquared)

        if len(legend_elements) != 0:
            plt.xlabel(dico_label[col_x], fontsize=14)
            plt.ylabel(dico_label[y_group], fontsize=14)
            legend_elements += [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[sp],
                                       label=f'{sp.replace("_", " ")}') for sp in species]
            plt.legend(handles=legend_elements)
            plt.tight_layout()
            plt.savefig(args.output.replace('.tsv', f'.{group}.scatter.pdf'), format="pdf")
        plt.clf()
        plt.close("all")

    df_out = pd.DataFrame(out_dict)
    df_out.to_csv(args.output.replace(".tsv", ".regression.tsv"), sep="\t", index=False)

    df = sort_df(df, args.sample_list)
    columns = [c for c in dico_label if c in df]
    sub_header = [dico_label[i] if i in dico_label else i for i in columns]
    df.to_csv(args.output, sep="\t", index=False, header=sub_header, columns=columns)

    o = open(args.output.replace(".tsv", ".tex"), 'w')
    o.write("\\section{Table} \n")
    o.write("\\begin{center}\n")

    cat_list = ['all'] + cat_snps.non_syn_list
    cols_suffix = [['logL'], ['S_b'], ['p_b'], ['S_d'], ['b'], ['S-'], ['S'], ['P-Spos'], ['P-Sweak'], ['P-Sneg'],
                   ['omega_Adiv']]

    cols = [
        ["tajima", 'pos', 'mut_sum_P-Spos', 'frac_pos_P-Spos', 'pos_P-Spos', 'bayes_P-Spos_P-pos'],
        ["tajima"] + [f'{prefix}_{cat}' for cat, prefix in product(cat_snps.non_syn_list, ['precision', 'sensitivity'])]
    ]
    if len(cat_snps.non_syn_list) == 2:
        cols.append(["tajima", 'proba_pos_div', 'all_omega_div', 'neg_omega_div', 'ratio_neg_omega_div'])
    else:
        cols.append(["tajima", 'proba_pos_div'] + [f'{cat}_omega_div' for cat in cat_list])

    cols.append(["tajima"] + [f'{prefix}_{cat}' for cat, prefix in product(polydfe_cat_dico, ['all', 'mut_sum'])])
    cols.extend([[f'{cat}_{suffix}' for suffix, cat in product(suffix_list, cat_list)] for suffix_list in cols_suffix])

    for c in cols:
        columns = [c for c in ["pop", "species"] + c if ((c in df) and not pd.isna(df[c]).all())]
        if len(columns) == 2:
            continue
        sub_header = [dico_label[i] if i in dico_label else i for i in columns]
        o.write(df.to_latex(index=False, escape=False, longtable=True, float_format=tex_f,
                            column_format=column_format(len(columns)), header=sub_header, columns=columns))
    o.write("\\end{center}\n")
    o.write("\\newpage\n")
    o.close()
    os.system(f"cp -f {args.tex_source} {os.path.dirname(args.output)}")
    tex_to_pdf = "pdflatex -synctex=1 -interaction=nonstopmode -output-directory={0} {0}/main-table.tex".format(
        os.path.dirname(args.output))
    os.system(tex_to_pdf)
    os.system(tex_to_pdf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', required=False, type=str, nargs="+", dest="tsv", help="Input tsv file")
    parser.add_argument('--tex_source', required=False, type=str, default="scripts/main-table.tex", dest="tex_source",
                        help="Main document source file")
    parser.add_argument('--method', required=False, type=str, dest="method", help="Sel coeff parameter")
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output tsv file")
    parser.add_argument('--sample_list', required=False, type=str, dest="sample_list", help="Sample list file")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    parser.add_argument('--windows', required=False, default=0, type=int, dest="windows", help="Number of windows")
    main(parser.parse_args())
