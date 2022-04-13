import os
import argparse
import numpy as np
import pandas as pd
from functools import reduce
from collections import defaultdict
import statsmodels.api as sm
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from libraries import my_dpi, plt, polydfe_cat_dico, tex_f, sort_df, sp_sorted, format_pop, CategorySNP

cat_snps = CategorySNP("MutSel")
markers = ["o", "d", "s", '.']


def open_tsv(filepath):
    ddf = pd.read_csv(filepath, sep="\t")
    if os.path.basename(filepath) == "Theta.results.tsv":
        df_theta = ddf[(ddf["method"] == "MutSel") & (ddf["category"] == "syn")]
        df_theta = df_theta.dropna(axis='columns')
        df_theta = df_theta.drop(["method", "species", "category"], axis=1)
        df_dfe = ddf[(ddf["method"] == "MutSel") & (ddf["category"] != "syn")]
        m_list = []
        for cat_poly in list(polydfe_cat_dico) + [f"P-{cat}" for cat in cat_snps.non_syn_list]:
            if cat_poly in df_dfe:
                matrix = df_dfe.pivot(index="pop", columns="category", values=cat_poly).add_suffix(f'_{cat_poly}')
                m_list.append(matrix)
        df_pivot = pd.concat(m_list, axis=1)
        ddf = pd.merge(df_pivot, df_theta, on=["pop"])
    elif "bounds" in os.path.basename(filepath):
        ddf = ddf[ddf["method"] == "MutSel"]
        ddf = ddf.pivot(index="pop", columns="cat", values="sampled_fraction").add_suffix('_snps')
    return ddf


def discard_col(col, df):
    return (col not in df) or (df.dtypes[col] == np.object) or (not np.all(np.isfinite(df[col])))


def generate_xy_plot():
    xy_dico = defaultdict(list)
    dico_label = {'pop': "Population", "species": "Species", "watterson": "Watterson $\\theta_W$",
                  "proba": "$\\mathbb{P}$"}

    y_dico = {"tajima": "Tajima $\\theta_{\\pi}$ ",
              "flowPos": "$\\Psi_{+}$", "flowNeg": "$\\Psi_{-}$", "flowRatio": "$\\Psi_{+} / \\Psi_{-}$",
              "logFitness": "Mean log-fitness", "SMean": "$\\overline{S}$",
              "betaS_ratio": "Slope of $\\overline{\\beta}(\\overline{S})$ at $\\overline{S}=0$",
              "fay_wu": "Fay and Wu $\\theta_{H}$", "D_tajima": "Tajima's $D$", "H_fay_wu": "Fay and Wu's $H$"}
    y_dico.update({f'all_{cat_poly}': v for cat_poly, v in polydfe_cat_dico.items()})

    for cat in cat_snps.non_syn_list + ['weak', 'all']:
        s = ""
        if cat == "weak":
            s = " -1 < S < 1"
        elif cat != "all":
            s = cat_snps.label(cat).replace("$", "")

        dico_label[cat] = "$\\mathbb{P}" + f"[{s}]$"
        dico_label[f"{cat}_snps"] = "$\\mathbb{P}_{obs}" + f"[{s}]$"
        for cat_poly, beta_tex in polydfe_cat_dico.items():
            beta = beta_tex[beta_tex.find("[")+1:beta_tex.rfind("]")]
            s_given_beta_key = f'{cat_poly}_P-{cat}'
            dico_label[s_given_beta_key] = "$\\mathbb{P}" + f"[ {s} | {beta}]$"

        if cat != "all":
            s = f" | {s}"
        y_dico.update({f'{cat}_{cat_poly}': v.replace(']', f'{s}]') for cat_poly, v in polydfe_cat_dico.items()})
        for cat_beta in cat_snps.non_syn_list + ['all']:
            beta = cat_snps.label(cat_beta).replace('S', '\\beta').replace("$", "")
            beta_given_s_key = f'{cat}_P-{cat_beta}'
            y_dico[beta_given_s_key] = "$\\mathbb{P}" + f"[{beta} {s}]$"

    for y, y_label in y_dico.items():
        xy_dico["x"].append("watterson")
        xy_dico["y"].append(y)

    xy_list = [("pos_snps", "all_P-Ssup0"), ("pos_snps", "pos_P-Ssup0"), ("pos_snps", "pos")]
    xy_list += [("pos", "all_P-Ssup0"), ("pos", "pos_P-Ssup0")]
    xy_list += [("all_P-Seq0", "weak_P-Seq0"), ("all_P-Seq0", "pos-weak_P-Seq0"), ("all_P-Seq0", "neg-weak_P-Seq0")]
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
    df_list = [open_tsv(filepath) for filepath in sorted(args.tsv)]
    df = reduce(lambda left, right: pd.merge(left, right, how="inner", on=["pop"]), df_list)
    if "species" not in df:
        df["species"] = df["species_x"]
    df = df.iloc[df.apply(lambda r: sp_sorted(format_pop(r["pop"]), r["species"]), axis=1).argsort()]
    df["flowRatio"] = df["flowPos"] / df["flowNeg"]

    '''
    for cat_poly in polydfe_cat_dico:
        if f'weak_{cat_poly}' not in df:
            num = (df[f'neg-weak_{cat_poly}'] * df[f'neg-weak'] + df[f'pos-weak_{cat_poly}'] * df[f'pos-weak'])
            df[f'weak_{cat_poly}'] = num / (df[f'neg-weak'] + df[f'pos-weak'])
    '''

    assert (np.abs(np.sum([df[cat] for cat in cat_snps.non_syn_list], axis=0) - 1.0) < 1e-6).all()
    assert (np.abs(np.sum([df[f"{cat}_snps"] for cat in cat_snps.non_syn_list], axis=0) - 1.0) < 1e-6).all()
    for cat in cat_snps.non_syn_list + ["all"]:
        assert (np.abs(np.sum([df[f'{cat}_{cat_poly}'] for cat_poly in polydfe_cat_dico], axis=0) - 1.0) < 1e-12).all()

    for cat_poly in polydfe_cat_dico:
        given_cat = cat_poly.replace("P-", '')
        for cat in cat_snps.non_syn_list:
            assert f'{given_cat}_P-{cat}' not in df
            df[f'{given_cat}_P-{cat}'] = df[cat] * df[f'{cat}_P-{given_cat}'] / df[f'all_P-{given_cat}']
        print(np.sum([df[f'{given_cat}_P-{cat}'] for cat in cat_snps.non_syn_list], axis=0))

    species = {k: None for k in df["species"]}
    cm = get_cmap('tab10')
    color_dict = {sp: cm((i + 1) / len(species)) for i, sp in enumerate(species)}
    color_list = [color_dict[sp] for sp in df["species"]]

    out_dict = defaultdict(list)
    df_xy, dico_label = generate_xy_plot()
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
            label = f'${a:.2f}$ ($r^2={results.rsquared:.2g}$)'
            label = f'Slope of {label}' if len(df_group) == 1 else f'{dico_label[col_y]}: slope of {label}'
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

        plt.xlabel(dico_label[col_x])
        plt.ylabel(dico_label[y_group])
        if len(legend_elements) != 0:
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
    output_columns = ["pop", "species", "tajima", "pos_snps", 'all_P-Ssup0', 'pos_P-Ssup0']
    columns = [c for c in output_columns if c in df]
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
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output tsv file")
    parser.add_argument('--sample_list', required=False, type=str, dest="sample_list", help="Sample list file")
    main(parser.parse_args())
