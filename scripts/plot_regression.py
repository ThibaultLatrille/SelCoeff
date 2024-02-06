import os
import argparse
import numpy as np
import pandas as pd
from functools import reduce
from collections import defaultdict
from itertools import product
from matplotlib import colormaps
from libraries import my_dpi, plt, polydfe_cat_dico, tex_f, sort_df, sp_sorted, format_pop, CategorySNP, row_color


def open_tsv(filepath, cat_snps, method):
    ddf = pd.read_csv(filepath, sep="\t")
    if os.path.basename(filepath) == "Theta.results.tsv":
        df_theta = ddf[(ddf["method"] == method) & (ddf["category"] == "syn")]
        df_theta = df_theta.dropna(axis='columns')
        df_theta = df_theta.drop(["method", "species", "category"], axis=1)
        df_dfe = ddf[(ddf["method"] == method) & (ddf["category"] != "syn")]
        cols = list(polydfe_cat_dico) + [f"P-{cat}" for cat in cat_snps.non_syn_list]
        cols += ['pop_size', 'div', "omega_div", "dN", "dS"]
        m_list = []
        for col in cols:
            if col in df_dfe:
                matrix = df_dfe.pivot(index="pop", columns="category", values=col).add_suffix(f'_{col}')
                m_list.append(matrix)
        df_pivot = pd.concat(m_list, axis=1)
        ddf = pd.merge(df_pivot, df_theta, on=["pop"])
    elif "bounds" in os.path.dirname(filepath).split("/")[-1]:
        ddf = ddf[ddf["method"] == method]
        ddf = ddf.pivot(index="pop", columns="cat", values="sampled_fraction").add_suffix('_snps')
    return ddf


def discard_col(col, df):
    return (col not in df) or (df.dtypes[col] == object) or (not np.all(np.isfinite(df[col])))


def generate_dico_labels(cat_snps: CategorySNP):
    dico_label = {'pop': "Population", "species": "Species", "pop_size": "Effective population size $N_e$ (x1000)",
                  "all_dS": "Total $dS$"}

    pr = "\\mathbb{P}"
    y_dico = {f'all_{cat_S}': v for cat_S, v in polydfe_cat_dico.items()}
    y_dico.update({f'mut_sum_{cat_S}': v for cat_S, v in polydfe_cat_dico.items()})

    for cat_S0 in cat_snps.non_syn_list + ['all']:
        s0 = ""
        if cat_S0 != "all":
            s0 = cat_snps.label(cat_S0).replace("$", "")
        given_s0 = f" | {s0}" if cat_S0 != "all" else s0
        bracket_s0 = f"({s0})" if cat_S0 != "all" else s0

        dico_label[cat_S0] = f"${pr}[{s0}]$"
        dico_label[f'{cat_S0}_omega_div'] = f"$\\omega {bracket_s0}$"
        dico_label[f"proba_{cat_S0}_div"] = f"${pr}_{{div}}[{s0}]$"
        dico_label[f'recall_{cat_S0}'] = f"TPR ${bracket_s0}$"
        dico_label[f'precision_{cat_S0}'] = f"PPV ${bracket_s0}$"

        for cat_S, s_tex in polydfe_cat_dico.items():
            s = s_tex[s_tex.find("[") + 1:s_tex.rfind("]")]
            y_dico[f'{cat_S0}_{cat_S}'] = f"${pr}[{s} {given_s0}]$"
            y_dico[f'frac_{cat_S0}_{cat_S}'] = f"$\\frac{{{pr}[{s0}]}}{{{pr}[{s}]}}$"
            y_dico[f'bayes_{cat_S}_P-{cat_S0}'] = f"${pr}[{s0}|{s}]$"
    dico_label |= y_dico

    return y_dico.keys(), dico_label


def column_format(size):
    return "|" + "|".join(["l"] * 2 + ["r"] * (size - 2)) + "|"


def plot_scatter(df, col_x, col_y, dico_label, pgls_dict, color_dict, output, xscale="linear"):
    plt.figure(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)

    plt.xlabel(dico_label[col_x], fontsize=14)
    plt.ylabel(dico_label[col_y], fontsize=14)
    if col_y in pgls_dict['pop_size']:
        plt.title(pgls_dict[col_x][col_y], fontsize=14)
    df_sub = df[["species", col_x, col_y]].copy()
    # group by species and compute mean
    for sp, df_sp in df_sub.groupby("species"):
        x_mean = df_sp[col_x].mean()
        y_mean = df_sp[col_y].mean()
        plt.scatter(x_mean, y_mean, s=25.0, edgecolors="black", linewidths=1.5, marker="s", color=color_dict[sp],
                    zorder=5, alpha=0.5)
        # Draw line between mean and each point
        for _, row in df_sp.iterrows():
            plt.plot([x_mean, row[col_x]], [y_mean, row[col_y]], '-', linewidth=0.5, color=color_dict[sp],
                     alpha=0.5, zorder=0)

        plt.scatter(df_sp[col_x], df_sp[col_y], s=60.0, color=color_dict[sp], edgecolors="dimgrey",
                    linewidths=0.25, zorder=10, marker="o", label=f'{sp.replace("_", " ")}')

    plt.xlim((min(df_sub[col_x]) * 0.95, max(df_sub[col_x]) * 1.05))
    if xscale == "log":
        plt.xscale("log")
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(output, format="pdf")
    plt.clf()
    plt.close("all")


def collapse_table_by_species(df):
    # Get the min and the max across populations for each species
    dico_df = defaultdict(list)
    df = df.drop("pop", axis=1)
    gb = df.groupby("species")
    for sp, df_sp in gb:
        dico_df["species"].append(sp)
        for column in df_sp.columns:
            if column == "species":
                continue
            min_val = tex_f(df_sp[column].min())
            if len(df_sp) == 1:
                dico_df[column].append(min_val)
            else:
                max_val = tex_f(df_sp[column].max())
                dico_df[column].append(f"[{min_val}, {max_val}]")
    return pd.DataFrame(dico_df)


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cat_snps = CategorySNP(args.method, bins=args.bins, windows=args.windows)
    y_list, dico_label = generate_dico_labels(cat_snps)

    df_list = [open_tsv(filepath, cat_snps, args.method) for filepath in sorted(args.tsv)]
    df = reduce(lambda left, right: pd.merge(left, right, how="inner", on=["pop"]), df_list)
    if "species" not in df:
        df["species"] = df["species_x"]
    df = df.iloc[df.apply(lambda r: sp_sorted(format_pop(r["pop"]), r["species"]), axis=1).argsort()]

    if args.bins <= 10:
        assert (np.abs(np.sum([df[cat] for cat in cat_snps.non_syn_list], axis=0) - 1.0) < 1e-6).all()
        for cat_S0 in cat_snps.non_syn_list + ["all"]:
            s_list = np.abs(np.sum([df[f'{cat_S0}_{cat_S}'] for cat_S in polydfe_cat_dico], axis=0) - 1.0) < 1e-6
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
                    df[f'recall_{cat_S0}'] = df[f'bayes_{cat_S}_P-{cat_S0}']
                    df[f'precision_{cat_S0}'] = df[f'{cat_S0}_{cat_S}']

    cat_list = ['all'] + cat_snps.non_syn_list
    for cat_S0 in cat_list:
        if f"{cat_S0}_div" in df and "all_div" in df:
            df[f"proba_{cat_S0}_div"] = df[f"{cat_S0}_div"] / df["all_div"]

        if f"{cat_S0}_omega_div" in df and "all_omega_div" in df:
            df[f"ratio_{cat_S0}_omega_div"] = 100 * (df["all_omega_div"] - df[f"{cat_S0}_omega_div"]) / df[
                "all_omega_div"]
            dico_label[f"ratio_{cat_S0}_omega_div"] = f"R({dico_label['all_omega_div']})"
        df = df.copy()  # To obtain a dataframe not fragmented

    species = {k: None for k in df["species"]}

    columns = [c for c in dico_label if c in df]
    df.to_csv(args.output, sep="\t", index=False, columns=columns)
    df["pop_size"] = df["pop_size"] // 1000
    # Run the following command to get the pgls results:
    # Rscript scripts/pgls.R --tsv <output> --output <output>.pgls.tsv
    stdout = args.pgls_output.replace(".tsv", ".log")
    cmd = f"Rscript {args.pgls_script} --input_tsv {args.output} --input_tree {args.pgls_tree} --output_tsv {args.pgls_output} > {stdout}"
    print(cmd)
    os.system(cmd)

    pgls_dict = defaultdict(dict)
    if os.path.exists(args.pgls_output) and open(args.pgls_output).read().strip() != "":
        pgls_df = pd.read_csv(args.pgls_output, sep="\t")
        for _, row in pgls_df.iterrows():
            if row["regression"] != "slope":
                continue
            y_label = row["y_label"]
            assert y_label in df.columns
            corr = "Positive correlation" if row["Estimate"] > 0 else "Negative correlation"
            pgls_dict[row["x_label"]][row["y_label"]] = f"{corr} ($r^2$={row['r2']:.2f}, p={tex_f(row['Pr(>|t|)'])})"

    cm = colormaps['tab10']
    color_dict = {sp: cm((i + 1) / len(species)) for i, sp in enumerate(species)}
    for col_y in y_list:
        if discard_col(col_y, df):
            continue
        output = args.output.replace('.tsv', f'.pop_size.{col_y}.scatter.pdf')
        plot_scatter(df, "pop_size", col_y, dico_label, pgls_dict, color_dict, output)

    output = args.output.replace('.tsv', f'.distance.recall_pos.scatter.pdf')
    plot_scatter(df, "all_dS", "recall_pos", dico_label, pgls_dict, color_dict, output, xscale="log")

    # Print number of populations for which the recall is between in 0.2 and 0.4
    print(df["recall_pos"])
    low_bound = 0.15
    high_bound = 0.45
    total = np.sum((df[f'recall_pos'] > low_bound) & (df[f'recall_pos'] < high_bound))
    print(f"{total} out of {len(df)} with {low_bound} < recall < {high_bound}")

    df = sort_df(df, args.sample_list)
    df = row_color(df)

    o = open(args.output.replace(".tsv", ".tex"), 'w')
    o.write("\\section{Table} \n")
    o.write("\\begin{center}\n")

    cols = [
        ["pop_size", 'pos', 'mut_sum_P-Spos', 'frac_pos_P-Spos', 'pos_P-Spos', 'bayes_P-Spos_P-pos'],
        ["pop_size"] + [f'{p}_{c}' for c, p in product(cat_snps.non_syn_list, ['precision', 'recall'])],
        cat_snps.non_syn_list + [f'proba_{c}_div' for c in cat_snps.non_syn_list]
    ]

    if len(cat_snps.non_syn_list) == 2:
        cols.append(['all_omega_div', 'neg_omega_div', 'ratio_neg_omega_div'])
    else:
        cols.append(['all_omega_div'] + [f'{cat}_omega_div' for cat in cat_snps.non_syn_list])

    for c in cols:
        columns = [c for c in ["pop", "species"] + c if ((c in df) and not pd.isna(df[c]).all())]
        if len(columns) == 2:
            continue
        data = df[columns]
        if "all_omega_div" in c:
            data = collapse_table_by_species(data)
        data = data.rename(columns={i: (dico_label[i] if i in dico_label else i) for i in columns})
        o.write(data.to_latex(index=False, escape=False, longtable=True, float_format=tex_f,
                              column_format=column_format(len(columns))))
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
    parser.add_argument('--pgls_script', required=True, type=str, dest="pgls_script", help="PGLS script")
    parser.add_argument('--pgls_tree', required=True, type=str, dest="pgls_tree", help="PGLS tree")
    parser.add_argument('--pgls_output', required=True, type=str, dest="pgls_output", help="PGLS output")
    parser.add_argument('--tex_source', required=False, type=str, default="scripts/main-table.tex", dest="tex_source",
                        help="Main document source file")
    parser.add_argument('--method', required=False, type=str, dest="method", help="Sel coeff parameter")
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output tsv file")
    parser.add_argument('--sample_list', required=False, type=str, dest="sample_list", help="Sample list file")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    parser.add_argument('--windows', required=False, default=0, type=int, dest="windows", help="Number of windows")
    main(parser.parse_args())
