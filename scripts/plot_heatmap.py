import argparse
import seaborn as sns
from libraries import *


def open_tsv(filepath):
    ddf = pd.read_csv(filepath, sep="\t")
    ddf["species"], pop, ddf["method"] = os.path.basename(filepath).replace("-", ".").split(".")[:3]
    ddf["pop"] = format_pop(pop.replace("_", " "))
    return ddf


def plot_stack_param(list_pops, df_all, pop_colors, title, output):
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 780 / my_dpi), dpi=my_dpi)
    x_pos = range(len(list_pops))
    hatches_list = ['', '', '//']
    colors_list = [LIGHTYELLOW_RGB, "silver", "white"]
    edgecolors_list = ["black", "black", "black"]
    bottom = np.array([0.0] * len(list_pops))
    for p_i, param in enumerate(polydfe_cat_dico):
        y = df_all[param].values
        ax.bar(x_pos, y, bottom=bottom, edgecolor=edgecolors_list[p_i], color=colors_list[p_i], hatch=hatches_list[p_i])
        bottom += y

    if title == 'All':
        ax.set_title("Estimation for all SNPs:", loc='left')
    else:
        ax.set_title("Estimation for SNPs with " + title + ":", loc='left')

    ax.set_xlabel("Populations")
    ax.set_ylabel("Proportion estimated")
    ax.set_xticks(x_pos)
    ax.set_ylim((0, 1))
    ax.set_xticklabels(list_pops)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")
    for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), pop_colors):
        ticklabel.set_color(tickcolor)
    plt.tight_layout()
    plt.savefig(output)
    plt.close("all")


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_theta = pd.concat([open_tsv(filepath) for filepath in sorted(args.tsv_theta)])
    df_dfe = pd.concat([open_tsv(filepath) for filepath in sorted(args.tsv_dfe)])

    df_mut_rates = pd.read_csv(args.tsv_mut_rate, sep="\t")
    # group mutationr rates per species and compute mean
    df_mut_rates = df_mut_rates.groupby("Species").mean().reset_index()
    dico_mut_rates = {row["Species"]: row["MutationRatePerGeneration"] for _, row in df_mut_rates.iterrows()}

    df_merge = pd.merge(df_theta, df_dfe, how="outer", on=["pop", "species", "method", "category"])
    df_merge["D_tajima"] = df_merge["tajima"] - df_merge["watterson"]
    df_merge["H_fay_wu"] = df_merge["tajima"] - df_merge["fay_wu"]
    # Apply the mutation rate to the Watterson estimator
    df_merge["pop_size"] = df_merge.apply(lambda r: r["tajima"] / (4 * dico_mut_rates[r["species"]]), axis=1)
    d_dict = {"watterson": "Watterson $\\theta_W$", "pop_size": "Effective population size $N_e$"}
    d_dict.update(polydfe_cat_dico)

    df_merge = df_merge.iloc[df_merge.apply(lambda r: sp_sorted(format_pop(r["pop"]), r["species"]), axis=1).argsort()]
    df_merge.to_csv(args.output, sep="\t", index=False)
    pop2sp = {pop: sp for (pop, sp), d in df_merge.groupby(["pop", "species"])}
    pop2theta = {pop: d["pop_size"].values[0] for pop, d in df_merge[df_merge["category"] == "syn"].groupby("pop")}

    species = {k: None for k in df_merge["species"]}
    cmtab10 = colormaps['tab10']
    colors = {sp: cmtab10((t_i + 1) / len(species)) for t_i, sp in enumerate(species)}

    sample_dico = sample_list_dico(args.sample_list)
    df_merge = df_merge[df_merge["category"] != "syn"]
    for method, df in df_merge.groupby("method"):
        cat_snps = CategorySNP(method, bins=args.bins, windows=args.windows)
        if cat_snps.bins <= 10:
            for cat, dfc in df.groupby("category"):
                dfc = dfc.iloc[dfc.apply(lambda r: pop2theta[r["pop"]], axis=1).argsort()]
                list_pops = [sample_dico[pop] for pop in dfc["pop"].values]
                pop_colors = [colors[pop2sp[pop]] for pop in dfc["pop"].values]
                plot_stack_param(list_pops, dfc, pop_colors, cat_snps.label(cat),
                                 args.output.replace('results.tsv', f'{method}.{cat}.stacked.pdf'))

        df = df[df["category"] != "all"]
        cats_df = set(df["category"])
        cats = [cat for cat in cat_snps.non_syn_list if cat in cats_df]
        assert len(cats) == len(cats_df)
        for d, d_label in d_dict.items():
            matrix = df.pivot(index="pop", columns="category", values=d)
            matrix = matrix.iloc[matrix.apply(lambda row: sp_sorted(row.name, pop2sp[row.name]), axis=1).argsort()]
            sp_list = [pop2sp[pop] for pop in matrix.index]
            vlist = [v + 1 for v in range(len(sp_list) - 1) if sp_list[v] != sp_list[v + 1]]
            matrix = matrix.reindex(cats, axis=1)
            matrix = matrix.rename(columns={cat: cat_snps.label(cat) for cat in cats}, index=sample_dico)
            if abs(np.max(matrix.values)) < 1e-2:
                matrix *= 1e4
                d_label += ' ($\\times 10^4$)'
            _, ax = plt.subplots(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
            start, end = np.nanmin(matrix), np.nanmax(matrix)
            if d.startswith("P-"):
                start, end = 0.0, 1.0
            rd_bu = colormaps['viridis_r']
            if start != 0.0 and np.sign(start) != np.sign(end):
                midpoint = - start / (np.nanmax(matrix) - start)
                rd_bu = shiftedColorMap(rd_bu, midpoint=midpoint, name='shifted')

            cbar_ws = {'label': d_label}
            if args.bins < 10:
                cbar_ws.update(dict(shrink=0.3, fraction=0.1))
            ax = sns.heatmap(matrix.T, linewidths=0.05, linecolor="black", ax=ax, cmap=rd_bu,
                             cbar_kws=cbar_ws, square=(args.bins < 10), vmin=start, vmax=end)
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
    parser.add_argument('--tsv_theta', required=False, type=str, nargs="+", dest="tsv_theta", help="Input theta files")
    parser.add_argument('--tsv_dfe', required=False, type=str, nargs="+", dest="tsv_dfe", help="Input dfe files")
    parser.add_argument('--tsv_mut_rate', required=False, type=str, dest="tsv_mut_rate", help="Input mutation rate file")
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output tsv file")
    parser.add_argument('--sample_list', required=False, type=str, dest="sample_list", help="Sample list file")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    parser.add_argument('--windows', required=False, default=0, type=int, dest="windows", help="Number of windows")
    main(parser.parse_args())
