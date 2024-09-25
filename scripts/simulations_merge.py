import argparse
from libraries import *

label_dico = {
    "recall_neg": "$\\mathbb{P} [ \\mathcal{D}_0 | \\mathcal{D} ]$",
    "recall_weak": "$\\mathbb{P} [ \\mathcal{N}_0 | \\mathcal{N} ]$",
    "recall_pos": "$\\mathbb{P} [ \\mathcal{B}_0 | \\mathcal{B} ]$",
}


def plot_boxplot(x, y, y_list, col, path_output):
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
    pops = [k for k, v in sorted(x.items(), key=lambda item: item[1])]

    pop_range = range(1, len(pops) + 1)
    labels = [f"Set {s}" for s in range(1, len(pops) + 1)]
    ax.bar(pop_range, height=[x[pop] for pop in pops], capsize=12, width=0.6, tick_label=labels,
           color="lightgrey", edgecolor="black", label="Ground truth (per set)")
    x_flat = np.array([[pop_range[i_pop]] * len(y_list[pop]) for i_pop, pop in enumerate(pops)]).flatten()
    y_flat = np.array([y_list[pop] for pop in pops]).flatten()
    ax.scatter(x_flat, y_flat, color=BLUE, edgecolor="black", s=96,
               label=f"Simulation replicates ({len(y_flat) // len(pops)} per set)", alpha=0.5)
    ax.scatter(pop_range, [y[pop] for pop in pops], color=RED, s=48,
               label="Mean over replicates (per set)", marker="x")
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel("Simulation replicates")
    ax.set_ylabel(label_dico[col])
    ax.legend()

    print(f"Plotting {path_output}")
    print([f"Set {s}: {pop}" for s, pop in zip(range(1, len(pops) + 1), pops)])
    plt.tight_layout()
    plt.savefig(path_output)
    plt.clf()
    plt.close("all")


def main(path_input, path_output):
    os.makedirs(os.path.dirname(path_output), exist_ok=True)

    pd_list = []
    for filepath in sorted(path_input):
        dff = pd.read_csv(filepath, sep="\t")
        dff["species"] = os.path.basename(filepath).split(".")[0]
        dff["pop"] = os.path.basename(filepath).split(".")[1]
        pd_list.append(dff)
    df = pd.concat(pd_list)
    df["simu"] = df["category"].apply(lambda x: x.split("_")[0] if "_" in x else x)
    df["seed"] = df["category"].apply(lambda x: x.split("_")[1] if "_" in x else "")
    df.to_csv(path_output, sep="\t", index=False)

    # Obtain the categories that are not "control"
    df_control = df[df["simu"] == "control"]
    for cat, df_cat in df.groupby("simu"):

        if cat == "control":
            continue
        for col in label_dico.keys():
            # Plot cat against the control
            df_merge = df_cat.merge(df_control, on=["species", "pop"], suffixes=("", "_control"))

            x, y, x_list, y_list = {}, {}, {}, {}
            for pop, df_pop in df_merge.groupby("pop"):
                xv, yv = df_pop[col + "_control"].values, df_pop[col].values
                assert len(set(xv)) == 1
                x_list[pop] = xv
                y_list[pop] = yv
                x[pop] = np.mean(xv)
                y[pop] = np.mean(yv)
            plot_boxplot(x, y, y_list, col, f"{path_output.replace(".tsv", "")}_{cat}_{col}_boxplot.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', required=True, type=str, nargs="+", dest="tsv", help="Input tsv file")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output tsv file")
    args = parser.parse_args()
    main(args.tsv, args.output)
