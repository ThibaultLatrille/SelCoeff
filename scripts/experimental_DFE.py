import os
import argparse
from matplotlib.patches import Rectangle
from libraries import *


def plot_histogram(counts, cat_snps, edges, file):
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
    cats_list_edges = [cat_snps.rate2cats(b) for b in edges[1:]]
    colors = [cat_snps.color(cats[0]) if len(cats) > 0 else "black" for cats in cats_list_edges]

    dico_opp = defaultdict(float)
    for tmp_cats, tmp_count in zip(cats_list_edges, counts):
        dico_opp[tmp_cats[0]] += tmp_count
    total = sum(dico_opp.values())
    for cat in cat_snps.non_syn_list:
        dico_opp[cat] /= total

    plt.bar(edges[:-1], height=counts, color=colors, width=np.diff(edges), linewidth=1.0, edgecolor="black",
            align="edge")
    if cat_snps.bins <= 10:
        handles = [Rectangle((0, 0), 1, 1, color=c) for c in [cat_snps.color(cat) for cat in cat_snps.non_syn_list]]
        labels = [cat_snps.label(cat).replace("S_", "s_") + f" ({dico_opp[cat] * 100:.2f}% of total)" for cat in
                  cat_snps.non_syn_list]
        plt.legend(handles, labels)
    plt.xlabel("Selection coefficient ($s_0$)")
    plt.ylabel("Density")
    for x in cat_snps.inner_bound:
        plt.axvline(x, color="grey", lw=1, ls='--')
    if min(edges) < -1.0 and max(edges) > 1.0:
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.axvline(0, color="black", lw=2)
    plt.xlim((min(edges), max(edges)))
    plt.tight_layout()
    plt.savefig(file, format="pdf")
    plt.clf()
    plt.close("all")


def main(input_folder, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    cat_snps = CategorySNP("MutSel", "", bins=2, windows=0)

    bins = np.linspace(-5e-2, 5e-2, 61)
    counts = np.histogram([], bins=bins)[0]

    for exp in os.listdir(input_folder):
        path = f"{input_folder}/{exp}"
        if not path.endswith(".csv"):
            continue

        df = pd.read_csv(path, sep=";")
        df = df[df["Mutation type"] == "Nonsynonymous_mutation"]
        for rep in range(1, 4):
            list_rates = df[f"Fitness from YPD replicate {rep}"].values - 1.0
            counts = counts + np.histogram(list_rates, bins=bins)[0]

    plot_histogram(counts, cat_snps, bins, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_folder', required=True, type=str, dest="exp_folder", help="The experiment folder path")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output pdf path")
    args = parser.parse_args()
    main(args.exp_folder, args.output)
