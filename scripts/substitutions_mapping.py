import argparse
from glob import glob
from ete3 import Tree
from matplotlib.patches import Rectangle
from libraries import *


def plot_histogram(score_list, cat_snps, method, file):
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
    xmin, xmax = xlim_dico[method][0], xlim_dico[method][1]
    n, bins, patches = plt.hist([s for s in score_list if np.isfinite(s)], bins=np.linspace(xmin, xmax, 61),
                                range=(xmin, xmax), edgecolor="black", linewidth=1.0)
    total_n = sum(n)
    if cat_snps.bins <= 10:
        n_cat = defaultdict(float)
        for i, b in enumerate(bins[1:]):
            cats = cat_snps.rate2cats(b)
            assert len(cats) >= 1
            cat = cats[0]
            patches[i].set_facecolor(cat_snps.color(cat))
            n_cat[cat] += n[i] / total_n
        handles = [Rectangle((0, 0), 1, 1, color=c) for c in [cat_snps.color(cat) for cat in cat_snps.non_syn_list]]
        labels = [cat_snps.label(cat) + f" ({n_cat[cat] * 100:.2f}% of total)" for cat in cat_snps.non_syn_list]
        plt.legend(handles, labels)
    plt.xlabel(rate_dico[method])
    plt.ylabel("Density")
    for x in cat_snps.inner_bound:
        plt.axvline(x, color="grey", lw=1, ls='--')
    if xmin < -1.0 and xmax > 1.0:
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.axvline(0, color="black", lw=2)
    plt.xlim((xmin, xmax))
    plt.tight_layout()
    plt.savefig(file, format="pdf")
    plt.clf()
    plt.close("all")


def open_prob(path):
    prob_list = []
    file = gzip.open(path, 'rt') if path.endswith(".gz") else open(path, 'r')
    with file as f:
        for line in f:
            if "joint probs" in line:
                f.readline()
                break
        for line in f:
            prob = line.strip().split(",")[1]
            prob_list.append(float(prob))
    return np.array(prob_list) / max(prob_list)


def main(args):
    assert 0 <= args.anc_proba <= 1
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    dico_div = defaultdict(float)
    cat_snps = CategorySNP("MutSel", args.bounds, bins=args.bins, windows=args.windows)
    cds_rates = CdsRates("MutSel", args.exp_folder)
    score_list = []

    mask_grouped = merge_mask_list(args.mask)

    for anc_file in glob(os.path.join(args.ancestral, "*.joint.fasta.gz")):
        seqs = open_fasta(anc_file)
        ensg = os.path.basename(anc_file).split(".")[0]
        cds_rates.add_ensg(ensg)
        cds_rates.add_mut_ensg(ensg)

        tree_path = anc_file.replace(".joint.fasta.gz", ".newick")
        t = Tree(tree_path, format=1)
        ancestor = t.get_leaves_by_name(args.species)[0].up.name
        anc_seq = seqs[ancestor]
        der_seq = seqs[args.species]
        assert len(anc_seq) == len(der_seq)

        proba_path = anc_file.replace(".joint.fasta.gz", ".join.prob.gz")
        prob_list = open_prob(proba_path)
        assert len(prob_list) == (len(anc_seq) // 3)

        for c_site in range(len(anc_seq) // 3):
            if prob_list[c_site] < args.anc_proba:
                continue

            anc_codon = anc_seq[c_site * 3:c_site * 3 + 3]
            der_codon = der_seq[c_site * 3:c_site * 3 + 3]
            if anc_codon == der_codon:
                continue

            anc_aa = codontable[anc_codon]
            der_aa = codontable[der_codon]
            if anc_aa == "X" or anc_aa == "-" or der_aa == "X" or der_aa == "-":
                continue

            diffs = [s for s in range(len(anc_codon)) if anc_codon[s] != der_codon[s]]
            assert len(diffs) > 0
            if len(diffs) != 1:
                continue

            if ensg in mask_grouped and c_site in mask_grouped[ensg]:
                dico_div["masked"] += 1.0
                continue

            if anc_aa == der_aa:
                dico_div["div_syn"] += 1
                continue

            rate = cds_rates.rate(ensg, anc_aa, der_aa, c_site)
            if not np.isfinite(rate):
                dico_div["div_NotFinite"] += 1.0
                continue

            dico_div["div_all"] += 1.0

            cats = cat_snps.rate2cats(rate)
            if len(cats) == 0:
                dico_div["div_Out"] += 1.0
                continue

            score_list.append(rate)
            for cat in cats:
                dico_div[f"div_{cat}"] += 1.0

        cds_rates.rm_ensg(ensg)

    plot_histogram(score_list, cat_snps, "MutSel", args.output.replace(".tsv", ".pdf"))

    df_opp = pd.read_csv(args.opp, sep="\t")
    for cat in cat_snps.non_syn_list:
        dico_div[f"L_{cat}"] = df_opp[cat].values[0] * df_opp["Ldn"].values[0]
    dico_div["L_syn"] = df_opp["Lds"].values[0]
    dico_div["L_all"] = df_opp["Ldn"].values[0]
    df = pd.DataFrame({k: [v] for k, v in dico_div.items()})
    df.to_csv(args.output, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ancestral', required=True, type=str, dest="ancestral", help="The ancestral folder")
    parser.add_argument('--anc_proba', required=True, type=float, dest="anc_proba", default=0.5,
                        help="Mask the substitutions with reconstruction probability lower than this threshold")
    parser.add_argument('--exp_folder', required=True, type=str, dest="exp_folder", help="The experiment folder path")
    parser.add_argument('--opp', required=True, type=str, dest="opp", help="The opportunities file path")
    parser.add_argument('--bounds', required=True, default="", type=str, dest="bounds", help="Input bound file path")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output tsv path")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    parser.add_argument('--windows', required=False, default=0, type=int, dest="windows", help="Number of windows")
    parser.add_argument('--species', required=True, type=str, dest="species", help="The focal species")
    parser.add_argument('--mask', required=False, default="", nargs="+", type=str, dest="mask",
                        help="List of input mask file path")
    main(parser.parse_args())
