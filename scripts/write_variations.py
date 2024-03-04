import argparse
from matplotlib.patches import Rectangle
from ete3 import Tree
from libraries import *

outgroup = "Outgroup"
outgroups = [f"{outgroup}_{i}" for i in range(1, 5)]


def masked_sub(row, mask_grouped):
    return (row.ENSG in mask_grouped) and (row.CODON_SITE in mask_grouped[row.ENSG])


def cat_sub(row, cat_snps):
    return cat_snps.rate2cats(row.SEL_COEFF)[0] if (row.SUB_TYPE != "Syn") else "syn"


def plot_histogram(score_list, cat_snps, file):
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
    xmin, xmax = xlim_dico["MutSel"][0], xlim_dico["MutSel"][1]
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
    plt.xlabel(rate_dico["MutSel"])
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

    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
    plt.xlabel(rate_dico["MutSel"])
    plt.ylabel("$\\frac{ \\mathbb{P} [ S_0 ]}{\\mathbb{P} [ -S_0 ]}$")
    # plot x against -x
    x_axis, y_axis = list(), list()

    minus_index = len(n) // 2 - 1
    for i in range(len(n) // 2, len(n)):
        plus_sel = (bins[i] + bins[i + 1]) / 2
        plus_p = n[i]
        minus_sel = (bins[minus_index] + bins[minus_index + 1]) / 2
        minus_p = n[minus_index]
        minus_index -= 1
        assert abs(plus_sel + minus_sel) < 1e-6, f"{plus_sel} and {minus_sel}"
        if minus_p == 0:
            continue
        x_axis.append(plus_sel)
        y_axis.append(plus_p / minus_p)
    plt.scatter(x_axis, y_axis, color="black")
    plt.tight_layout()
    plt.savefig(file.replace(".pdf", ".folded_ratio.pdf"), format="pdf")
    plt.clf()
    plt.close("all")


def flushleft_text(text: str, max_len: int, blank_sep="_") -> str:
    text = text.strip().replace(" ", "_")
    if len(text) > max_len:
        print(f"Text too long: {text} for max_len {max_len}")
        return text[:max_len - 1] + "-"
    out_text = text + blank_sep * (max_len - len(text))
    assert len(out_text) == max_len, f"{len(out_text)} != {max_len}"
    return out_text


def main(input_path, mask_list, bins, windows, bounds, fasta_folder, tree_file, focal_species, output_fasta,
         output_tree,
         output_hist):
    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)
    os.makedirs(os.path.dirname(output_tree), exist_ok=True)
    os.makedirs(os.path.dirname(output_hist), exist_ok=True)
    cat_snps = CategorySNP("MutSel", bounds, bins=bins, windows=windows)
    mask_grouped = merge_mask_list(mask_list)

    df = pd.read_csv(input_path, sep='\t')
    df["MASKED"] = df.apply(lambda row: masked_sub(row, mask_grouped), axis=1)
    n_masked = len(df[df.MASKED])
    print(f"Number of masked sites: {n_masked} out of {len(df)} ({n_masked / len(df) * 100:.2f}%)")
    df = df[~df.MASKED].copy()
    df["CAT"] = df.apply(lambda row: cat_sub(row, cat_snps), axis=1)

    # plot the histogram of counts
    df_non_syn = df[df.SUB_TYPE != "Syn"]
    plot_histogram(df_non_syn["SEL_COEFF"].tolist(), cat_snps, output_hist)

    df_pos = df_non_syn[df_non_syn["CAT"] == "pos"]
    print(f"Number of positive selected sites: {len(df_pos)}")
    # whole_tree = Tree(tree_file, format=1)
    # groups of 50 sites
    sep = "|||"
    pt = Tree(tree_file, quoted_node_names=True, format=1)
    all_species = pt.get_leaf_names()
    # group the substitutions by gene
    # Use ete3 to plot the tree and the substitutions
    subset_fasta = {species: sep for species in outgroups + all_species}
    for ensg, ensg_df in df_pos.groupby("ENSG"):
        print(f"Processing {ensg}")
        fasta_path = clean_ensg_path(os.path.join(fasta_folder, f"{ensg}_NT.fasta"))

        fasta = open_fasta(fasta_path)
        assert focal_species in fasta, f"{focal_species} not in {fasta_path}"

        for index, row in ensg_df.iterrows():
            # Take the two sites before and after the site
            min_site = max(0, row.CODON_SITE - 2)
            max_site = min(len(fasta[focal_species]) // 3 - 1, row.CODON_SITE + 2)
            nb_chars = (max_site - min_site + 1) * 3
            left, right = "", ""

            ensg_name = ensg.split("_")[1]
            subset_fasta[f"{outgroup}_1"] += flushleft_text(ensg_name, nb_chars) + sep
            subset_fasta[f"{outgroup}_2"] += flushleft_text(f"pos:{row.ENSG_POS}", nb_chars) + sep
            blanks = ["_"] * nb_chars
            subset_fasta[f"{outgroup}_3"] += "".join(blanks) + sep

            for species in all_species:
                if focal_species == species:
                    left = fasta[species][min_site * 3: row.CODON_SITE * 3]
                    right = fasta[species][(row.CODON_SITE + 1) * 3: (max_site + 1) * 3]
                    site_seq = left + row.CODON_DER + right + sep
                elif species in fasta:
                    site_seq = fasta[species][min_site * 3: (max_site + 1) * 3] + sep
                else:
                    site_seq = "-" * nb_chars + sep
                assert len(site_seq) == nb_chars + len(sep), f"{len(site_seq)} != {nb_chars + len(sep)}"
                subset_fasta[species] += site_seq

            for i in range(3):
                blanks[len(left) + i] = "-"
            blanks[len(left) + row.ENSG_POS % 3] = "↓"
            subset_fasta[f"{outgroup}_4"] += "".join(blanks) + sep

    clean_subset = {k: v for k, v in subset_fasta.items() if len(set(v)) > 2 or k in outgroups}

    crown = Tree('(A,B);', quoted_node_names=True, format=1)
    A = crown.search_nodes(name='A')[0]
    for o in outgroups:
        A.add_child(name=o, dist=0.0)
    B = crown.search_nodes(name='B')[0]
    B.add_child(pt)
    crown.prune(clean_subset.keys(), preserve_branch_length=True)
    assert len(set(crown.get_leaf_names()) & set(outgroups)) == len(outgroups)
    assert len(crown.get_leaf_names()) == len(clean_subset), f"{len(crown.get_leaf_names())} != {len(clean_subset)}"

    crown.write(outfile=output_tree, format=1)
    write_fasta(clean_subset, output_fasta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True, type=str, dest="input", help="Input vcf")
    parser.add_argument('--mask', required=False, default="", nargs="+", type=str, dest="mask")
    parser.add_argument('--mask_CpG', required=False, default=False, action="store_true", dest="mask_CpG",
                        help="Mask CpG opportunities")
    parser.add_argument('--bounds', required=True, default="", type=str, dest="bounds", help="Input bound file path")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    parser.add_argument('--windows', required=False, default=0, type=int, dest="windows", help="Number of windows")
    parser.add_argument('--fasta_folder', required=True, type=str, dest="fasta_folder", help="The fasta folder path")
    parser.add_argument('--tree', required=True, type=str, dest="tree", help="The tree path")
    parser.add_argument('--species', required=True, type=str, dest="species", help="The focal species")

    parser.add_argument('--output_fasta', required=True, type=str, dest="output_fasta", help="Output fasta")
    parser.add_argument('--output_tree', required=True, type=str, dest="output_tree", help="Output tree")
    parser.add_argument('--output_hist', required=True, type=str, dest="output_hist", help="Output histogram")

    args = parser.parse_args()
    assert args.mask_CpG is False, "Mask CpG is not implemented for this script."
    main(args.input, args.mask, args.bins, args.windows, args.bounds, args.fasta_folder,
         args.tree, args.species, args.output_fasta, args.output_tree, args.output_hist)
