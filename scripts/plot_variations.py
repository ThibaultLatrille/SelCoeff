import argparse
from ete3 import PhyloTree, TreeStyle
from matplotlib.patches import Rectangle
from libraries import *
import ete3_custom_faces as faces


def masked_sub(row, mask_grouped):
    return (row.ENSG in mask_grouped) and (row.CODON_SITE in mask_grouped[row.ENSG])


def cat_sub(row, cat_snps):
    return cat_snps.rate2cats(row.SEL_COEFF)[0] if (row.SUB_TYPE != "Syn") else "syn"


def plot_histogram(score_list, cat_snps, file):
    plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    ax = plt.subplot(1, 2, 1)
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
    plt.subplot(1, 2, 2)
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
    plt.savefig(file, format="pdf")
    plt.clf()
    plt.close("all")


def custom_layout(node, focal_species, light_cols):
    leaf_color, node_size, bold, text_prefix = "#000000", 2, False, ""
    if node.is_leaf():
        if "name" in node.features and node.name == focal_species:
            leaf_color = "red"
            node_size = 8
            bold = True
            text_prefix = " "
        node.img_style["shape"] = "square"
        node.img_style["size"] = node_size
        node.img_style["fgcolor"] = leaf_color
        attr_face = faces.AttrFace("name", "Verdana", 11, leaf_color, None, bold=bold, text_prefix=text_prefix)
        faces.add_face_to_node(attr_face, node, 0)
        if hasattr(node, "sequence"):
            SequenceFace = faces.SequenceFace(node.sequence, "nt", 13, bold=bold, light_cols=light_cols)
            faces.add_face_to_node(SequenceFace, node, 1, aligned=True)
    else:
        node.img_style["size"] = node_size
        node.img_style["shape"] = "circle"


def clean_seq(seq):
    nucs = {"A", "C", "G", "T", "-", " "}
    return "".join([n if n in nucs else "-" for n in seq])


def main(input_path, mask_list, output, bins, windows, bounds, fasta_folder, tree_file, focal_species):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    cat_snps = CategorySNP("MutSel", bounds, bins=bins, windows=windows)
    mask_grouped = merge_mask_list(mask_list)

    df = pd.read_csv(input_path, sep='\t')
    df["MASKED"] = df.apply(lambda row: masked_sub(row, mask_grouped), axis=1)
    df["CAT"] = df.apply(lambda row: cat_sub(row, cat_snps), axis=1)

    # plot the histogram of counts
    df_non_syn = df[df.SUB_TYPE != "Syn"]
    plot_histogram(df_non_syn["SEL_COEFF"].tolist(), cat_snps, output)

    df_pos = df_non_syn[df_non_syn["CAT"] == "pos"]
    print(f"Number of positive selected sites: {len(df_pos)}")
    plot_folder = output.replace(".pdf", "")
    os.makedirs(plot_folder, exist_ok=True)
    # whole_tree = Tree(tree_file, format=1)
    # groups of 50 sites
    n_sites, flanks, sep = 10, 2, "|"
    for i_row in range(0, len(df_pos), n_sites):
        pt = PhyloTree(tree_file, quoted_node_names=True, format=1)
        all_species = pt.get_leaf_names()
        sub_df = df_pos.iloc[i_row: i_row + n_sites]
        # group the substitutions by gene
        # Use ete3 to plot the tree and the substitutions
        subset_fasta = {species: "" for species in all_species}
        light_cols = []
        nuc_pos = 1
        for ensg, ensg_df in sub_df.groupby("ENSG"):
            fasta_path = clean_ensg_path(os.path.join(fasta_folder, f"{ensg}_NT.fasta"))

            fasta = open_fasta(fasta_path)
            assert focal_species in fasta, f"{focal_species} not in {fasta_path}"

            for index, row in ensg_df.iterrows():
                # Take the two sites before and after the site
                min_site = max(0, row.CODON_SITE - 2)
                max_site = min(len(fasta[focal_species]) // 3, row.CODON_SITE + 2)
                for species in all_species:
                    if focal_species == species:
                        left = fasta[species][min_site * 3: row.CODON_SITE * 3]
                        right = fasta[species][(row.CODON_SITE + 1) * 3: (max_site + 1) * 3]
                        subset_fasta[species] += left + row.CODON_DER + right + sep
                        light_cols.extend([nuc_pos + i for i in range(len(left))])
                        nuc_pos += len(left) + 3
                        light_cols.extend([nuc_pos + i for i in range(len(right))])
                        nuc_pos += len(right) + 1
                    elif species in fasta:
                        subset_fasta[species] += fasta[species][min_site * 3: (max_site + 1) * 3] + sep
                    else:
                        subset_fasta[species] += "-" * (max_site - min_site + 1) * 3 + sep

        clean_subset = {k: sep + v for k, v in subset_fasta.items() if len(set(v)) > 2}
        pos_fasta = fasta_txt(clean_subset)
        pt.prune(clean_subset.keys())
        assert len(pt.get_leaf_names()) == len(clean_subset)
        n_block = i_row // n_sites

        pt.link_to_alignment(alignment=pos_fasta, alg_format="fasta")
        ts = TreeStyle()
        ts.show_leaf_name = False
        ts.draw_guiding_lines = False
        ts.layout_fn = lambda node: custom_layout(node, focal_species, light_cols)
        pt.render(f"{plot_folder}/{n_block}.pdf", tree_style=ts, w=1200, units="mm")
        print(f"{len(clean_subset)} species in the {n_block}th block")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True, type=str, dest="input", help="Input vcf")
    parser.add_argument('--mask', required=False, default="", nargs="+", type=str, dest="mask")
    parser.add_argument('--bounds', required=True, default="", type=str, dest="bounds", help="Input bound file path")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    parser.add_argument('--windows', required=False, default=0, type=int, dest="windows", help="Number of windows")
    parser.add_argument('-o', '--output', required=False, type=str, dest="output", help="Output pdf")
    parser.add_argument('--fasta_folder', required=True, type=str, dest="fasta_folder", help="The fasta folder path")
    parser.add_argument('--tree', required=True, type=str, dest="tree", help="The tree path")
    parser.add_argument('--species', required=True, type=str, dest="species", help="The focal species")
    args = parser.parse_args()
    main(args.input, args.mask, args.output, args.bins, args.windows, args.bounds, args.fasta_folder,
         args.tree, args.species)
