import os
import gzip
import argparse
from collections import defaultdict
from ete3 import PhyloTree, TreeStyle
import ete3_custom_faces as faces

codontable = defaultdict(lambda: "-")
codontable.update({
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': 'X', 'TAG': 'X',
    'TGC': 'C', 'TGT': 'C', 'TGA': 'X', 'TGG': 'W'})


def open_file(path):
    return gzip.open(path, 'rt') if path.endswith(".gz") else open(path, 'r')


def open_fasta(path):
    outfile = {}
    ali_file = open_file(path)
    for seq_id in ali_file:
        outfile[seq_id.replace('>', '').strip()] = ali_file.readline().strip()
    return outfile


def fasta_txt(dico_fasta):
    return "\n".join([f">{seq_id}\n{seq}" for seq_id, seq in dico_fasta.items()])


outgroup = "Outgroup"
outgroups = [f"{outgroup}_{i}" for i in range(1, 5)]


def custom_layout(node, focal_species, light_cols, aa_cols):
    leaf_color, node_size, bold, bw, text_prefix = "#000000", 2, False, False, ""
    if node.is_leaf():
        if "name" in node.features and node.name == focal_species:
            leaf_color = "red"
            node_size = 8
            bold = True
            text_prefix = " "
        elif node.name in outgroups:
            node_size = 0
            leaf_color = "#FFFFFF"
            node.img_style["hz_line_color"] = "#FFFFFF"
            node.img_style["vt_line_color"] = "#FFFFFF"
            bw = True
            bold = True
        node.img_style["shape"] = "square"
        node.img_style["size"] = node_size
        node.img_style["fgcolor"] = leaf_color
        attr_face = faces.AttrFace("name", "Verdana", 11, leaf_color, None, bold=bold, text_prefix=text_prefix)
        faces.add_face_to_node(attr_face, node, 0)
        if hasattr(node, "sequence"):
            SequenceFace = faces.SequenceFace(node.sequence, "nt", 13, bold=bold, light_cols=light_cols,
                                              black_and_white=bw, aa_cols=aa_cols)
            faces.add_face_to_node(SequenceFace, node, 1, aligned=True)
    else:
        if len(set(outgroups) & set(node.get_leaf_names())) == len(set(outgroups)):
            node_size = 0
            node.img_style["hz_line_color"] = "#FFFFFF"
            node.img_style["vt_line_color"] = "#FFFFFF"
        node.img_style["size"] = node_size
        node.img_style["shape"] = "circle"


def translate_codons(sp: str, seq: str) -> str:
    if sp in outgroups:
        if sp == outgroups[-1]:
            suffix = "_" + "".join(["↓" if ("↓" in seq[i:i + 3]) else "_" for i in range(0, len(seq), 3)])
        elif sp == outgroups[-2]:
            seq = seq[:-3] + "NUC"
            suffix = "→" + "AA" + "_" * (len(seq) // 3 - 2)
        else:
            suffix = "_" + "_" * (len(seq) // 3)
    else:
        suffix = "→" + "".join([codontable[seq[i:i + 3]] for i in range(0, len(seq), 3)])
    return seq + suffix


def is_aa(seq: str) -> str:
    return "F" * (len(seq) + 1) + "T" * (len(seq) // 3)


def keep_seq(name: str, pos: str, bypass: bool = False) -> bool:
    if bypass:
        return True
    keep_dico = {"TFB1M": {979}, "FMO1": {1240}, "OAS2": {1381}, "SELE": {1722}}
    # keep_dico = {"THSD7A": {3145}, "LIG3": {645}, "DNAH9": {1614}, "NCAPD2": {309}}
    name_s = name.replace("_", "").strip()
    if name_s in keep_dico:
        pos_i = int(pos.replace("_", "").replace("pos:", "").strip())
        return pos_i in keep_dico[name_s]


def main(fasta_file, tree_file, output, focal_species):
    os.makedirs(os.path.dirname(output), exist_ok=True)

    input_seqs = open_fasta(fasta_file)
    if focal_species == "":
        focal_species = os.path.basename(fasta_file).split(".")[0]
    # groups of 50 sites
    n_sites, max_plots, sep = 10, 10, "|"

    input_splits = {k: v.split("|||")[1:-1] for k, v in input_seqs.items()}
    iter_out = enumerate(zip(input_splits[outgroups[0]], input_splits[outgroups[1]]))
    filter_list = [g for g, (seq_0, seq_1) in iter_out if keep_seq(seq_0, seq_1, True)]
    input_splits = {k: [v[i] for i in filter_list] for k, v in input_splits.items()}
    outgroup_split = input_splits[outgroups[-1]]
    fasta_n_sites = len(outgroup_split)
    max_sites = min(max_plots * n_sites, fasta_n_sites)
    print(f"Plotting {max_sites} sites out of {fasta_n_sites}.")

    for i_block in range(0, max_sites, n_sites):
        n_block = i_block // n_sites
        pt = PhyloTree(tree_file, quoted_node_names=True, format=1)
        subset_fasta = {}

        for k, v in input_splits.items():
            tr = [translate_codons(k, seq) for seq in v[i_block:i_block + n_sites]]
            subset_fasta[k] = sep + sep.join(tr) + sep
        clean_subset = {k: v for k, v in subset_fasta.items() if len(set(v)) > 3 or k in outgroups}
        pt.prune(clean_subset.keys(), preserve_branch_length=True)

        light_cols = set([p for p, nuc in enumerate(clean_subset[outgroups[-1]]) if nuc == "_"])

        is_aa_str = sep + sep.join([is_aa(seq) for seq in input_splits[focal_species][i_block:i_block + n_sites]]) + sep
        foc_seq_len = len(clean_subset[focal_species])
        assert len(is_aa_str) == foc_seq_len, f"{len(is_aa_str)} != {foc_seq_len}"
        aa_cols = set([p for p, c in enumerate(is_aa_str) if c == "T"])

        tmp_fasta = fasta_txt(clean_subset)
        pt.link_to_alignment(alignment=tmp_fasta, alg_format="fasta")
        ts = TreeStyle()
        ts.show_leaf_name = False
        ts.draw_guiding_lines = False
        ts.layout_fn = lambda node: custom_layout(node, focal_species, light_cols, aa_cols)
        pt.render(output.replace("-0.pdf", f"-{n_block}.pdf"), tree_style=ts, w=1200, units="mm")
        print(f"{n_block}th block out of {max_sites // n_sites} done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output pdf")
    parser.add_argument('--species', required=False, type=str, dest="species", help="The focal species")
    parser.add_argument('--fasta', required=True, type=str, dest="fasta", help="The fasta path")
    parser.add_argument('--tree', required=True, type=str, dest="tree", help="The tree path")
    args = parser.parse_args()
    main(args.fasta, args.tree, args.output, args.species)
