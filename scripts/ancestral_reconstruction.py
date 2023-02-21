import os
import argparse
from ete3 import Tree
import subprocess
import numpy as np
from libraries import open_fasta, write_fasta, codontable, zip_file


def most_common(lst):
    return max(set(lst), key=lst.count)


def clean_sequence(seq):
    codon_list = []
    for pos in range(len(seq) // 3):
        codon = seq[pos * 3: pos * 3 + 3]
        codon = codon if (codontable[codon] != "-" and codontable[codon] != "X") else "---"
        codon_list.append(codon)
    return "".join(codon_list)


def SubsetMostCommon(alignment, specie, tree_path, fasta_pop, depth=4):
    if not os.path.exists(tree_path):
        tree_path = tree_path.replace("_null_", "__")
    t = Tree(tree_path)
    leaves = t.get_leaves_by_name(specie)
    assert len(leaves) == 1
    node = leaves[0]

    subali = {specie: clean_sequence(fasta_pop)}
    while len(subali) < depth:
        if node is None:
            break

        sister_nodes = node.get_sisters()
        if len(sister_nodes) > 0:
            sister_names = node.get_sisters()[0].get_leaf_names()
            assert specie not in sister_names

            diff_genus_names = [k for k in sister_names if k.split("_")[0] != specie.split("_")[0]]
            if len(diff_genus_names) > 0:

                seqs = [alignment[sp] for sp in diff_genus_names]
                ref_seq = []
                for pos in range(len(seqs[0]) // 3):
                    codons = [s[pos * 3: pos * 3 + 3] for s in seqs]
                    codons = [c for c in codons if codontable[c] != "-" and codontable[c] != "X"]
                    ref_seq.append("---" if len(codons) == 0 else most_common(codons))

                subali[sister_names[0]] = "".join(ref_seq)

        node = node.up
    tree = (t if node is None else node)
    tree.prune(subali.keys())
    # print(subali.keys())
    return subali, tree


def main(args):
    '''
    USAGE:	./fastml [-options]
    |-------------------------------- HELP: -------------------------------------+
    | VALUES IN [] ARE DEFAULT VALUES                                            |
    |-h   help                                                                   |
    |-s sequence input file (for example use -s D:\mySequences\seq.txt )       |
    |-t tree input file                                                          |
    |   (if tree is not given, a neighbor joining tree is computed).             |
    |-g Assume among site rate variation model (Gamma) [By default the program   |
    |   will assume an homogenous model. very fast, but less accurate!]          |
    |-m     model name                                                           |
    |-mj    [JTT]                                                                |
    |-ml    LG                                                                   |
    |-mr    mtREV (for mitochondrial genomes)                                    |
    |-md    DAY                                                                  |
    |-mw    WAG                                                                  |
    |-mc    cpREV (for chloroplasts genomes)                                     |
    |-ma    Jukes and Cantor (JC) for amino acids                                |
    |-mn    Jukes and Cantor (JC) for nucleotides                                |
    |-mh    HKY Model for nucleotides                                            |
    |-mg    nucgtr Model for nucleotides                                         |
    |-mt    tamura92 Model for nucleotides                                       |
    |-my    yang M5 codons model                                                 |
    |-me    empirical codon matrix                                               |
    +----------------------------------------------------------------------------+
    |Controling the output options:                                              |
    |-x   tree file output in Newick format [tree.newick.txt]                    |
    |-y   tree file output in ANCESTOR format [tree.ancestor.txt]                |
    |-j   joint sequences output file [seq.joint.txt]                            |
    |-k   marginal sequences output file [seq.marginal.txt]                      |
    |-d   joint probabilities output file [prob.joint.txt]                       |
    |-e   marginal probabilities output file [prob.marginal.txt]                 |
    |-q   ancestral sequences output format.  -qc = [CLUSTAL], -qf = FASTA       |
    |     -qm = MOLPHY, -qs = MASE, -qp = PHLIYP, -qn = Nexus                    |
    +----------------------------------------------------------------------------+
    |Advances options:                                                           |
    |-a   Treshold for computing again marginal probabilities [0.9]              |
    |-b   Do not optimize branch lengths on starting tree                        |
    |     [by default branches and alpha are ML optimized from the data]         |
    |-c   number of discrete Gamma categories for the gamma distribution [8]     |
    |-f   don't compute Joint reconstruction (good if the branch and bound       |
    |     algorithm takes too much time, and the goal is to compute the          |
    |     marginal reconstruction with Gamma).                                   |
    |-z   The bound used. -zs - bound based on sum. -zm based on max. -zb [both] |
    |-p   user alpha parameter of the gamma distribution [if alpha is not given, |
    |     alpha and branches will be evaluated from the data (override -b)       |
    +----------------------------------------------------------------------------+
    '''
    os.makedirs(args.output, exist_ok=True)

    fasta_pop = open_fasta(args.fasta_pop)
    # Open fasta files, for each extract the species sequence
    # Update seqs with fixed poly dict
    files = sorted(os.listdir(args.fasta_folder))

    size = len(files) if args.subsample_genes < 1 else min(args.subsample_genes, len(files))
    np.random.seed(seed=0)
    rd_fasta_list = np.random.choice(files, size=size, replace=False)
    for fasta_file in rd_fasta_list:
        ensg = fasta_file.replace("__", "_null_").replace("_NT.fasta", "")
        fasta_dict = open_fasta(f"{args.fasta_folder}/{fasta_file}")
        tree_path = f"{args.tree_folder}/{ensg}_NT.rootree"
        if args.species not in fasta_dict:
            continue

        o = f"{args.output}/{ensg}.fastml"
        if os.path.exists(f"{o}.joint.fasta") and os.path.isfile(f"{o}.joint.fasta"):
            continue
        subali, subtree = SubsetMostCommon(fasta_dict, args.species, tree_path, fasta_pop[ensg])

        # Write fasta file
        s = f"{args.output}/{ensg}_NT.fasta"
        write_fasta(subali, s)
        # Write the tree
        t = f"{args.output}/{ensg}_NT.rootree"
        subtree.write(outfile=t, format=5, format_root_node=True)

        cmd = f"{args.exec} -g -my -qf -s {s} -t {t} -x {o}.newick -y {o}.ancestor -j {o}.joint.fasta -k {o}.marginal" \
              f".fasta -d {o}.join.prob -e {o}.marginal.prob "
        # print(cmd)
        subprocess.check_output(cmd, shell=True)
        os.remove(f"{o}.marginal.prob")
        os.remove(f"{o}.marginal.fasta")
        zip_file(s)
        zip_file(f"{o}.join.prob")
        zip_file(f"{o}.joint.fasta")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fasta_pop', required=True, type=str, dest="fasta_pop", help="The fasta file path")
    parser.add_argument('--fasta_folder', required=True, type=str, dest="fasta_folder", help="The fasta folder path")
    parser.add_argument('--tree_folder', required=True, type=str, dest="tree_folder", help="The tree folder path")
    parser.add_argument('--exec', required=True, type=str, dest="exec", help="The exec path")
    parser.add_argument('--species', required=True, type=str, dest="species", help="The focal species")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output folder")
    parser.add_argument('--subsample_genes', required=False, default=-1, type=int, dest="subsample_genes",
                        help="Number of genes to subsample")
    main(parser.parse_args())
