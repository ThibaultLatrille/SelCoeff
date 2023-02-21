import os
import argparse
import pandas as pd
from collections import defaultdict
from libraries import open_fasta, codontable
from ancestral_reconstruction import SubsetMostCommon


def main(fasta_folder, tree_folder, species, fasta_path, output, flank):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    output_dict = defaultdict(list)
    fasta_pop = open_fasta(fasta_path)
    for ensg in fasta_pop:
        fasta_dict = open_fasta(f"{fasta_folder}/{ensg}_NT.fasta")
        tree_path = f"{tree_folder}/{ensg}_NT.rootree"

        subali, _ = SubsetMostCommon(fasta_dict, species, tree_path, fasta_pop[ensg], depth=2)

        unconserved_c_site = []
        nbr_codons = len(subali[species]) // 3
        seqs = [subali[sp] for sp in subali]
        for c_site in range(nbr_codons):
            column = [seq[c_site * 3:c_site * 3 + 3] for seq in seqs]
            unconserved_c_site.append(len(set(column)) > 1 or codontable[column[0]] in ["X", "-"])

        mask = []
        for c_site in range(nbr_codons):
            for i in range(max(0, c_site - flank), min(c_site + flank + 1, nbr_codons)):
                if i == c_site:
                    continue
                if unconserved_c_site[i]:
                    mask.append(c_site)
                    break
        if len(mask) > 0:
            output_dict["ensg"].extend([ensg] * len(mask))
            output_dict["pos"].extend(mask)

    df = pd.DataFrame(output_dict)
    df.to_csv(args.output, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fasta_folder', required=True, type=str, dest="fasta_folder", help="The fasta folder path")
    parser.add_argument('--tree_folder', required=True, type=str, dest="tree_folder", help="The tree folder path")
    parser.add_argument('--species', required=True, type=str, dest="species", help="The focal species")
    parser.add_argument('--fasta', required=True, type=str, dest="fasta", help="The fasta file path")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output tsv path")
    parser.add_argument('--flank', required=False, type=int, default=2, dest="flank", help="Flank size")
    args = parser.parse_args()
    main(args.fasta_folder, args.tree_folder, args.species, args.fasta, args.output, args.flank)
