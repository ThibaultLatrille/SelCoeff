import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from libraries import open_fasta, codontable


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_dict = defaultdict(list)

    # Open fasta files, for each extract the species sequence
    # Update seqs with fixed poly dict
    files = sorted(os.listdir(args.fasta_folder))
    for fasta_file in files:
        ensg = fasta_file.replace("__", "_null_").replace("_NT.fasta", "")
        fasta_dict = open_fasta(f"{args.fasta_folder}/{fasta_file}")
        aa_len = len(next(iter(fasta_dict.values()))) // 3
        if not np.all([(sp in fasta_dict) for sp in args.species]):
            output_dict["ensg"].extend([ensg] * aa_len)
            output_dict["pos"].extend(range(aa_len))
            continue

        unconserved = list()
        for c_site in range(aa_len):
            for sp in args.species:
                seq = fasta_dict[sp]
                ref_codon = seq[c_site * 3:c_site * 3 + 3]
                ref_aa = codontable[ref_codon]
                if ref_aa == "X" or ref_aa == "-":
                    unconserved.append(c_site)
                    break

        assert len(unconserved) == len(set(unconserved))
        if len(unconserved) > 0:
            output_dict["ensg"].extend([ensg] * len(unconserved))
            output_dict["pos"].extend(sorted(unconserved))

    df = pd.DataFrame(output_dict)
    df.to_csv(args.output, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fasta_folder', required=True, type=str, dest="fasta_folder", help="The fasta folder path")
    parser.add_argument('--species', required=True, type=str, nargs="+", dest="species", help="The focal species")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output tsv path")
    main(parser.parse_args())
