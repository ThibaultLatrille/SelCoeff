import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from libraries import open_fasta, codontable


def main(fasta_path, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    output_dict = defaultdict(list)

    fasta = open_fasta(fasta_path)
    for ensg, seq in fasta.items():
        unconserved = set()
        for nuc_site in range(len(seq) - 1):
            if seq[nuc_site:nuc_site + 2] == "CG":
                unconserved.add(nuc_site // 3)
                unconserved.add((nuc_site + 1) // 3)

        if len(unconserved) > 0:
            output_dict["ensg"].extend([ensg] * len(unconserved))
            output_dict["pos"].extend(sorted(unconserved))

    df = pd.DataFrame(output_dict)
    df.to_csv(args.output, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fasta', required=True, type=str, dest="fasta", help="The fasta file path")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output tsv path")
    args = parser.parse_args()
    main(args.fasta, args.output)
