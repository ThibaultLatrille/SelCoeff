import os
import argparse
import pandas as pd
from collections import defaultdict
from libraries import open_fasta, translate_cds


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_dict = defaultdict(list)

    files = sorted(os.listdir(args.fasta_folder))
    for fasta_file in files:
        ensg = fasta_file.replace("__", "_null_").replace("_NT.fasta", "")
        fasta_dict = open_fasta(f"{args.fasta_folder}/{fasta_file}")
        seqs = [translate_cds(s) for s in fasta_dict.values()]
        unconserved = []
        for c_pos in range(len(seqs[0])):
            column = "".join([s[c_pos] for s in seqs])
            pct_id = (len(column) - (column.count('X') + column.count('-'))) / len(column)
            if pct_id < args.identity:
                unconserved.append(c_pos)
        if len(unconserved) > 0:
            output_dict["ensg"].extend([ensg] * len(unconserved))
            output_dict["pos"].extend(sorted(unconserved))

    df = pd.DataFrame(output_dict)
    df.to_csv(args.output, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fasta_folder', required=True, type=str, dest="fasta_folder", help="The fasta folder path")
    parser.add_argument('--identity', required=True, type=float, dest="identity",
                        help="The identity (between 0.0 and 1.0) threshold for each column")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output tsv path")
    main(parser.parse_args())
