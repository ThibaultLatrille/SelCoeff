import os
import argparse
import pandas as pd
from collections import defaultdict
from libraries import open_fasta, translate_cds


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_dict = defaultdict(list)

    df = pd.read_csv(args.ontology_tsv, sep="\t",
                     dtype={"go_id": str, "go_name": str, "go_cds_count": int, "go_cds": str})
    df = df[df["go_name"] == args.ontology_key]
    if len(df) > 1:
        print(f"Multiple rows found for {args.ontology_key}, taking the first one")
    ensg_list = set(df["go_cds"].values[0].split("; "))
    assert len(ensg_list) == df["go_cds_count"].values[0]

    files = sorted(os.listdir(args.fasta_folder))
    for fasta_file in files:
        ensg = fasta_file.replace("__", "_null_").replace("_NT.fasta", "")
        if ensg in ensg_list:
            print(f"Masking {ensg}")
        else:
            print(f"Not masking {ensg}")
            continue
        fasta_dict = open_fasta(f"{args.fasta_folder}/{fasta_file}")
        first_key = next(iter(fasta_dict))
        nbr_sites = len(fasta_dict[first_key])
        output_dict["ensg"].extend([ensg] * nbr_sites)
        output_dict["pos"].extend(range(nbr_sites))

    df = pd.DataFrame(output_dict)
    df.to_csv(args.output, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fasta_folder', required=True, type=str, dest="fasta_folder", help="The fasta folder path")
    parser.add_argument('--ontology_tsv', required=True, type=str, dest="ontology_tsv", help="The ontology tsv path")
    parser.add_argument('--ontology_key', required=True, type=str, dest="ontology_key", help="The ontology key")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output tsv path")
    main(parser.parse_args())
