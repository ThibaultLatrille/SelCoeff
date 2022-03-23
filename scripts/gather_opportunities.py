import os
import argparse
import pandas as pd
from libraries import format_pop


def open_tsv(filepath):
    ddf = pd.read_csv(filepath, sep="\t")
    ddf["species"], pop, ddf["method"] = os.path.basename(filepath).replace(".tsv", "").split(".")[-3:]
    ddf["pop"] = format_pop(pop.replace("_", " "))
    return ddf


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_merge = pd.concat([open_tsv(filepath) for filepath in sorted(args.tsv)])
    df_merge.to_csv(args.output, sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', required=False, type=str, nargs="+", dest="tsv", help="Input tsv file")
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output tsv file")
    parser.add_argument('--sample_list', required=False, type=str, dest="sample_list", help="Sample list file")
    main(parser.parse_args())
