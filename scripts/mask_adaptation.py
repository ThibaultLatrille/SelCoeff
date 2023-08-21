import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_dict = defaultdict(list)

    for ensg in os.listdir(args.exp_folder):
        f_path = f"{args.exp_folder}/{ensg}"

        path_mutsel = f"{f_path}/sitemutsel_1.run.ci0.025.tsv"
        mutsel_upper = pd.read_csv(path_mutsel, sep="\t")["omega_upper"].values[1:]
        path_omega = f"{f_path}/siteomega_1.run.ci0.025.tsv"
        omega_lower = pd.read_csv(path_omega, sep="\t")["omega_lower"].values[1:]

        adaptation = np.argwhere(omega_lower < mutsel_upper)
        output_dict["ensg"].extend([ensg.replace('__', '_null_').replace('_NT', '')] * len(adaptation))
        output_dict["pos"].extend([i[0] for i in adaptation])

    df = pd.DataFrame(output_dict)
    df.to_csv(args.output, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_folder', required=True, type=str, dest="exp_folder", help="The experiment folder path")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output tsv path")
    main(parser.parse_args())
