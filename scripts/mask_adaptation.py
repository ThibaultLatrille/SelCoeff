import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
gene_level = True


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_dict = defaultdict(list)

    for ensg in os.listdir(args.exp_folder):
        if "ENSG" not in ensg:
            continue
        f_path = f"{args.exp_folder}/{ensg}"
        path_mutsel = f"{f_path}/sitemutsel_1.run.ci0.025.tsv"
        path_omega = f"{f_path}/siteomega_1.run.ci0.025.tsv"
        array_mutsel = pd.read_csv(path_mutsel, sep="\t")["omega_upper"].values
        array_omega = pd.read_csv(path_omega, sep="\t")["omega_lower"].values
        assert len(array_mutsel) == len(array_omega)
        nbr_sites = len(array_mutsel) - 1
        ensg_name = ensg.replace('__', '_null_').replace('_NT', '')
        if gene_level:
            mutsel_upper = array_mutsel[0]
            omega_lower = array_omega[0]
            if omega_lower > mutsel_upper:
                output_dict["ensg"].extend([ensg_name] * nbr_sites)
                output_dict["pos"].extend(range(nbr_sites))
        else:
            mutsel_upper = pd.read_csv(path_mutsel, sep="\t")["omega_upper"].values[1:]
            omega_lower = pd.read_csv(path_omega, sep="\t")["omega_lower"].values[1:]
            adaptation = np.argwhere(omega_lower > mutsel_upper)
            output_dict["ensg"].extend([ensg_name] * len(adaptation))
            output_dict["pos"].extend([i[0] for i in adaptation])

    df = pd.DataFrame(output_dict)
    df.to_csv(args.output, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_folder', required=True, type=str, dest="exp_folder", help="The experiment folder path")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output tsv path")
    main(parser.parse_args())
