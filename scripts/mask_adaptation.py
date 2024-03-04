import os
import argparse
import pandas as pd
from collections import defaultdict


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_dict = defaultdict(list)
    print("Site level:", args.site_level, "Output:", args.output, "ωᴬ:", args.omega_A)
    for ensg in os.listdir(args.exp_folder):
        if "ENSG" not in ensg:
            continue
        f_path = f"{args.exp_folder}/{ensg}"
        ensg_name = ensg.replace('__', '_null_').replace('_NT', '')
        ci = "0.025"
        path_mutsel = f"{f_path}/sitemutsel_1.run.ci{ci}.tsv"
        path_omega = f"{f_path}/siteomega_1.run.ci{ci}.tsv"
        array_mutsel = pd.read_csv(path_mutsel, sep="\t")["omega_upper"].values
        array_omega = pd.read_csv(path_omega, sep="\t")["omega_lower"].values
        assert len(array_mutsel) == len(array_omega)
        mutsel_upper = array_mutsel[0]
        omega_lower = array_omega[0]
        if omega_lower > mutsel_upper:
            nbr_sites = len(array_mutsel) - 1
            output_dict["ensg"].extend([ensg_name] * nbr_sites)
            output_dict["pos"].extend(range(nbr_sites))
        elif args.site_level:
            if args.omega_A:
                path_mutsel = f"{f_path}/sitemutsel_1.run.omegaA.tsv"
                omegaA = pd.read_csv(path_mutsel, sep="\t")["p(ωᴬ>0)"].values[1:]
                adaptation = [i for i, v in enumerate(omegaA) if v > 0.95]
            else:
                ci = "0.0025"
                path_mutsel = f"{f_path}/sitemutsel_1.run.ci{ci}.tsv"
                path_omega = f"{f_path}/siteomega_1.run.ci{ci}.tsv"
                array_mutsel = pd.read_csv(path_mutsel, sep="\t")["omega_upper"].values[1:]
                array_omega = pd.read_csv(path_omega, sep="\t")["omega_lower"].values[1:]
                assert len(array_mutsel) == len(array_omega)
                adaptation = [i for i, (m, o) in enumerate(zip(array_mutsel, array_omega)) if o > m]
            output_dict["ensg"].extend([ensg_name] * len(adaptation))
            output_dict["pos"].extend(adaptation)
    df = pd.DataFrame(output_dict)
    df.to_csv(args.output, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_folder', required=True, type=str, dest="exp_folder", help="The experiment folder path")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output tsv path")
    parser.add_argument('--site_level', required=False, default=False, action="store_true", dest="site_level",
                        help="At the site level")
    parser.add_argument('--omega_A', required=False, default=False, action="store_true", dest="omega_A",
                        help="omega_A for site level")
    main(parser.parse_args())
