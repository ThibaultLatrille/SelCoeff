import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict
from ete3 import Tree
from libraries import CdsRates, CategorySNP, open_fasta, codontable


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    dico_div = defaultdict(float)
    cat_snps = CategorySNP("MutSel", args.bounds, bins=args.bins, windows=args.windows)
    cds_rates = CdsRates("MutSel", args.exp_folder)

    for anc_file in glob(os.path.join(args.ancestral, "*.joint.fasta")):
        seqs = open_fasta(anc_file)
        ensg = os.path.basename(anc_file).split(".")[0]
        cds_rates.add_ensg(ensg)
        cds_rates.add_mut_ensg(ensg)

        tree_path = anc_file.replace(".joint.fasta", ".newick")
        t = Tree(tree_path, format=1)
        ancestor = t.get_leaves_by_name(args.species)[0].up.name
        anc_seq = seqs[ancestor]
        der_seq = seqs[args.species]

        assert len(anc_seq) == len(der_seq)

        for c_site in range(len(anc_seq) // 3):
            anc_codon = anc_seq[c_site * 3:c_site * 3 + 3]
            der_codon = der_seq[c_site * 3:c_site * 3 + 3]
            if anc_codon == der_codon:
                continue

            anc_aa = codontable[anc_codon]
            der_aa = codontable[der_codon]
            if anc_aa == "X" or anc_aa == "-" or der_aa == "X" or der_aa == "-":
                continue

            if anc_aa == der_aa:
                dico_div["div_syn"] += 1
                continue

            rate = cds_rates.rate(ensg, anc_aa, der_aa, c_site)
            if not np.isfinite(rate):
                dico_div["div_NotFinite"] += 1.0
                continue

            dico_div["div_all"] += 1.0

            cats = cat_snps.rate2cats(rate)
            if len(cats) == 0:
                dico_div["div_Out"] += 1.0
                continue

            for cat in cats:
                dico_div[f"div_{cat}"] += 1.0

        cds_rates.rm_ensg(ensg)

    df_opp = pd.read_csv(args.opp, sep="\t")
    for cat in cat_snps.non_syn_list:
        dico_div[f"L_{cat}"] = df_opp[cat].values[0] * df_opp["Ldn"].values[0]
    dico_div["L_syn"] = df_opp["Lds"].values[0]
    dico_div["L_all"] = df_opp["Ldn"].values[0]
    df = pd.DataFrame({k: [v] for k, v in dico_div.items()})
    df.to_csv(args.output, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ancestral', required=True, type=str, dest="ancestral", help="The ancestral folder")
    parser.add_argument('--exp_folder', required=True, type=str, dest="exp_folder", help="The experiment folder path")
    parser.add_argument('--opp', required=True, type=str, dest="opp", help="The opportunities file path")
    parser.add_argument('--bounds', required=True, default="", type=str, dest="bounds", help="Input bound file path")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output tsv path")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    parser.add_argument('--windows', required=False, default=0, type=int, dest="windows", help="Number of windows")
    parser.add_argument('--species', required=True, type=str, dest="species", help="The focal species")
    main(parser.parse_args())
