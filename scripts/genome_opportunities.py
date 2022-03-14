import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from libraries import nucleotides, codontable, CategorySNP, CdsRates, open_fasta

transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}


def main(args):
    if args.method == "Omega":
        cat_snps = CategorySNP(args.method, transform_bound=lambda s: s / (1 - np.exp(-s)) if s != 0 else 1.0)
    else:
        cat_snps = CategorySNP(args.method)

    cds_rates = CdsRates(args.method, args.exp_folder)
    output_dict, dico_opp_sp = defaultdict(list), defaultdict(int)
    seqs = open_fasta(args.fasta_pop)
    rd_ensg_list = np.random.choice(list(seqs.keys()), size=min(1500, len(seqs)), replace=False)
    for ensg in rd_ensg_list:
        seq = seqs[ensg]
        for c_site in range(len(seq) // 3):
            ref_codon = seq[c_site * 3:c_site * 3 + 3]
            ref_aa = codontable[ref_codon]
            if ref_aa == "X" or ref_aa == "-":
                continue
            for frame, ref_nuc in enumerate(ref_codon):
                for alt_nuc in [i for i in nucleotides if i != ref_nuc]:
                    alt_codon = ref_codon[:frame] + alt_nuc + ref_codon[frame + 1:]
                    alt_aa = codontable[alt_codon]
                    if alt_aa != 'X' and alt_aa != ref_aa:
                        rate = cds_rates.rate(ensg, ref_aa, alt_aa, c_site)
                        if np.isfinite(rate):
                            mutation_rate = 2.0 if (alt_nuc, ref_nuc) in transitions else 1.0
                            dico_opp_sp[cat_snps.rate2cat(rate)] += mutation_rate

    tot_opp = sum(dico_opp_sp.values())
    for cat in cat_snps.non_syn():
        output_dict[cat].append(dico_opp_sp[cat] / tot_opp)
    df = pd.DataFrame(output_dict)
    df.to_csv(args.output, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_folder', required=True, type=str, dest="exp_folder", help="The experiment folder path")
    parser.add_argument('--fasta_pop', required=True, type=str, dest="fasta_pop", help="The fasta path")
    parser.add_argument('--method', required=True, type=str, dest="method", help="The method (MutSel or Omega)")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output path")
    main(parser.parse_args())
