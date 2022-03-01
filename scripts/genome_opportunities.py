import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from libraries import nucleotides, codontable, CategorySNP

transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}


class CdsSelCoeff(dict):
    def __init__(self, method, exp_folder):
        self.method = method
        assert self.method in ["Omega", "MutSel"], 'Method must be either "Omega" or "MutSel"'
        self.exp_folder = exp_folder
        super().__init__()

    def add_engs(self, ensg):
        f_path = f"{self.exp_folder}/{ensg}"
        if self.method == "MutSel":
            path = f"{f_path}/sitemutsel_1.run.siteprofiles"
            self[ensg] = pd.read_csv(path, sep="\t", skiprows=1, header=None,
                                     names="site,A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y".split(","))
        elif self.method == "Omega":
            path = f"{f_path}/siteomega_1.run.ci0.025.tsv"
            self[ensg] = pd.read_csv(path, sep="\t")["gene_omega"].values[1:]

    def sel_coeff(self, ensg, ref_aa, alt_aa, c_site):
        if ensg not in self:
            self.add_engs(ensg)
        if self.method == "MutSel":
            return np.log(self[ensg][alt_aa][c_site] / self[ensg][ref_aa][c_site])
        elif self.method == "Omega":
            return self[ensg][c_site]


def ali_seq(fasta_folder, fasta_file, species):
    path = f"{fasta_folder}/{fasta_file}.fasta"
    ali_file = open(path, 'r')
    for sp in ali_file:
        seq = ali_file.readline()
        if sp.replace(">", "").strip() == species:
            return seq.strip()
    return None


def main(args):
    if args.method == "Omega":
        cat_snps = CategorySNP(args.method, transform_bound=lambda s: s / (1 - np.exp(-s)) if s != 0 else 1.0)
    else:
        cat_snps = CategorySNP(args.method)

    dico_profiles = CdsSelCoeff(args.method, args.exp_folder)
    output_dict = defaultdict(list)
    for species in args.species:
        dico_opp_sp = defaultdict(int)
        for ensg in sorted(os.listdir(args.exp_folder))[:500]:
            seq = ali_seq(args.fasta_folder, ensg, species)
            if not seq:
                continue

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
                            sel_coeff = dico_profiles.sel_coeff(ensg, ref_aa, alt_aa, c_site)
                            if np.isfinite(sel_coeff):
                                mutation_rate = 2.0 if (alt_nuc, ref_nuc) in transitions else 1.0
                                dico_opp_sp[cat_snps.selcoeff2cat(sel_coeff)] += mutation_rate

        tot_opp = sum(dico_opp_sp.values())
        output_dict["species"].append(species)
        for cat in cat_snps.non_syn():
            output_dict[cat].append(dico_opp_sp[cat] / tot_opp)
        print(output_dict)
    df = pd.DataFrame(output_dict)
    df.to_csv(args.output, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--species', required=True, type=str, nargs="+", dest="species", help="Focal species")
    parser.add_argument('--exp_folder', required=True, type=str, dest="exp_folder", help="The experiment folder path")
    parser.add_argument('--fasta_folder', required=True, type=str, dest="fasta_folder", help="The fasta folder path")
    parser.add_argument('--method', required=True, type=str, dest="method", help="The method (MutSel or Omega)")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output path")
    main(parser.parse_args())
