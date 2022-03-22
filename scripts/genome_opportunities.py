import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from libraries import nucleotides, codontable, CdsRates, open_fasta, plt, CategorySNP, my_dpi, xlim_dico, rate_dico

transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}


class Stat:
    def __init__(self):
        self.total = 0.0
        self.n = 0.0

    def add(self, x, w=1.0):
        self.total += x * w
        self.n += w

    def mean(self):
        return self.total / self.n


def pfix(s):
    return s / (1 - np.exp(-s))


def plot_histogram(counts, edges, method, file):
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
    plt.bar(edges[:-1], counts, width=np.diff(edges), edgecolor="black", align="edge")
    plt.xlabel(rate_dico[method])
    plt.ylabel("Density")
    if min(edges) < -1.0 and max(edges) > 1.0:
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.axvline(0, color="black", lw=2)
    plt.xlim((min(edges), max(edges)))
    plt.tight_layout()
    plt.savefig(file, format="pdf")
    plt.clf()
    plt.close("all")


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cat_snps = CategorySNP(args.method, args.bounds, bins=args.bins)

    bins = np.linspace(xlim_dico[args.method][0], xlim_dico[args.method][1], 61)
    counts_mu, counts_q = np.histogram([], bins=bins)[0], np.histogram([], bins=bins)[0]
    log_fitness, sel_coeff, flow_pos, flow_neg = Stat(), Stat(), Stat(), Stat()

    cds_rates = CdsRates(args.method, args.exp_folder)
    output_dict, dico_opp_sp = defaultdict(list), defaultdict(int)
    for cat in cat_snps:
        dico_opp_sp[cat] = 0

    seqs = open_fasta(args.fasta_pop)
    size = len(seqs) if args.subsample_genes < 1 else min(args.subsample_genes, len(seqs))
    rd_ensg_list = np.random.choice(list(seqs.keys()), size=size, replace=False)

    for ensg in rd_ensg_list:
        seq = seqs[ensg]
        list_rates, list_mu, list_q = [], [], []

        for c_site in range(len(seq) // 3):
            ref_codon = seq[c_site * 3:c_site * 3 + 3]
            ref_aa = codontable[ref_codon]
            if ref_aa == "X" or ref_aa == "-":
                continue
            lf = cds_rates.log_fitness(ensg, ref_aa, c_site)
            if args.method == "MutSel":
                log_fitness.add(lf)
            for frame, ref_nuc in enumerate(ref_codon):
                for alt_nuc in [i for i in nucleotides if i != ref_nuc]:
                    alt_codon = ref_codon[:frame] + alt_nuc + ref_codon[frame + 1:]
                    alt_aa = codontable[alt_codon]
                    if alt_aa == 'X' or alt_aa == ref_aa:
                        continue
                    rate = cds_rates.rate(ensg, ref_aa, alt_aa, c_site)
                    if not np.isfinite(rate):
                        continue

                    mutation_rate = 2.0 if (alt_nuc, ref_nuc) in transitions else 1.0
                    cat = cat_snps.rate2cat(rate)
                    dico_opp_sp[cat] += mutation_rate
                    if cat == "OutOfBounds":
                        continue

                    list_rates.append(rate)
                    list_mu.append(mutation_rate)
                    if args.method != "MutSel":
                        continue

                    sel_coeff.add(rate, w=mutation_rate)
                    q = mutation_rate * pfix(rate)
                    list_q.append(q)
                    (flow_pos if rate > 0 else flow_neg).add(q)
        counts_mu = counts_mu + np.histogram(list_rates, bins=bins, weights=list_mu)[0]
        if args.method == "MutSel":
            counts_q = counts_q + np.histogram(list_rates, bins=bins, weights=list_q)[0]

    tot_opp = sum(dico_opp_sp.values())
    for cat in dico_opp_sp:
        output_dict[cat].append(dico_opp_sp[cat] / tot_opp)

    if "OutOfBounds" in dico_opp_sp:
        print(f'{output_dict["OutOfBounds"][0] * 100}% of opportunities out of bounds')

    if args.method == "MutSel":
        output_dict["S_mean"].append(sel_coeff.mean())
        output_dict["log_fitness"].append(log_fitness.mean())
        output_dict["flow_pos"].append(flow_pos.total)
        output_dict["flow_neg"].append(flow_pos.total)
        plot_histogram(counts_q, bins, args.method, args.output.replace(".tsv", ".Flow.pdf"))

    plot_histogram(counts_mu, bins, args.method, args.output.replace(".tsv", ".pdf"))
    df = pd.DataFrame(output_dict)
    df.to_csv(args.output, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_folder', required=True, type=str, dest="exp_folder", help="The experiment folder path")
    parser.add_argument('--fasta_pop', required=True, type=str, dest="fasta_pop", help="The fasta path")
    parser.add_argument('--bounds', required=False, default="", type=str, dest="bounds", help="Input bound file path")
    parser.add_argument('--method', required=True, type=str, dest="method", help="The method (MutSel or Omega)")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output tsv path")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    parser.add_argument('--subsample_genes', required=False, default=-1, type=int, dest="subsample_genes",
                        help="Number of genes to subsample")
    main(parser.parse_args())
