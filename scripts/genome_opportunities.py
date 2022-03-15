import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib.patches import Rectangle
from libraries import nucleotides, codontable, CdsRates, open_fasta, plt, CategorySNP, my_dpi

transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}


def pfix(s):
    return s / (1 - np.exp(-s))


def plot_histogram(list_rates, list_mut_rates, cat_snps, file, xmax=10):
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
    xmin, xmax = (-10, xmax) if min(list_rates) < 0.0 else (0, 1)
    n, bins, patches = plt.hist(list_rates, weights=list_mut_rates, bins=np.linspace(xmin, xmax, 61),
                                range=(xmin, xmax))
    n_cat = defaultdict(int)
    for i, b in enumerate(bins[1:]):
        cat = cat_snps.rate2cat(b)
        patches[i].set_facecolor(cat_snps.color(cat))
        n_cat[cat] += n[i]
    sum_ncat = sum(n_cat.values())
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in [cat_snps.color(cat) for cat in cat_snps.non_syn()]]
    labels = [cat_snps.label(cat) + f" $({n_cat[cat] * 100 / sum_ncat:.2f}\\%)$" for cat in cat_snps.non_syn()]
    plt.legend(handles, labels, loc="upper left")
    plt.xlabel("Scaled selection coefficient (S)")
    plt.ylabel("Density")
    for x in cat_snps.inner_bound:
        plt.axvline(x, color="grey", lw=1, ls='--')
    if xmin < -1.0 and xmax > 1.0:
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.axvline(0, color="black", lw=2)
    plt.xlim((xmin, xmax))
    plt.tight_layout()
    plt.savefig(file, format="pdf")
    plt.clf()
    plt.close("all")
    return n_cat


def main(args):
    if args.method == "Omega":
        cat_snps = CategorySNP(args.method, transform_bound=lambda s: (pfix(s) if s != 0 else 1.0))
    else:
        cat_snps = CategorySNP(args.method)

    cds_rates = CdsRates(args.method, args.exp_folder)
    output_dict, dico_opp_sp = defaultdict(list), defaultdict(int)
    log_fitness, list_rates, list_mut_rates = [], [], []
    seqs = open_fasta(args.fasta_pop)
    for ensg in seqs:
        seq = seqs[ensg]
        for c_site in range(len(seq) // 3):
            ref_codon = seq[c_site * 3:c_site * 3 + 3]
            ref_aa = codontable[ref_codon]
            if ref_aa == "X" or ref_aa == "-":
                continue
            lf = cds_rates.log_fitness(ensg, ref_aa, c_site)
            if np.isfinite(lf):
                log_fitness.append(lf)
            for frame, ref_nuc in enumerate(ref_codon):
                for alt_nuc in [i for i in nucleotides if i != ref_nuc]:
                    alt_codon = ref_codon[:frame] + alt_nuc + ref_codon[frame + 1:]
                    alt_aa = codontable[alt_codon]
                    if alt_aa != 'X' and alt_aa != ref_aa:
                        rate = cds_rates.rate(ensg, ref_aa, alt_aa, c_site)
                        if np.isfinite(rate):
                            mutation_rate = 2.0 if (alt_nuc, ref_nuc) in transitions else 1.0
                            dico_opp_sp[cat_snps.rate2cat(rate)] += mutation_rate
                            list_rates.append(rate)
                            list_mut_rates.append(mutation_rate)
    tot_opp = sum(dico_opp_sp.values())
    for cat in cat_snps.non_syn():
        output_dict[cat].append(dico_opp_sp[cat] / tot_opp)
    output_dict["S_mean"].append(np.mean([mu * r for mu, r in zip(list_mut_rates, list_rates)]))
    if len(log_fitness) > 0:
        output_dict["log_fitness"].append(np.mean(log_fitness))
    if args.method != "SIFT":
        output_dict["flow_pos"].append(sum([mu * pfix(r) for mu, r in zip(list_mut_rates, list_rates) if r > 0]))
        output_dict["flow_neg"].append(sum([mu * pfix(r) for mu, r in zip(list_mut_rates, list_rates) if r < 0]))
        w = [mu * pfix(r) for mu, r in zip(list_mut_rates, list_rates)]
        plot_histogram(list_rates, w, cat_snps, args.output_pdf.replace("DFE.", "DFlow."), xmax=10)
    plot_histogram(list_rates, list_mut_rates, cat_snps, args.output_pdf)

    df = pd.DataFrame(output_dict)
    df.to_csv(args.output_tsv, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_folder', required=True, type=str, dest="exp_folder", help="The experiment folder path")
    parser.add_argument('--fasta_pop', required=True, type=str, dest="fasta_pop", help="The fasta path")
    parser.add_argument('--method', required=True, type=str, dest="method", help="The method (MutSel or Omega)")
    parser.add_argument('--output_tsv', required=True, type=str, dest="output_tsv", help="Output tsv path")
    parser.add_argument('--output_pdf', required=True, type=str, dest="output_pdf", help="Output pdf path")
    main(parser.parse_args())
