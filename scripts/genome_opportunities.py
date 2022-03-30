import argparse
from matplotlib.patches import Rectangle
from libraries import *

transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}


class Stat:
    def __init__(self):
        self.total = 0.0
        self.n = 0.0

    def add(self, x, w=1.0):
        if np.isfinite(x) and np.isfinite(w):
            self.total += x * w
            self.n += w

    def mean(self):
        return self.total / self.n


def pfix(s):
    if s == 0.0:
        return 1.0
    else:
        return s / (1 - np.exp(-s))


def plot_histogram(counts, edges, cat_snps, dico_opp_sp, method, file):
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
    cats_list_edges = [cat_snps.rate2cats(b) for b in edges[1:]]
    colors = [cat_snps.color(cats[0]) if len(cats) > 0 else "black" for cats in cats_list_edges]
    plt.bar(edges[:-1], height=counts, color=colors, width=np.diff(edges), edgecolor="black", align="edge")
    if cat_snps.bins <= 10:
        handles = [Rectangle((0, 0), 1, 1, color=c) for c in [cat_snps.color(cat) for cat in cat_snps.non_syn_list]]
        labels = [cat_snps.label(cat) + f" ({dico_opp_sp[cat] * 100:.2f}% of total)" for cat in
                  cat_snps.non_syn_list]
        plt.legend(handles, labels)
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
    cat_snps = CategorySNP(args.method, args.bounds, bins=args.bins, windows=args.windows)

    bins = np.linspace(xlim_dico[args.method][0], xlim_dico[args.method][1], 61)
    counts_mu, counts_q = np.histogram([], bins=bins)[0], np.histogram([], bins=bins)[0]
    log_fitness, sel_coeff, flow_pos, flow_neg = Stat(), Stat(), Stat(), Stat()

    mask_grouped = open_mask(args.mask)
    unconserved_grouped = open_mask(args.unconserved)

    cds_rates = CdsRates(args.method, args.exp_folder)
    output_dict, dico_opp_sp = defaultdict(list), {cat: 0 for cat in cat_snps.non_syn_list}
    dico_opp_sp["OutOfBounds"], dico_opp_sp["Adaptive"] = 0, 0
    dico_flow_sp = dico_opp_sp.copy()
    tot_opp = 0

    seqs = open_fasta(args.fasta_pop)
    size = len(seqs) if args.subsample_genes < 1 else min(args.subsample_genes, len(seqs))
    np.random.seed(seed=0)
    rd_ensg_list = np.random.choice(list(seqs.keys()), size=size, replace=False)

    for ensg in rd_ensg_list:
        seq = seqs[ensg]
        list_rates, list_mu, list_q = [], [], []

        for c_site in range(len(seq) // 3):
            adaptive = ensg in mask_grouped and c_site in mask_grouped[ensg]

            ref_codon = seq[c_site * 3:c_site * 3 + 3]
            ref_aa = codontable[ref_codon]
            if ref_aa == "X" or ref_aa == "-":
                continue
            lf = cds_rates.log_fitness(ensg, ref_aa, c_site)
            if args.method == "MutSel":
                if (ensg not in unconserved_grouped) or (c_site not in unconserved_grouped[ensg]):
                    log_fitness.add(lf)

            for frame, ref_nuc in enumerate(ref_codon):
                for alt_nuc in [nuc for nuc in nucleotides if nuc != ref_nuc]:
                    alt_codon = ref_codon[:frame] + alt_nuc + ref_codon[frame + 1:]
                    alt_aa = codontable[alt_codon]
                    if alt_aa == 'X' or alt_aa == ref_aa:
                        continue

                    rate = cds_rates.rate(ensg, ref_aa, alt_aa, c_site)
                    if not np.isfinite(rate):
                        continue

                    mutation_rate = 2.0 if (alt_nuc, ref_nuc) in transitions else 1.0
                    tot_opp += mutation_rate

                    if adaptive:
                        dico_opp_sp["Adaptive"] += mutation_rate
                        continue

                    cats = cat_snps.rate2cats(rate)
                    if len(cats) == 0:
                        dico_opp_sp["OutOfBounds"] += mutation_rate
                        continue

                    for cat in cats:
                        dico_opp_sp[cat] += mutation_rate

                    list_rates.append(rate)
                    list_mu.append(mutation_rate)
                    if args.method != "MutSel":
                        continue

                    sel_coeff.add(rate, w=mutation_rate)
                    q = mutation_rate * pfix(rate)
                    list_q.append(q)
                    if np.isfinite(q):
                        for cat in cats:
                            dico_flow_sp[cat] += q

                    (flow_pos if rate > 0 else flow_neg).add(q)
        counts_mu = counts_mu + np.histogram(list_rates, bins=bins, weights=list_mu)[0]
        if args.method == "MutSel":
            counts_q = counts_q + np.histogram(list_rates, bins=bins, weights=list_q)[0]

    for cat in dico_opp_sp:
        dico_opp_sp[cat] = dico_opp_sp[cat] / tot_opp
        output_dict[cat].append(dico_opp_sp[cat])
        if args.method == "MutSel":
            dico_flow_sp[cat] = dico_flow_sp[cat] / (flow_neg.total + flow_pos.total)

    print(f'{output_dict["OutOfBounds"][0] * 100:.2f}% of opportunities out of bounds')
    print(f'{output_dict["Adaptive"][0] * 100:.2f}% of opportunities are discarded because their are adaptive.')

    if args.method == "MutSel":
        output_dict["S_mean"].append(sel_coeff.mean())
        output_dict["log_fitness"].append(log_fitness.mean())
        output_dict["flow_pos"].append(flow_pos.total)
        output_dict["flow_neg"].append(flow_neg.total)
        plot_histogram(counts_q, bins, cat_snps, dico_flow_sp, args.method, args.output.replace(".tsv", ".Flow.pdf"))

    plot_histogram(counts_mu, bins, cat_snps, dico_opp_sp, args.method, args.output.replace(".tsv", ".pdf"))
    df = pd.DataFrame(output_dict)
    df.to_csv(args.output, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_folder', required=True, type=str, dest="exp_folder", help="The experiment folder path")
    parser.add_argument('--fasta_pop', required=True, type=str, dest="fasta_pop", help="The fasta path")
    parser.add_argument('--bounds', required=False, default="", type=str, dest="bounds", help="Input bound file path")
    parser.add_argument('--mask', required=False, default="", type=str, dest="mask", help="Input mask file path")
    parser.add_argument('--unconserved', required=False, default="", type=str, dest="unconserved",
                        help="Input unconserved file path")
    parser.add_argument('--method', required=True, type=str, dest="method", help="The method (MutSel or Omega)")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output tsv path")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    parser.add_argument('--windows', required=False, default=0, type=int, dest="windows", help="Number of windows")
    parser.add_argument('--subsample_genes', required=False, default=-1, type=int, dest="subsample_genes",
                        help="Number of genes to subsample")
    main(parser.parse_args())
