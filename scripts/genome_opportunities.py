import argparse
from matplotlib.patches import Rectangle
from libraries import *


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


def plot_histogram(counts, edges, cat_snps, dico_opp, method, file):
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
    cats_list_edges = [cat_snps.rate2cats(b) for b in edges[1:]]
    colors = [cat_snps.color(cats[0]) if len(cats) > 0 else "black" for cats in cats_list_edges]
    plt.bar(edges[:-1], height=counts, color=colors, width=np.diff(edges), linewidth=1.0, edgecolor="black",
            align="edge")
    if cat_snps.bins <= 10:
        handles = [Rectangle((0, 0), 1, 1, color=c) for c in [cat_snps.color(cat) for cat in cat_snps.non_syn_list]]
        labels = [cat_snps.label(cat) + f" ({dico_opp[cat] * 100:.2f}% of total)" for cat in
                  cat_snps.non_syn_list]
        plt.legend(handles, labels)
    plt.xlabel(rate_dico[method])
    plt.ylabel("Density")
    for x in cat_snps.inner_bound:
        plt.axvline(x, color="grey", lw=1, ls='--')
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
    codon_neighbors = build_codon_neighbors()
    cat_snps = CategorySNP(args.method, args.bounds, bins=args.bins, windows=args.windows)

    bins = np.linspace(xlim_dico[args.method][0], xlim_dico[args.method][1], 61)
    counts_mu, counts_q = np.histogram([], bins=bins)[0], np.histogram([], bins=bins)[0]
    log_fitness, sel_coeff, flow_pos, flow_neg = Stat(), Stat(), Stat(), Stat()

    mask_grouped = merge_mask_list(args.mask)
    unconserved_grouped = open_mask(args.unconserved)

    cds_rates = CdsRates(args.method, args.exp_folder, args.sift_folder)
    dico_opp, dico_flow = defaultdict(float), defaultdict(float)
    for cat in cat_snps.non_syn_list:
        dico_opp[cat] = 0.0
    seqs = open_fasta(args.fasta_pop)
    size = len(seqs) if args.subsample_genes < 1 else min(args.subsample_genes, len(seqs))
    np.random.seed(seed=0)
    rd_ensg_list = np.random.choice(list(seqs.keys()), size=size, replace=False)

    outfile = open_file(f"{args.output.replace('.tsv', '.log.gz')}", "w")
    for ensg in rd_ensg_list:
        cds_rates.add_ensg(ensg)
        cds_rates.add_mut_ensg(ensg)
        seq = seqs[ensg]
        list_rates, list_mu, list_q = [], [], []

        for c_site in range(len(seq) // 3):
            ref_codon = seq[c_site * 3:c_site * 3 + 3]
            ref_aa = codontable[ref_codon]
            if ref_aa == "X" or ref_aa == "-":
                continue

            dico_opp["nTotal"] += 1.0

            if ensg in mask_grouped and c_site in mask_grouped[ensg]:
                dico_opp["nMasked"] += 1.0
                continue

            dico_opp["nNonMasked"] += 1.0

            if args.method == "MutSel":
                lf = cds_rates.log_fitness(ensg, ref_aa, c_site)
                if (ensg not in unconserved_grouped) or (c_site not in unconserved_grouped[ensg]):
                    log_fitness.add(lf)

            for (syn, anc_nuc, der_nuc, alt_codon, alt_aa, diff_pos) in codon_neighbors[ref_codon]:
                if args.mask_CpG:
                    pos = c_site * 3 + diff_pos
                    if pos == 0 or pos == len(seq) - 1:
                        continue
                    anc_context = seq[pos - 1:pos + 2]
                    der_context = anc_context[0] + der_nuc + anc_context[-1]
                    assert anc_context[1] == anc_nuc
                    assert len(anc_context) == 3
                    assert len(der_context) == 3
                    if "CG" in anc_context or "CG" in der_context:
                        dico_opp["nMasked"] += 1.0 / len(codon_neighbors[ref_codon])
                        continue

                mutation_rate = cds_rates.mutation_rate(ensg, anc_nuc, der_nuc)
                assert np.isfinite(mutation_rate)
                dico_opp["μTotal"] += mutation_rate

                if syn:
                    dico_opp["μSyn"] += mutation_rate
                    continue

                rate = cds_rates.rate(ensg, ref_aa, alt_aa, c_site)
                if not np.isfinite(rate):
                    dico_opp["μNotFinite"] += mutation_rate
                    continue

                dico_opp["μNonSyn"] += mutation_rate

                cats = cat_snps.rate2cats(rate)
                if len(cats) == 0:
                    dico_opp["μOut"] += mutation_rate
                    continue

                for cat in cats:
                    dico_opp[cat] += mutation_rate

                list_rates.append(rate)
                list_mu.append(mutation_rate)
                if args.method != "MutSel":
                    continue

                sel_coeff.add(rate, w=mutation_rate)
                q = mutation_rate * pfix(rate)
                list_q.append(q)
                if np.isfinite(q):
                    for cat in cats:
                        dico_flow[cat] += q
                    (flow_pos if rate > 0 else flow_neg).add(q)
        cds_rates.rm_ensg(ensg)

        counts_mu = counts_mu + np.histogram(list_rates, bins=bins, weights=list_mu)[0]
        if args.method == "MutSel":
            counts_q = counts_q + np.histogram(list_rates, bins=bins, weights=list_q)[0]
            # Output the list rate one rate per line
            if len(list_rates) > 0:
                outfile.write(";".join([str(rate) for rate in list_rates]) + "\n")

    outfile.close()

    for cat in cat_snps.non_syn_list:
        dico_opp[cat] /= dico_opp["μNonSyn"]
        if args.method == "MutSel":
            dico_flow[cat] = dico_flow[cat] / (flow_neg.total + flow_pos.total)

    dico_opp["Lds"] = dico_opp["nNonMasked"] * dico_opp["μSyn"] / dico_opp["μTotal"]
    dico_opp["Ldn"] = dico_opp["nNonMasked"] * dico_opp["μNonSyn"] / dico_opp["μTotal"]

    print(f'{dico_opp["μOut"] * 100 / dico_opp["μNonSyn"]:.2f}% of opportunities out of bounds')
    print(f'{dico_opp["nMasked"] * 100 / dico_opp["nTotal"]:.2f}% of sites are discarded because their are masked.')

    if args.method == "MutSel":
        dico_opp["SMean"] = sel_coeff.mean()
        dico_opp["logFitness"] = log_fitness.mean()
        dico_opp["flowPos"] = flow_pos.total
        dico_opp["flowNeg"] = flow_neg.total
        plot_histogram(counts_q, bins, cat_snps, dico_flow, args.method, args.output.replace(".tsv", ".Flow.pdf"))

    plot_histogram(counts_mu, bins, cat_snps, dico_opp, args.method, args.output.replace(".tsv", ".pdf"))
    df = pd.DataFrame({k: [v] for k, v in dico_opp.items()})
    df.to_csv(args.output, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_folder', required=True, type=str, dest="exp_folder", help="The experiment folder path")
    parser.add_argument('--sift_folder', required=False, default="", type=str, dest="sift_folder", help="SIFT path")
    parser.add_argument('--fasta_pop', required=True, type=str, dest="fasta_pop", help="The fasta path")
    parser.add_argument('--bounds', required=False, default="", type=str, dest="bounds", help="Input bound file path")
    parser.add_argument('--mask', required=False, default="", nargs="+", type=str, dest="mask",
                        help="List of input mask file path")
    parser.add_argument('--mask_CpG', required=False, default=False, action="store_true", dest="mask_CpG",
                        help="Mask CpG opportunities")
    parser.add_argument('--unconserved', required=False, default="", type=str, dest="unconserved",
                        help="Input unconserved file path")
    parser.add_argument('--method', required=True, type=str, dest="method", help="The method (MutSel or Omega)")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output tsv path")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    parser.add_argument('--windows', required=False, default=0, type=int, dest="windows", help="Number of windows")
    parser.add_argument('--subsample_genes', required=False, default=-1, type=int, dest="subsample_genes",
                        help="Number of genes to subsample")
    main(parser.parse_args())
