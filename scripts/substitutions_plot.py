import os
import argparse
import pandas as pd
from collections import defaultdict
from matplotlib.patches import Rectangle
from libraries import open_mask, plt, my_dpi, CategorySNP


def main(input_path, mask_list, output, bins, windows, bounds, opp):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    cat_snps = CategorySNP("MutSel", bounds, bins=bins, windows=windows)

    masks = []
    for mask_file in mask_list:
        assert os.path.isfile(mask_file)
        masks.append(open_mask(mask_file))

    df = pd.read_csv(input_path, sep='\t')
    dico_hist, dico_hist_not_masked = defaultdict(int), defaultdict(int)
    dico_div, dico_div_not_masked = defaultdict(float), defaultdict(float)

    def masked_sub(row):
        for mask_grouped in masks:
            if row.ENSG in mask_grouped and row.CODON_SITE in mask_grouped[row.ENSG]:
                return True
        return False

    def cat_sub(row):
        if row.SUB_TYPE == "Syn":
            return "syn"
        else:
            return cat_snps.rate2cats(row.SEL_COEFF)[0]

    for ensg, df in df.groupby("ENSG"):
        df["MASKED"] = df.apply(masked_sub, axis=1)
        df["CAT"] = df.apply(cat_sub, axis=1)

        dico_hist[len(df)] += 1
        mask_cds = len(df) > 75
        if not mask_cds:
            dico_hist_not_masked[len(~df['MASKED'])] += 1
        for cat, df_cat in df.groupby("CAT"):
            dico_div[cat] += len(df_cat)
            if not mask_cds:
                dico_div_not_masked[cat] += len(df_cat[~df_cat['MASKED']])

    # plot the histogram of counts
    df_opp = pd.read_csv(opp, sep="\t")
    cat_opp = {cat: df_opp[cat].values[0] for cat in df_opp.columns}
    print("cat_opp: ", cat_opp)
    plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    for div, hist in [(dico_div, dico_hist), (dico_div_not_masked, dico_hist_not_masked)]:
        plt.subplot(1, 2, 1 if div == dico_div else 2)
        plt.bar(hist.keys(), hist.values())
        plt.xlabel("Number of substitutions per gene")
        plt.ylabel("Number of genes")
        plt.yscale("log")
        plt.tight_layout()
        # Add dns/ds to the plot
        handles = [Rectangle((0, 0), 1, 1, color=c) for c in [cat_snps.color(cat) for cat in cat_snps.non_syn_list]]
        dnds = {cat: (div[cat] / (cat_opp[cat] * cat_opp["Ldn"])) / (div['syn'] / cat_opp["Lds"]) for cat in
                cat_snps.non_syn_list}
        print("div: ", div)
        print("dnds: ", dnds)
        labels = [cat_snps.label(cat) + f": dN/dS=${dnds[cat]:.2f}$" for cat in cat_snps.non_syn_list]
        plt.legend(handles, labels)
    plt.savefig(output, format="pdf")
    plt.clf()
    plt.close("all")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True, type=str, dest="input", help="Input vcf")
    parser.add_argument('--mask', required=False, default="", nargs="+", type=str, dest="mask")
    parser.add_argument('--opp', required=True, type=str, dest="opp", help="The opportunities file path")
    parser.add_argument('--bounds', required=True, default="", type=str, dest="bounds", help="Input bound file path")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    parser.add_argument('--windows', required=False, default=0, type=int, dest="windows", help="Number of windows")
    parser.add_argument('-o', '--output', required=False, type=str, dest="output", help="Output pdf")
    args = parser.parse_args()
    main(args.input, args.mask, args.output, args.bins, args.windows, args.bounds, args.opp)
