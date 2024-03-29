import os
import gzip
import argparse
import numpy as np
from collections import defaultdict
from libraries import merge_mask_list, plt, my_dpi, format_pop
from upsetplot import UpSet, from_contents


def read_vcf(vcf, mask_grouped, anc_proba, subsample):
    vcf_file = gzip.open(vcf, 'rt')
    set_snp, sample_sizes = set(), set()
    discarded, filtered = 0, 0
    header = {}
    for vcf_line in vcf_file:
        if vcf_line[0] == '#':
            if vcf_line[1] != '#':
                line_strip = vcf_line.strip()
                header = {k: i for i, k in enumerate(line_strip.split("\t"))}
            continue
        assert len(header) > 0

        split_line = vcf_line.strip().split("\t")
        info = str(split_line[header["INFO"]])
        dico_info = {k: v for k, v in [s.split('=') for s in info.split(';') if '=' in s]}
        ensg = dico_info["ENSG"]
        c_site = int(dico_info["ENSG_POS"]) // 3

        if ensg in mask_grouped and c_site in mask_grouped[ensg]:
            discarded += 1
            continue

        if float(dico_info["ANC_PROBA"]) < anc_proba:
            filtered += 1
            continue

        sample_size = int(dico_info["SAMPLE_SIZE"])
        k = int(dico_info["COUNT_POLARIZED"])
        max_daf = min(sample_size, subsample)
        sampled_k = np.random.hypergeometric(k, sample_size - k, max_daf)
        if sampled_k == 0 or k == max_daf:
            continue

        if dico_info["SNP_TYPE"] != "NonSyn" and float(dico_info["SEL_COEFF"]) < 1.0:
            continue

        sample_sizes.add(sample_size)
        set_snp.add((ensg, int(dico_info["ENSG_POS"]), dico_info["AA_ANC"], dico_info["AA_ALT"]))
    vcf_file.close()
    assert len(sample_sizes) == 1
    return set_snp, sample_sizes.pop()


def plot_upset(dico_snps, output):
    plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    plt.subplot(1, 1, 1)
    df_pop = from_contents(dico_snps)
    UpSet(df_pop, subset_size='count', show_counts=True, min_subset_size=10, sort_by="cardinality").plot()
    plt.savefig(output, format="pdf")
    plt.clf()
    plt.close("all")


def main(vcf_list, mask_list, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    mask_grouped = merge_mask_list(mask_list)

    dico_pop, dico_sp = defaultdict(dict), defaultdict(set)
    for vcf_file in sorted(vcf_list):
        sp, pop = os.path.basename(vcf_file).replace("_", " ").replace(".vcf.gz", "").split(".")
        print(sp, pop)
        set_snps, sample_size = read_vcf(vcf_file, mask_grouped, args.anc_proba, args.subsample)
        dico_pop[sp][f"{format_pop(pop)} (k={sample_size})"] = set_snps
        dico_sp[sp].update(set_snps)

    plot_upset(dico_sp, output)
    for sp in dico_pop:
        if len(dico_pop[sp]) > 1:
            plot_upset(dico_pop[sp], output.replace(".pdf", f".{sp}.pdf"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--vcf', required=False, type=str, nargs="+", dest="vcf")
    parser.add_argument('--subsample', required=False, type=int, default=16, dest="subsample")
    parser.add_argument('--anc_proba', required=True, type=float, dest="anc_proba", default=0.5,
                        help="Mask the dico_snp with reconstruction probability lower than this threshold")
    parser.add_argument('--mask', required=False, default="", nargs="+", type=str, dest="mask",
                        help="List of input mask file path")
    parser.add_argument('--mask_CpG', required=False, default=False, action="store_true", dest="mask_CpG",
                        help="Mask CpG opportunities")
    parser.add_argument('-o', '--output', required=False, type=str, dest="output", help="Output pdf")
    args = parser.parse_args()
    assert args.mask_CpG is False, "Mask CpG is not implemented for this script."
    main(args.vcf, args.mask, args.output)
