import os
import gzip
import argparse
from collections import defaultdict
from libraries import open_mask, plt, my_dpi, format_pop
from upsetplot import UpSet, from_contents


def read_vcf(vcf, masks):
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
        masked = False
        for mask_grouped in masks:
            if ensg in mask_grouped and c_site in mask_grouped[ensg]:
                masked = True
                break

        if masked:
            discarded += 1
            continue

        if float(dico_info["ANC_PROBA"]) < 0.95:
            filtered += 1
            continue

        sample_size = int(dico_info["SAMPLE_SIZE"])
        k = int(dico_info["COUNT_POLARIZED"])
        if k == 0 or k == sample_size:
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

    masks = []
    for mask_file in mask_list:
        assert os.path.isfile(mask_file)
        masks.append(open_mask(mask_file))

    dico_pop, dico_sp = defaultdict(dict), defaultdict(set)
    for vcf_file in sorted(vcf_list):
        sp, pop = os.path.basename(vcf_file).replace("_", " ").replace(".vcf.gz", "").split(".")
        print(sp, pop)
        set_snps, sample_size = read_vcf(vcf_file, masks)
        dico_pop[sp][f"{format_pop(pop)} (k={sample_size})"] = set_snps
        dico_sp[sp].update(set_snps)

    plot_upset(dico_sp, output)
    for sp in dico_pop:
        if len(dico_pop[sp]) > 1:
            plot_upset(dico_pop[sp], output.replace(".pdf", f".{sp}.pdf"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--vcf', required=False, type=str, nargs="+", dest="vcf")
    parser.add_argument('--mask', required=False, default="", nargs="+", type=str, dest="mask",
                        help="List of input mask file path")
    parser.add_argument('-o', '--output', required=False, type=str, dest="output", help="Output pdf")
    args = parser.parse_args()
    main(args.vcf, args.mask, args.output)
