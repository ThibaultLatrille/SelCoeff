import os
import argparse
import gzip
import pandas as pd
from libraries import CdsRates, open_fasta
from collections import defaultdict, namedtuple

SNP = namedtuple('SNP', ['chr', 'pos', "ensg", "anc_aa", "der_aa", "c_site"])


def open_vcf(vcf):
    vcf_file = gzip.open(vcf, 'rt')
    header, snp_list = {}, []
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

        snp_chr = str(split_line[header["#CHROM"]])
        snp_pos = int(split_line[header["POS"]])
        anc_aa = dico_info["AA_ANC"]
        der_aa = dico_info["AA_DER"]
        c_site = int(dico_info["ENSG_POS"]) // 3
        ensg = dico_info["ENSG"]
        snp_list.append(SNP(snp_chr, snp_pos, ensg, anc_aa, der_aa, c_site))

    vcf_file.close()
    return snp_list


def main(args):
    all_snps = open_vcf(args.vcf)
    dict_ensg_sift = CdsRates("SIFT", args.sift_folder)
    dict_output = defaultdict(list)
    for snp in all_snps:
        if not os.path.exists(f"{args.sift_folder}/{snp.ensg}.fasta"):
            continue

        sift_score = dict_ensg_sift.rate(snp.ensg, snp.anc_aa, snp.der_aa, snp.c_site)
        dict_output["chr"].append(snp.chr)
        dict_output["pos"].append(snp.pos)
        dict_output["sift_score"].append(sift_score)

    df = pd.DataFrame(dict_output)
    df.to_csv(args.output, sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--species', required=True, type=str, dest="species", help="Focal species")
    parser.add_argument('--sift_folder', required=True, type=str, dest="sift_folder", help="The sift folder path")
    parser.add_argument('--vcf', required=False, type=str, dest="vcf", help="Input vcf file")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output path")
    main(parser.parse_args())
