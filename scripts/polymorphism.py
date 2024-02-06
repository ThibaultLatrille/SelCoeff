import os
import gzip
import argparse
import pandas as pd
from collections import defaultdict
from substitutions import build_dict_trID


def read_vcf(vcf, anc_proba, trid):
    vcf_file = gzip.open(vcf, 'rt')
    dico_snp = defaultdict(list)
    filtered = 0
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

        if float(dico_info["ANC_PROBA"]) < anc_proba:
            filtered += 1
            continue

        sample_size = int(dico_info["SAMPLE_SIZE"])
        k = int(dico_info["COUNT_POLARIZED"])
        if k == 0 or k == sample_size:
            continue
        s_mutsel = float(dico_info["SEL_COEFF"]) if "SEL_COEFF" in dico_info else 0.0

        dico_snp["ENSG"].append(ensg)
        dico_snp["TR_ID"].append(trid[ensg])
        dico_snp["ENSG_POS"].append(dico_info["ENSG_POS"])
        dico_snp["NUC_ANC"].append(dico_info["NUC_ANC"])
        dico_snp["NUC_DER"].append(dico_info["NUC_DER"])
        dico_snp["CODON_SITE"].append(c_site)
        dico_snp["CODON_ANC"].append(dico_info["CODON_ANC"])
        dico_snp["CODON_DER"].append(dico_info["CODON_DER"])
        dico_snp["AA_ANC"].append(dico_info["AA_ANC"])
        dico_snp["AA_DER"].append(dico_info["AA_DER"])
        dico_snp["SUB_TYPE"].append(dico_info["SNP_TYPE"])
        dico_snp["SAMPLE_SIZE"].append(sample_size)
        dico_snp["COUNT"].append(k)
        dico_snp["FREQ"].append(k / sample_size)
        dico_snp["SEL_COEFF"].append(s_mutsel)
    vcf_file.close()
    return dico_snp, filtered


def main(args):
    assert 0 <= args.anc_proba <= 1
    os.makedirs(os.path.dirname(args.output_tsv), exist_ok=True)
    trid = build_dict_trID(args.xml_folder, args.species)

    dico_snp, filtered = read_vcf(args.vcf, args.anc_proba, trid)
    total = len(dico_snp["SUB_TYPE"]) + filtered
    print(f'{filtered * 100 / total:.2f}% of SNPs are discarded because they are filtered.')
    print(f'{len(dico_snp["SUB_TYPE"])} SNPs are kept ({len(dico_snp["SUB_TYPE"]) * 100 / total:.2f}%).')

    df = pd.DataFrame(dico_snp)
    df.to_csv(args.output_tsv, sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vcf', required=False, type=str, dest="vcf", help="Input vcf file")
    parser.add_argument('--xml_folder', required=True, type=str, dest="xml_folder", help="The xml folder path")
    parser.add_argument('--anc_proba', required=True, type=float, dest="anc_proba", default=0.5,
                        help="Mask the dico_snp with reconstruction probability lower than this threshold")
    parser.add_argument('--species', required=True, type=str, dest="species", help="The focal species")
    parser.add_argument('--output_tsv', required=False, type=str, dest="output_tsv", help="Output tsv file")
    main(parser.parse_args())
