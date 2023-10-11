import os
import gzip
import argparse
import numpy as np
import pandas as pd
import scipy.stats as st
from collections import defaultdict
from libraries import CategorySNP, tex_f, merge_mask_list


def read_effect_vcf(vcf, set_ids: set):
    print(f"Reading file {vcf}")
    keywords = ["benign", "pathogenic", "risk", 'protective']
    print("Keywords: {}".format(", ".join(keywords)))
    vcf_file = gzip.open(vcf, 'rt')
    dico_snps, header = defaultdict(set), {}
    for vcf_line in vcf_file:
        while vcf_line[0] == '#':
            if vcf_line[1] != '#':
                line_strip = vcf_line.strip()
                header = {k: i for i, k in enumerate(line_strip.split("\t"))}
            vcf_line = vcf_file.readline()
        assert len(header) > 0

        split_line = vcf_line.strip().split("\t")
        id_snp = (split_line[header["#CHROM"]], split_line[header["POS"]])
        if id_snp not in set_ids:
            continue

        info = str(split_line[header["INFO"]])
        print(info)
        if "CLIN_" not in info:
            continue

        for clin in [i for i in info.split(";") if "CLIN_" in i]:
            if sum([k in clin for k in keywords]) > 0:
                dico_snps[clin.replace("CLIN_", "").replace("_", " ")].add(id_snp)

    vcf_file.close()
    print("Number of SNPs: {}".format(sum([len(v) for v in dico_snps.values()])))
    for cat in dico_snps:
        print("{}: {}".format(cat, len(dico_snps[cat])))
    return dico_snps


def read_vcf(vcf, cat_snps: CategorySNP, mask_grouped):
    print(f"Reading file {vcf}")
    vcf_file = gzip.open(vcf, 'rt')
    dico_snps, header = defaultdict(set), {}
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
            continue

        if float(dico_info["ANC_PROBA"]) < 0.99:
            continue

        k = int(dico_info["COUNT_POLARIZED"])
        if k == 0 or k == int(dico_info["SAMPLE_SIZE"]):
            continue

        if dico_info["POLARIZED"] != "True":
            continue

        id_snp = (split_line[header["#CHROM"]], split_line[header["POS"]])
        if dico_info["SNP_TYPE"] == "Syn":
            assert float(dico_info["SEL_COEFF"]) == 0
            dico_snps["syn"].add(id_snp)
        else:
            cats = cat_snps.rate2cats(float(dico_info["SEL_COEFF"]))
            assert len(cats) in [1, 2]
            dico_snps[cats[0]].add(id_snp)
    vcf_file.close()
    print("Number of SNPs: {}".format(sum([len(v) for v in dico_snps.values()])))
    for cat in dico_snps:
        print("{}: {}".format(cat, len(dico_snps[cat])))
    return dico_snps


def format_pval_df(d, prefix="", alpha=0.05):
    col = prefix + "pval_adj"
    d[col] = d.apply(lambda r: tex_f(r[col], r[col] < alpha, "~~"), axis=1)
    return d


def adjusted_holm_pval(d, prefix="", alpha=0.05, format_p=True):
    n = len(d[prefix + "pval"])
    sorted_pval = sorted(zip(d[prefix + "pval"], d.index))
    sorted_adjpval = [[min(1, pval * (n - i)), p] for i, (pval, p) in enumerate(sorted_pval)]
    for i in range(1, len(sorted_adjpval)):
        if sorted_adjpval[i][0] <= sorted_adjpval[i - 1][0]:
            sorted_adjpval[i][0] = sorted_adjpval[i - 1][0]
    holm = {p: pval for pval, p in sorted_adjpval}
    d[prefix + "pval_adj"] = [holm[p] for p in d.index]
    if format_p:
        d = format_pval_df(d, prefix=prefix, alpha=alpha)
    return d


def main(input_vcf, input_effect, tex_source, output, mask_list):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    mask_grouped = merge_mask_list(mask_list)

    cat_snps = CategorySNP("MutSel", bins=3)
    dico_cat_snps = read_vcf(input_vcf, cat_snps, mask_grouped)
    set_ids = set().union(*dico_cat_snps.values())
    dico_clin_snps = read_effect_vcf(input_effect, set_ids)

    control_cat = "weak"
    dico_ouput = defaultdict(list)
    for clin_id in ['protective', "benign", "likely benign", "risk factor", "likely pathogenic", "pathogenic"]:
        if clin_id not in dico_clin_snps:
            continue
        clin_snps_set = dico_clin_snps[clin_id]
        syn_snps_clin_set = dico_cat_snps[control_cat] & clin_snps_set
        syn_snps_no_clin_set = dico_cat_snps[control_cat] - syn_snps_clin_set
        assert len(syn_snps_clin_set) + len(syn_snps_no_clin_set) == len(dico_cat_snps[control_cat])

        for cat in [cat for cat in cat_snps.non_syn_list if cat in dico_cat_snps and cat != control_cat]:
            cat_snps_clin_set = dico_cat_snps[cat] & clin_snps_set
            cat_snps_no_clin_set = dico_cat_snps[cat] - cat_snps_clin_set
            obs = np.array([[len(cat_snps_clin_set), len(cat_snps_no_clin_set)],
                            [len(syn_snps_clin_set), len(syn_snps_no_clin_set)]])

            assert sum(obs[0, :]) == len(dico_cat_snps[cat])
            assert sum(obs[1, :]) == len(dico_cat_snps[control_cat])
            assert np.sum(obs) == len(dico_cat_snps[control_cat].union(dico_cat_snps[cat]))

            if np.min(obs) > 1:
                oddsratio, pval = st.fisher_exact(obs, alternative='greater')
                dico_ouput["GO"].append(clin_id.capitalize())
                dico_ouput["cat"].append(cat)
                dico_ouput["obs"].append(int(obs[0, 0]))
                dico_ouput["exp"].append(float(obs[0, 0]) / oddsratio)
                dico_ouput["oddsratio"].append(oddsratio)
                dico_ouput["pval"].append(pval)

    df = pd.DataFrame(dico_ouput)
    df.to_csv(output, sep="\t", index=False)

    header = ["SNP clinical ontology", "$n_{\\mathrm{Observed}}$", "$n_{\\mathrm{Expected}}$", "Odds ratio",
              "$p_{\\mathrm{v}}$", "$p_{\\mathrm{v-adjusted}}$"]
    # Output as a latex table and compile
    with open(f"{os.path.dirname(output)}/results.tex", "w") as o:
        for cat, df_sub in df.groupby("cat"):
            o.write("\\begin{center}\n")
            o.write(f"{cat_snps.dico[cat].label}\\\\\n")
            df_sub = adjusted_holm_pval(df_sub, alpha=0.05, format_p=False)
            df_sub = format_pval_df(df_sub, alpha=0.05)
            columns = [c for c in df_sub if c != "cat"]
            df_out = df_sub[columns]
            df_out = df_out.rename(columns={k: v for k, v in zip(columns, header)})
            tex = df_out.to_latex(index=False, escape=False, float_format=tex_f, column_format="|l|r|r|r|r|r|")
            o.write(tex)
            o.write("\\end{center}\n")

    os.system(f"cp -f {tex_source} {os.path.dirname(output)}")
    tex_to_pdf = "pdflatex -synctex=1 -interaction=nonstopmode -output-directory={0} {0}/main-table.tex".format(
        os.path.dirname(output))
    os.system(tex_to_pdf)
    os.system(tex_to_pdf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--tex', required=True, type=str, dest="tex", help="The tex source file")
    parser.add_argument('-v', '--vcf', required=True, type=str, dest="vcf", help="Vcf file")
    parser.add_argument('-e', '--effect', required=True, type=str, dest="effect", help="The effect vcf file")
    parser.add_argument('-o', '--output', required=True, type=str, dest="output", help="Output path")
    parser.add_argument('--mask', required=False, default="", nargs="+", type=str, dest="mask",
                        help="List of input mask file path")
    args = parser.parse_args()
    main(args.vcf, args.effect, args.tex, args.output, args.mask)
