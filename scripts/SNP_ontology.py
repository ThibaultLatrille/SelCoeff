import os
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.stats as st
from libraries import CategorySNP, merge_mask_list, open_fasta, clean_ensg_path
from table_ontology import ontology_table
from SNP_effect import adjusted_holm_pval, format_pval_df, tex_f
from write_variations import masked_sub, cat_sub
import matplotlib.pyplot as plt

# LaTeX preamble


preamble_list = [r'\documentclass{article}', r'\usepackage{lmodern}', r'\usepackage[export]{adjustbox}',
                 r'\usepackage{tabu}', r'\tabulinesep=0.6mm', r'\newcommand\cellwidth{\TX@col@width}',
                 r'\usepackage{hhline}', r'\setlength{\arrayrulewidth}{1.2pt}', r'\usepackage{multicol,multirow,array}',
                 r'\usepackage{etoolbox}', r'\usepackage{bm}', r'\AtBeginEnvironment{tabu}{\footnotesize}',
                 r'\usepackage{booktabs}', r'\usepackage[margin=40pt]{geometry}', r'\usepackage{longtable}']


def trim_sentence(sentence: str, n: int) -> str:
    # Trim the sentence to n characters, split the sentence at the last space and add "..."
    if len(sentence) > n:
        sentence = sentence[:n]
        sentence = sentence[:sentence.rfind(" ")] + "..."
    return sentence


def n_sites_ensg(fasta_folder, mask_grouped):
    dico_n_sites = {}
    for file in os.listdir(fasta_folder):
        if file.endswith(".fasta") and file.startswith("ENSG"):
            fasta = open_fasta(clean_ensg_path(os.path.join(fasta_folder, file)))
            ensg = file[:-9]
            if ensg.endswith("_"):
                ensg = ensg + "null"
            n_sites = len(fasta[list(fasta.keys())[0]]) // 3
            if ensg in mask_grouped:
                n_sites -= len(mask_grouped[ensg])
            dico_n_sites[ensg] = n_sites
    return dico_n_sites


def open_df(input_path, mask_list, bins, windows, bounds):
    cat_snps = CategorySNP("MutSel", bounds, bins=bins, windows=windows)
    mask_grouped = merge_mask_list(mask_list)
    df = pd.read_csv(input_path, sep='\t')
    df["MASKED"] = df.apply(lambda row: masked_sub(row, mask_grouped), axis=1)
    n_masked = len(df[df.MASKED])
    print(f"Number of masked sites: {n_masked} out of {len(df)} ({n_masked / len(df) * 100:.2f}%)")
    df = df[~df.MASKED].copy()
    df["CAT"] = df.apply(lambda row: cat_sub(row, cat_snps), axis=1)
    df_non_syn = df[df.SUB_TYPE != "Syn"]
    df_pos = df_non_syn[df_non_syn["CAT"] == "pos"]
    print(f"Number of snps: {len(df)}")
    print(f"Number of B0 snps: {len(df_pos)}")
    return df, df_pos, mask_grouped


def plot_histogram(cds_p_dico, output_hist):
    proportion = np.array(list(cds_p_dico.values()))
    # find the p for which 99.9% of the data is below
    p_positives = proportion[proportion > 0]
    p_threshold = np.percentile(p_positives, 99.5)
    print(f"Threshold: {p_threshold}")
    # Clip the data to this threshold
    p_hist = np.clip(p_positives, 0, p_threshold)
    plt.hist(p_hist, bins=30, alpha=0.5,
             label=f"{(len(p_hist) / len(proportion) * 100):.2f}% of genes with non-zero value")
    plt.xlabel("Proportion of sites that are $\\mathcal{B}_0$")
    plt.ylabel("Number of genes")
    plt.legend()
    plt.savefig(output_hist)
    plt.close("all")
    plt.clf()


def plot_histogram_pvalues(p_hist, output_hist):
    plt.hist(p_hist, bins=30, alpha=0.5)
    plt.xlabel("p-value of the Mann-Whitney U test")
    plt.ylabel("Number of ontology terms")
    plt.savefig(output_hist)
    plt.close("all")
    plt.clf()


def ontology_enrichment(cds_p_dico, cds_n_dico, go_id2cds_list, go_id2name, set_all_go_cds, output_tsv):
    dico_ouput = defaultdict(list)
    plot_histogram(cds_p_dico, output_tsv.replace(".tsv", ".hist.pdf"))
    for go_id, go_cds_set in go_id2cds_list.items():
        x = [cds_p_dico[cds] for cds in go_cds_set if cds in cds_p_dico]
        no_go_cds_set = set_all_go_cds - go_cds_set
        y = [cds_p_dico[cds] for cds in no_go_cds_set if cds in cds_p_dico]
        n_go = np.sum([cds_n_dico[cds] for cds in go_cds_set if cds in cds_n_dico])
        p = n_go / (n_go + np.sum([cds_n_dico[cds] for cds in no_go_cds_set if cds in cds_n_dico]))
        if len(x) > 30 and len(y) > 30 and n_go > 0:
            statistic, pval = st.mannwhitneyu(x, y, alternative='two-sided')
            go_term = go_id2name[go_id].replace("_", "-").replace("[", "").replace("]", "")
            # Cut the name if too long
            go_term = trim_sentence(go_term, 50)
            dico_ouput["GOid"].append(go_id)
            dico_ouput["GO"].append(go_term)
            dico_ouput["p"].append(p)
            dico_ouput["r"].append(np.mean(x) / np.mean(y))
            dico_ouput["U"].append(statistic)
            dico_ouput["pval"].append(pval)

    header = ["GO id", "GO name", "p", "r", "Mann-Whitney U", "pval", "pvalHolm"]
    df_onto = pd.DataFrame(dico_ouput)
    df_onto = adjusted_holm_pval(df_onto, alpha=0.05, format_p=False)
    df_onto.sort_values(by=["pval_adj", "pval"], inplace=True)
    df_onto.to_csv(output_tsv, index=False, sep="\t")
    plot_histogram_pvalues(df_onto["pval"], output_tsv.replace(".tsv", ".hist_pval.pdf"))

    text_core = f"{len(df_onto['pval'])} tests performed."
    text_core += "\\tiny\n"
    df_head = format_pval_df(df_onto).head(100)
    text_core += df_head.to_latex(index=False, escape=False, longtable=True, column_format="|l|r|r|r|r|r|r|",
                                  header=header, float_format=tex_f)

    output_tex = output_tsv.replace(".tsv", ".tex")
    with open(output_tex, "w") as table_tex:
        table_tex.write("\n".join(preamble_list) + "\n")
        table_tex.write("\\begin{document}\n")
        table_tex.write(text_core)
        table_tex.write("\\end{document}\n")

    print('Table generated')
    tex_to_pdf = "pdflatex -synctex=1 -interaction=nonstopmode -output-directory={0} {1}".format(
        os.path.dirname(output_tex), output_tex)
    os.system(tex_to_pdf)
    os.system(tex_to_pdf)
    os.system(tex_to_pdf)
    print('Pdf generated')
    print('Ontology computed')


def main(input_path, mask_list, bins, windows, bounds, xml_folder, fasta_folder, species, output_tsv):
    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)

    df, df_pos, mask_grouped = open_df(input_path, mask_list, bins, windows, bounds)
    n_sites_dico = n_sites_ensg(fasta_folder, mask_grouped)

    fasta_list_ensg = set(n_sites_dico.keys())
    xml_list_ensg = set([i[:-4] for i in os.listdir(xml_folder)])
    df_list_ensg = set(df["ENSG"])
    print(f"Number of ENSG in the xml folder: {len(xml_list_ensg)}")
    print(f"Number of ENSG in the fasta folder: {len(fasta_list_ensg)}")
    print(f"Number of ENSG in the xml/fasta intersection: {len(xml_list_ensg & fasta_list_ensg)}")
    print(f"Number of ENSG in the input file: {len(df_list_ensg)}")
    print(f"Number of ENSG in the xml/input intersection: {len(xml_list_ensg & df_list_ensg)}")
    print(f"Number of ENSG in the fasta/input intersection: {len(fasta_list_ensg & df_list_ensg)}")
    go_id2cds_list, go_id2name, set_all_go_cds = ontology_table(xml_folder, xml_list_ensg)
    pos_dico = {k: len(d) for k, d in df_pos.groupby("ENSG")}

    print(f"{len(pos_dico)} genes out of {len(n_sites_dico)} with at least one pos site detected.")
    cds_p_dico = {k: ((pos_dico[k] / n_sites_dico[k]) if k in pos_dico else 0.0) for k in set_all_go_cds}
    ontology_enrichment(cds_p_dico, pos_dico, go_id2cds_list, go_id2name, set_all_go_cds, output_tsv)

    n_dico = {k: len(d) for k, d in df.groupby("ENSG")}
    cds_p_dico = {k: ((pos_dico[k] / n_dico[k]) if k in pos_dico else 0.0) for k in n_dico}
    ontology_enrichment(cds_p_dico, pos_dico, go_id2cds_list, go_id2name, set_all_go_cds,
                        output_tsv.replace(".tsv", "_all.tsv"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True, type=str, dest="input", help="Input vcf")
    parser.add_argument('--mask', required=False, default="", nargs="+", type=str, dest="mask")
    parser.add_argument('--mask_CpG', required=False, default=False, action="store_true", dest="mask_CpG",
                        help="Mask CpG opportunities")
    parser.add_argument('--bounds', required=True, default="", type=str, dest="bounds", help="Input bound file path")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    parser.add_argument('--windows', required=False, default=0, type=int, dest="windows", help="Number of windows")
    parser.add_argument('--xml_folder', required=True, type=str, dest="xml_folder", help="The xml folder path")
    parser.add_argument('--fasta_folder', required=True, type=str, dest="fasta_folder", help="The fasta folder path")

    parser.add_argument('--species', required=True, type=str, dest="species", help="The focal species")
    parser.add_argument('--output_tsv', required=True, type=str, dest="output_tsv", help="Output tsv")

    args = parser.parse_args()
    assert args.mask_CpG is False, "Mask CpG is not implemented for this script."
    main(args.input, args.mask, args.bins, args.windows, args.bounds, args.xml_folder, args.fasta_folder, args.species,
         args.output_tsv)
