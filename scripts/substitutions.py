import argparse
from glob import glob
from ete3 import Tree
from libraries import *
from substitutions_mapping import open_prob


def build_dict_trID(xml_folder, specie):
    print('Converting TR_ID to ENSG.')
    from lxml import etree
    dico_trid = {}
    for file in os.listdir(xml_folder):
        root = etree.parse(xml_folder + "/" + file).getroot()
        for info in root.findall(".//infoCDS[@specy='{0}']".format(specie)):
            ensg = file.replace(".xml", "")
            assert ensg not in dico_trid
            trid = str(info.find('ensidTr').text)
            dico_trid[ensg] = trid
    print('TR_ID to ENSG conversion done.')
    return dico_trid


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    trid = build_dict_trID(args.xml_folder, args.species)
    cds_rates = CdsRates("MutSel", args.exp_folder)
    substitutions = defaultdict(list)

    for anc_file in glob(os.path.join(args.ancestral, "*.joint.fasta")):
        seqs = open_fasta(anc_file)
        ensg = os.path.basename(anc_file).split(".")[0]
        cds_rates.add_ensg(ensg)
        cds_rates.add_mut_ensg(ensg)

        tree_path = anc_file.replace(".joint.fasta", ".newick")
        t = Tree(tree_path, format=1)
        ancestor = t.get_leaves_by_name(args.species)[0].up.name
        anc_seq = seqs[ancestor]
        der_seq = seqs[args.species]
        assert len(anc_seq) == len(der_seq)

        proba_path = anc_file.replace(".joint.fasta", ".join.prob")
        prob_list = open_prob(proba_path)
        assert len(prob_list) == (len(anc_seq) // 3)

        for c_site in range(len(anc_seq) // 3):
            if prob_list[c_site] < args.threshold:
                continue

            anc_codon = anc_seq[c_site * 3:c_site * 3 + 3]
            der_codon = der_seq[c_site * 3:c_site * 3 + 3]
            if anc_codon == der_codon:
                continue

            anc_aa = codontable[anc_codon]
            der_aa = codontable[der_codon]
            if anc_aa == "X" or anc_aa == "-" or der_aa == "X" or der_aa == "-":
                continue

            diffs = [s for s in range(len(anc_codon)) if anc_codon[s] != der_codon[s]]
            assert len(diffs) > 0
            if len(diffs) != 1:
                continue

            rate = cds_rates.rate(ensg, anc_aa, der_aa, c_site)
            if not np.isfinite(rate):
                continue

            frame = diffs[0]
            substitutions["ENSG"].append(ensg)
            substitutions["TR_ID"].append(trid[ensg])
            substitutions["ENSG_POS"].append(c_site * 3 + frame)
            substitutions["NUC_ANC"].append(anc_codon[frame])
            substitutions["NUC_DER"].append(der_codon[frame])
            substitutions["CODON_SITE"].append(c_site)
            substitutions["CODON_ANC"].append(anc_codon)
            substitutions["CODON_DER"].append(der_codon)
            substitutions["AA_ANC"].append(anc_aa)
            substitutions["AA_DER"].append(der_aa)
            substitutions["SUB_TYPE"].append("Syn" if anc_aa == der_aa else "NonSyn")
            substitutions["SEL_COEFF"].append(rate)

        cds_rates.rm_ensg(ensg)

    df = pd.DataFrame(substitutions)
    df.to_csv(args.output, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ancestral', required=True, type=str, dest="ancestral", help="The ancestral folder")
    parser.add_argument('--exp_folder', required=True, type=str, dest="exp_folder", help="The experiment folder path")
    parser.add_argument('--xml_folder', required=True, type=str, dest="xml_folder", help="The xml folder path")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output tsv path")
    parser.add_argument('--species', required=True, type=str, dest="species", help="The focal species")
    parser.add_argument('--threshold', required=False, default=0.0, type=float, dest="threshold",
                        help="The threshold for the probability")
    main(parser.parse_args())
