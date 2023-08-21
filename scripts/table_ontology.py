import os
import argparse
import pandas as pd
from lxml import etree
from collections import defaultdict


def ontology_table(xml_folder: str, set_ensg: set) -> (dict, dict, set):
    print('Finding CDS ontologies.')
    go_id2name, go_id2cds_list = {}, {}
    all_go_set = set()
    for file in os.listdir(xml_folder):
        root = etree.parse(xml_folder + "/" + file).getroot()
        for annot in root.find('goAnnot').findall("annot"):
            go_id = annot.find('goId').text
            go_name = annot.find('goName').text
            if go_id not in go_id2name:
                go_id2name[go_id] = go_name.replace('"', '')
            if go_id not in go_id2cds_list:
                go_id2cds_list[go_id] = set()
            ensg = file.replace(".xml", "")
            assert ensg in set_ensg
            go_id2cds_list[go_id].add(ensg)
            all_go_set.add(ensg)
    print('CDS ontologies found.')
    return go_id2cds_list, go_id2name, all_go_set


def main(folder: str, xml_folder: str, output: str):
    list_ensg = set([i[:-3] for i in os.listdir(folder)])
    go_id2cds_list, go_id2name, set_all_go_cds = ontology_table(xml_folder, list_ensg)

    # Create a dataframe with 4 columns: go_id, go_name, go_cds_count, go_cds
    output_dict = defaultdict(list)
    for go_id in go_id2cds_list:
        output_dict["go_id"].append(go_id)
        go_name = go_id2name[go_id].replace(' ', '-').replace("_", "-").replace("/", "-")
        output_dict["go_name"].append(go_name)
        output_dict["go_cds_count"].append(len(go_id2cds_list[go_id]))
        output_dict["go_cds"].append("; ".join(list(go_id2cds_list[go_id])))

    df = pd.DataFrame(output_dict)
    df = df.sort_values(by=['go_cds_count'], ascending=False)
    df.to_csv(output, sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--folder', required=True, type=str, dest="folder",
                        help="folder containing OrthoMam results")
    parser.add_argument('-x', '--xml', required=True, type=str, dest="xml", metavar="<xml>", help="The xml folder")
    parser.add_argument('-o', '--output', required=True, type=str, dest="output", help="Output path")
    args = parser.parse_args()
    main(args.folder, args.xml, args.output)
