import os
import gzip
import argparse
from collections import namedtuple, defaultdict
from libraries import open_fasta, write_fasta, nucleotides
from ete3 import Tree

SNP = namedtuple('SNP', ['pos', 'ref', 'anc', 'der', 'k', 'n', 'polarized'])


def list_poly_from_vcf(vcf):
    vcf_file = gzip.open(vcf, 'rt')
    header, list_poly = {}, defaultdict(list)
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

        polarized = dico_info["POLARIZED"] == "True"
        ensg = dico_info["ENSG"]
        pos = int(dico_info["ENSG_POS"])
        sample_size = int(dico_info["SAMPLE_SIZE"])
        k = int(dico_info["COUNT_POLARIZED"])
        nuc_ref = dico_info["ENSG_REF"]
        nuc_anc = dico_info["NUC_ANC"]
        nuc_der = dico_info["NUC_DER"]
        list_poly[ensg].append(SNP(pos, nuc_ref, nuc_anc, nuc_der, k, sample_size, polarized))

    vcf_file.close()
    return list_poly


def most_common(lst):
    return max(set(lst), key=lst.count)


class Outgroup(object):
    def __init__(self, alignment, specie, tree_path):

        if not os.path.exists(tree_path):
            tree_path = tree_path.replace("_null_", "__")

        t = Tree(tree_path)
        leaves = t.get_leaves_by_name(specie)
        assert len(leaves) == 1
        leaf = leaves[0]
        self.seqs = [list(), list(), list(), list()]
        self.names = [leaf.get_leaf_names(), list(), list(), list()]
        for gr in [1, 2, 3]:
            if leaf is not None and len(leaf.get_sisters()) > 0:
                self.names[gr] = leaf.get_sisters()[0].get_leaf_names()
            if leaf is not None:
                leaf = leaf.up
        for gr in [0, 1, 2, 3]:
            self.seqs[gr] = [alignment[group] for group in self.names[gr]]

    def position(self, cds_pos):
        codon_pos = cds_pos // 3
        out = []
        for out_seqs in self.seqs:
            states = [s[codon_pos * 3:codon_pos * 3 + 3] for s in out_seqs if s[cds_pos] in nucleotides]
            out.append("-" if len(states) == 0 else most_common(states))
        return out


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_fasta = {}
    # Open vcf file: fixed poly dict by ensg : (pos, anc, der)
    list_poly = list_poly_from_vcf(args.vcf)

    # Open fasta files, for each extract the species sequence
    # Update seqs with fixed poly dict
    files = sorted(os.listdir(args.fasta_folder))
    for fasta_file in files:
        ensg = fasta_file.replace("__", "_null_").replace("_NT.fasta", "")
        fasta_dict = open_fasta(f"{args.fasta_folder}/{fasta_file}")
        ali_folder = args.fasta_folder.replace("omm_NT_fasta.v10c_116", "omm_RooTree.v10b_116")
        tree_path = f"{ali_folder}/{ensg}_NT.rootree"
        if args.species in fasta_dict:
            seq = list(fasta_dict[args.species])
            # outgroup = Outgroup(fasta_dict, args.species, tree_path)
            for f in list_poly[ensg]:
                assert seq[f.pos] == f.ref
                if f.k == f.n:
                    seq[f.pos] = f.der
                else:
                    seq[f.pos] = f.anc
                # out_pos = outgroup.position(f.pos)
                # print(f"polarized:{f.polarized}. ref:{f.ref}. anc:{f.anc}. der:{f.der}. k:{f.k}. n:{f.n}", out_pos)

            output_fasta[ensg] = "".join(seq)

    # Put the seqs per ensg in a single fasta file
    write_fasta(output_fasta, args.output)
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vcf', required=True, type=str, dest="vcf", help="The vcf file for a population")
    parser.add_argument('--fasta_folder', required=True, type=str, dest="fasta_folder", help="The fasta folder path")
    parser.add_argument('--species', required=True, type=str, dest="species", help="The focal species")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output path")
    main(parser.parse_args())
