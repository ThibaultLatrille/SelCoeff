import argparse
import pandas as pd
from libraries import open_fasta, write_fasta, translate_cds
import subprocess


def mask_seq(seq, mask):
    return "".join([seq[pos] for pos in mask])


def main(args):
    # Open fasta file containing all CDS for the specific pop
    cds_dico = open_fasta(args.fasta_pop)

    # For each CDS retrieve the alignment in fasta folder
    for ensg, seq in cds_dico.items():
        # Change the species sequence by the updated population sequence (with fixed poly)
        # Export alignment in fasta (Remove empty sites in focal pop? -> Map the sites)
        # Convert to amino acids
        focal_seq = translate_cds(seq)

        mask, mask_pos, mask_tmp_pos = [], [], 0
        for pos, aa in enumerate(focal_seq):
            if aa != "-":
                mask.append(pos)
                mask_pos.append(mask_tmp_pos)
                mask_tmp_pos += 1
            else:
                mask_pos.append("NaN")

        convert_path = f"{args.output}/{ensg}.mask.tsv"
        df = pd.DataFrame({"pos": range(len(mask_pos)), "mask_pos": mask_pos})
        df.to_csv(convert_path, index=False, sep="\t")

        ensg_seqs = {args.species: mask_seq(focal_seq, mask)}
        fastas = open_fasta(f"{args.fasta_folder}/{ensg}_NT.fasta")
        ensg_seqs.update({sp: mask_seq(translate_cds(s), mask) for sp, s in fastas.items() if sp != args.species})

        ensg_path = f"{args.output}/{ensg}.fasta"
        write_fasta(ensg_seqs, ensg_path)

        sift_path = f"{args.output}/{ensg}.SIFTprediction"
        # Run sift for each alignment.
        sift_cmd = f"export BLIMPS_DIR={args.blimps_dir} && {args.sift_exec} {ensg_path} - {sift_path}"
        output = subprocess.check_output(sift_cmd, shell=True)
        stdout = f"{args.output}/{ensg}.stdout"
        f = open(stdout, 'wb')
        f.write(output)
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--species', required=True, type=str, dest="species", help="Focal species")
    parser.add_argument('--fasta_folder', required=True, type=str, dest="fasta_folder", help="The fasta folder path")
    parser.add_argument('--fasta_pop', required=True, type=str, dest="fasta_pop", help="The fasta population path")
    parser.add_argument('--sift_exec', required=True, type=str, dest="sift_exec", help="The sift_exec path")
    parser.add_argument('--blimps_dir', required=True, type=str, dest="blimps_dir", help="The blimps directory path")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output path")
    main(parser.parse_args())
