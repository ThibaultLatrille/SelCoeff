import numpy as np
from random import choices
from libraries import open_file
import argparse


def do_sample(file, k):
    fichier = open_file(file, 'r')
    liste_S = []
    for line in fichier:
        txt = line.strip().split(";")
        liste_S.append(txt)
        if len(txt) < 10:
            print(txt)
    fichier.close()
    sample = choices(liste_S, k=k)
    return [x for lst in sample for x in lst]


def density(s: float, x: float) -> float:
    return (1 - np.exp(-s * (1 - x))) / (x * (1 - x) * (1 - np.exp(-s)))


def build_mixed_dfes(dfe_path: str, fenetre: str, adapt: float = 10) -> list:
    p_pospos = 0.721
    p_posneut = 0.279
    p_posneg = 1.4 * 10 ** (-5)
    p_negneg = 0.911
    p_negpos = 0.007
    p_negneut = 0.082
    p_neutneut = 0.579
    p_neutpos = 0.099
    p_neutneg = 0.322
    sample = choices(do_sample(dfe_path, 100), k=100000)
    pos_dfe = [s for s in sample if float(s) > 1]
    neutral_dfe = [s for s in sample if (1 > float(s) > -1)]
    neg_dfe = [s for s in sample if float(s) < -1]
    n = 100000
    if fenetre == 'positive':
        mixed_pos_dfe = (choices(pos_dfe, k=round(n * p_pospos)) +
                         choices(neg_dfe, k=round(n * p_posneg)) +
                         choices(neutral_dfe, k=round(n * p_posneut)))
        return mixed_pos_dfe
    elif fenetre == 'negative':
        mixed_neg_dfe = (round(n * p_negpos) * [adapt] +
                         choices(neg_dfe, k=round(n * p_negneg)) +
                         choices(neutral_dfe, k=round(n * p_negneut)))
        return mixed_neg_dfe
    elif fenetre == 'neutral':
        mixed_neutral_dfe = (round(n * p_neutpos) * [adapt] +
                             choices(neg_dfe, k=round(n * p_neutneg)) +
                             choices(neutral_dfe, k=round(n * p_neutneut)))
        return mixed_neutral_dfe


def open_sfs(sfs_file: str) -> tuple:
    with open(sfs_file, 'r') as opened_sfs:
        header = opened_sfs.readline().split()
        name = header[0] + 'Simulated'
        data_signature = opened_sfs.readline()
        k = int(data_signature.split()[2])
        syn_sfs = opened_sfs.readline().split()
        syn = np.sum([int(x) for x in syn_sfs[:-1]])
        syn_opp = syn_sfs[-1]
        non_syn_sfs = opened_sfs.readline().split()
        non_syn = np.sum([int(x) for x in non_syn_sfs[:-1]])
        non_syn_opp = non_syn_sfs[-1]
    return name, data_signature, k, syn, syn_opp, non_syn, non_syn_opp


def write_sfs_file(output_path: str, name: str, data_signature: str, sfs_syn: list, sfs_non_syn: list):
    with open(output_path, 'w') as sortie:
        sortie.write(name + '\n')
        sortie.write(data_signature)
        sortie.write(' '.join([str(x) for x in sfs_syn]) + '\n')
        sortie.write(' '.join([str(x) for x in sfs_non_syn]) + '\n')


def simulate_sfs(sfs_path, dfe, output_path):
    name, data_signature, k, syn, syn_opp, non_syn, non_syn_opp = open_sfs(sfs_path)
    x = []
    y_sel = []
    y_neu = []
    for i in range(k - 1):
        x.append((i + 1) / k)
        y_sel.append([])
        y_neu.append([])
        for S in dfe:
            y_sel[i].append(density(float(S), (i + 1) / k))
            y_neu[i].append(1 / ((i + 1) / k))
        y_sel[i] = np.mean(y_sel[i])
        y_neu[i] = np.mean(y_neu[i])
        print(y_sel[i])
    tot_y_sel = np.sum(y_sel)
    tot_y_neu = np.sum(y_neu)
    for i in range(len(y_sel)):
        y_sel[i] = y_sel[i] / tot_y_sel
        y_neu[i] = y_neu[i] / tot_y_neu
    sfs_syn = [int(y_neu[i] * syn) for i in range(k - 2)] + [int(y_neu[-1] * syn), syn_opp]
    sfs_non_syn = [int(y_sel[i] * non_syn) for i in range(k - 2)] + [int(y_sel[-1] * non_syn), non_syn_opp]
    write_sfs_file(output_path, name, data_signature, sfs_syn, sfs_non_syn)


def main(dfe_file: str, sfs_file: str, fenetre: str, adapt: float, output_path: str):
    dfe = build_mixed_dfes(dfe_file, fenetre, adapt)
    simulate_sfs(sfs_file, dfe, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dfe_file', type=str, help='Path to the DFE file', required=True)
    parser.add_argument('-s', '--sfs_file', type=str, help='Path to the SFS file', required=True)
    parser.add_argument('-f', '--fenetre', type=str, help='Window to simulate', required=True)
    parser.add_argument("-a", "--adapt", type=float, help="Adaptation strength", required=False)
    parser.add_argument('-o', '--output', type=str, help='Path to the output file', required=True)
    args = parser.parse_args()
    main(args.dfe_file, args.sfs_file, args.fenetre, args.adapt, args.output)
