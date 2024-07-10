import numpy as np
from random import choices
import argparse
import gzip
import time


def open_file(path, rw="r"):
    return gzip.open(path, f'{rw}t') if path.endswith(".gz") else open(path, rw)


def do_sample(dfe_file, k):
    liste_per_cds = []
    with open_file(dfe_file, 'r') as opened_dfe:
        for line in opened_dfe:
            txt = line.strip().split(";")
            if len(txt) < 10:
                continue
            liste_per_cds.append(txt)
    sample = choices(liste_per_cds, k=k)
    out_list = list(map(float, [s for lst_s in sample for s in lst_s]))
    return out_list


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
    pos_dfe = [s for s in sample if s > 1]
    neutral_dfe = [s for s in sample if (1 > s > -1)]
    neg_dfe = [s for s in sample if s < -1]
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
    y_sel, y_neu = [0.0], [0.0]
    for i in range(1, k):
        x = float(i) / k
        y_neu.append(1 / x)
        d_list = [density(s, x) for s in dfe]
        y_sel.append(np.mean(d_list))
        print(f"SFS for {i}/{k}: syn={y_neu[-1]:.3f}, non_syn={y_sel[-1]:.3f}")
    y_neu = np.array(y_neu) * (syn / np.sum(y_neu))
    y_sel = np.array(y_sel) * (non_syn / np.sum(y_sel))
    sfs_syn = [int(y_neu[i]) for i in range(1, k)] + [syn_opp]
    sfs_non_syn = [int(y_sel[i]) for i in range(1, k)] + [non_syn_opp]
    write_sfs_file(output_path, name, data_signature, sfs_syn, sfs_non_syn)


def main(dfe_file: str, sfs_file: str, fenetre: str, adapt: float, output_path: str):
    # Count the amount of time spent in this function
    start = time.time()
    dfe = build_mixed_dfes(dfe_file, fenetre, adapt)
    print(f"Time spent building DFE: {time.time() - start:.2f}s")
    start = time.time()
    simulate_sfs(sfs_file, dfe, output_path)
    print(f"Time spent simulating SFS: {time.time() - start:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dfe_file', type=str, help='Path to the DFE file', required=True)
    parser.add_argument('-s', '--sfs_file', type=str, help='Path to the SFS file', required=True)
    parser.add_argument('-f', '--fenetre', type=str, help='Window to simulate', required=True)
    parser.add_argument("-a", "--adapt", type=float, help="Adaptation strength", required=False)
    parser.add_argument('-o', '--output', type=str, help='Path to the output file', required=True)
    args = parser.parse_args()
    main(args.dfe_file, args.sfs_file, args.fenetre, args.adapt, args.output)
