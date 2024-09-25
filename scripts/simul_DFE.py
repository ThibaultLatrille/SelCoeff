import os
import numpy as np
import argparse
import gzip
import time
import pandas as pd
import numba as nb
from functools import lru_cache
import scipy.integrate as integrate
import math


def open_file(path: str, rw: str):
    return gzip.open(path, f'{rw}t') if path.endswith(".gz") else open(path, rw)


def open_dfe(dfe_file: str):
    t = time.time()
    liste_per_cds = []
    with open_file(dfe_file, 'r') as opened_dfe:
        for line in opened_dfe:
            txt = line.strip().split(";")
            if len(txt) < 10:
                continue
            liste_per_cds.append(txt)
    print(f"Time spent in open_dfe: {time.time() - t:.2f}s")
    return np.asarray(liste_per_cds, dtype="object")


def do_sample(liste_per_cds: np.ndarray) -> np.ndarray:
    sample = np.random.choice(liste_per_cds, size=int(1e3))
    out_list = list(map(float, [s for lst_s in sample for s in lst_s]))
    sample = np.random.choice(out_list, size=int(1e6))
    return sample


def extract_data(filepath: str):
    df = pd.read_csv(filepath, sep="\t")
    row_neg0 = df[df["category"] == "neg"]
    row_weak0 = df[df["category"] == "weak"]
    row_pos0 = df[df["category"] == "pos"]
    return {"pos_neg0": row_neg0["P-Spos"].values[0],
            "pos_weak0": row_weak0["P-Spos"].values[0],
            "pos_pos0": row_pos0["P-Spos"].values[0],
            "neg_neg0": row_neg0["P-Sneg"].values[0],
            "neg_weak0": row_weak0["P-Sneg"].values[0],
            "neg_pos0": row_pos0["P-Sneg"].values[0],
            "weak_neg0": row_neg0["P-Sweak"].values[0],
            "weak_weak0": row_weak0["P-Sweak"].values[0],
            "weak_pos0": row_pos0["P-Sweak"].values[0]
            }


def build_mixed_dfes(sample: np.ndarray, fenetre: str, adapt: float = 10, params_file: str = "") -> np.ndarray:
    dp = extract_data(params_file)
    print(f"{len(sample)} S in the DFE sample")
    pos_dfe = sample[sample > 1]
    neutral_dfe = sample[np.abs(sample) <= 1]
    neg_dfe = sample[sample < -1]
    n = int(1e5)
    if fenetre == 'positive':
        mixed_pos_dfe = np.concatenate([np.random.choice(pos_dfe, size=round(n * dp["pos_pos0"])),
                                        np.random.choice(neg_dfe, size=round(n * dp["neg_pos0"])),
                                        np.random.choice(neutral_dfe, size=round(n * dp["weak_pos0"]))])
        return mixed_pos_dfe
    elif fenetre == 'negative':
        # pos_array = np.random.exponential(adapt, size=round(n * dp["pos_neg0"]))
        pos_array = np.array(round(n * dp["pos_neg0"]) * [adapt])
        mixed_neg_dfe = np.concatenate([pos_array,
                                        np.random.choice(neg_dfe, size=round(n * dp["neg_neg0"])),
                                        np.random.choice(neutral_dfe, size=round(n * dp["weak_neg0"]))])
        return mixed_neg_dfe
    elif fenetre == 'neutral':
        # pos_array = np.random.exponential(adapt, size=round(n * dp["pos_weak0"]))
        pos_array = np.array(round(n * dp["pos_weak0"]) * [adapt])
        mixed_neutral_dfe = np.concatenate([pos_array,
                                            np.random.choice(neg_dfe, size=round(n * dp["neg_weak0"])),
                                            np.random.choice(neutral_dfe, size=round(n * dp["weak_weak0"]))])
        return mixed_neutral_dfe


def open_sfs(sfs_file: str) -> tuple:
    with open(sfs_file, 'r') as opened_sfs:
        header = opened_sfs.readline().split()
        name = header[0] + 'Simulated'
        data_signature = opened_sfs.readline()
        n = int(data_signature.split()[2])
        syn_sfs = opened_sfs.readline().split()
        syn = np.sum([int(x) for x in syn_sfs[:-1]])
        syn_opp = syn_sfs[-1]
        non_syn_sfs = opened_sfs.readline().split()
        non_syn = np.sum([int(x) for x in non_syn_sfs[:-1]])
        non_syn_opp = non_syn_sfs[-1]
    return name, data_signature, n, syn, syn_opp, non_syn, non_syn_opp


def write_sfs_file(output_path: str, name: str, data_signature: str, sfs_syn: np.ndarray, sfs_non_syn: np.ndarray):
    with open(output_path, 'w') as sortie:
        sortie.write(name + '\n')
        sortie.write(data_signature)
        sortie.write(' '.join([str(int(float(x))) for x in sfs_syn]) + '\n')
        sortie.write(' '.join([str(int(float(x))) for x in sfs_non_syn]) + '\n')


@lru_cache(maxsize=None)
def comb(n: int, i: int) -> int:
    return math.comb(n, i)


@nb.jit(nopython=True)
def neutral_distribution(i: int, n: int, x: float) -> float:
    return np.power(x, i - 1) * np.power(1 - x, n - i)


@nb.jit(nopython=True)
def selection_distribution(i: int, n: int, s: float, x: float) -> float:
    if s == 0:
        return neutral_distribution(i, n, x)
    return (1 - np.exp(-s * (1 - x))) * np.power(x, i - 1) * np.power(1 - x, n - i - 1) / (1 - np.exp(-s))


def generate_sampled_sfs(dfe: np.ndarray, n: int, syn: int, non_syn: int) -> tuple[np.ndarray, np.ndarray]:
    y_neu, y_sel = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        f_i = comb(n, i) * integrate.quad(lambda x: neutral_distribution(i, n, x), 0, 1)[0]
        y_neu[i] = f_i

    for i in range(1, n):
        mean_fi = np.mean([integrate.quad(lambda x: selection_distribution(i, n, s, x), 0, 1)[0] for s in dfe])
        y_sel[i] = comb(n, i) * mean_fi

    y_neu = y_neu * (syn / np.sum(y_neu))
    y_sel = y_sel * (non_syn / np.sum(y_sel))
    for i in range(1, n):
        y_neu[i] = np.random.poisson(y_neu[i])
        y_sel[i] = np.random.poisson(y_sel[i])
    return y_neu[1:], y_sel[1:]


@nb.jit(nopython=True)
def density(s: float, x: float) -> float:
    return (1 - np.exp(-s * (1 - x))) / (x * (1 - x) * (1 - np.exp(-s)))


@nb.jit(nopython=True)
def generate_sfs(dfe: np.ndarray, k: int, syn: int, non_syn: int) -> tuple[np.ndarray, np.ndarray]:
    y_sel, y_neu = [], []
    for i in range(1, k):
        x = float(i) / k
        y_neu.append(1 / x)
        d_list = np.array([density(s, x) for s in dfe])
        y_sel.append(np.mean(d_list))
    y_neu = np.array(y_neu)
    y_neu = y_neu * (syn / np.sum(y_neu))
    y_sel = np.array(y_sel)
    y_sel = y_sel * (non_syn / np.sum(y_sel))
    return np.round(y_neu), np.round(y_sel)


def simulate_sfs(sfs_path, dfe, output_path):
    name, data_signature, n, syn, syn_opp, non_syn, non_syn_opp = open_sfs(sfs_path)
    t = time.time()
    sfs_syn, sfs_non_syn = generate_sampled_sfs(dfe, n, syn, non_syn)
    sfs_syn = np.append(sfs_syn, syn_opp)
    sfs_non_syn = np.append(sfs_non_syn, non_syn_opp)
    print(f"Time spent in simulate_sfs: {time.time() - t:.2f}s")
    write_sfs_file(output_path, name, data_signature, sfs_syn, sfs_non_syn)


def main(dfe_file: str, parsed_DFE_file: str, sfs_file: str, fenetre: str, adapt: float, seed: int, output_path: str):
    # Count the amount of time spent in this function
    # seed of the random number generator
    np.random.seed(seed)
    sample = do_sample(open_dfe(dfe_file))
    dfe = build_mixed_dfes(sample, fenetre, adapt, parsed_DFE_file)
    simulate_sfs(sfs_file, dfe, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dfe_file', type=str, help='Path to the DFE file', required=True)
    parser.add_argument('-p', '--parsed_DFE', type=str, help='Path to the parameter file', required=True)
    parser.add_argument('-s', '--sfs_file', type=str, help='Path to the SFS file', required=True)
    parser.add_argument('-f', '--fenetre', type=str, help='Window to simulate', required=True)
    parser.add_argument("-a", "--adapt", type=float, help="Adaptation strength", required=False)
    parser.add_argument('-r', '--seed', type=int, help='Seed for the random number generator', required=False)
    parser.add_argument('-o', '--output', type=str, help='Path to the output file', required=True)
    args = parser.parse_args()
    main(args.dfe_file, args.parsed_DFE, args.sfs_file, args.fenetre, args.adapt, args.seed, args.output)
