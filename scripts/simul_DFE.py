import numpy as np
from random import choices
from libraries import open_file
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, add_help=False)
parser.add_argument("-d", "--dfe_file")
parser.add_argument("-s", "--sfs_file")
parser.add_argument("-f", "--fenetre")

args = parser.parse_args()


def do_sample(file, N):
    fichier = open_file(file, 'r')
    txt = fichier.readline().replace("\n", "").split(";")
    liste_S = []
    while txt != ['']:
        liste_S.append(txt)
        txt = fichier.readline().replace("\n", "").split(";")
        if len(txt) < 10:
            print(txt)
    fichier.close()
    sample = choices(liste_S, k=N)
    return [x for lst in sample for x in lst]


def density(S, x):
    return (1 - np.exp(-S * (1 - x))) / (x * (1 - x) * (1 - np.exp(-S)))


def build_mixed_dfes(dfe, fenetre):
    p_pospos = 0.721
    p_posneut = 0.279
    p_posneg = 1.4 * 10 ** (-5)
    p_negneg = 0.911
    p_negpos = 0.007
    p_negneut = 0.082
    p_neutneut = 0.579
    p_neutpos = 0.099
    p_neutneg = 0.322
    sample = choices(do_sample(dfe, 100), k=100000)
    pos_dfe = [s for s in sample if float(s) > 1]
    neutral_dfe = [s for s in sample if (float(s) < 1 and float(s) > -1)]
    neg_dfe = [s for s in sample if float(s) < -1]
    S_adapt = 10
    if fenetre == 'positive':
        mixed_pos_dfe = choices(pos_dfe, k=round(10000 * p_pospos)) + choices(neg_dfe,
                                                                              k=round(10000 * p_posneg)) + choices(
            neutral_dfe, k=round(10000 * p_posneut))
        return mixed_pos_dfe
    elif fenetre == 'negative':
        mixed_neg_dfe = round(10000 * p_negpos) * [S_adapt] + choices(neg_dfe, k=round(10000 * p_negneg)) + choices(
            neutral_dfe, k=round(10000 * p_negneut))
        return mixed_neg_dfe
    elif fenetre == 'neutral':
        mixed_neutral_dfe = round(10000 * p_neutpos) * [S_adapt] + choices(neg_dfe,
                                                                           k=round(10000 * p_neutneg)) + choices(
            neutral_dfe, k=round(10000 * p_neutneut))
        return mixed_neutral_dfe


def write_polyDFE_file(sfs, dfe):
    fichier = open(sfs, 'r')
    txt = fichier.readline().split()
    name = txt[0] + '_simulated'
    txt = fichier.readline()
    subname = txt
    L = int(txt.split()[2])
    txt = fichier.readline().split()
    Syn = np.asarray([int(x) for x in txt[:-1]]).sum()
    Syn_opp = txt[-1]
    txt = fichier.readline().split()
    Non_Syn = np.asarray([int(x) for x in txt[:-1]]).sum()
    Non_Syn_opp = txt[-1]
    fichier.close()
    sortie = open(sfs + '.simulated', 'w')
    sortie.write(name + '\n')
    sortie.write(subname)
    X = []
    Y1 = []
    Y2 = []
    for i in range(L - 1):
        X.append((i + 1) / L)
        Y1.append([])
        Y2.append([])
        for S in dfe:
            Y1[i].append(density(float(S), (i + 1) / L))
            Y2[i].append(1 / ((i + 1) / L))
        Y1[i] = np.asarray(Y1[i]).mean()
        Y2[i] = np.asarray(Y2[i]).mean()
        print(Y1[i])
    totY1 = np.asarray(Y1).sum()
    totY2 = np.asarray(Y2).sum()
    for i in range(len(Y1)):
        Y1[i] = Y1[i] / totY1
        Y2[i] = Y2[i] / totY2
    for i in range(L - 2):
        sortie.write(str(int(Y2[i] * Syn)) + ' ')
    sortie.write(str(int(Y2[-1] * Syn)) + '\t' + Syn_opp + '\n')
    for i in range(L - 2):
        sortie.write(str(int(Y1[i] * Non_Syn)) + ' ')
    sortie.write(str(int(Y1[-1] * Non_Syn)) + '\t' + Non_Syn_opp + '\n')
    return ()


dfe = build_mixed_dfes(args.dfe_file, args.fenetre)
write_polyDFE_file(args.sfs_file, dfe)
