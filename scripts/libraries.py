#!/usr/bin/env python3
from collections import namedtuple
import numpy as np
from math import floor

confidence_interval = namedtuple('confidence_interval', ['low', 'mean', 'up'])
sfs_weight = {"watterson": lambda i, n: 1.0 / i, "tajima": lambda i, n: n - i, "fay_wu": lambda i, n: i}


def theta(sfs_epsilon, daf_n, weight_method):
    sfs_theta = sfs_epsilon * np.array(range(1, daf_n))
    weights = np.array([sfs_weight[weight_method](i, daf_n) for i in range(1, daf_n)])
    return sum(sfs_theta * weights) / sum(weights)


def write_dofe(sfs_syn, sfs_non_syn, l_non_syn, d_non_syn, l_syn, d_syn, k, filepath, sp_focal, sp_sister, is_unfolded,
               L):
    sfs_list = [k]
    for sfs, nbr_site in [(sfs_non_syn, l_non_syn), (sfs_syn, l_syn)]:
        if is_unfolded:
            sfs_list += [nbr_site] + [sfs[i] for i in range(1, k)]
        else:
            range_sfs = range(1, int(floor(k // 2)) + 1)
            assert len(range_sfs) * 2 == k
            sfs_list += [nbr_site] + [(sfs[i] + sfs[k - i]) if k - i != i else sfs[i] for i in range_sfs]
    sfs_list += [l_non_syn, d_non_syn, l_syn, d_syn]

    dofe_file = open(filepath + ".dofe", 'w')
    dofe_file.write(f"{sp_focal}+{sp_sister} ({int(L)} sites)\n")
    if is_unfolded:
        dofe_file.write("#unfolded\n")
    dofe_file.write("Summed\t" + "\t".join(map(lambda i: str(int(i)), sfs_list)) + "\n")
    dofe_file.close()


def write_sfs(sfs_syn, sfs_non_syn, l_non_syn, d_non_syn, l_syn, d_syn, k, filepath, sp_focal, sp_sister, div=True):
    sfs_syn_str = " ".join([str(int(sfs_syn[i])) for i in range(1, k)]) + f"\t{int(l_syn)}"
    sfs_non_syn_str = " ".join([str(int(sfs_non_syn[i])) for i in range(1, k)]) + f"\t{int(l_non_syn)}"
    if div:
        sfs_syn_str += f"\t{int(d_syn)}\t{int(l_syn)}\n"
        sfs_non_syn_str += f"\t{int(d_non_syn)}\t{int(l_non_syn)}\n"
    sfs_file = open(filepath + ".sfs", 'w')
    sfs_file.write(f"#{sp_focal}+{sp_sister}\n")
    sfs_file.write("1 1 {0}".format(k) + "\n")
    sfs_file.write(sfs_syn_str + "\n")
    sfs_file.write(sfs_non_syn_str + "\n")
    sfs_file.close()
