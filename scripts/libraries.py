#!/usr/bin/env python3
import os
import re
import gzip
import numpy as np
import pandas as pd
from collections import namedtuple, defaultdict
import matplotlib

matplotlib.rcParams["font.family"] = ["Latin Modern Sans"]
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.cm import get_cmap

fontsize = 16
fontsize_legend = 14
my_dpi = 256
GREEN = "#8FB03E"
RED = "#EB6231"
YELLOW = "#E29D26"
BLUE = "#5D80B4"
LIGHTGREEN = "#6ABD9B"
RED_RGB = to_rgb(RED)
BLUE_RGB = to_rgb(BLUE)
GREEN_RGB = to_rgb(GREEN)
GREY_RGB = to_rgb("grey")
BLACK_RGB = to_rgb("black")

complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
nucleotides = list(sorted(complement.keys()))
codontable = defaultdict(lambda: "-")
codontable.update({
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': 'X', 'TAG': 'X',
    'TGC': 'C', 'TGT': 'C', 'TGA': 'X', 'TGG': 'W', '---': '-'})
confidence_interval = namedtuple('confidence_interval', ['low', 'mean', 'up'])
sfs_weight = {"watterson": lambda i, n: 1.0 / i, "tajima": lambda i, n: n - i, "fay_wu": lambda i, n: i}
polydfe_cat_dico = {"S": "$S^{pop}$", "P-Sinf0": "$p_{-}$", "P-Seq0": "$p_{neutral}$", "P-Ssup0": "$p_{+}$"}
polydfe_cat_list = list(polydfe_cat_dico.keys())
xlim_dico = {"Omega": (0.0, 2.0), "MutSel": (-10, 10), "SIFT": (0.0, 1.0)}
rate_dico = {"MutSel": "Scaled selection coefficient (S)",
             "Omega": "Rate of evolution ($\\omega$)",
             "SIFT": "SIFT score"}


def translate_cds(seq):
    return "".join([codontable[seq[codon_pos * 3:codon_pos * 3 + 3]] for codon_pos in range(len(seq) // 3)])


def open_fasta(path):
    if not os.path.isfile(path):
        path = path.replace("_null_", "__")
    assert os.path.isfile(path)

    outfile = {}
    ali_file = gzip.open(path, 'rt') if path.endswith(".gz") else open(path, 'r')
    for seq_id in ali_file:
        outfile[seq_id.replace('>', '').strip()] = ali_file.readline().strip()
    return outfile


def write_fasta(dico_fasta, output):
    outfile = gzip.open(output, 'wt') if output.endswith(".gz") else open(output, 'w')
    outfile.write("\n".join([f">{seq_id}\n{seq}" for seq_id, seq in dico_fasta.items()]))
    outfile.close()


class CdsRates(dict):
    def __init__(self, method, exp_folder):
        self.method = method
        assert self.method in ["Omega", "MutSel", "SIFT"], 'Method must be either "Omega", "MutSel" or "SIFT"'
        self.exp_folder = exp_folder
        super().__init__()

    def add_sift(self, ensg, f_path):
        mask_path = f"{f_path}.mask.tsv.gz"
        mask_file = gzip.open(mask_path, 'rt')
        mask_file.readline()
        mask_convert = {}
        for line in mask_file:
            pos, mask_pos = line.strip().split("\t")
            mask_convert[pos] = mask_pos
        mask_file.close()

        pred_path, header_line = f"{f_path}.SIFTprediction.gz", []
        sift_file = gzip.open(pred_path, 'rt')
        for line in sift_file:
            header_line = re.sub(' +', ' ', line.strip()).split(" ")
            if len(header_line) != 25:
                continue
            else:
                break
        self[ensg] = {k: [] for k in header_line}

        for pos, mask_pos in mask_convert.items():
            if mask_pos == "NaN":
                for col in header_line:
                    self[ensg][col].append(np.nan)
            else:
                strip_line = sift_file.readline().strip()
                line_split = re.sub(' +', ' ', strip_line).split(" ")
                for col, value in zip(header_line, line_split):
                    self[ensg][col].append(float(value))

        strip_line = sift_file.readline().strip()
        assert strip_line == "//"
        sift_file.close()

    def add_ensg(self, ensg):
        f_path = f"{self.exp_folder}/{ensg}"
        if self.method == "MutSel":
            path = f"{f_path}_NT/sitemutsel_1.run.siteprofiles"
            if not os.path.isfile(path):
                path = path.replace("_null_", "__")
            if not os.path.isfile(path):
                path = path.replace("__", "_null_")
            assert os.path.isfile(path)
            self[ensg] = pd.read_csv(path, sep="\t", skiprows=1, header=None,
                                     names="site,A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y".split(","))
        elif self.method == "Omega":
            path = f"{f_path}_NT/siteomega_1.run.ci0.025.tsv"
            if not os.path.isfile(path):
                path = path.replace("_null_", "__")
            if not os.path.isfile(path):
                path = path.replace("__", "_null_")
            assert os.path.isfile(path)
            self[ensg] = pd.read_csv(path, sep="\t")["gene_omega"].values[1:]
        elif self.method == "SIFT":
            self.add_sift(ensg, f_path)

    def log_fitness(self, ensg, ref_aa, c_site):
        if self.method == "MutSel":
            if ensg not in self:
                self.add_ensg(ensg)
            return np.log(self[ensg][ref_aa][c_site])
        else:
            return np.float("infinity")

    def rate(self, ensg, ref_aa, alt_aa, c_site):
        if ensg not in self:
            self.add_ensg(ensg)
        if self.method == "MutSel":
            return np.log(self[ensg][alt_aa][c_site] / self[ensg][ref_aa][c_site])
        elif self.method == "Omega":
            return self[ensg][c_site]
        if self.method == "SIFT":
            return self[ensg][alt_aa][c_site]

    def seq_len(self, ensg):
        if ensg not in self:
            self.add_ensg(ensg)
        if self.method == "MutSel" or self.method == "Omega":
            return len(self[ensg])
        if self.method == "SIFT":
            return len(self[ensg]["A"])


def theta(sfs_epsilon, daf_n, weight_method):
    sfs_theta = sfs_epsilon * np.array(range(1, daf_n))
    weights = np.array([sfs_weight[weight_method](i, daf_n) for i in range(1, daf_n)])
    return sum(sfs_theta * weights) / sum(weights)


def write_dofe(sfs_syn, sfs_non_syn, l_non_syn, d_non_syn, l_syn, d_syn, k, filepath, sp_focal, sp_sister, L):
    sfs_list = [k]
    for sfs, nbr_site in [(sfs_non_syn, l_non_syn), (sfs_syn, l_syn)]:
        sfs_list += [nbr_site] + [sfs[i] for i in range(1, k)]
    sfs_list += [l_non_syn, d_non_syn, l_syn, d_syn]

    dofe_file = open(filepath + ".dofe", 'w')
    dofe_file.write(f"{sp_focal}+{sp_sister} ({int(L)} sites)\n")
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


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    if cbar_kw is None:
        cbar_kw = {}
    im = ax.imshow(data, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, orientation='horizontal')
    cbar.ax.set_xlabel(cbarlabel)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
             rotation_mode="anchor")
    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar


def annotate_heatmap(im, data=None, div=False, valfmt="{x:.2f}", textcolors=("white", "black"), **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    threshold_high = im.norm(data.max()) * (0.75 if div else 0.5)
    threshold_low = (im.norm(data.max()) * 0.25) if div else 0.0
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(threshold_high >= im.norm(data[i, j]) >= threshold_low)])
            text = im.axes.text(j, i, valfmt(data[i, j]), **kw)
            texts.append(text)
    return texts


def format_pop(t):
    if "up" == t:
        return "Equus"
    elif "dogs" == t:
        return "Canis"
    elif " " in t:
        return "".join([s[0] for s in t.split(" ")])
    else:
        return t


def sp_sorted(pop, sp):
    out = sp + "_" + pop
    if "Homo" in out:
        return "Z" + out
    elif "Chloro" in out:
        return "Y" + out
    elif "Ovis" in out:
        return "X" + out
    elif "Capra" in out:
        return "W" + out
    elif "Bos" in out:
        return "V" + out
    elif "Canis" in out:
        return "U" + out
    elif "Equus" in out:
        return "T" + out
    else:
        return out


def multiline(xs, ys, c, ax, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax: Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def append_int(num):
    if num > 9:
        second_to_last_digit = str(num)[-2]
        if second_to_last_digit == '1':
            return 'th'
    last_digit = num % 10
    if last_digit == 1:
        return 'st'
    elif last_digit == 2:
        return 'nd'
    elif last_digit == 3:
        return 'rd'
    else:
        return 'th'


def quantile(b, num):
    if num == 100:
        return f"{b}{append_int(b)} percentiles"
    if num == 10:
        return f"{b}{append_int(b)} deciles"
    if num == 5:
        return f"{b}{append_int(b)} quintiles"
    elif num == 4:
        return f"{b}{append_int(b)} quarters"
    else:
        return f"{b}/{num}"


BOUND = namedtuple('BOUND', ['cat', 'lower', 'upper'])
P = namedtuple('P', ['label', 'color', 'lower', 'upper'])


class CategorySNP(list):
    def __init__(self, method="", bound_file="", bins=0, windows=0):
        super().__init__()
        self.bins, self.windows = bins, windows
        self.non_syn_list, self.inner_bound = [], []
        self.dico, self.mean = {}, {}
        if bins > 0:
            intervals = []
            if bound_file != "":
                df = pd.read_csv(bound_file, sep="\t")
                df = df[df["method"] == method]
                assert 1 <= len(df) <= self.bins
                for row_id, row in df.iterrows():
                    intervals.append(BOUND(row["cat"], row["lower"], row["upper"]))
                    self.mean[row["cat"]] = row["mean"]
            else:
                for b in range(bins):
                    intervals.append(BOUND(f"{'b' if self.windows == 0 else 'w'}_{b + 1}", b, b + 1))
            self.add_intervals(intervals)

        elif method == "SIFT":
            self.inner_bound = [0.05, 0.1, 0.3, 0.8]
            self.dico = {
                "neg-strong": P("$SIFT<0.05$", BLUE, 0, 0.05),
                "neg": P("$0.05<SIFT<0.1$", GREEN, 0.05, 0.1),
                "neg-weak": P("$0.1<SIFT<0.3$", LIGHTGREEN, 0.1, 0.3),
                "pos-weak": P("$0.3<SIFT<0.8$", YELLOW, 0.3, 0.8),
                "pos": P("$0.8<SIFT$", RED, 0.8, 1.0),
                "syn": P("$Synonymous$", 'black', None, None)
            }
            self.update()
        else:
            self.inner_bound = [-3, -1, 0, 1]
            self.dico = {
                "neg-strong": P("$S<-3$", BLUE, -np.float("infinity"), -3),
                "neg": P("$-3<S<-1$", GREEN, -3, -1),
                "neg-weak": P("$-1<S<0$", LIGHTGREEN, -1, 0),
                "syn": P("$Synonymous$", 'black', None, None),
                "pos-weak": P("$0<S<1$", YELLOW, 0, 1),
                "pos": P("$S>1$", RED, 1, np.float("infinity"))
            }
            self.update()

    def update(self):
        super().__init__(self.dico.keys())
        self.non_syn_list = [i for i in self if i != "syn"]

    def add_intervals(self, intervals):
        cmap = get_cmap('viridis_r')
        self.dico = {"syn": P("Synonymous", 'black', None, None)}
        for bound in intervals:
            b = int(bound.cat.split("_")[-1])
            color = cmap((b - 1) / (len(intervals) - 1))
            self.dico[bound.cat] = P(f"{quantile(b, len(intervals))}", color, bound.lower, bound.upper)
        self.update()

    def color(self, cat):
        return self.dico[cat].color if cat != "all" else "grey"

    def label(self, cat):
        return self.dico[cat].label if cat != "all" else "All"

    def rate2cats(self, s):
        cats = []
        for cat in self.non_syn_list:
            if self.dico[cat].lower <= s <= self.dico[cat].upper:
                cats.append(cat)
        return cats

    def nbr_non_syn(self):
        return len(self.non_syn_list)

    def all(self):
        return self.non_syn_list + ["all"]
