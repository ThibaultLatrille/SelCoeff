#!/usr/bin/env python3
import os
import re
import gzip
import shutil
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
LIGHTYELLOW = "#FFFCBF"
RED_RGB = to_rgb(RED)
BLUE_RGB = to_rgb(BLUE)
GREEN_RGB = to_rgb(GREEN)
GREY_RGB = to_rgb("grey")
BLACK_RGB = to_rgb("black")
LIGHTYELLOW_RGB = to_rgb(LIGHTYELLOW)

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
    'TGC': 'C', 'TGT': 'C', 'TGA': 'X', 'TGG': 'W'})
confidence_interval = namedtuple('confidence_interval', ['low', 'mean', 'up'])
sfs_weight = {"watterson": lambda i, n: 1.0 / i, "tajima": lambda i, n: n - i, "fay_wu": lambda i, n: i}
polydfe_cat_dico = {
    "P-Spos": "$\\mathbb{P} [ S > 1 ]$",
    "P-Sweak": "$\\mathbb{P} [ -1 < S < 1 ]$",
    "P-Sneg": "$\\mathbb{P} [ S < -1 ]$",
}
alpha_sup_limits = [0, 1, 3, 5]
polydfe_cat_list = list(polydfe_cat_dico.keys())
xlim_dico = {"Omega": (0.0, 2.0), "MutSel": (-10, 10), "SIFT": (0.0, 1.0)}
rate_dico = {"MutSel": "Scaled selection coefficient ($S_0$)",
             "Omega": "Rate of evolution ($\\omega$)",
             "SIFT": "SIFT score"}


def zip_file(input_path, output_path=None):
    if output_path is None:
        output_path = input_path + ".gz"
    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(input_path)


def build_codon_neighbors():
    codon_neighbors = defaultdict(list)
    for ref_codon, ref_aa in codontable.items():
        if ref_aa == "-" or ref_aa == "X":
            continue
        for frame, ref_nuc in enumerate(ref_codon):
            for alt_nuc in [nuc for nuc in nucleotides if nuc != ref_nuc]:
                alt_codon = ref_codon[:frame] + alt_nuc + ref_codon[frame + 1:]
                alt_aa = codontable[alt_codon]
                if alt_aa != 'X':
                    syn = alt_aa == ref_aa
                    codon_neighbors[ref_codon].append((syn, ref_nuc, alt_nuc, alt_codon, alt_aa))
    return codon_neighbors


def translate_cds(seq):
    return "".join([codontable[seq[i * 3:i * 3 + 3]] for i in range(len(seq) // 3)])


def clean_ensg_path(path):
    if not os.path.isfile(path):
        path = path.replace("_null_", "__")
        if not os.path.isfile(path):
            path = path.replace("__", "_null_")
            assert os.path.isfile(path)
    return path


def open_fasta(path):
    outfile = {}
    path = clean_ensg_path(path)
    ali_file = gzip.open(path, 'rt') if path.endswith(".gz") else open(path, 'r')
    for seq_id in ali_file:
        outfile[seq_id.replace('>', '').strip()] = ali_file.readline().strip()
    return outfile


def write_fasta(dico_fasta, output):
    outfile = gzip.open(output, 'wt') if output.endswith(".gz") else open(output, 'w')
    outfile.write("\n".join([f">{seq_id}\n{seq}" for seq_id, seq in dico_fasta.items()]))
    outfile.close()


def open_mask(file: str):
    if file != "" and os.path.isfile(file):
        df_mask = pd.read_csv(file, sep="\t", dtype={"ensg": 'string', "pos": int})
        return {ensg: set(df["pos"].values) for ensg, df in df_mask.groupby("ensg")}
    else:
        print(f"No mask found at: {file}.")
        return {}


def merge_mask_list(mask_list_path: list):
    if len(mask_list_path) == 0:
        print(f"No mask provided.")
        return {}
    elif len(mask_list_path) == 1:
        return open_mask(mask_list_path[0])
    else:
        mask_merged = defaultdict(set)
        for mask_path in mask_list_path:
            mask = open_mask(mask_path)
            for ensg, pos in mask.items():
                mask_merged[ensg] = mask_merged[ensg].union(pos)
        return mask_merged


class CdsRates(dict):
    def __init__(self, method, exp_folder="", sift_folder=""):
        self.method = method
        assert self.method in ["Omega", "MutSel", "SIFT"], 'Method must be either "Omega", "MutSel" or "SIFT"'
        if self.method == "SIFT":
            assert sift_folder != ""
        self.exp_folder = exp_folder
        self.sift_folder = sift_folder
        self.nuc_matrix = defaultdict(dict)
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

    def add_nuc_matrix(self, ensg, nuc_path):
        df = pd.read_csv(nuc_path, sep="\t")
        self.nuc_matrix[ensg] = {name: q for name, q in zip(df["Name"], df["Rate"])}

    def add_mut_ensg(self, ensg):
        nuc_path = clean_ensg_path(f"{self.exp_folder}/{ensg}_NT/sitemutsel_1.run.nucmatrix.tsv")
        self.add_nuc_matrix(ensg, nuc_path)

    def add_ensg(self, ensg):
        if self.method == "MutSel":
            path = clean_ensg_path(f"{self.exp_folder}/{ensg}_NT/sitemutsel_1.run.siteprofiles")
            self[ensg] = pd.read_csv(path, sep="\t", skiprows=1, header=None,
                                     names="site,A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y".split(","))
        elif self.method == "Omega":
            path = clean_ensg_path(f"{self.exp_folder}/{ensg}_NT/siteomega_1.run.ci0.025.tsv")
            self[ensg] = pd.read_csv(path, sep="\t")["gene_omega"].values[1:]
        elif self.method == "SIFT":
            self.add_sift(ensg, f"{self.sift_folder}/{ensg}")

    def rm_ensg(self, ensg):
        if ensg in self:
            self.pop(ensg)
        if ensg in self.nuc_matrix:
            self.nuc_matrix.pop(ensg)

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

    def mutation_rate(self, ensg, ref_nuc, alt_nuc):
        if ensg not in self:
            self.add_mut_ensg(ensg)
        return self.nuc_matrix[ensg][f"q_{ref_nuc}_{alt_nuc}"]

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


def write_sfs(sfs_syn, sfs_non_syn, l_non_syn, d_non_syn, l_syn, d_syn, k, filepath, sp_focal, sp_sister):
    sfs_syn_str = " ".join([str(int(sfs_syn[i])) for i in range(1, k)]) + f"\t{int(l_syn)}"
    sfs_non_syn_str = " ".join([str(int(sfs_non_syn[i])) for i in range(1, k)]) + f"\t{int(l_non_syn)}"
    if d_syn is not None and d_non_syn is not None:
        sfs_syn_str += f"\t{int(d_syn)}\t{int(l_syn)}"
        sfs_non_syn_str += f"\t{int(d_non_syn)}\t{int(l_non_syn)}"
    sfs_file = open(filepath + ".sfs", 'w')
    sfs_file.write(f"#{sp_focal}+{sp_sister}\n")
    sfs_file.write("1 1 {0}".format(k) + "\n")
    sfs_file.write(sfs_syn_str + "\n")
    sfs_file.write(sfs_non_syn_str + "\n")
    sfs_file.close()


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

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


def extend_pop(p, sample):
    fp = format_pop(p)
    if sample[p] == fp:
        return p
    else:
        return f"{sample[p]} ({fp})"


def sample_list_dico(sample_list_path):
    df = pd.read_csv(sample_list_path, sep='\t')
    df["pop"] = df.apply(lambda r: format_pop(r["SampleName"].replace("_", " ")), axis=1)
    dico_sample = {r["pop"]: r["DisplayName"] for _, r in df.iterrows()}
    return dico_sample


def sort_df(df, sample_list_path):
    df = df.iloc[df.apply(lambda r: sp_sorted(format_pop(r["pop"]), r["species"]), axis=1).argsort()]
    df["species"] = df.apply(lambda r: r["species"].replace("_", " "), axis=1)
    dico_sample = sample_list_dico(sample_list_path)
    df["pop"] = df.apply(lambda r: dico_sample[r["pop"]], axis=1)
    return df


def tex_f(x, highlight=False, pad=""):
    if x == 0:
        s = "0.0"
    elif 0.001 < abs(x) < 10:
        s = f"{x:6.3f}"
    elif 10 <= abs(x) < 10000:
        s = f"{x:6.1f}"
    else:
        s = f"{x:6.2g}"
        if "e" in s:
            mantissa, exp = s.split('e')
            s = mantissa + '\\times 10^{' + str(int(exp)) + '}'
    if highlight:
        return "$\\bm{" + s + "{^*}}$"
    else:
        return f"${s}{pad}$"


def multiline(xs, ys, c, ax, **kwargs):
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
        assert method in ["MutSel", "Omega", "SIFT"]
        super().__init__()
        self.bins, self.windows = bins, windows
        self.non_syn_list, self.inner_bound = [], []
        self.dico, self.mean = {}, {}
        if bins > 0 and (bins != 2 and bins != 3 and bins != 5):
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
            if self.bins == 2:
                self.inner_bound = [0.8]
                self.dico = {
                    "neg": P("$SIFT < 0.8$", RED, 0, 0.8),
                    "pos": P("$SIFT > 0.8$", BLUE, 0.8, 1.0),
                    "syn": P("$Synonymous$", 'black', None, None)
                }
            elif self.bins == 3:
                self.inner_bound = [0.05, 0.8]
                self.dico = {
                    "neg": P("$SIFT<0.05$", RED, 0, 0.05),
                    "weak": P("$0.05<SIFT<0.8$", GREEN, 0.05, 0.8),
                    "pos": P("$SIFT > 0.8$", BLUE, 0.8, 1.0),
                    "syn": P("$Synonymous$", 'black', None, None)
                }
            else:
                self.inner_bound = [0.05, 0.1, 0.3, 0.8]
                self.dico = {
                    "neg-strong": P("$SIFT<0.05$", RED, 0, 0.05),
                    "neg": P("$0.05<SIFT<0.1$", YELLOW, 0.05, 0.1),
                    "neg-weak": P("$0.1<SIFT<0.3$", LIGHTGREEN, 0.1, 0.3),
                    "pos-weak": P("$0.3<SIFT<0.8$", GREEN, 0.3, 0.8),
                    "pos": P("$SIFT > 0.8$", BLUE, 0.8, 1.0),
                    "syn": P("$Synonymous$", 'black', None, None)
                }
            self.update()
        elif method == "Omega":
            if self.bins == 2:
                self.inner_bound = [1.0]
                self.dico = {
                    "neg": P("$\\omega < 1$", RED, 0, 1.0),
                    "pos": P("$\\omega > 1$", BLUE, 1.0, np.float("infinity")),
                    "syn": P("$Synonymous$", 'black', None, None)
                }
            elif self.bins == 3:
                self.inner_bound = [0.05, 1.0]
                self.dico = {
                    "neg": P("$\\omega<0.05$", RED, 0, 0.05),
                    "weak": P("$0.05<\\omega<1.0$", GREEN, 0.05, 1.0),
                    "pos": P("$\\omega > 1$", BLUE, 1.0, np.float("infinity")),
                    "syn": P("$Synonymous$", 'black', None, None)
                }
            else:
                self.inner_bound = [0.05, 0.1, 0.3, 1.0]
                self.dico = {
                    "neg-strong": P("$\\omega<0.05$", RED, 0, 0.05),
                    "neg": P("$0.05<\\omega<0.1$", YELLOW, 0.05, 0.1),
                    "neg-weak": P("$0.1<\\omega<0.3$", LIGHTGREEN, 0.1, 0.3),
                    "pos-weak": P("$0.3<\\omega < 1$", GREEN, 0.3, 1.0),
                    "pos": P("$\\omega > 1$", BLUE, 1.0, np.float("infinity")),
                    "syn": P("$Synonymous$", 'black', None, None)
                }
            self.update()
        else:
            assert method == "MutSel"
            if self.bins == 2:
                self.inner_bound = [1]
                self.dico = {
                    "neg": P("$S_0 < 1$", RED, -np.float("infinity"), 1),
                    "syn": P("$Synonymous$", 'black', None, None),
                    "pos": P("$S_0 > 1$", BLUE, 1, np.float("infinity"))
                }
            elif self.bins == 3:
                self.inner_bound = [-1, 1]
                self.dico = {
                    "neg": P("$S_0 < -1$", RED, -np.float("infinity"), -1),
                    "weak": P("$-1 < S_0 < 1$", GREEN, -1, 1),
                    "pos": P("$S_0 > 1$", BLUE, 1, np.float("infinity")),
                    "syn": P("$Synonymous$", 'black', None, None)
                }
            else:
                self.inner_bound = [-5, -1, 0, 1]
                self.dico = {
                    "neg-strong": P("$S_0 < -5$", RED, -np.float("infinity"), -5),
                    "neg": P("$-5 < S_0 < -1$", YELLOW, -5, -1),
                    "neg-weak": P("$-1 < S_0 < 0$", LIGHTGREEN, -1, 0),
                    "syn": P("$Synonymous$", 'black', None, None),
                    "pos-weak": P("$0 < S_0 < 1$", GREEN, 0, 1),
                    "pos": P("$S_0 >1$", BLUE, 1, np.float("infinity"))
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
