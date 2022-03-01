#!/usr/bin/env python3
from collections import namedtuple, defaultdict
import numpy as np
from math import floor
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
        return f"{b}/{num} quantile"


class CategorySNP(list):
    def __init__(self, method="", bins=0, transform_bound=lambda s: s):
        P = namedtuple('P', ['label', 'color', 'interval'])
        self.bins = bins
        if bins > 0:
            self.inner_bound = []
            cmap = get_cmap('viridis_r')
            assert bins > 1
            self.dico = {"syn": P("Synonymous", 'black', lambda s: False)}
            for b in range(1, bins + 1):
                color = cmap((b - 1) / (bins - 1))
                self.dico[f"bin{b}"] = P(f"{quantile(b, bins)}", color, lambda s: False)
        elif method == "SIFT":
            self.inner_bound = [0.1, 0.3, 0.8]
            self.dico = {
                "neg-strong": P("$SIFT<0.1$", BLUE, lambda s: 0.1 >= s >= 0),
                "neg": P("$0.1<SIFT<0.3$", LIGHTGREEN, lambda s: 0.3 >= s > 0.1),
                "weak": P("$0.3<SIFT<0.8$", YELLOW, lambda s: 0.8 >= s > 0.3),
                "pos": P("$0.8<SIFT$", RED, lambda s: 1 >= s > 0.8),
                "syn": P("$Synonymous$", 'black', lambda x: False),
            }
        else:
            self.inner_bound = [-3, -1, 0, 1]
            self.dico = {
                "neg-strong": P("$S<-3$", BLUE, lambda s: transform_bound(-3) >= s),
                "neg": P("$-3<S<-1$", GREEN, lambda s: transform_bound(-1) >= s > transform_bound(-3)),
                "neg-weak": P("$-1<S<0$", LIGHTGREEN, lambda s: transform_bound(0) >= s > transform_bound(-1)),
                "syn": P("$Synonymous$", 'black', lambda s: False),
                "pos-weak": P("$0<S<1$", YELLOW, lambda s: transform_bound(1) >= s >= transform_bound(0)),
                "pos": P("$1<S$", RED, lambda s: s > transform_bound(1))
            }
        super().__init__(self.dico.keys())

    def color(self, cat):
        return self.dico[cat].color if cat != "all" else "grey"

    def label(self, cat):
        return self.dico[cat].label if cat != "all" else "All"

    def selcoeff2cat(self, s):
        for cat in self:
            if self.dico[cat].interval(s):
                return cat

    def nbr_non_syn(self):
        return len(self) - 1

    def non_syn(self):
        return [i for i in self if i != "syn"]

    def all(self):
        return self.non_syn() + ["all"]
