import argparse

import matplotlib.pyplot as plt
from scipy.stats import expon, gamma
from matplotlib.patches import Rectangle
from libraries import *


def main(output):
    plt.figure(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
    p_b = 0.111
    b = 0.198
    S_d = -50.1
    S_b = 1.0
    p_pos = p_b
    shape_neg = b
    scale_neg = -S_d / shape_neg
    scale_pos = S_b
    d_neg = gamma(shape_neg, scale=scale_neg)
    x = np.linspace(-10, 0, 1000)
    print(x[-1])
    y = [(1 - p_pos) * d_neg.pdf(-s) for s in x]
    print(y[-1])
    plt.plot(x, y, color=RED)
    plt.fill_between(x, y, color=RED, alpha=0.5)
    d_pos = expon(scale=scale_pos)
    x = np.linspace(0, 10, 1000)
    y = [p_pos * d_pos.pdf(s) for s in x]
    plt.plot(x, y, color=BLUE)
    plt.ylim((0, 0.2))
    plt.fill_between(x, y, color=BLUE, alpha=0.5)
    plt.axvline(-1, color="grey", lw=1, ls='--')
    plt.axvline(1, color="grey", lw=1, ls='--')
    plt.axvline(0, color="black", lw=2, ls='--')
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in [RED, BLUE]]
    labels = [f"$S < 0$ ({(1 - p_b) * 100:.2f}% of total)", f"$S > 0$ ({p_b * 100:.2f}% of total)"]

    plt.legend(handles, labels)
    plt.xlabel("Selection coefficient ($s_0$)")
    plt.ylabel("All SNPs")
    plt.tight_layout()
    plt.savefig(output, format="pdf")
    plt.close("all")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output pdf file")
    args = parser.parse_args()
    main(args.output)
