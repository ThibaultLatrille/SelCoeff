import argparse
from scipy.stats import expon, gamma
from libraries import *
from matplotlib import cm


def read_poly_dfe(path):
    out_poly_dfe = {}
    with open(path, 'r') as f:
        for line in f:
            if "Model: C" in line or "Model: D" in line:
                f.readline()
                f.readline()
                header = re.sub(' +', ' ', f.readline().replace("S_p-", "S_p -").replace("--", "").strip()).split(" ")
                values = re.sub(' +', ' ', f.readline().replace("--", "").strip()).split(" ")
                if "Model: C" in line:
                    for h_i, h in enumerate(header):
                        out_poly_dfe[h] = float(values[h_i])
                    out_poly_dfe["S_d"] = -out_poly_dfe["S_d"]
                elif "Model: D" in line:
                    for v_i, v in enumerate(values):
                        out_poly_dfe[f"S_{v_i + 1}"] = float(header[v_i * 2 + 1])
                        out_poly_dfe[f"p(s={header[v_i * 2 + 1]})"] = float(v)
            if "alpha_dfe" in line:
                k, v = line.strip().split("=")
                out_poly_dfe["$\\alpha$"] = float(v)
    return out_poly_dfe


def read_grapes(path):
    dfem_df = pd.read_csv(path)
    ge_df = dfem_df[dfem_df["model"] == "GammaExpo"]
    return {"p_b": float(ge_df["GammaExpo:pos_prop"]), "S_b": float(ge_df["GammaExpo:posGmean"]),
            "S_d": float(ge_df["GammaExpo:negGmean"]), "b": float(ge_df["GammaExpo:negGshape"]),
            "$\\alpha$": float(ge_df["alpha"])}


def plot_stack_param(list_cat, cat_snps, s_dico, output):
    n = len(polydfe_cat_dico)
    fig, axs = plt.subplots(n, 1, sharex='all', figsize=(1920 / my_dpi, 280 * (n + 1) / my_dpi), dpi=my_dpi)
    x_pos = range(len(list_cat))
    for p_i, (param, param_label) in enumerate(polydfe_cat_dico.items()):
        axs[p_i].bar(x_pos, [s_dico[cat][param] for cat in list_cat], color=[cat_snps.color(cat) for cat in list_cat])
        axs[p_i].axhline(0, color="black", lw=1)
        axs[p_i].set_ylabel(param_label)
        axs[p_i].set_xticks(x_pos)
    axs[len(polydfe_cat_dico) - 1].set_xticklabels([cat_snps.label(cat) for cat in list_cat])
    plt.tight_layout()
    plt.savefig(output)
    plt.close("all")


def plot_heatmap(cat_snps, cat_poly_snps, s_dico, output):
    cat_labels_cols = [cat_snps.label(cat) for cat in cat_snps.non_syn()]
    cat_labels_rows = [cat_poly_snps.label(cat) for cat in cat_poly_snps.non_syn()]
    matrix = np.zeros((len(cat_labels_cols), len(cat_labels_rows)))
    for col, cat_col in enumerate(cat_snps.non_syn()):
        for row, cat_row in enumerate(cat_poly_snps.non_syn()):
            matrix[(row, col)] = s_dico[cat_col][cat_row]
    _, ax = plt.subplots(figsize=(1920 / my_dpi, 880 / my_dpi), dpi=my_dpi)

    rd_bu = cm.get_cmap('RdBu_r')
    im, _ = heatmap(matrix, cat_labels_rows, cat_labels_cols, ax=ax, cmap=rd_bu, cbarlabel="$p$",
                    cbar_kw={"fraction": 0.046}, origin="lower")
    annotate_heatmap(im, valfmt=lambda p: "{0:.2f}".format(p), div=True, fontsize=5)
    plt.tight_layout()
    plt.savefig(output, format="pdf")
    plt.clf()
    plt.close("all")


def plot_dfe_stack_cat(list_cat, cat_snps, s_dico, output):
    n = len(list_cat)
    fig, axs = plt.subplots(n, 1, sharex='all', sharey='all', figsize=(1920 / my_dpi, 280 * (n + 1) / my_dpi),
                            dpi=my_dpi)
    if "b" in s_dico["all"]:
        for cat_i, cat in enumerate(list_cat):
            p_pos = s_dico[cat]["p_b"]
            shape_neg = s_dico[cat]["b"]
            scale_neg = s_dico[cat]["S_d"] / shape_neg
            d_neg = gamma(shape_neg, scale=scale_neg)
            x = np.linspace(-10, 0, 100)
            y = [(1 - p_pos) * d_neg.pdf(-s) for s in x]
            axs[cat_i].plot(x, y, color=cat_snps.color(cat))
            scale_pos = s_dico[cat]["S_b"]
            d_pos = expon(scale=scale_pos)
            x = np.linspace(0, 5, 100)
            y = [p_pos * d_pos.pdf(s) for s in x]
            axs[cat_i].plot(x, y, color=cat_snps.color(cat))
            axs[cat_i].axvline(-1, color="grey", lw=1, ls='--')
            axs[cat_i].axvline(1, color="grey", lw=1, ls='--')
            axs[cat_i].axvline(0, color="black", lw=2, ls='--')
            axs[cat_i].set_ylabel(cat)
    else:
        x_pos = range(len(polydfe_cat_dico))
        for cat_i, cat in enumerate(list_cat[::-1]):
            axs[cat_i].bar(x_pos, [s_dico[cat][param] for param in polydfe_cat_dico], color=cat_snps.color(cat))
            axs[cat_i].set_ylabel(cat_snps.label(cat))
            axs[cat_i].set_xticks(x_pos)
        axs[len(list_cat) - 1].set_xticklabels([label for label in polydfe_cat_dico.values()])
    plt.tight_layout()
    plt.savefig(output, format="pdf")
    plt.close("all")


def main(args):
    cat_snps = CategorySNP(args.method, args.bins)
    cat_poly_snps = CategorySNP("MutSel", args.bins)
    list_cat = cat_snps.non_syn()
    s_dico = dict()
    for file in args.input:
        cat = os.path.basename(file).replace(".out", "").split(".")[-2]
        out = read_poly_dfe(file) if "polyDFE" in file else read_grapes(file)
        if "polyDFE_D" in file:
            p_list = np.array([v for k, v in out.items() if "p(s=" in k])
            s_list = np.array([v for k, v in out.items() if "S_" in k])
            out["S+"] = sum(p_list[3:] * s_list[3:])
            out["S-"] = sum(p_list[:3] * s_list[:3])
            out[polydfe_cat_list[0]] = sum(p_list * s_list)
            out[polydfe_cat_list[1]] = sum(p_list[:2])
            out[polydfe_cat_list[2]] = p_list[2]
            out[polydfe_cat_list[3]] = sum(p_list[3:])
        else:
            p_pos = out["p_b"]
            shape_neg = out["b"]
            scale_neg = out["S_d"] / shape_neg
            d_neg = gamma(shape_neg, scale=scale_neg)
            scale_pos = out["S_b"]
            d_pos = expon(scale=scale_pos)
            out[polydfe_cat_list[0]] = d_pos.stats("m") * p_pos - d_neg.stats("m") * (1 - p_pos)
            out[polydfe_cat_list[1]] = (1 - p_pos) * (1 - d_neg.cdf(1.0))
            out[polydfe_cat_list[2]] = (1 - p_pos) * d_neg.cdf(1.0) + p_pos * d_pos.cdf(1.0)
            out[polydfe_cat_list[3]] = p_pos * (1 - d_pos.cdf(1.0))

            for cat_poly in cat_poly_snps.non_syn():
                if cat_poly == "neg-strong":
                    out[cat_poly] = (1 - p_pos) * (1 - d_neg.cdf(3.0))
                elif cat_poly == "neg":
                    out[cat_poly] = (1 - p_pos) * (d_neg.cdf(3.0) - d_neg.cdf(1.0))
                elif cat_poly == "neg-weak":
                    out[cat_poly] = (1 - p_pos) * d_neg.cdf(1.0)
                elif cat_poly == "pos-weak":
                    out[cat_poly] = p_pos * d_pos.cdf(1.0)
                elif cat_poly == "pos":
                    out[cat_poly] = p_pos * (1 - d_pos.cdf(1.0))
        s_dico[cat] = out

    df_dico = {p: [s_dico[cat][p] for cat in list_cat] for p in polydfe_cat_list}
    df_dico["category"] = list_cat
    pd.DataFrame(df_dico).to_csv(args.output.replace(".pdf", ".tsv"), sep="\t", index=False)

    plot_stack_param(list_cat, cat_snps, s_dico, args.output)
    plot_heatmap(cat_snps, cat_poly_snps, s_dico, args.output.replace(".pdf", ".heatmap.pdf"))
    plot_dfe_stack_cat(list_cat, cat_snps, s_dico, args.output.replace(".pdf", ".predictedDFE.pdf"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', required=False, type=str, nargs="+", dest="input", help="Input polyDFE file")
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output tex file")
    parser.add_argument('--sample_list', required=False, type=str, dest="sample_list", help="Sample list file")
    parser.add_argument('--method', required=False, type=str, dest="method", help="Sel coeff parameter")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    main(parser.parse_args())
