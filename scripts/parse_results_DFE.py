import os
import re
import argparse
import numpy as np
import pandas as pd
from scipy.stats import expon, gamma
from libraries import plt, CategorySNP, my_dpi


def read_polyDFE(path):
    out_polyDFE = {}
    with open(path, 'r') as f:
        for line in f:
            if "Model: C" in line or "Model: D" in line:
                f.readline()
                f.readline()
                header = re.sub(' +', ' ', f.readline().replace("S_p-", "S_p -").replace("--", "").strip()).split(" ")
                values = re.sub(' +', ' ', f.readline().replace("--", "").strip()).split(" ")
                if "Model: C" in line:
                    for h_i, h in enumerate(header):
                        out_polyDFE[h] = float(values[h_i])
                    out_polyDFE["S_d"] = -out_polyDFE["S_d"]
                elif "Model: D" in line:
                    for v_i, v in enumerate(values):
                        out_polyDFE[f"S_{v_i + 1}"] = float(header[v_i * 2 + 1])
                        out_polyDFE[f"p(S={header[v_i * 2 + 1]})"] = float(v)
            if "alpha_dfe" in line:
                k, v = line.strip().split("=")
                out_polyDFE["$\\alpha$"] = float(v)
    return out_polyDFE


def read_grapes(path):
    dfem_df = pd.read_csv(path)
    ge_df = dfem_df[dfem_df["model"] == "GammaExpo"]
    return {"p_b": float(ge_df["GammaExpo:pos_prop"]), "S_b": float(ge_df["GammaExpo:posGmean"]),
            "S_d": float(ge_df["GammaExpo:negGmean"]), "b": float(ge_df["GammaExpo:negGshape"]),
            "$\\alpha$": float(ge_df["alpha"])}


def main(args):
    cat_snps = CategorySNP(args.method)
    list_cat = cat_snps.non_syn() + ["all"]
    s_dico = dict()
    for file in args.input:
        cat = os.path.basename(file).replace(".out", "").split(".")[-2]
        out = read_polyDFE(file) if "polyDFE" in file else read_grapes(file)
        if "polyDFE_D" in file:
            p_list = np.array([v for k, v in out.items() if "p(S=" in k])
            s_list = np.array([v for k, v in out.items() if "S_" in k])
            out["S"] = sum(p_list * s_list)
            out["S+"] = sum(p_list[3:] * s_list[3:])
            out["S-"] = sum(p_list[:3] * s_list[:3])
            out["P(S<0)"] = sum(p_list[:2])
            out["P(S=0)"] = p_list[2]
            out["P(S>0)"] = sum(p_list[3:])
        else:
            out["S"] = out["S_b"] * out["p_b"] - out["S_d"] * (1 - out["p_b"])
            out["P(S>0)"] = out["p_b"]
            out["P(S=0)"] = 0
            out["P(S<0)"] = 1 - out["p_b"]
        s_dico[cat] = out

    params_list = [p for p in s_dico["all"].keys() if "P(S" in p]
    df_dico = {p: [s_dico[cat][p] for cat in list_cat] for p in params_list}
    df_dico["category"] = list_cat
    pd.DataFrame(df_dico).to_csv(args.output.replace(".pdf", ".tsv"), sep="\t", index=False)

    n = len(params_list)
    fig, axs = plt.subplots(n, 1, sharex='all', figsize=(1920 / my_dpi, 240 * (n + 1) / my_dpi), dpi=my_dpi)
    x_pos = range(len(list_cat))
    for p_i, param in enumerate(params_list):
        axs[p_i].bar(x_pos, [s_dico[cat][param] for cat in list_cat], color=[cat_snps.color(cat) for cat in list_cat])
        axs[p_i].axhline(0, color="black", lw=1)
        axs[p_i].set_ylabel(param)
        axs[p_i].set_xticks(x_pos)
    axs[len(params_list) - 1].set_xticklabels([cat_snps.label(cat) for cat in list_cat])
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close("all")

    n = len(list_cat)
    fig, axs = plt.subplots(n, 1, sharex='all', sharey='all', figsize=(1920 / my_dpi, 240 * (n + 1) / my_dpi),
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
            y = [p_pos * expon.pdf(s) for s in x]
            axs[cat_i].plot(x, y, color=cat_snps.color(cat))
            axs[cat_i].axvline(-1, color="grey", lw=1, ls='--')
            axs[cat_i].axvline(1, color="grey", lw=1, ls='--')
            axs[cat_i].axvline(0, color="black", lw=2, ls='--')
            axs[cat_i].set_ylabel(cat)
    else:
        x_pos = range(len([1 for k in s_dico['all'].keys() if "S_" in k]))
        for cat_i, cat in enumerate(list_cat):
            axs[cat_i].bar(x_pos, [v for k, v in s_dico[cat].items() if "p(S=" in k], color=cat_snps.color(cat))
            axs[cat_i].set_ylabel(cat_snps.label(cat))
            axs[cat_i].set_xticks(x_pos)
        axs[len(list_cat) - 1].set_xticklabels([v for k, v in s_dico['all'].items() if "S_" in k])
    axs[len(list_cat) - 1].set_xlabel("Scaled selection coefficient (S)")
    plt.tight_layout()
    plt.savefig(args.output.replace(".pdf", ".predictedDFE.pdf"), format="pdf")
    plt.close("all")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', required=False, type=str, nargs="+", dest="input", help="Input polyDFE file")
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output tex file")
    parser.add_argument('--sample_list', required=False, type=str, dest="sample_list", help="Sample list file")
    parser.add_argument('--method', required=False, type=str, dest="method", help="Sel coeff parameter")
    main(parser.parse_args())
