import argparse
from scipy.stats import expon, gamma
import scipy.integrate as integrate
from libraries import *
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage


def read_poly_dfe(path, proc_R):
    out_poly_dfe = {}
    with open(path, 'r') as f:
        for line in f:
            start = 'Best joint likelihood found '
            if start in line:
                out_poly_dfe["logL"] = float(line[line.find(start) + len(start):line.rfind(' with gradient')])
            if "Model: B" in line or "Model: C" in line or "Model: D" in line:
                f.readline()
                f.readline()
                header = re.sub(' +', ' ', f.readline().replace("S_p-", "S_p -").replace("--", "").strip()).split(" ")
                values = re.sub(' +', ' ', f.readline().replace("--", "").strip()).split(" ")
                if "Model: B" in line or "Model: C" in line:
                    for h_i, h in enumerate(header):
                        out_poly_dfe[h] = float(values[h_i])

                    if "Model: C" in line:
                        estimates = proc_R.parseOutput(path)[0]
                        for lim_left in alpha_sup_limits:
                            out_poly_dfe[f"alpha{lim_left}"] = proc_R.estimateAlpha(estimates, supLimit=lim_left)[0]
                elif "Model: D" in line:
                    for v_i, v in enumerate(values):
                        out_poly_dfe[f"S_{v_i + 1}"] = float(header[v_i * 2 + 1])
                        out_poly_dfe[f"p(s={header[v_i * 2 + 1]})"] = float(v)

    return out_poly_dfe


def plot_stack_param(list_cat, cat_snps, s_dico, output):
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 960 / my_dpi), dpi=my_dpi)
    x_pos = range(len(list_cat))
    hatches_list = ['', '', '//']
    colors = [cat_snps.color(cat) for cat in list_cat]
    colors_list = ["black", "silver", "white"]
    edgecolors_list = ["black", "black", "black"]
    bottom = np.array([0.0] * len(list_cat))
    for p_i, param in enumerate(polydfe_cat_dico):
        y = np.array([s_dico[cat][param] for cat in list_cat])
        ax.bar(x_pos, y, bottom=bottom, edgecolor=edgecolors_list[p_i], color=colors_list[p_i], hatch=hatches_list[p_i])
        bottom += y
    ax.set_xlabel("Category of $S_0$")
    ax.set_ylabel("Proportion estimated at the population scale")
    ax.set_xticks(x_pos)
    ax.set_ylim((0, 1))
    ax.set_xticklabels([cat_snps.label(cat) for cat in list_cat])
    for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), colors):
        ticklabel.set_color(tickcolor)
    plt.tight_layout()
    plt.savefig(output)
    plt.close("all")


def plot_dfe_stack_cat(list_cat, cat_snps, s_dico, output):
    fig, axs = plt.subplots(len(list_cat), 1, sharex='all', dpi=my_dpi,
                            figsize=(1920 / my_dpi, 280 * (len(list_cat) + 1) / my_dpi))
    if "b" in s_dico["all"]:
        for cat_i, cat in enumerate(list_cat):
            p_pos = s_dico[cat]["p_b"]
            shape_neg = s_dico[cat]["b"]
            scale_neg = -s_dico[cat]["S_d"] / shape_neg
            d_neg = gamma(shape_neg, scale=scale_neg)
            x = np.linspace(-20, 0, 100)
            y = [(1 - p_pos) * d_neg.pdf(-s) * pfix(s) for s in x]
            y[-1] = np.nan
            axs[cat_i].plot(x, y, color=cat_snps.color(cat))
            scale_pos = s_dico[cat]["S_b"]
            d_pos = expon(scale=scale_pos)
            x = np.linspace(0, 20, 100)
            y = [p_pos * d_pos.pdf(s) * pfix(s) for s in x]
            y[0] = np.nan
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


def pfix(s):
    if s < -100:
        return 0.0
    elif s == 0.0:
        return 1.0
    else:
        return s / (1 - np.exp(-s))


def omega_div(df_div, cat):
    div_non_syn = float(df_div[f"div_{cat}"].values[0])
    l_non_syn = float(df_div[f"L_{cat}"].values[0])
    div_syn = float(df_div[f"div_syn"].values[0])
    l_syn = float(df_div[f"L_syn"].values[0])
    return (div_non_syn / div_syn) * (l_syn / l_non_syn)


def omega_model_C(p_pos, d_neg, d_pos, min_s, max_s):
    out = 0.0
    if min_s < 0:
        out += integrate.quad(lambda s: (1 - p_pos) * d_neg.pdf(-s) * pfix(s), min_s, 0)[0]
    if max_s > 0:
        out += integrate.quad(lambda s: p_pos * d_pos.pdf(s) * pfix(s), 0, max_s)[0]
    return out


def omega_model_D(p_list, s_list, min_s, max_s):
    return sum([p * pfix(s) for p, s in zip(p_list, s_list) if min_s <= s < max_s])


def omega_dico(df_div, cat, omega_dfe, omega_na_dfe, omega_a_dfe):
    out = {f'omega_dfe': omega_dfe, f'omega_NAdfe': omega_na_dfe, f'omega_Adfe': omega_a_dfe}
    if df_div is not None:
        omega = omega_div(df_div, cat)
        div = float(df_div[f"div_{cat}"].values[0])
        out |= {f"div": div, f'omega_div': omega, f'omega_Adiv': omega - omega_na_dfe,
                f'alpha_div': (omega - omega_na_dfe) / omega}
    return out


def alpha_model_D(p_list, s_list, lim_left=0.0):
    q_linspace = [p * pfix(s) for p, s in zip(p_list, s_list)]
    return sum([q for s, q in zip(s_list, q_linspace) if lim_left < s]) / sum(q_linspace)



def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    string = ''.join(open(args.postprocessing, "r").readlines())
    proc_R = SignatureTranslatedAnonymousPackage(string, "postprocessing")

    df_div = pd.read_csv(args.substitutions, sep="\t") if args.substitutions != "" else None
    cat_snps = CategorySNP(args.method, args.bounds, bins=args.bins, windows=args.windows)
    cat_poly_snps = CategorySNP("MutSel", bins=0, windows=0)
    list_cat = cat_snps.all()
    s_dico = dict()
    for file in args.input:
        cat = os.path.basename(file).replace(".out", "").split(".")[-2]
        out = read_poly_dfe(file, proc_R)
        if "polyDFE_D" in file:
            p_list = np.array([v for k, v in out.items() if "p(s=" in k])
            s_list = np.array([v for k, v in out.items() if "S_" in k])
            out_list = [f"p(s={s})={p}" for s, p in zip(s_list, p_list)]
            if cat == "all":
                print(f"{file} - {cat}:\n{out_list}")
            assert abs(sum(p_list) - 1.0) < 1e-4
            out["S+"] = sum([p * s for p, s in zip(p_list, s_list) if s > 0])
            out["S-"] = sum([p * s for p, s in zip(p_list, s_list) if s < 0])
            out["S"] = sum([p * s for p, s in zip(p_list, s_list)])
            out[polydfe_cat_list[0]] = sum([p for p, s in zip(p_list, s_list) if s >= 1])
            out[polydfe_cat_list[1]] = sum([p for p, s in zip(p_list, s_list) if -1 <= s < 1])
            out[polydfe_cat_list[2]] = sum([p for p, s in zip(p_list, s_list) if s < -1])
            out["p_b"] = sum([p for p, s in zip(p_list, s_list) if s >= 0])
            for lim_left in alpha_sup_limits:
                out[f'alpha{lim_left}'] = alpha_model_D(p_list, s_list, lim_left=lim_left)

            omega_dfe = omega_model_D(p_list, s_list, -100, 100)
            omega_na_dfe = omega_model_D(p_list, s_list, -100, 0)
            omega_a_dfe = omega_model_D(p_list, s_list, 0, 100)
            out |= omega_dico(df_div, cat, omega_dfe, omega_na_dfe, omega_a_dfe)
        else:
            p_pos = out["p_b"]
            shape_neg = out["b"]
            scale_neg = -out["S_d"] / shape_neg
            d_neg = gamma(shape_neg, scale=scale_neg)
            scale_pos = out["S_b"]
            d_pos = expon(scale=scale_pos)
            out["S"] = d_pos.stats("m") * p_pos - d_neg.stats("m") * (1 - p_pos)
            out[polydfe_cat_list[0]] = p_pos * (1 - d_pos.cdf(1.0))
            out[polydfe_cat_list[1]] = (1 - p_pos) * d_neg.cdf(1.0) + p_pos * d_pos.cdf(1.0)
            out[polydfe_cat_list[2]] = (1 - p_pos) * (1 - d_neg.cdf(1.0))

            for cat_poly in cat_poly_snps.non_syn_list:
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

            omega_dfe = omega_model_C(p_pos, d_neg, d_pos, -100, 100)
            omega_na_dfe = omega_model_C(p_pos, d_neg, d_pos, -100, 0)
            omega_a_dfe = omega_model_C(p_pos, d_neg, d_pos, 0, 100)
            out |= omega_dico(df_div, cat, omega_dfe, omega_na_dfe, omega_a_dfe)
        s_dico[cat] = out

    list_cat = [cat for cat in list_cat if cat in s_dico]
    df_dico = {p: [s_dico[cat][p] for cat in list_cat] for p in s_dico[list_cat[0]]}
    df_dico["category"] = list_cat
    pd.DataFrame(df_dico).to_csv(args.output.replace(".pdf", ".tsv"), sep="\t", index=False)
    plot_stack_param([c for c in list_cat if c != 'all'], cat_snps, s_dico, args.output)

    if args.bins <= 10 and args.windows == 0:
        plot_dfe_stack_cat(list_cat, cat_snps, s_dico, args.output.replace(".pdf", ".predictedDFE.pdf"))
    else:
        s_pop_list = [s_dico[cat]["S"] for cat in list_cat if cat != 'all']
        s_phy_list = [cat_snps.mean[cat] for cat in list_cat if cat != 'all']
        pd.DataFrame({"S_pop": s_pop_list, "S_phy": s_phy_list}).to_csv(args.output.replace(".pdf", ".scatter.tsv"),
                                                                        sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', required=False, type=str, nargs="+", dest="input", help="Input polyDFE file")
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output tex file")
    parser.add_argument('--sample_list', required=False, type=str, dest="sample_list", help="Sample list file")
    parser.add_argument('--method', required=False, type=str, dest="method", help="Sel coeff parameter")
    parser.add_argument('--postprocessing', required=True, type=str, dest="postprocessing", help="polyDFE processing")
    parser.add_argument('--substitutions', required=False, default="", type=str, dest="substitutions",
                        help="Substitutions mapping")
    parser.add_argument('--bins', required=False, default=0, type=int, dest="bins", help="Number of bins")
    parser.add_argument('--windows', required=False, default=0, type=int, dest="windows", help="Number of windows")
    parser.add_argument('--bounds', required=False, default="", type=str, dest="bounds", help="Input bound file path")
    main(parser.parse_args())
