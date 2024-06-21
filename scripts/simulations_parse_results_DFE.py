import argparse
from scipy.stats import expon, gamma
from libraries import *
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from parse_results_DFE import read_poly_dfe


def main(path_input, path_output, path_postprocessing):
    os.makedirs(os.path.dirname(path_output), exist_ok=True)
    string = ''.join(open(path_postprocessing, "r").readlines())
    proc_R = SignatureTranslatedAnonymousPackage(string, "postprocessing")

    cat_snps = CategorySNP("MutSel", bins=3, windows=0)
    cat_poly_snps = CategorySNP("MutSel", bins=0, windows=0)
    list_cat = cat_snps.all()
    s_dico = dict()
    for file in path_input:
        cat = os.path.basename(file).replace(".out", "").split(".")[-2]
        out = read_poly_dfe(file, proc_R)
        if "polyDFE_D" in file:
            p_list = np.array([v for k, v in out.items() if "p(s=" in k])
            s_list = np.array([v for k, v in out.items() if "S_" in k])
            out_list = [f"p(s={s})={p}" for s, p in zip(s_list, p_list)]
            print(f"{file} - {cat}:\n{out_list}")
            assert abs(sum(p_list) - 1.0) < 1e-4
            out["S+"] = sum([p * s for p, s in zip(p_list, s_list) if s > 0])
            out["S-"] = sum([p * s for p, s in zip(p_list, s_list) if s < 0])
            out["S"] = sum([p * s for p, s in zip(p_list, s_list)])
            out[polydfe_cat_list[0]] = sum([p for p, s in zip(p_list, s_list) if s > 1])
            out[polydfe_cat_list[1]] = sum([p for p, s in zip(p_list, s_list) if -1 <= s <= 1])
            out[polydfe_cat_list[2]] = sum([p for p, s in zip(p_list, s_list) if s < -1])
            out["p_b"] = sum([p for p, s in zip(p_list, s_list) if s >= 0])

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
        s_dico[cat] = out

    list_cat = [cat for cat in list_cat if cat in s_dico]
    df_dico = {p: [s_dico[cat][p] for cat in list_cat] for p in s_dico[list_cat[0]]}
    df_dico["category"] = list_cat
    pd.DataFrame(df_dico).to_csv(path_output, sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', required=False, type=str, nargs="+", dest="input", help="Input polyDFE file")
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output file")
    parser.add_argument('--postprocessing', required=True, type=str, dest="postprocessing", help="polyDFE processing")
    args = parser.parse_args()
    main(args.input, args.output, args.postprocessing)
