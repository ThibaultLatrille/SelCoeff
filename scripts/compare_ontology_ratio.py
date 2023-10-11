import os
import argparse
import pandas as pd
from collections import defaultdict
from scipy.stats import wilcoxon
from libraries import sort_df, tex_f, row_color


def open_results(path: str, sample_list: str) -> dict:
    df = pd.read_csv(os.path.join(path, "regression-MutSel", "results.tsv"), sep="\t")
    df = sort_df(df, args.sample_list)  # assign the pop name to the index
    df = row_color(df)  # assign the pop name to the index
    df_out = df[["pop", "species", "recall_pos"]]
    return df_out


def main(experiments: str, sample_list: str, output: str):
    # Iter over folders with "Onto" in the name
    control_name = "3bins-mC"
    control_path = os.path.join(experiments, control_name)
    control_df = open_results(control_path, sample_list)
    control = control_df["recall_pos"].values

    case_path_list = [os.path.join(experiments, f) for f in os.listdir(experiments)]
    case_path_list = [f for f in case_path_list if os.path.isdir(f)]
    case_path_list = [f for f in case_path_list if (("Adaptation" in f))]

    print(control)
    out_dict = defaultdict(list)
    tex_file = open(output.replace(".tsv", f".{control_name}.tex"), "w")
    for case_path in case_path_list:
        case_name = os.path.basename(case_path)
        print(f"Comparing {case_name} to control")

        tex_file.write(f"\\section{{{case_name.replace('-', ' ')}}}\n")
        case_df = open_results(case_path, sample_list)

        assert case_df["species"].equals(
            control_df["species"]), f"Species differ between {case_name} and {control_name}"
        assert case_df["pop"].equals(control_df["pop"]), f"Populations differ between {case_name} and {control_name}"

        # T-test paired between case and control
        case = case_df["recall_pos"].values
        res = wilcoxon(control, case, alternative="less")

        out_df = pd.DataFrame({"pop": case_df["pop"], "species": case_df["species"], "control": control, "case": case})
        tex_file.write(out_df.to_latex(index=False, escape=False, float_format=tex_f))
        r = (case / control).mean()
        s, p = res[0], res[1]
        print(f"{case}\n\ts={s:.2g}, p={p:.2g}, r={r:.2g}")
        tex_file.write(f"\\textbf{{s={s:.2g}, p={p:.2g}, r={r:.2g}}}\n")
        out_dict["case"].append(case_name)
        out_dict["s"].append(s)
        out_dict["p"].append(p)
    df = pd.DataFrame(out_dict)
    df.to_csv(output.replace(".tsv", f".{control_name}.tsv"), sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiments', required=False, type=str, dest="experiments",
                        default="experiments", help="folder containing experiments")
    parser.add_argument('--output', required=False, type=str, dest="output",
                        default="experiments/compare.tsv", help="Output path")
    parser.add_argument('--sample_list', required=False, type=str, dest="sample_list",
                        default="config/sample_all.tsv", help="Sample list")
    args = parser.parse_args()
    main(args.experiments, args.output, args.output)
