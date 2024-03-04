import os
import argparse
import pandas as pd
from itertools import product
from collections import defaultdict
from scipy.stats import wilcoxon
from libraries import sort_df, tex_f, row_color


def open_results(path: str, sample_list: str) -> pd.DataFrame:
    path_tsv = os.path.join(path, "regression-MutSel", "results.tsv")
    if not os.path.exists(path_tsv):
        print(f"\nSkipping {path} as it does not contain a results.tsv")
        return pd.DataFrame()
    df = pd.read_csv(path_tsv, sep="\t")
    df = sort_df(df, sample_list)  # assign the pop name to the index
    df = row_color(df)  # assign the pop name to the index
    df_out = df[["pop", "species", "recall_pos"]]
    return df_out


def main(experiments: str, sample_list: str, output: str):
    # Iter over folders with "Onto" in the name
    case_path_list = sorted([os.path.join(experiments, f) for f in os.listdir(experiments)])
    case_path_list = [f for f in case_path_list if os.path.isdir(f)]
    case_path_list = [f for f in case_path_list if ("NonAdaptive" in f)]
    control_list = ["3bins-mC-nodiv", "3bins-mD"]

    out_dict = defaultdict(list)
    tex_file = open(output.replace(".tsv", f".tex"), "w")
    for case_path, control_name in product(case_path_list, control_list):
        case_name = os.path.basename(case_path)
        if case_name == control_name:
            continue
        if ("mC" in case_name + control_name) and ("mD" in case_name + control_name):
            continue
        control_path = os.path.join(experiments, control_name)
        control_df = open_results(control_path, sample_list)
        if control_df.empty:
            continue
        control = control_df["recall_pos"].values

        section_name = case_name.replace('-', ' ') + " vs " + control_name.replace('-', ' ')
        tex_file.write(f"\\section{{{section_name}}}\n")
        case_df = open_results(case_path, sample_list)
        if case_df.empty:
            continue

        assert case_df["species"].equals(control_df["species"]), f"Species diff between {case_name} and {control_name}"
        assert case_df["pop"].equals(control_df["pop"]), f"Populations diff between {case_name} and {control_name}"

        # T-test paired between case and control
        case = case_df["recall_pos"].values
        res = wilcoxon(control, case, alternative="less")

        out_df = pd.DataFrame({"pop": case_df["pop"], "species": case_df["species"], "control": control, "case": case})
        tex_file.write(out_df.to_latex(index=False, escape=False, float_format=tex_f))
        r = (case / control).mean()
        s, p = res[0], res[1]
        print(f"\n{case_name}: {case.mean():.2g}\n{control_name}: {control.mean():.2g}")
        print(f"\ts={s:.2g}, p={p:.2g}, r={r:.2g}")
        tex_file.write(f"\\textbf{{s={s:.2g}, p={p:.2g}, r={r:.2g}}}\n")
        out_dict["control"].append(control_name)
        out_dict["case"].append(case_name)
        out_dict["s"].append(s)
        out_dict["p"].append(p)
    df = pd.DataFrame(out_dict)
    df.to_csv(output, sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiments', required=False, type=str, dest="experiments",
                        default="experiments", help="folder containing experiments")
    parser.add_argument('--output', required=False, type=str, dest="output",
                        default="experiments/compare.tsv", help="Output path")
    parser.add_argument('--sample_list', required=False, type=str, dest="sample_list",
                        default="config/sample_all.tsv", help="Sample list")
    args = parser.parse_args()
    main(args.experiments, args.sample_list, args.output)
