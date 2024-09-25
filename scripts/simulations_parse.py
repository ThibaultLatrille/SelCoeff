import argparse
from libraries import *
from simul_DFE import extract_data


def bayes_factor(dp, neg0, pos0, weak0):
    weak = dp["weak_weak0"] * weak0 + dp["weak_neg0"] * neg0 + dp["weak_pos0"] * pos0
    neg = dp["neg_weak0"] * weak0 + dp["neg_neg0"] * neg0 + dp["neg_pos0"] * pos0
    pos = dp["pos_weak0"] * weak0 + dp["pos_neg0"] * neg0 + dp["pos_pos0"] * pos0
    pos0_pos = dp["pos_pos0"] * pos0 / pos
    neg0_neg = dp["neg_neg0"] * neg0 / neg
    weak0_weak = dp["weak_weak0"] * weak0 / weak
    return neg0_neg, weak0_weak, pos0_pos


def main(path_input, path_control, path_bounds, path_output):
    os.makedirs(os.path.dirname(path_output), exist_ok=True)

    bounds = pd.read_csv(path_bounds, sep="\t")
    assert len(bounds) == 1
    neg0 = bounds["neg"].values[0]
    pos0 = bounds["pos"].values[0]
    weak0 = bounds["weak"].values[0]

    out_dict = defaultdict(list)
    for filepath in sorted(path_input):
        dp = extract_data(filepath)
        neg0_neg, weak0_weak, pos0_pos = bayes_factor(dp, neg0, pos0, weak0)
        out_dict["category"].append(os.path.basename(filepath).split(".")[0])
        out_dict["recall_neg"].append(neg0_neg)
        out_dict["recall_weak"].append(weak0_weak)
        out_dict["recall_pos"].append(pos0_pos)

    dpc = extract_data(path_control)
    neg0_neg, weak0_weak, pos0_pos = bayes_factor(dpc, neg0, pos0, weak0)
    out_dict["category"].append("control")
    out_dict["recall_neg"].append(neg0_neg)
    out_dict["recall_weak"].append(weak0_weak)
    out_dict["recall_pos"].append(pos0_pos)
    df_out = pd.DataFrame(out_dict)
    df_out.to_csv(args.output, sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', required=False, type=str, nargs="+", dest="tsv", help="Input tsv file")
    parser.add_argument('--control', required=False, type=str, dest="control", help="Control file path")
    parser.add_argument('--bounds', required=False, default="", type=str, dest="bounds", help="Input bound file path")
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output tsv file")
    args = parser.parse_args()
    main(args.tsv, args.control, args.bounds, args.output)
