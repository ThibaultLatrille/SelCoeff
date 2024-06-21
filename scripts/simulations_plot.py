import argparse
from libraries import *


def main(path_input, path_bounds, path_output):
    os.makedirs(os.path.dirname(path_output), exist_ok=True)

    bounds = pd.read_csv(path_bounds, sep="\t")
    assert len(bounds) == 1
    neg0 = bounds["neg"].values[0]
    pos0 = bounds["pos"].values[0]
    weak0 = bounds["weak"].values[0]

    out_dict = defaultdict(list)
    for filepath in sorted(path_input):
        df = pd.read_csv(filepath, sep="\t")
        row_neg0 = df[df["category"] == "neg"]
        row_weak0 = df[df["category"] == "weak"]
        row_pos0 = df[df["category"] == "pos"]

        pos_neg0 = row_neg0["P-Spos"].values[0]
        pos_weak0 = row_weak0["P-Spos"].values[0]
        pos_pos0 = row_pos0["P-Spos"].values[0]

        neg_neg0 = row_neg0["P-Sneg"].values[0]
        neg_weak0 = row_weak0["P-Sneg"].values[0]
        neg_pos0 = row_pos0["P-Sneg"].values[0]

        weak_neg0 = row_neg0["P-Sweak"].values[0]
        weak_weak0 = row_weak0["P-Sweak"].values[0]
        weak_pos0 = row_pos0["P-Sweak"].values[0]

        weak = weak_weak0 * weak0 + weak_neg0 * neg0 + weak_pos0 * pos0
        neg = neg_weak0 * weak0 + neg_neg0 * neg0 + neg_pos0 * pos0
        pos = pos_weak0 * weak0 + pos_neg0 * neg0 + pos_pos0 * pos0

        pos0_pos = pos_pos0 * pos0 / pos
        neg0_neg = neg_neg0 * neg0 / neg
        weak0_weak = weak_weak0 * weak0 / weak
        out_dict["category"].append(os.path.basename(filepath).replace(".tsv", ""))
        out_dict["recall_neg"].append(neg0_neg)
        out_dict["recall_weak"].append(weak0_weak)
        out_dict["recall_pos"].append(pos0_pos)
    df_out = pd.DataFrame(out_dict)
    df_out.to_csv(args.output, sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', required=False, type=str, nargs="+", dest="tsv", help="Input tsv file")
    parser.add_argument('--bounds', required=False, default="", type=str, dest="bounds", help="Input bound file path")
    parser.add_argument('--output', required=False, type=str, dest="output", help="Output tsv file")
    args = parser.parse_args()
    main(args.tsv, args.bounds, args.output)
