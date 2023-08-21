import pandas as pd
from scipy.stats import wilcoxon


def open_results(exp):
    tmp_df = pd.read_csv(f"experiments/{exp}/regression-MutSel/results.tsv", sep="\t", index_col=0)
    return tmp_df["bayes_P-Spos_P-pos"]


control = "3bins-mC"
dico_df = {control: open_results(control)}
cases = ["3bins-mC-OntoCytoplasme", "3bins-mC-OntoNucleus", "3bins-mC-OntoMembrane", "3bins-mC-OntoProtBind",
         "3bins-mC-OntoRegTrans"]

for case in cases:
    dico_df[case] = open_results(case)
    res = wilcoxon(dico_df[control], dico_df[case], alternative="two-sided")
    print(case, res, (dico_df[case] / dico_df[control]).mean())

df = pd.DataFrame(dico_df)
