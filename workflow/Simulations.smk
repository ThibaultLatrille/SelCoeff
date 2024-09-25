import os
import sys
import pandas as pd
import numpy as np

sys.path.append(".")

configfile: "config/simulations_config.yaml"

FOLDER = os.path.abspath(".")
EXP_FOLDER = f"{FOLDER}/data_simulations"
EXPERIMENT = config["EXPERIMENT"]

from scripts.libraries import CategorySNP

model = config['MODEL']
assert model in ["polyDFE_D", "polyDFE_C", "polyDFE_B"]
polyDFE_param = f"-m {model.split('_')[-1]}"
polyDFE_param += " 6" if model == "polyDFE_D" else ""

SIMULATIONS = {sim: sim_param for sim, sim_param in config["SIMULATIONS"].items()}

CATEGORIES = CategorySNP("MutSel",bound_file="",bins=3).non_syn_list
assert len(CATEGORIES) == 3
print(f"Categories: {CATEGORIES}")
print(f"Simulations: {SIMULATIONS}")

sample_list = f"{FOLDER}/config/{config['SAMPLE_LIST']}"
pop2sp = {}
LIST_RESULTS = []
for row_id, row in pd.read_csv(sample_list,sep="\t").iterrows():
    pop2sp[row['SampleName']] = row['Species']
    LIST_RESULTS.append(f"{EXP_FOLDER}/results/{row['Species']}.{row['SampleName']}.tsv")
    for simulation in SIMULATIONS:
        os.makedirs(f"{EXP_FOLDER}/analysis/{simulation}",exist_ok=True)

rule all:
    input: f"{EXP_FOLDER}/results.tsv"


def cat_to_param(cat):
    dico_cat = {"neg": "negative", "weak": "neutral", "pos": "positive"}
    return dico_cat[cat]


rule simul_SFS:
    input:
        script=f"{FOLDER}/scripts/simul_DFE.py",
        dfe=f"{FOLDER}/data_processed/opportunities_bins3Mask0.9/DFE.{{species}}.{{popu}}.MutSel.log.gz",
        parsed_DFE=f"{FOLDER}/experiments/{EXPERIMENT}/analysis/{{species}}.{{popu}}.MutSel.polyDFE_C.tsv",
        sfs=f"{FOLDER}/experiments/{EXPERIMENT}/analysis/{{species}}.{{popu}}.MutSel/{{cat}}.sfs",
    output:
        sfs=f"{EXP_FOLDER}/sfs/{{simulation}}.{{species}}.{{popu}}/{{cat}}.sfs"
    params:
        fenetre=lambda wildcards: cat_to_param(wildcards.cat),
        simu=lambda wildcards: SIMULATIONS[wildcards.simulation]
    shell:
        "python3 {input.script} -p {input.parsed_DFE} -s {input.sfs} -d {input.dfe} -f {params.fenetre} {params.simu} --output {output.sfs}"


rule polyDFE:
    input:
        sfs=rules.simul_SFS.output.sfs,
        polyDFE=f"{FOLDER}/utils/polyDFE/polyDFE-2.0-macOS-64-bit",
        init_file=f"{FOLDER}/config/{model}_init.tsv",
        range_file=f"{FOLDER}/config/{model}_range.tsv"
    output:
        out=f"{EXP_FOLDER}/analysis/{{simulation}}/{{species}}.{{popu}}/{{cat}}.{model}.out",
        stderr=f"{EXP_FOLDER}/analysis/{{simulation}}/{{species}}.{{popu}}/{{cat}}.{model}.stderr"
    shell:
        "{input.polyDFE} -d {input.sfs} -i {input.init_file} 1 -r {input.range_file} 1 {polyDFE_param} 1> {output.out} 2> {output.stderr}"

rule parse_results_DFE:
    input:
        script=f"{FOLDER}/scripts/simulations_parse_results_DFE.py",
        postprocessing=f"{FOLDER}/scripts/postprocessing.R",
        modelfile=lambda wildcards: expand(rules.polyDFE.output.out,cat=CATEGORIES,simulation=wildcards.simulation,species=wildcards.species,popu=wildcards.popu)
    output:
        tsv=f"{EXP_FOLDER}/analysis/{{simulation}}.{{species}}.{{popu}}.{model}.tsv"
    shell:
        "python3 {input.script} --postprocessing {input.postprocessing} --input {input.modelfile} --output {output.tsv}"

rule parse_simulations:
    input:
        script=f"{FOLDER}/scripts/simulations_parse.py",
        bounds=f"{FOLDER}/data_processed/opportunities_bins3Mask0.9/DFE.{{species}}.{{popu}}.MutSel.tsv",
        control=f"{FOLDER}/experiments/{EXPERIMENT}/analysis/{{species}}.{{popu}}.MutSel.polyDFE_C.tsv",
        tsv=lambda wildcards: expand(rules.parse_results_DFE.output.tsv,simulation=SIMULATIONS,species=wildcards.species,popu=wildcards.popu)
    output:
        tsv=f"{EXP_FOLDER}/results/{{species}}.{{popu}}.tsv"
    shell:
        "python3 {input.script} --control {input.control} --tsv {input.tsv} --bounds {input.bounds} --output {output.tsv}"

rule merge_simulations:
    input:
        script=f"{FOLDER}/scripts/simulations_merge.py",
        tsv=LIST_RESULTS
    output:
        tsv=f"{EXP_FOLDER}/results.tsv"
    shell:
        "python3 {input.script} --tsv {input.tsv} --output {output.tsv}"
