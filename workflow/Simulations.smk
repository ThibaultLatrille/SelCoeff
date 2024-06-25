import os
import sys

sys.path.append(".")

configfile: "config/simulations_config.yaml"

FOLDER = os.path.abspath(".")
EXP_FOLDER = f"{FOLDER}/data_simulations"
from scripts.libraries import CategorySNP

model = config['MODEL']
assert model in ["polyDFE_D", "polyDFE_C", "polyDFE_B"]
polyDFE_param = f"-m {model.split('_')[-1]}"
polyDFE_param += " 6" if model == "polyDFE_D" else ""

SIMULATIONS = {sim: sim_param for sim, sim_param in config["SIMULATIONS"].items()}
for simulation in SIMULATIONS:
    os.makedirs(f"{EXP_FOLDER}/analysis/{simulation}",exist_ok=True)

CATEGORIES = CategorySNP("MutSel",bound_file="",bins=3).non_syn_list
assert len(CATEGORIES) == 3
print(f"Categories: {CATEGORIES}")
print(f"Simulations: {SIMULATIONS}")

rule all:
    input:
        f"{EXP_FOLDER}/results.tsv",


def cat_to_param(cat):
    dico_cat = {"neg": "negative", "weak": "neutral", "pos": "positive"}
    return dico_cat[cat]


rule simul_DFE:
    output:
        sfs=f"{EXP_FOLDER}/sfs/{{simulation}}/{{cat}}.sfs"
    params:
        script=f"{FOLDER}/scripts/simul_DFE.py",
        dfe=f"{EXP_FOLDER}/DFE.Homo_sapiens.AFR.MutSel.log.gz",
        sfs=f"{EXP_FOLDER}/sfs/control/{{cat}}.sfs",
        fenetre=lambda wildcards: cat_to_param(wildcards.cat),
        simu=lambda wildcards: SIMULATIONS[wildcards.simulation]
    shell:
        "python3 {params.script} -s {params.sfs} -d {params.dfe} -f {params.fenetre} {params.simu} --output {output.sfs}"


rule polyDFE:
    input:
        sfs=f"{EXP_FOLDER}/sfs/{{simulation}}/{{cat}}.sfs",
        polyDFE=f"{FOLDER}/utils/polyDFE/polyDFE-2.0-macOS-64-bit",
        init_file=f"{FOLDER}/config/{model}_init.tsv",
        range_file=f"{FOLDER}/config/{model}_range.tsv"
    output:
        out=f"{EXP_FOLDER}/analysis/{{simulation}}/{{cat}}.{model}.out",
        stderr=f"{EXP_FOLDER}/analysis/{{simulation}}/{{cat}}.{model}.stderr"
    shell:
        "{input.polyDFE} -d {input.sfs} -i {input.init_file} 1 -r {input.range_file} 1 {polyDFE_param} 1> {output.out} 2> {output.stderr}"

rule parse_results_DFE:
    input:
        script=f"{FOLDER}/scripts/simulations_parse_results_DFE.py",
        postprocessing=f"{FOLDER}/scripts/postprocessing.R",
        modelfile=lambda wildcards: expand(rules.polyDFE.output.out,cat=CATEGORIES,simulation=wildcards.simulation),
    output:
        tsv=f"{EXP_FOLDER}/analysis/{{simulation}}.{model}.tsv"
    shell:
        "python3 {input.script} --postprocessing {input.postprocessing} --input {input.modelfile} --output {output.tsv}"

rule plot_simulations:
    input:
        script=f"{FOLDER}/scripts/simulations_plot.py",
        bounds=f"{EXP_FOLDER}/DFE.Homo_sapiens.AFR.MutSel.tsv",
        tsv=lambda wildcards: expand(rules.parse_results_DFE.output.tsv,simulation=SIMULATIONS),
    output:
        tsv=f"{EXP_FOLDER}/results.tsv"
    shell:
        "python3 {input.script} --tsv {input.tsv} --bounds {input.bounds} --output {output.tsv}"
