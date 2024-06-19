import os
import sys
import pandas as pd

sys.path.append(".")
from scripts.libraries import CategorySNP

configfile: "config/config.yaml"
pop2sp = {}
SFS_LIST, LIST_SUPP, VCF_LIST, MASK_LIST = [], [], [], []
FOLDER = os.path.abspath(".")
FASTA_FOLDER = f"{FOLDER}/data_div/omm_NT_fasta.v10c_116"
TREE_FOLDER = f"{FOLDER}/data_div/omm_RooTree.v10b_116"
EXP_FOLDER = f"{FOLDER}/experiments/{config['experiment']}"
os.makedirs(EXP_FOLDER,exist_ok=True)
sample_list = f"{FOLDER}/config/{config['SAMPLE_LIST']}"
model = config['MODEL']
assert model in ["polyDFE_D", "polyDFE_C", "polyDFE_B"]
USE_DIVERGENCE = "USE_DIVERGENCE" in config and config['USE_DIVERGENCE']
if USE_DIVERGENCE:
    assert config['ALPHA']
polyDFE_param = f"-m {model.split('_')[-1]}"
polyDFE_param += " 6" if model == "polyDFE_D" else ""
polyDFE_param += " -w" if not USE_DIVERGENCE else ""
bins = config['bins']
windows = config['windows']
ontology_subset = ""
method_cats = {m: CategorySNP(m,bound_file="",bins=bins,windows=windows).all() for m in config['METHOD_LIST']}
binning = (f"bins{bins}" if windows == 0 else f"windows{bins}x{windows}")
if 'mask_adaptation' in config and config['mask_adaptation']:
    level = "site" if config['mask_adaptation'] == "site" else "gene"
    binning += "NonAdaptive" + level.capitalize()
    MASK_LIST.append(f"{FOLDER}/data_processed/Mask/adaptation_{level}.tsv.gz")
if "mask_identity" in config and config['mask_identity'] > 0.0:
    binning += f"Mask{config['mask_identity']}"
    MASK_LIST.append(f"{FOLDER}/data_processed/Mask/identity.{config['mask_identity']}.tsv.gz")
if "ontology_subset" in config and config['ontology_subset'] != "":
    ontology_subset = config['ontology_subset']
    binning += ontology_subset
    MASK_LIST.append(f"{FOLDER}/data_processed/Mask/Ontology-{ontology_subset}.tsv.gz")
if "mask_unaligned_flanks" in config and config['mask_unaligned_flanks'] > 0:
    binning += f"MaskFlanks{config['mask_unaligned_flanks']}"
if "mask_CpG" in config and config['mask_CpG']:
    binning += "MaskCpG"

output_supp = "OUTPUT_SUPP" in config and config['OUTPUT_SUPP']
for row_id, row in pd.read_csv(sample_list,sep="\t").iterrows():
    pop2sp[row['SampleName']] = row['Species']
    if output_supp:
        LIST_SUPP.append(f"{FOLDER}/supp/pdf_subs/{row['Species']}.{row['SampleName']}/{binning}-0.pdf")
        LIST_SUPP.append(f"{FOLDER}/supp/pdf_poly/{row['Species']}.{row['SampleName']}/{binning}-0.pdf")
        LIST_SUPP.append(f"{FOLDER}/supp/ontology_subs/{row['Species']}.{row['SampleName']}.tsv")
        LIST_SUPP.append(f"{FOLDER}/supp/ontology_poly/{row['Species']}.{row['SampleName']}.tsv")
    vcf = f"{FOLDER}/data_poly/{row['Species']}.{row['SampleName']}.vcf.gz"
    if os.path.exists(vcf):
        VCF_LIST.append(vcf)
    for method in config['METHOD_LIST']:
        SFS_LIST.append(f"{EXP_FOLDER}/analysis/{row['Species']}.{row['SampleName']}.{method}")

if output_supp:
    LIST_SUPP.append(f"{FOLDER}/data_processed/upset_plot/{binning}.pdf")
    LIST_SUPP.append(f"{FOLDER}/data_processed/SNP_effect/{binning}_clinically/output.tsv")
    LIST_SUPP.append(f"{FOLDER}/data_processed/experimental_DFE.pdf")
    LIST_SUPP.append(f"{FOLDER}/data_processed/population_DFE.pdf")

wildcard_constraints:
    # constrain to only alphanumeric characters and underscore
    popu=r"[a-zA-Z_]+",species=r"[a-zA-Z_]+",method=r"[a-zA-Z]+"

rule all:
    input:
        f"{FOLDER}/data_processed/ontology.tsv",
        f"{FOLDER}/data_processed/merged_opportunities/{binning}.tsv",
        f"{FOLDER}/data_processed/merged_bounds/{binning}.tsv",
        f"{EXP_FOLDER}/supp-mat.pdf",
        f"{FOLDER}/data_processed/trimmed_tree_{config['SAMPLE_LIST']}.tree",
        LIST_SUPP


rule sfs:
    input:
        script=f"{FOLDER}/scripts/plot_sfs.py",
        tsv=rules.annotate_vcf.output.tsv,
        opportunities=rules.opportunities.output.tsv,
        bounds=rules.annotate_vcf.output.bounds,
        subs=lambda wildcards: rules.substitutions_mapping.output.tsv if reconstruct_substitutions(wildcards) else []
    output:
        dir=directory(f"{EXP_FOLDER}/analysis/{{species}}.{{popu}}.{{method}}"),
        pdf=f"{EXP_FOLDER}/analysis/{{species}}.{{popu}}.{{method}}-sfs.pdf",
        tsv=f"{EXP_FOLDER}/analysis/{{species}}.{{popu}}.{{method}}-sfs-summary-stats.tsv"
    params:
        config=f"--nbr_replicates {config['nbr_replicates']} --subsample {config['subsample']}",
        substitutions=lambda wildcards, input, output: f"--substitutions {input.subs}" if reconstruct_substitutions(wildcards) else ""
    shell:
        "python3 {input.script} {params.substitutions} --bins {bins} --windows {windows} --bounds {input.bounds} {params.config} --tsv {input.tsv} --pop {wildcards.popu} --method {wildcards.method} --opportunities {input.opportunities} --output_tsv {output.tsv} --output_pdf {output.pdf} --output_dir {output.dir}"


rule polyDFE:
    input:
        sfs=rules.sfs.output.dir,
        polyDFE=f"{FOLDER}/utils/polyDFE/polyDFE-2.0-macOS-64-bit",
        init_file=f"{FOLDER}/config/{model}_init.tsv",
        range_file=f"{FOLDER}/config/{model}_range.tsv"
    output:
        out=f"{EXP_FOLDER}/analysis/{{species}}.{{popu}}.{{method}}/{{cat}}.{model}.out",
        stderr=f"{EXP_FOLDER}/analysis/{{species}}.{{popu}}.{{method}}/{{cat}}.{model}.stderr"
    params:
        sfs=lambda wildcards, input, output: os.path.join(input.sfs,f"{wildcards.cat}.sfs")
    shell:
        "{input.polyDFE} -d {params.sfs} -i {input.init_file} 1 -r {input.range_file} 1 {polyDFE_param} 1> {output.out} 2> {output.stderr}"


rule parse_results_DFE:
    input:
        script=f"{FOLDER}/scripts/parse_results_DFE.py",
        postprocessing=f"{FOLDER}/scripts/postprocessing.R",
        modelfile=lambda wildcards: expand(rules.polyDFE.output.out,species=wildcards.species,popu=wildcards.popu,
            method=wildcards.method,cat=method_cats[wildcards.method]),
        substitutions=rules.substitutions_mapping.output.tsv,
        bounds=rules.annotate_vcf.output.bounds
    output:
        pdf=f"{EXP_FOLDER}/analysis/{{species}}.{{popu}}.{{method}}.{model}.pdf",
        tsv=f"{EXP_FOLDER}/analysis/{{species}}.{{popu}}.{{method}}.{model}.tsv"
    shell:
        "python3 {input.script} --postprocessing {input.postprocessing} --substitutions {input.substitutions} --bins {bins} --windows {windows} --bounds {input.bounds} --input {input.modelfile} --method {wildcards.method} --output {output.pdf}"

rule plot_dfe_heatmap:
    input:
        script=f"{FOLDER}/scripts/plot_heatmap.py",
        tsv_theta=map(lambda p: f"{p}-sfs-summary-stats.tsv",SFS_LIST),
        tsv_dfe=map(lambda p: f"{p}.{model}.tsv",SFS_LIST),
        tsv_mut_rate=f"{FOLDER}/config/mutation_rates.tsv",
        sample_list=sample_list
    output:
        tsv=f"{EXP_FOLDER}/results/Theta.results.tsv"
    shell:
        "python3 {input.script} --tsv_theta {input.tsv_theta} --tsv_dfe {input.tsv_dfe} --tsv_mut_rate {input.tsv_mut_rate} --sample_list {input.sample_list} --bins {bins} --windows {windows} --output {output.tsv}"

rule plot_multiple_regression:
    input:
        script=f"{FOLDER}/scripts/plot_multiple_regression.py",
        tsv=map(lambda p: f"{p}.{model}.tsv",SFS_LIST),
        sample_list=sample_list
    output:
        tsv=f"{EXP_FOLDER}/results/regression.results.tsv"
    shell:
        "python3 {input.script} --tsv {input.tsv} --sample_list {input.sample_list} --output {output.tsv}"


