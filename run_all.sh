#!/usr/bin/env bash
for FILE in config/config_3cat_Onto*.yaml;
do
  cp -f "${FILE}" config/config.yaml
  snakemake --unlock
  snakemake -j 8 -k --rerun-incomplete
done
