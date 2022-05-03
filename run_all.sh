#!/usr/bin/env bash
for FILE in config/config_*.yaml;
do
  cp -f "${FILE}" config/config.yaml
  snakemake -j 8 -k --rerun-incomplete
done
