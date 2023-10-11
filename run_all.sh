for FILE in config/config_*.yaml;
do
  echo "Running ${FILE}"
  cp -f "${FILE}" config/config.yaml
  snakemake --unlock
  snakemake -j 10 -k --rerun-incomplete
done