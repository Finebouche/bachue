#!/bin/bash
# Merges the parts of the jobs of the studies (outputs/<kind>/hpo/parts/<dataset>/<study>/) into the databases of the
# datasets (outputs/<kind>/hpo/<dataset>_db.sqlite3): python -m bachue.hpo.studies merge [kinds]. Run from the root of a
# checkout of bachue, after the jobs of the studies (slurm/submit_hpo_rules.sh, submit_hpo_readouts.sh do it).
#   bash slurm/merge_studies.sh [kinds]
set -euo pipefail
cd "$(dirname "$0")/.."
set +u
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate hag_env
set -u
PYTHONPATH=$PWD:${HAG_DIR:-$HOME/hag_core} python -m bachue.hpo.studies merge "$@"
