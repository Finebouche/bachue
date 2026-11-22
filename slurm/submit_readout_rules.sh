#!/bin/bash
# Learning rules of the readout of the next phrase (bachue.performances.learning_rules_evaluation) on PlaFRIM, one job per
# condition (reservoirs and readouts of the studies of the readout, BACHUE_SEARCH_READOUT=1, or reservoirs of the studies
# with the linear ridge readout, BACHUE_SEARCH_READOUT=0), on the zonda nodes (no rare resource), then the merge of their
# results and the paired comparisons with ridge. Run from the root of a checkout of bachue:
#   BACHUE_TASK=next_phrase BACHUE_WITH_CALLS=1 BACHUE_SEARCH_READOUT=1 [BACHUE_LEAKY_RESERVOIRS=1] bash slurm/submit_readout_rules.sh [sizes] [nb_seeds]
#   (default 500,1000 and 10 seeds; BACHUE_EVAL_RULES, BACHUE_ONLINE_MAX_EPOCHS and BACHUE_EVAL_RULES_SUFFIX
#   passed to the jobs; CPUS cores and MEM memory of each job, its configurations trained in CPUS processes of one
#   thread; TIME the time limit of each job, default 1 day)
# The rules on the monomials of the state (order 2), on the reservoirs of the ridge studies, the online rules for at most
# 1000 epochs (a kernel of ~1 GB per process):
#   BACHUE_TASK=next_phrase BACHUE_WITH_CALLS=1 BACHUE_SEARCH_READOUT=0 [BACHUE_LEAKY_RESERVOIRS=1] \
#   BACHUE_EVAL_RULES=ridge,ridge:2,nlms,nlms:2,neh,neh:2 BACHUE_ONLINE_MAX_EPOCHS=1000 \
#   BACHUE_EVAL_RULES_SUFFIX=_orders CPUS=16 MEM=96G bash slurm/submit_readout_rules.sh
set -euo pipefail
source "$(dirname "$0")/env_names.sh"  # (the former names HAG_CANARY_*, until the jobs that use them are done)
SIZES=${1:-500,1000}
NB_SEEDS=${2:-10}
MAX_RUNNING=${MAX_RUNNING:-40}
CPUS=${CPUS:-8}
MEM=${MEM:-48G}
TIME=${TIME:-1-00:00:00}
NODES=(--constraint=zonda)
cd "$(dirname "$0")/.."
mkdir -p slurm/logs slurm/tasks

set +u
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate hag_env
set -u
rule=${BACHUE_STUDY_RULE:-}
TASKS=slurm/tasks/readout_rules_$(date +%Y%m%d_%H%M%S)_${BACHUE_DATASET:-marron1}_${BACHUE_LEAKY_RESERVOIRS:-0}${rule:+_${rule//:/-}}${BACHUE_EVAL_RULES_SUFFIX:-}.txt
PYTHONPATH=$PWD:${HAG_DIR:-$HOME/hag_core} \
    python -m bachue.performances.learning_rules_evaluation list "$SIZES" 2>/dev/null | grep -E "^[^ ]+ [0-9]+$" > "$TASKS"
n=$(wc -l < "$TASKS")

array=$(sbatch --parsable "${NODES[@]}" --array=0-$((n - 1))%$MAX_RUNNING --cpus-per-task=$CPUS --mem=$MEM \
    --time=$TIME --job-name=canary-readout-rules \
    --export=ALL,BACHUE_JOB_MODE=readout_rules,BACHUE_WORKERS=$CPUS,BACHUE_RULE_PROCESSES=$CPUS,BACHUE_TASK_FILE="$TASKS" \
    slurm/canary.sbatch "$NB_SEEDS")
echo "readout rules: job array $array ($n conditions, $TASKS)"
merge=$(sbatch --parsable "${NODES[@]}" --cpus-per-task=1 --mem=8G --time=01:00:00 --dependency=afterany:$array \
    --job-name=readout-rules-merge --export=ALL,BACHUE_JOB_MODE=readout_rules,BACHUE_WORKERS=1 \
    --output=slurm/logs/%x-%j.out slurm/canary.sbatch merge)
echo "merge and comparisons: job $merge (after $array)"
