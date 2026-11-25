#!/bin/bash
# Optimization of the reservoirs of the next-phrase prediction for each learning rule of the readout "last" -> "last"
# (bachue.hpo.learning_rules_study, bachue.hpo.partial_order2_study for "<rule>:<selection>", with BACHUE_STUDY_RULE, the hyperparameters of the rule in the search: rate of
# LMS, NLMS and EH, noise of EH, alpha of RLS; studies next_phrase_calls_[leak_]<rule>_<model>_<size>, "<rule>-order<n>"
# for the rules on the monomials "<rule>:<n>", "<rule>-<selection>" for the rules on selected products
# "<rule>:<selection>", e.g. RULES="nlms:pca:64 eh_np_local:scores ridge:random_pairs:2080"), so that the rules are
# compared on reservoirs optimized for each of them
# (the rule "ridge": the studies of the linear ridge readout, without rule, done again here with the same annotations),
# then their final evaluation on the reservoirs of their studies (slurm/submit_readout_rules.sh for each rule and leak,
# slurm/submit_next_phrase.sh for ridge, 10 seeds, two conditions at a time each, from the next BEGIN)
# once all the studies are done. A single job array of all the studies (one per job, MAX_RUNNING at once), the rules on
# the monomials and the largest reservoirs first. Run from the root of a checkout of bachue with the audio and the MFCC
# caches (the compute nodes have no internet):
#   [RULES="ridge lms eh_np_local rls nlms neh ridge:2 nlms:2 neh:2"] [PURE="mean_hag var_hag random_ee random_ei"] \
#   [SIZES="500 1000"] [LEAKS="0 1"] [DATASETS="marron1"] [N_TRIALS=400] [MAX_EPOCHS=3000] [BEGIN=20:00] \
#   [MAX_RUNNING=30] bash slurm/submit_hpo_rules.sh
# DATASETS: "marron1" (the next phrase of the canary songs) and classification datasets of hag ("JapaneseVowels": the
# last state of each instance; its data in bachue/datasets/datasets/JapaneseVowels, downloaded before, the compute nodes having
# no internet; their final evaluations: those of the rules on selected products "<rule>:<selection>" only). After the
# studies, their parts are merged into the databases of the datasets (slurm/merge_studies.sh), then the final
# evaluations are submitted.
# Each job: a whole node (CPUS=all, sbatch --exclusive) of the CPU nodes CONSTRAINT (default zonda, suroit and bora, no
# GPU nor big memory), WORKERS processes optimizing the study (default auto: as many as the cores, 2 threads each, and
# the memory, 11 GB each, 14 for the rules on the monomials, allow), or CPUS cores and MEM memory; time limit TIME
# (default 3 days, the limit outside the weekends). The studies resume where they stopped if resubmitted. MAX_EPOCHS:
# the number of epochs of the online rules chosen up to this one (BACHUE_STUDY_MAX_EPOCHS).
set -euo pipefail
source "$(dirname "$0")/env_names.sh"  # (the former names HAG_CANARY_*, until the jobs that use them are done)
read -r -a RULES <<< "${RULES:-ridge lms eh_np_local rls nlms neh ridge:2 nlms:2 neh:2}"
read -r -a PURE <<< "${PURE:-mean_hag var_hag random_ee random_ei}"
read -r -a SIZES <<< "${SIZES:-500 1000}"
read -r -a LEAKS <<< "${LEAKS:-0 1}"
read -r -a DATASETS <<< "${DATASETS:-marron1}"
N_TRIALS=${N_TRIALS:-400}
MAX_EPOCHS=${MAX_EPOCHS:-3000}
WORKERS=${WORKERS:-auto}
CPUS=${CPUS:-all}
MEM=${MEM:-90G}
if [ "$CPUS" = all ]; then RESOURCES=(--exclusive --mem=0); else RESOURCES=(--cpus-per-task="$CPUS" --mem="$MEM"); fi
CONSTRAINT=${CONSTRAINT:-zonda|suroit|bora}
TIME=${TIME:-3-00:00:00}
BEGIN=${BEGIN:-20:00}
MAX_RUNNING=${MAX_RUNNING:-30}
cd "$(dirname "$0")/.."
mkdir -p slurm/logs slurm/tasks
for dataset in "${DATASETS[@]}"; do
    if [ "$dataset" = marron1 ]; then
        if [ -z "$(find bachue/datasets/datasets/marron1/audio -name "*.wav" -print -quit)" ] || [ ! -f bachue/datasets/datasets/marron1/boundary_features_44100_no_floor.pkl ]; then
            echo "bachue/datasets/datasets/marron1: audio or MFCC cache missing (the compute nodes have no internet)" >&2
            exit 1
        fi
    elif [ -z "$(ls -A "bachue/datasets/datasets/$dataset" 2>/dev/null)" ]; then
        echo "bachue/datasets/datasets/$dataset: missing (download it before: the compute nodes have no internet)" >&2
        exit 1
    fi
done

# one line per study: "<model> <size> BACHUE_STUDY_RULE=<rule> BACHUE_LEAKY_RESERVOIRS=<leak> BACHUE_DATASET=<dataset>
# [BACHUE_WORKER_MEM_GB=14]" (slurm/canary.sbatch), the rules on the monomials first, then the largest sizes
TASKS=slurm/tasks/hpo_rules_$(date +%Y%m%d_%H%M%S).txt
for dataset in "${DATASETS[@]}"; do
for order in 2 1; do for size in $(printf '%s\n' "${SIZES[@]}" | sort -rn); do for leak in "${LEAKS[@]}"; do
    for rule in "${RULES[@]}"; do
        case "$rule" in *:*) [ "$order" = 2 ] || continue; memory=" BACHUE_WORKER_MEM_GB=14" ;;
                        *) [ "$order" = 1 ] || continue; memory="" ;; esac
        assignment="BACHUE_STUDY_RULE=$rule "
        # (the studies of the linear ridge readout: on the songs, those of readout_window_study, without rule; on a
        # classification dataset, those of learning_rules_study with the rule ridge, readout_window_study being for
        # the songs only)
        [ "$rule" = ridge ] && [ "$dataset" = marron1 ] && assignment=""
        for model in "${PURE[@]}"; do
            echo "$model $size ${assignment}BACHUE_LEAKY_RESERVOIRS=$leak BACHUE_DATASET=$dataset$memory"
        done
    done
done; done; done
done > "$TASKS"
n=$(wc -l < "$TASKS")
array=$(BACHUE_TASK=next_phrase BACHUE_WITH_CALLS=1 BACHUE_SEARCH_READOUT=0 BACHUE_N_TRIALS=$N_TRIALS \
    BACHUE_STUDY_MAX_EPOCHS=$MAX_EPOCHS \
    sbatch --parsable --constraint="$CONSTRAINT" --begin="$BEGIN" --array=0-$((n - 1))%$MAX_RUNNING "${RESOURCES[@]}" \
    --time=$TIME --job-name=hpo-rules --export=ALL,BACHUE_TASK_FILE="$TASKS",BACHUE_WORKERS=$WORKERS \
    slurm/canary.sbatch)
echo "studies: job array $array ($n studies, $MAX_RUNNING at once, from $BEGIN; $TASKS)"

SIZE_LIST=$(IFS=,; echo "${SIZES[*]}")
EVALUATIONS="bash slurm/merge_studies.sh; "
for dataset in "${DATASETS[@]}"; do for leak in "${LEAKS[@]}"; do for rule in "${RULES[@]}"; do
    if [ "$rule" = ridge ]; then  # (the linear ridge readout: on the songs only, slurm/submit_next_phrase.sh)
        [ "$dataset" = marron1 ] || continue
        EVALUATIONS+="BACHUE_TASK=next_phrase BACHUE_WITH_CALLS=1 BACHUE_LEAKY_RESERVOIRS=$leak MAX_RUNNING=2"
        EVALUATIONS+=" SBATCH_BEGIN=$BEGIN bash slurm/submit_next_phrase.sh $SIZE_LIST 10; "
        continue
    fi
    case "$rule" in *:*[!0-9]*) ;; *) [ "$dataset" = marron1 ] || continue ;; esac  # (classification: the selections)
    EVALUATIONS+="BACHUE_DATASET=$dataset BACHUE_TASK=next_phrase BACHUE_WITH_CALLS=1 BACHUE_SEARCH_READOUT=0"
    EVALUATIONS+=" BACHUE_LEAKY_RESERVOIRS=$leak BACHUE_STUDY_RULE=$rule BACHUE_ONLINE_MAX_EPOCHS=$MAX_EPOCHS CPUS=16"
    EVALUATIONS+=" MEM=96G MAX_RUNNING=2 SBATCH_BEGIN=$BEGIN bash slurm/submit_readout_rules.sh $SIZE_LIST 10; "
done; done; done
evaluation=$(sbatch --parsable --constraint="$CONSTRAINT" --cpus-per-task=1 --mem=4G --time=02:00:00 \
    --dependency=afterany:$array --job-name=submit-rule-evaluations --output=slurm/logs/%x-%j.out \
    --wrap="$EVALUATIONS")
echo "merge of the studies, then final evaluations (one submit_readout_rules.sh or submit_next_phrase.sh per dataset," \
    "rule and leak): job $evaluation, after $array"
