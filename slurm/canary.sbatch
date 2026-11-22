#!/bin/bash
# Canary songs on PlaFRIM (bachue.tasks: boundaries and next phrase), from a checkout of bachue:
#   - BACHUE_JOB_MODE=hpo (default): optimization of the reservoirs python -m bachue.hpo.<study> <model> <size>, the
#     study given by BACHUE_STUDY_RULE (or, with BACHUE_KERNEL, kernel_study: <model> <size> the kernel and its
#     window): none, readout_window_study (the readout BACHUE_STUDY_READOUT, or "last" -> "last");
#     a rule on the states or all the monomials ("nlms", "ridge:2"), learning_rules_study; a rule and a selection of products
#     ("nlms:pca:64"), partial_order2_study. BACHUE_WORKERS processes sharing the cores; in a job array,
#     "<model> <size>" is the line SLURM_ARRAY_TASK_ID (from 0) of the file BACHUE_TASK_FILE;
#   - BACHUE_JOB_MODE=evaluate: final evaluation python -m bachue.performances.phrase_boundaries_evaluation [sizes] [nb_seeds];
#   - BACHUE_JOB_MODE=next_phrase: prediction of the next phrase python -m bachue.performances.readout_window_evaluation [sizes]
#     [nb_seeds] (reservoirs of the next-phrase studies, BACHUE_TASK=next_phrase);
#   - BACHUE_JOB_MODE=readout_rules: learning rules of the readout of the next phrase python -m
#     bachue.performances.learning_rules_evaluation (slurm/submit_readout_rules.sh);
#   - BACHUE_JOB_MODE=kernel_evaluation: final evaluation of a kernel study (BACHUE_KERNEL, bachue.hpo.kernel_study)
#     python -m bachue.performances.kernels_evaluation <kernel> <window>;
#   - BACHUE_JOB_MODE=monomial_selection: selection of the monomials of the readout of order 2 python -m
#     bachue.performances.partial_order2_evaluation (slurm/submit_monomial_selection.sh), and
#     BACHUE_JOB_MODE=monomial_selection_rules: the online rules on the selected monomials (its command "rules"; the
#     same script, MODULE).
# The Optuna database of a study is in /dev/shm during its job (SQLite does not support the network file systems well,
# and the suroit nodes have no local disk), copied from and back to its part (every 10 minutes and at the end); the
# evaluations read the databases of the datasets, outputs/<kind>/hpo/<dataset>_db.sqlite3 (bachue.hpo.studies).
#SBATCH --job-name=canary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm/logs/%x-%A_%a.out
set -euo pipefail
# (the former names HAG_CANARY_*, until the jobs that use them are done: here, and after the variables of a task)
ENV_NAMES=${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/slurm/env_names.sh
source "$ENV_NAMES"
if [ -n "${BACHUE_TASK_FILE:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    read -r MODEL SIZE EXTRA < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$BACHUE_TASK_FILE")
    if [[ "$EXTRA" == *=* ]]; then
        # (optional fields KEY=VALUE: variables of the study, e.g. slurm/submit_hpo_rules.sh)
        for assignment in $EXTRA; do export "$assignment"; done
        source "$ENV_NAMES"
    elif [ -n "$EXTRA" ]; then
        # (optional third and fourth fields: the readout of the study and the leak, slurm/submit_hpo_readouts.sh)
        read -r READOUT LEAK <<< "$EXTRA"
        export BACHUE_STUDY_READOUT=$READOUT BACHUE_LEAKY_RESERVOIRS=$LEAK
        # (training rows "all": all the states of the phrases, ~1 million rows, ~9 GB at 1000 neurons, more memory per
        # worker)
        case "$READOUT" in all:*) export BACHUE_WORKER_MEM_GB=${BACHUE_WORKER_MEM_GB_ALL_ROWS:-28} ;; esac
    fi
    set -- "$MODEL" "$SIZE" "$@"
fi
set +u
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate hag_env
set -u
cd "$SLURM_SUBMIT_DIR"
# this checkout and hag (HAG_DIR, default ~/hag_core, branch main), not the editable install of ~/hag (branch rl)
export PYTHONPATH=$PWD:${HAG_DIR:-$HOME/hag_core}

# threads of each worker (NumPy and JAX), on the cores of the job (all those of the node with sbatch --exclusive);
# BACHUE_WORKERS=auto: as many workers as the cores (2 threads each) and the memory of the job
# (BACHUE_WORKER_MEM_GB GB each, default 11: about 9.5 GB at 1000 neurons) allow
CPUS=${SLURM_CPUS_ON_NODE:-$SLURM_CPUS_PER_TASK}
WORKERS=${BACHUE_WORKERS:-4}
if [ "$WORKERS" = auto ]; then
    WORKERS=$((CPUS / 2))
    # memory of the job in MB (that of the node with sbatch --mem=0)
    MEMORY=${SLURM_MEM_PER_NODE:-0}
    if [ "$MEMORY" -eq 0 ]; then MEMORY=$(($(awk '/^MemTotal/ {print $2}' /proc/meminfo) / 1024)); fi
    BY_MEMORY=$((MEMORY / 1024 / ${BACHUE_WORKER_MEM_GB:-11}))
    if [ "$BY_MEMORY" -lt "$WORKERS" ]; then WORKERS=$((BY_MEMORY > 0 ? BY_MEMORY : 1)); fi
fi
THREADS=$((CPUS / WORKERS))
export BACHUE_WORKERS=$WORKERS OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$THREADS"
# numba cache of librosa per job: the jobs writing it at the same time in the shared environment fail (stale file handle)
export NUMBA_CACHE_DIR=/dev/shm/numba_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}

mkdir -p slurm/logs
if [ "${BACHUE_JOB_MODE:-hpo}" = hpo ]; then
    # the study in a database of its own (several jobs writing the same SQLite file would corrupt it), its part
    # outputs/<kind>/hpo/parts/<dataset>/<study>/, merged into the database of the dataset afterwards
    # (slurm/merge_studies.sh: python -m bachue.hpo.studies merge); the study starts from that of the database of the
    # dataset if it is there (bachue.hpo.studies.run_study)
    PART=$(python -m bachue.hpo.studies part "$1" "$2" 2>/dev/null | tail -n 1)
    export BACHUE_JOB_STUDIES_DIR=/dev/shm/bachue_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
    mkdir -p "$PART" "$BACHUE_JOB_STUDIES_DIR"
    rsync -a "$PART/" "$BACHUE_JOB_STUDIES_DIR/"
    sync_back() { rsync -a "$BACHUE_JOB_STUDIES_DIR/" "$PART/"; }
    trap 'sync_back; rm -rf "$BACHUE_JOB_STUDIES_DIR" "$NUMBA_CACHE_DIR"' EXIT
    trap 'exit 143' TERM
    (while true; do sleep 600; sync_back; done) &
else
    trap 'rm -rf "$NUMBA_CACHE_DIR"' EXIT
fi

echo "[slurm] job ${SLURM_JOB_ID} ${SLURM_ARRAY_TASK_ID:-}: ${BACHUE_JOB_MODE:-hpo} $*, on $(hostname)," \
    "$CPUS cores, $WORKERS workers"
if [ "${BACHUE_JOB_MODE:-hpo}" = evaluate ]; then
    python -m bachue.performances.phrase_boundaries_evaluation "$@"
elif [ "${BACHUE_JOB_MODE:-hpo}" = next_phrase ]; then
    # in a job array: one condition ("<condition> <size>" of BACHUE_TASK_FILE)
    if [ -n "${BACHUE_TASK_FILE:-}" ]; then set -- run "$@"; fi
    python -m bachue.performances.readout_window_evaluation "$@"
elif [ "${BACHUE_JOB_MODE:-hpo}" = readout_rules ]; then
    if [ -n "${BACHUE_TASK_FILE:-}" ]; then set -- run "$@"; fi
    case "${BACHUE_STUDY_RULE:-}" in
        *:*[!0-9]*) python -m bachue.performances.partial_order2_evaluation studies "$@" ;;  # (<rule>:<selection>)
        *) python -m bachue.performances.learning_rules_evaluation "$@" ;;
    esac
elif [ "${BACHUE_JOB_MODE:-hpo}" = kernel_evaluation ]; then
    python -m bachue.performances.kernels_evaluation "$@"  # (<kernel> <window>, with BACHUE_KERNEL)
elif [ "${BACHUE_JOB_MODE:-hpo}" = monomial_selection ] || [ "${BACHUE_JOB_MODE:-hpo}" = monomial_selection_rules ]; then
    if [ -n "${BACHUE_TASK_FILE:-}" ]; then set -- run "$@"; fi
    if [ "$BACHUE_JOB_MODE" = monomial_selection_rules ]; then set -- rules "$@"; fi
    python -m bachue.performances.partial_order2_evaluation "$@"
else
    case "${BACHUE_KERNEL:+kernel}${BACHUE_STUDY_RULE:-}" in
        kernel*) study=kernel_study ;;  # (a kernel of the inputs, without reservoir: <kernel> <window>)
        "") study=readout_window_study ;;
        *:*[!0-9]*) study=partial_order2_study ;;  # (<rule>:<selection>)
        *) study=learning_rules_study ;;  # (<rule> or <rule>:<order>)
    esac
    python -m "bachue.hpo.$study" "$@"
fi
