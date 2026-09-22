#!/bin/bash

if [ "$#" -gt 0 ]; then
    N_SAMPLES_VALUES=("$@")
else
    N_SAMPLES_VALUES=(2 4 8 16)
fi

# Sanity-check: each N_SAMPLES value must be a positive integer.
for v in "${N_SAMPLES_VALUES[@]}"; do
    if ! [[ "$v" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: N_SAMPLES value '$v' is not a positive integer." >&2
        echo "Usage: $0 [N_SAMPLES ...]   (e.g. '$0 2'  or  '$0 2 4 8 16')" >&2
        exit 1
    fi
done

# Per-invocation failure log, named after the value(s) so parallel sessions
# each write their own file instead of interleaving into a shared one.
FAILED_LOG="failed_runs_sensitivity_${N_SAMPLES_VALUES[*]}.txt"
FAILED_LOG="${FAILED_LOG// /_}"

BACKBONES=(
    "GCN"
    "GAT"
    "GIN"
    "GraphGPS"
    "GrokFormer"
    "DualFormer"
)

MODELS=(
    "CIF"
)

DATASETS=(
    "Graph_SST2"
    "Molhiv"
    "SPMotif_b_05"
    "SPMotif_b_07"
    "SPMotif_b_09"
    "SYN_multi_b_01"
    "SYN_multi_b_03"
    "SYN_multi_b_05"
    "SYN_multi_b_07"
    "SYN_multi_b_09"
    "MNIST_75sp_n02"
    "MNIST_75sp_n04"
    "MNIST_75sp_n06"
    "MNIST_75sp_n08"
)

run_experiment() {
    local dataset=$1
    local backbone=$2
    local model=$3
    local n_samples=$4
    echo "----------------------------------------"
    echo "Running: $dataset | $backbone | $model | N_SAMPLES=$n_samples"
    echo "----------------------------------------"
    N_SAMPLES="$n_samples" python run_experiments.py --dataset "$dataset" --backbone "$backbone" --model "$model"
    if [ $? -ne 0 ]; then
        echo "FAILED: $dataset | $backbone | $model | N_SAMPLES=$n_samples" >> "$FAILED_LOG"
    fi
}

echo "========================================"
echo "Starting N_SAMPLES sensitivity analysis"
echo "========================================"
for n_samples in "${N_SAMPLES_VALUES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for backbone in "${BACKBONES[@]}"; do
            for model in "${MODELS[@]}"; do
                run_experiment "$dataset" "$backbone" "$model" "$n_samples"
            done
        done
    done
done

echo "========================================"
echo "Sensitivity analysis done!"
if [ -f "$FAILED_LOG" ]; then
    echo "Failed runs:"
    cat "$FAILED_LOG"
fi
echo "========================================"
