#!/bin/bash

BACKBONES=(
    "GCN"
    "GAT"
    "GIN"
    "GraphGPS"
    "GrokFormer"
    "DualFormer"
)

MODELS=(
    "GNN"
    "CIF"
    "DIR"
    "CAL"
    "ICL"
    "ACE"
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

FAILED_LOG="failed_runs.txt"

run_experiment() {
    local dataset=$1
    local backbone=$2
    local model=$3
    echo "----------------------------------------"
    echo "Running: $dataset | $backbone | $model"
    echo "----------------------------------------"
    python run_experiments.py --dataset "$dataset" --backbone "$backbone" --model "$model"
    if [ $? -ne 0 ]; then
        echo "FAILED: $dataset | $backbone | $model" >> "$FAILED_LOG"
    fi
}

echo "========================================"
echo "Starting experiments"
echo "========================================"
for dataset in "${DATASETS[@]}"; do
    for backbone in "${BACKBONES[@]}"; do
        for model in "${MODELS[@]}"; do
            run_experiment "$dataset" "$backbone" "$model"
        done
    done
done

echo "========================================"
echo "Experiments done!"
if [ -f "$FAILED_LOG" ]; then
    echo "Failed runs:"
    cat "$FAILED_LOG"
fi
echo "========================================"
