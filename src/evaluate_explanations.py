import argparse
import os
import sys
import json
import csv
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent) + "/data_scripts")
from data_scripts.dataset_names import GET_DATASET, DATASET_N_CLASSES
from hyper_parameters import model_class_hparams
from backbones.MPNNs.gcn import GCN_encoder
from backbones.MPNNs.gat import GAT_encoder
from backbones.MPNNs.gin import GIN_encoder
from backbones.GTs.graph_GPS import GraphGPS_encoder
from backbones.GTs.GrokFormer import GrokFormer_encoder
from backbones.GTs.DualFormer import DualFormer_encoder
from src.baselines.cgnn.cgnn import CGNN
from baselines.cgnn.cal import CAL
from baselines.cgnn.icl import ICL
from baselines.cgnn.ace import ACE
from baselines.cgnn.dir import DIR
from cif.cif import CIF, CIF_NoJ
import warnings
warnings.filterwarnings("ignore")

DEFAULT_DATASETS = [
    "SPMotif_b_05", "SPMotif_b_07", "SPMotif_b_09",
    "SYN_multi_b_01", "SYN_multi_b_03", "SYN_multi_b_05",
    "SYN_multi_b_07", "SYN_multi_b_09",
]
BACKBONE_MAP = {
    "GCN": GCN_encoder, "GAT": GAT_encoder, "GIN": GIN_encoder,
    "GraphGPS": GraphGPS_encoder, "GrokFormer": GrokFormer_encoder,
    "DualFormer": DualFormer_encoder,
}
MODEL_MAP = {
    "DIR": DIR, "CAL": CAL, "ICL": ICL, "ACE": ACE,
    "CIF_NoJ": CIF_NoJ,
    "CIF": CIF,
}
SEEDS = [28, 1999, 1130, 5898, 820] 
_RUN_TAG_ENV = os.environ.get("RUN_TAG", "")
if _RUN_TAG_ENV and not _RUN_TAG_ENV.startswith("_"):
    _RUN_TAG_ENV = "_" + _RUN_TAG_ENV

device = "cuda" if torch.cuda.is_available() else "cpu"


def roc_auc(scores, labels):
    labels = labels.bool()
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)                                   
    ranks = torch.empty_like(scores, dtype=torch.float)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float)
    return ((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def graph_scores(scores, gt):
    n = gt.numel()
    n_pos = int(gt.sum())
    if n_pos == 0 or n_pos == n:        
        return None
    k = n_pos                           
    topk = torch.topk(scores, k).indices
    hits = gt[topk].sum().item()        
    accuracy = hits / k                 
    return accuracy, roc_auc(scores, gt.float()), n_pos / n  


@torch.no_grad()
def evaluate_loader(model, loader):
    model.eval()
    accs, aucs, randoms = [], [], []
    for batch in loader:
        batch = batch.to(device)
        scores = model.extract_causal_attention(batch)        
        bvec = batch.batch.cpu()
        gt = batch.node_gt.cpu()
        for g in range(batch.num_graphs):
            m = bvec == g
            out = graph_scores(scores[m], gt[m])
            if out is None:
                continue
            a, auc, rnd = out
            accs.append(a)
            aucs.append(auc)
            randoms.append(rnd)
    return np.array(accs), np.array(aucs), np.array(randoms)


def load_model(model_class, gnn_backbone, input_channels, num_classes, results_dir, run_index):
    ckpt = (f"{results_dir}/run_{run_index + 1}/checkpoints/"
            f"{model_class.__name__}_{gnn_backbone.__name__}{_RUN_TAG_ENV}.ckpt")
    if not os.path.exists(ckpt):
        return None
    best_params_file = (f"{results_dir}/hparam_tuning/hparams_best/"
                        f"{model_class.__name__}_{gnn_backbone.__name__}.json")
    if os.path.exists(best_params_file):
        with open(best_params_file) as f:
            best_params = json.load(f)
    else:
        best_params = {"lr": 1e-3, "wd": 1e-3}
    model = model_class.load_from_checkpoint(
        ckpt,
        gnn_backbone=gnn_backbone,
        in_channels=input_channels["in_channels"],
        in_channels_e=input_channels["in_channels_e"],
        num_classes=num_classes,
        model_hparams=model_class_hparams[model_class.__name__],
        optimizer_hparams=best_params,
    )
    return model.to(device)


def evaluate(datasets, backbones, models):
    out_dir = "../results/explanation_metrics"
    os.makedirs(out_dir, exist_ok=True)
    rows = []                                  
    nested = {}                                 

    for dataset in datasets:
        if dataset not in GET_DATASET:
            print(f"[skip] unknown dataset {dataset}")
            continue
        entry = GET_DATASET[dataset]
        input_channels, loader = entry() if callable(entry) else entry
        num_classes = DATASET_N_CLASSES[dataset]
        results_dir = f"../results/{dataset}"
        nested[dataset] = {}

        probe = next(iter(loader["test"]))
        if not hasattr(probe, "node_gt"):
            print(f"[skip] {dataset}: test graphs have no `node_gt` "
                  f"(rebuild the dataset to add it)")
            continue

        for bb_name in backbones:
            gnn_backbone = BACKBONE_MAP[bb_name]
            nested[dataset][bb_name] = {}
            for m_name in models:
                model_class = MODEL_MAP[m_name]
                if not issubclass(model_class, CGNN):
                    continue                   

                run_means, run_sample_stds, run_aucs, run_rand, n_graphs = [], [], [], [], 0
                for run_index in range(len(SEEDS)):
                    model = load_model(model_class, gnn_backbone, input_channels,
                                       num_classes, results_dir, run_index)
                    if model is None:
                        continue
                    accs, aucs, rnd = evaluate_loader(model, loader["test"])
                    if accs.size == 0:
                        continue
                    run_means.append(float(accs.mean()))
                    run_sample_stds.append(float(accs.std()))   
                    run_aucs.append(float(aucs.mean()))
                    run_rand.append(float(rnd.mean()))
                    n_graphs = accs.size

                if not run_means:
                    print(f"[ ] {dataset:<16} {bb_name:<10} {m_name:<6} : no checkpoints")
                    continue

                rec = {
                    "n_runs": len(run_means),
                    "n_graphs": n_graphs,
                    "prec_at_k_mean": float(np.mean(run_means)),
                    "prec_at_k_std":  float(np.std(run_means)),
                    "prec_at_k_sample_std": float(np.mean(run_sample_stds)),
                    "auc_mean": float(np.mean(run_aucs)),
                    "auc_std":  float(np.std(run_aucs)),
                    "random_baseline": float(np.mean(run_rand)),
                    "per_run_prec_means": run_means,
                }
                nested[dataset][bb_name][m_name] = rec
                rows.append({
                    "dataset": dataset, "backbone": bb_name, "model": m_name,
                    "prec_at_k_mean": rec["prec_at_k_mean"],
                    "prec_at_k_std": rec["prec_at_k_std"],
                    "prec_at_k_sample_std": rec["prec_at_k_sample_std"],
                    "auc_mean": rec["auc_mean"], "auc_std": rec["auc_std"],
                    "random_baseline": rec["random_baseline"],
                    "n_runs": rec["n_runs"], "n_graphs": rec["n_graphs"],
                })
                print(f"[x] {dataset:<16} {bb_name:<10} {m_name:<6} : "
                      f"P@k={rec['prec_at_k_mean']:.3f}+/-{rec['prec_at_k_std']:.3f}  "
                      f"AUROC={rec['auc_mean']:.3f}  (random={rec['random_baseline']:.3f}, "
                      f"{rec['n_runs']} runs)")

    json_path = os.path.join(out_dir, f"explanation_metrics{_RUN_TAG_ENV}.json")
    csv_path = os.path.join(out_dir, f"explanation_metrics{_RUN_TAG_ENV}.csv")
    with open(json_path, "w") as f:
        json.dump(nested, f, indent=2)
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nSaved: {json_path}\n       {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Explanation-quality (Precision@k + node AUROC) evaluation")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                   help="datasets with a node_gt motif mask")
    p.add_argument("--backbones", nargs="+", default=list(BACKBONE_MAP.keys()),
                   choices=list(BACKBONE_MAP.keys()))
    p.add_argument("--models", nargs="+", default=list(MODEL_MAP.keys()),
                   choices=list(MODEL_MAP.keys()))
    return p.parse_args()


def main():
    args = parse_args()
    print(f"DEVICE is {device}")
    evaluate(args.datasets, args.backbones, args.models)


if __name__ == "__main__":
    main()
