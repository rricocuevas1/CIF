import os
import re
import sys
import csv
import numpy as np
import json
import pandas as pd
from collections import OrderedDict

DATASET_SHORT_LATEX_TABLES = {
    "Graph_SST2": "G-SST2",
    "MNIST_75sp_n02": "MNIST-75sp",
    "MNIST_75sp_n04": "MNIST-75sp",
    "MNIST_75sp_n06": "MNIST-75sp",
    "MNIST_75sp_n08": "MNIST-75sp",
    "Molhiv": "Molhiv",
    "SPMotif_b_05": "SpMotif",
    "SPMotif_b_07": "SpMotif",
    "SPMotif_b_09": "SpMotif",
    "SYN_binary_b_01": "SYN",
    "SYN_binary_b_03": "SYN",
    "SYN_binary_b_05": "SYN",
    "SYN_binary_b_07": "SYN",
    "SYN_binary_b_09": "SYN",
    "SYN_multi_b_01": "SYN", 
    "SYN_multi_b_03": "SYN", 
    "SYN_multi_b_05": "SYN", 
    "SYN_multi_b_07": "SYN", 
    "SYN_multi_b_09": "SYN",
}

DATASET_SUBLABELS = {
    "SPMotif_b_05": "$b=0.5$",
    "SPMotif_b_07": "$b=0.7$",
    "SPMotif_b_09": "$b=0.9$",
    "MNIST_75sp_n02": "$n=0.2$",
    "MNIST_75sp_n04": "$n=0.4$",
    "MNIST_75sp_n06": "$n=0.6$",
    "MNIST_75sp_n08": "$n=0.8$",
    "SYN_binary_b_01": "$b=0.1$",
    "SYN_binary_b_03": "$b=0.3$",
    "SYN_binary_b_05": "$b=0.5$",
    "SYN_binary_b_07": "$b=0.7$",
    "SYN_binary_b_09": "$b=0.9$",
    "SYN_multi_b_01": "$b=0.1$", 
    "SYN_multi_b_03": "$b=0.3$", 
    "SYN_multi_b_05": "$b=0.5$", 
    "SYN_multi_b_07": "$b=0.7$", 
    "SYN_multi_b_09": "$b=0.9$",
}

BACKBONES_LATEX_TABLES = [
    "GCN", 
    "GAT", 
    "GIN", 
    "GraphGPS", 
    "GrokFormer", 
    "DualFormer"
]

BACKBONE_DISPLAY_LATEX_TABLES = {
    "GCN": "GCN",
    "GAT": "GAT",
    "GIN": "GIN",
    "GraphGPS": "GraphGPS",
    "GrokFormer": "GrokFormer",
    "DualFormer": "DualFormer",
}

SYSTEMS_LATEX_TABLES = [
    ("GNN",   None,              False),
    ("CAL",   "- CAL",           False),
    ("ICL",   "- ICL",           False),
    ("ACE",   "- ACE",           False),
    ("CIF", "- CIF",         True),
]

CLASSIFICATION_METRICS = [
    ("av_test_auroc", "stdev_test_auroc", "AUROC"),
    ("av_test_auprc", "stdev_test_auprc", "AUPRC"),
]

def get_gnn_kwargs(in_channels, in_channels_e, model_hparams, gnn_backbone_name):
    """Auxiliary function to set the GNN kwargs"""
    # GCN, GIN
    gnn_kwargs = {
        'in_channels': in_channels,
        'hidden_channels' : model_hparams["hidden_channels_gnn"], 
        'out_channels': model_hparams["gnn_out_channels"],
    }

    # GAT
    if gnn_backbone_name == "GAT_encoder":
        gnn_kwargs["in_channels_e"] = in_channels_e
    
    # GraphGPS
    elif gnn_backbone_name == "GraphGPS_encoder":
        gnn_kwargs["in_channels_e"] = in_channels_e
        gnn_kwargs["num_layers"] =  model_hparams["num_layers"]
        gnn_kwargs["rwse_dim"] = model_hparams["rwse_walk_length"]
        gnn_kwargs["pe_dim"] = model_hparams["pe_dim"]
        gnn_kwargs["attn_type"] = model_hparams["attn_type"]
        gnn_kwargs["attn_heads"] = model_hparams["attn_heads"]
        gnn_kwargs["attn_kwargs"] = model_hparams["attn_kwargs"]
    
    # GrokFormer
    elif gnn_backbone_name == 'GrokFormer_encoder':
        gnn_kwargs['in_channels_e'] = in_channels_e
        gnn_kwargs['num_layers'] = model_hparams['num_layers']
        gnn_kwargs['k'] = model_hparams['k']
        gnn_kwargs['nheads'] = model_hparams['nheads']
        gnn_kwargs['sine_dim'] = model_hparams['sine_dim']
        gnn_kwargs['tran_dropout'] = model_hparams['tran_dropout']
        gnn_kwargs['feat_dropout'] = model_hparams['feat_dropout']
        gnn_kwargs['prop_dropout'] = model_hparams['prop_dropout']
    
    # DualFormer
    elif gnn_backbone_name == 'DualFormer_encoder':
        gnn_kwargs['activation'] = model_hparams['activation']
        gnn_kwargs['num_gnns'] = model_hparams['num_gnns']
        gnn_kwargs['num_trans'] = model_hparams['num_sa']
        gnn_kwargs['num_heads'] = model_hparams['num_heads']
        gnn_kwargs['dropout_trans'] = model_hparams['dropout_sa']
        gnn_kwargs['dropout'] = model_hparams['dropout']
        gnn_kwargs['alpha'] = model_hparams['alpha']
        gnn_kwargs['lammda'] = model_hparams['lammda']
        gnn_kwargs['GraphConv'] = model_hparams['GraphConv']
        gnn_kwargs['use_bn'] = model_hparams['use_bn']
    
    return gnn_kwargs


def get_filename(system_prefix, backbone):
    if system_prefix == "GNN":
        return f"GNN_{backbone}_encoder_integrated.csv"
    else:
        return f"{system_prefix}_{backbone}_encoder_integrated.csv"


def read_csv(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)
        parsed = {}
        for k, v in row.items():
            try:
                parsed[k] = float(v)
            except (TypeError, ValueError):
                # Empty cell, e.g. stdev_* of a config with a single completed run
                parsed[k] = float('nan')
        return parsed


def format_val(mean, std, bold=False, underline=False, decimals=2):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return r"{}"
    mean_str = f"{mean:.{decimals}f}"
    std_str = "--" if (std is None or (isinstance(std, float) and np.isnan(std))) else f"{std:.{decimals}f}"
    if bold:
        return r"{$\mathbf{" + mean_str + r"\pm {\scriptstyle " + std_str + r"}}$}"
    elif underline:
        return r"{$\underline{" + mean_str + r"\pm {\scriptstyle " + std_str + r"}}$}"
    else:
        return r"{$" + mean_str + r"\pm {\scriptstyle " + std_str + r"}$}"


def collect_results(results_dir, datasets, metrics, systems=None):
    if systems is None:
        systems = SYSTEMS_LATEX_TABLES
    results = {}
    for backbone in BACKBONES_LATEX_TABLES:
        results[backbone] = {}
        for system_prefix, _, _ in systems:
            results[backbone][system_prefix] = {}
            for dataset in datasets:
                results[backbone][system_prefix][dataset] = {}
                filename = get_filename(system_prefix, backbone)
                filepath = os.path.join(results_dir, dataset, "integrated", filename)
                data = read_csv(filepath)
                for metric_mean, metric_std, metric_name in metrics:
                    if data and metric_mean in data and metric_std in data:
                        results[backbone][system_prefix][dataset][metric_name] = (
                            data[metric_mean], data[metric_std]
                        )
                    else:
                        results[backbone][system_prefix][dataset][metric_name] = (None, None)
    return results


def build_header(datasets, metrics):
    n_metric_cols = len(metrics)
    n_data_cols = len(datasets) * n_metric_cols
    metric_labels = " & ".join(f"{{{m[2]}}}" for m in metrics)

    groups = OrderedDict()
    for dataset in datasets:
        short = DATASET_SHORT_LATEX_TABLES[dataset]
        if short not in groups:
            groups[short] = []
        groups[short].append(dataset)

    header1 = r"\textbf{Methods}"
    group_list = list(groups.items())
    for i, (group_name, group_datasets) in enumerate(group_list):
        n_cols = len(group_datasets) * n_metric_cols
        sep = "|" if i < len(group_list) - 1 else ""
        header1 += r" & \multicolumn{" + str(n_cols) + r"}{c" + sep + r"}{\textbf{" + group_name + r"}}"
    header1 += r" \\"

    header2 = ""
    for i, dataset in enumerate(datasets):
        sublabel = DATASET_SUBLABELS.get(dataset, "")
        sep = "|" if i < len(datasets) - 1 else ""
        header2 += r" & \multicolumn{" + str(n_metric_cols) + r"}{c" + sep + r"}{" + sublabel + r"}"
    header2 += r" \\"

    header3 = ""
    for _ in datasets:
        header3 += f" & {metric_labels}"
    header3 += r" \\"

    return header1, header2, header3, n_data_cols


def integrate_results(base_dir, integrated_dir, name_filter=""):
    runs = [
        d for d in os.listdir(base_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))
    ]
    if not runs:
        raise ValueError(f"No run_* directories found in {base_dir}")

    metric_files = set()
    for run in runs:
        run_metrics_dir = os.path.join(base_dir, run, "test_metrics")
        if os.path.exists(run_metrics_dir):
            for f in os.listdir(run_metrics_dir):
                if f.endswith(f"{name_filter}.json"):
                    metric_files.add(f)

    os.makedirs(integrated_dir, exist_ok=True)

    for metric_file in metric_files:
        metrics_list = []
        for run in runs:
            file_path = os.path.join(base_dir, run, "test_metrics", metric_file)
            if not os.path.exists(file_path):
                print(f"MISSING: {file_path}")
                continue
            with open(file_path, "r") as f:
                data = json.load(f)
            if data is None:
                print(f"NULL: {file_path}")
                continue
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError(f"Unexpected JSON structure in {file_path}: {type(data)}")
            metrics_list.append(df)


        if not metrics_list:
            continue

        combined_metrics = pd.concat(metrics_list, axis=0, ignore_index=True)
        agg_data = {}
        for col in combined_metrics.columns:
            agg_data[f"av_{col}"] = [combined_metrics[col].mean()]
            agg_data[f"stdev_{col}"] = [combined_metrics[col].std()]

        aggregated_df = pd.DataFrame(agg_data)
        output_file = os.path.join(
            integrated_dir,
            metric_file.replace(".json", "_integrated.csv")
        )
        aggregated_df.to_csv(output_file, index=False)


EXPLANATION_METRICS = [
    ("prec_at_k_mean", "prec_at_k_std", "P@k"),
    ("auc_mean",       "auc_std",       "AUROC"),
]


def collect_explanation_results(json_path, datasets, metrics, systems):
    with open(json_path) as f:
        data = json.load(f)
    results = {}
    for backbone in BACKBONES_LATEX_TABLES:
        results[backbone] = {}
        for system_prefix, _, _ in systems:
            results[backbone][system_prefix] = {}
            for dataset in datasets:
                rec = (data.get(dataset) or {}).get(backbone, {}).get(system_prefix)
                results[backbone][system_prefix][dataset] = {}
                for m_mean, m_std, m_name in metrics:
                    if rec is not None and m_mean in rec and m_std in rec:
                        results[backbone][system_prefix][dataset][m_name] = (rec[m_mean], rec[m_std])
                    else:
                        results[backbone][system_prefix][dataset][m_name] = (None, None)
    return results


def find_best_second_systems(results, datasets, metrics, systems):
    best, second = {}, {}
    for dataset in datasets:
        best[dataset], second[dataset] = {}, {}
        for _, _, metric_name in metrics:
            scored = []
            for system_prefix, _, _ in systems:
                for backbone in BACKBONES_LATEX_TABLES:
                    mean, _ = results[backbone][system_prefix][dataset][metric_name]
                    if mean is not None:
                        scored.append((round(mean, 2), system_prefix, backbone))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored:
                top_val = scored[0][0]
                best[dataset][metric_name] = [(s, b) for v, s, b in scored if v == top_val]
            else:
                best[dataset][metric_name] = []
            top_set = set(best[dataset][metric_name])
            remaining = [(v, s, b) for v, s, b in scored if (s, b) not in top_set]
            if remaining:
                second_val = remaining[0][0]
                second[dataset][metric_name] = [(s, b) for v, s, b in remaining if v == second_val]
            else:
                second[dataset][metric_name] = []
    return best, second


CLASSIFICATION_TABLE1_DATASETS = [
    "Graph_SST2", "Molhiv",
    "SYN_multi_b_01", "SYN_multi_b_03", "SYN_multi_b_05", "SYN_multi_b_07", "SYN_multi_b_09",
]
CLASSIFICATION_TABLE2_DATASETS = [
    "MNIST_75sp_n02", "MNIST_75sp_n04", "MNIST_75sp_n06", "MNIST_75sp_n08",
    "SPMotif_b_05", "SPMotif_b_07", "SPMotif_b_09",
]
PREVALENCE = {
    "Graph_SST2": 0.5587,
    "Molhiv": 0.0316,
    "SYN_multi_b_01": 0.25, "SYN_multi_b_03": 0.25, "SYN_multi_b_05": 0.25,
    "SYN_multi_b_07": 0.25, "SYN_multi_b_09": 0.25,
    "MNIST_75sp_n02": 0.10, "MNIST_75sp_n04": 0.10,
    "MNIST_75sp_n06": 0.10, "MNIST_75sp_n08": 0.10,
    "SPMotif_b_05": 1.0 / 3.0, "SPMotif_b_07": 1.0 / 3.0, "SPMotif_b_09": 1.0 / 3.0,
}
SUMMARY_SYSTEM_ROWS = [
    ("DIR",            "DIR",                False),
    ("CAL",            "CAL",                False),
    ("ICL",            "ICL",                False),
    ("ACE",            "ACE",                False),
    ("CIF_J_NoMC",   r"CIF\_J\_NoMC",    False),
    ("CIF",          "CIF",              True),   # ours (full model)
]
MAIN_SYSTEM_ROWS = [
    ("DIR",   "DIR",        False),
    ("CAL",   "CAL",        False),
    ("ICL",   "ICL",        False),
    ("ACE",   "ACE",        False),
    ("CIF", r"CIF (us)",  True),   # ours (full model)
]
CE_CIF_ABLATION_ROWS = [
    ("GNN",          "CE",             False),
    ("CIF_NoJ_MC", r"CIF$_{\leq}$",  False),
    ("CIF",        "CIF",            True),   # ours (full model)
]


def _deagg_label_config():
    return 2, r"\textbf{Backbone} & \textbf{Method}", " & ", " & "


def _deagg_row_lead(bb_cell, system_prefix, display_name, is_ours):
    method_lbl = (r"\textbf{" + display_name + r"}") if is_ours else display_name
    return bb_cell + " & " + method_lbl


def minprc(pi):
    return 1.0 + (1.0 - pi) * np.log(1.0 - pi) / pi


def aucnpr(auprc, pi):
    mp = minprc(pi)
    return max(0.0, (auprc - mp) / (1.0 - mp))


def _agg_mean_std(values):
    xs = [v for v in values
          if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not xs:
        return None, None
    mean = float(np.mean(xs))
    std = float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan")
    return mean, std


def collect_aggregated_classification(results_dir, datasets, systems=None):
    if systems is None:
        systems = SUMMARY_SYSTEM_ROWS
    results = collect_results(results_dir, datasets, CLASSIFICATION_METRICS, systems)

    missing = [d for d in datasets if d not in PREVALENCE]
    if missing:
        raise KeyError(f"No prevalence for datasets {missing}; add them to PREVALENCE.")

    agg = {}
    for backbone in BACKBONES_LATEX_TABLES:
        agg[backbone] = {}
        for system_prefix, _, _ in systems:
            aurocs, aucnprs = [], []
            for dataset in datasets:
                auroc_mean, _ = results[backbone][system_prefix][dataset]["AUROC"]
                auprc_mean, _ = results[backbone][system_prefix][dataset]["AUPRC"]
                aurocs.append(auroc_mean)
                if auprc_mean is None or (isinstance(auprc_mean, float) and np.isnan(auprc_mean)):
                    aucnprs.append(None)
                else:
                    aucnprs.append(aucnpr(auprc_mean, PREVALENCE[dataset]))
            au_m, au_s = _agg_mean_std(aurocs)
            an_m, an_s = _agg_mean_std(aucnprs)
            agg[backbone][system_prefix] = (au_m, au_s, an_m, an_s)
    return agg


def _best_second_means(pairs, decimals=2):
    scored = [(round(m, decimals), sp) for sp, m in pairs
              if m is not None and not (isinstance(m, float) and np.isnan(m))]
    if not scored:
        return set(), set()
    uniq = sorted({v for v, _ in scored}, reverse=True)
    best_val = uniq[0]
    second_val = uniq[1] if len(uniq) > 1 else None
    best = {sp for v, sp in scored if v == best_val}
    second = {sp for v, sp in scored if second_val is not None and v == second_val}
    return best, second


def _summary_cell(mean, std, bold=False, underline=False, show_std=False, decimals=2):
    if show_std:
        return format_val(mean, std, bold=bold, underline=underline, decimals=decimals)
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return r"{}"
    s = f"{mean:.{decimals}f}"
    if bold:
        return r"{$\mathbf{" + s + r"}$}"
    if underline:
        return r"{$\underline{" + s + r"}$}"
    return r"{$" + s + r"$}"


ALL_CLASSIFICATION_DATASETS = CLASSIFICATION_TABLE1_DATASETS + CLASSIFICATION_TABLE2_DATASETS


def generate_table_backbone_first(results_dir, datasets, caption, metrics=None,
                                  systems=None, empty_zero=False):
    if metrics is None:
        metrics = CLASSIFICATION_METRICS
    if systems is None:
        systems = MAIN_SYSTEM_ROWS
    results = collect_results(results_dir, datasets, metrics, systems)
    best, second = find_best_second_systems(results, datasets, metrics, systems)

    n_metric_cols = len(metrics)
    n_data_cols = len(datasets) * n_metric_cols
    col_spec = "c|" * (n_data_cols - 1) + "c"
    n_methods = len(systems)
    n_label_cols, header_label, h2_prefix, h3_prefix = _deagg_label_config()
    first_data_col = n_label_cols + 1
    last_data_col = n_data_cols + n_label_cols

    lines = [r"\begin{table*}[t!]", r"\centering", r"\caption{" + caption + r"}",
             r"\resizebox{\textwidth}{!}{",
             r"\begin{tabular}{" + "l|" * n_label_cols + " " + col_spec + r"}",
             r"\toprule"]

    h1, h2, h3, _ = build_header(datasets, metrics)
    h1 = h1.replace(r"\textbf{Methods}", header_label)
    h2 = h2_prefix + h2
    h3 = h3_prefix + h3
    cmid = r"\cmidrule(lr){" + str(first_data_col) + "-" + str(last_data_col) + r"}"
    cmid_h1 = cmid
    lines.append(h1)
    lines.append(cmid_h1)
    lines.append(h2)
    if len(metrics) > 1:
        lines.append(cmid)
        lines.append(h3)
    lines.append(r"\midrule")

    for b_idx, backbone in enumerate(BACKBONES_LATEX_TABLES):
        for m_idx, (system_prefix, display_name, is_ours) in enumerate(systems):
            bb_cell = (r"\multirow{" + str(n_methods) + r"}{*}{" +
                       BACKBONE_DISPLAY_LATEX_TABLES[backbone] + r"}") if m_idx == 0 else ""
            lead = _deagg_row_lead(bb_cell, system_prefix, display_name, is_ours)
            vals = []
            for dataset in datasets:
                for _, _, metric_name in metrics:
                    mean, std = results[backbone][system_prefix][dataset][metric_name]
                    is_best = (mean is not None
                               and (system_prefix, backbone) in best[dataset][metric_name])
                    is_second = (mean is not None
                                 and (system_prefix, backbone) in second[dataset][metric_name])
                    if mean is None and empty_zero:
                        mean, std = 0.0, 0.0
                    vals.append(format_val(mean, std, bold=is_best, underline=is_second))
            row = lead + "\n"
            for v in vals:
                row += f"    &{v}\n"
            row = row.rstrip("\n") + r"\\"
            if is_ours and m_idx > 0:
                lines.append(r"\cmidrule(lr){2-" + str(last_data_col) + r"}")
            lines.append(row)
        if b_idx < len(BACKBONES_LATEX_TABLES) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
    return "\n".join(lines)


def save_deaggregated_backbone_first_tables(
    results_dir="../results",
    output_dir="../results/results_tables",
):
    os.makedirs(output_dir, exist_ok=True)
    cap = r"Experimental results on classification, AUROC (Mean $\pm$ Std) and AUPRC (Mean $\pm$ Std)."
    out = {}
    for tag, datasets in (("1", CLASSIFICATION_TABLE1_DATASETS),
                          ("2", CLASSIFICATION_TABLE2_DATASETS)):
        table = generate_table_backbone_first(results_dir, datasets, cap)
        label = r"\label{table:results_" + tag + r"}"
        table = table.replace(r"\caption{" + cap + "}", r"\caption{" + cap + r"}" + label)
        path = os.path.join(output_dir, f"table_results_backbone_{tag}.txt")
        with open(path, "w") as f:
            f.write(table)
        print(f"Backbone-first classification table {tag} saved to {path}")
        out[tag] = table
    return out


EXPLANATION_TABLE1_DATASETS = [
    "SYN_multi_b_01", "SYN_multi_b_03", "SYN_multi_b_05", "SYN_multi_b_07", "SYN_multi_b_09",
]
EXPLANATION_TABLE2_DATASETS = [
    "SPMotif_b_05", "SPMotif_b_07", "SPMotif_b_09",
]
EXPLANATION_ALL_DATASETS = EXPLANATION_TABLE1_DATASETS + EXPLANATION_TABLE2_DATASETS


def generate_explanation_table_backbone_first(json_path, datasets, caption,
                                             metrics=None, systems=None,
                                             empty_dash=False):
    if metrics is None:
        metrics = EXPLANATION_METRICS
    if systems is None:
        systems = MAIN_SYSTEM_ROWS
    results = collect_explanation_results(json_path, datasets, metrics, systems)
    best, second = find_best_second_systems(results, datasets, metrics, systems)

    n_metric_cols = len(metrics)
    n_data_cols = len(datasets) * n_metric_cols
    col_spec = "c|" * (n_data_cols - 1) + "c"
    n_methods = len(systems)
    n_label_cols, header_label, h2_prefix, h3_prefix = _deagg_label_config()
    first_data_col = n_label_cols + 1
    last_data_col = n_data_cols + n_label_cols

    lines = [r"\begin{table*}[t!]", r"\centering", r"\caption{" + caption + r"}",
             r"\resizebox{\textwidth}{!}{",
             r"\begin{tabular}{" + "l|" * n_label_cols + " " + col_spec + r"}",
             r"\toprule"]

    h1, h2, h3, _ = build_header(datasets, metrics)
    h1 = h1.replace(r"\textbf{Methods}", header_label)
    h2 = h2_prefix + h2
    h3 = h3_prefix + h3
    cmid = r"\cmidrule(lr){" + str(first_data_col) + "-" + str(last_data_col) + r"}"
    cmid_h1 = cmid
    lines.append(h1)
    lines.append(cmid_h1)
    lines.append(h2)
    lines.append(cmid)
    lines.append(h3)
    lines.append(r"\midrule")

    for b_idx, backbone in enumerate(BACKBONES_LATEX_TABLES):
        for m_idx, (system_prefix, display_name, is_ours) in enumerate(systems):
            bb_cell = (r"\multirow{" + str(n_methods) + r"}{*}{" +
                       BACKBONE_DISPLAY_LATEX_TABLES[backbone] + r"}") if m_idx == 0 else ""
            lead = _deagg_row_lead(bb_cell, system_prefix, display_name, is_ours)
            vals = []
            for dataset in datasets:
                for _, _, metric_name in metrics:
                    mean, std = results[backbone][system_prefix][dataset][metric_name]
                    is_best = (mean is not None
                               and (system_prefix, backbone) in best[dataset][metric_name])
                    is_second = (mean is not None
                                 and (system_prefix, backbone) in second[dataset][metric_name])
                    if mean is None and empty_dash:
                        vals.append(r"{$-$}")
                    else:
                        vals.append(format_val(mean, std, bold=is_best, underline=is_second))
            row = lead + "\n"
            for v in vals:
                row += f"    &{v}\n"
            row = row.rstrip("\n") + r"\\"
            if is_ours and m_idx > 0:
                lines.append(r"\cmidrule(lr){2-" + str(last_data_col) + r"}")
            lines.append(row)
        if b_idx < len(BACKBONES_LATEX_TABLES) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
    return "\n".join(lines)


def save_ce_cif_ablation_tables(
    results_dir="../results",
    json_path="../results/explanation_metrics/explanation_metrics.json",
    output_dir="../results/results_tables",
):
    os.makedirs(output_dir, exist_ok=True)

    cls_cap = (r"Ablation on classification, AUROC (Mean $\pm$ Std) and AUPRC (Mean $\pm$ Std): "
               r"CE (bare cross-entropy backbone), CIF$_{\leq}$ (No-Jensen ablation) and CIF "
               r"(full model). Best per backbone in \textbf{bold}, second-best \underline{underlined}.")
    for tag, datasets in (("1", CLASSIFICATION_TABLE1_DATASETS),
                          ("2", CLASSIFICATION_TABLE2_DATASETS)):
        table = generate_table_backbone_first(
            results_dir, datasets, cls_cap, systems=CE_CIF_ABLATION_ROWS,
            empty_zero=True)
        label = r"\label{table:results_" + tag + r"_ablation}"
        table = table.replace(r"\caption{" + cls_cap + "}", r"\caption{" + cls_cap + r"}" + label)
        table = rebold(table, per_backbone=True)
        path = os.path.join(output_dir, f"table_results_backbone_{tag}_ablation.txt")
        with open(path, "w") as f:
            f.write(table)
        print(f"CE/CIF ablation classification table {tag} saved to {path}")

    exp_cap = (r"Ablation on explanation quality (invariant graph rationales): Precision@$k$ "
               r"($k=\#$motif nodes, $=$ Recall@$k$) and per-node AUROC (Mean $\pm$ Std): "
               r"CIF$_{\leq}$ (No-Jensen ablation) and CIF (full model). Best per backbone in "
               r"\textbf{bold}, second-best \underline{underlined}.")
    exp_systems = [r for r in CE_CIF_ABLATION_ROWS if r[0] != "GNN"]
    table = generate_explanation_table_backbone_first(
        json_path, EXPLANATION_ALL_DATASETS, exp_cap,
        systems=exp_systems, empty_dash=True)
    label = r"\label{table:explanation_ablation}"
    table = table.replace(r"\caption{" + exp_cap + "}", r"\caption{" + exp_cap + r"}" + label)
    table = rebold(table, per_backbone=True)
    path = os.path.join(output_dir, "table_explanation_backbone_ablation.txt")
    with open(path, "w") as f:
        f.write(table)
    print(f"CE/CIF ablation explanation table saved to {path}")


def save_deaggregated_backbone_first_explanation_table(
    json_path="../results/explanation_metrics/explanation_metrics.json",
    output_path=None,
):
    if output_path is None:
        output_path = os.path.join("../results/results_tables", "table_explanation_backbone.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = (r"Explanation quality (invariant graph rationales): Precision@$k$ "
           r"($k=\#$motif nodes, $=$ Recall@$k$) and per-node AUROC (Mean $\pm$ Std). "
           r"Best per column in \textbf{bold}, second-best \underline{underlined}.")
    label = r"\label{table:explanation}"
    table = generate_explanation_table_backbone_first(
        json_path, EXPLANATION_ALL_DATASETS, cap)
    table = table.replace(r"\caption{" + cap + "}", r"\caption{" + cap + r"}" + label)
    with open(output_path, "w") as f:
        f.write(table)
    print(f"Backbone-first explanation table saved to {output_path}")
    return table


def generate_cif_ce_absolute_ablation_table(results_dir, json_path, caption, decimals=2):
    OURS, ABL, CE = "CIF", "CIF_NoJ_MC", "GNN"
    sys_c = [(OURS, "CIF", True), (ABL, r"CIF$_{\leq}$", False), (CE, "CE", False)]
    sys_e = [(OURS, "CIF", True), (ABL, r"CIF$_{\leq}$", False)]   # CE has no rationale
    method_lbl = {OURS: r"\textbf{CIF}", ABL: r"CIF$_{\leq}$", CE: "CE"}
    class_methods = [OURS, ABL, CE]
    rat_methods = [OURS, ABL]
    class_ds, expl_ds = ALL_CLASSIFICATION_DATASETS, EXPLANATION_ALL_DATASETS

    agg_c = collect_aggregated_classification(results_dir, class_ds, sys_c)
    expl = collect_explanation_results(json_path, expl_ds, EXPLANATION_METRICS, sys_e)
    bbs = BACKBONES_LATEX_TABLES

    def rat_mean(bb, sp, key):
        return _agg_mean_std([expl[bb][sp][d][key][0] for d in expl_ds])[0]

    tasks = [
        ("Classification", [
            ("AUROC",       lambda bb, sp: agg_c[bb][sp][0], class_methods),
            ("AUNPRC",      lambda bb, sp: agg_c[bb][sp][2], class_methods),
        ]),
        ("Graph Rationale Discovery", [
            (r"Prec@$k$",   lambda bb, sp: rat_mean(bb, sp, "P@k"),   rat_methods),
            ("AUROC",       lambda bb, sp: rat_mean(bb, sp, "AUROC"), rat_methods),
        ]),
    ]
    groups = [g for _, gs in tasks for g in gs]

    col_spec = "l|" + "|".join("c" * sum(len(ms) for _, _, ms in gs) for _, gs in tasks)
    task_hdr, metric_hdr, task_cmids, metric_cmids = [], [], [], []
    col = 2
    for ti, (tlabel, gs) in enumerate(tasks):
        width = sum(len(ms) for _, _, ms in gs)
        sep = "|" if ti < len(tasks) - 1 else ""
        task_hdr.append(r"\multicolumn{" + str(width) + r"}{c" + sep + r"}{\textbf{" + tlabel + r"}}")
        task_cmids.append(r"\cmidrule(lr){" + f"{col}-{col + width - 1}" + r"}")
        for gi, (mlabel, _, ms) in enumerate(gs):
            last_in_task = gi == len(gs) - 1
            msep = "|" if (last_in_task and ti < len(tasks) - 1) else ""
            metric_hdr.append(r"\multicolumn{" + str(len(ms)) + r"}{c" + msep + r"}{" + mlabel + r"}")
            metric_cmids.append(r"\cmidrule(lr){" + f"{col}-{col + len(ms) - 1}" + r"}")
            col += len(ms)

    lines = [
        r"\begin{table*}[t!]", r"\centering", r"\caption{" + caption + r"}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{" + col_spec + r"}", r"\toprule",
        r"\textbf{Task} & " + " & ".join(task_hdr) + r" \\",
        "".join(task_cmids),
        r"\textbf{Metric} & " + " & ".join(metric_hdr) + r" \\",
        "".join(metric_cmids),
        r"\textbf{Backbone \textbackslash\ Model}"
        + "".join(" & " + method_lbl[m] for _, _, ms in groups for m in ms) + r" \\",
        r"\midrule",
    ]
    for bb in bbs:
        cells = []
        for _, getter, ms in groups:
            pairs = [(m, getter(bb, m)) for m in ms]
            best, second = _best_second_means(pairs, decimals=decimals)
            for m in ms:
                v = getter(bb, m)
                if v is None:
                    cells.append(r"{$-$}")
                else:
                    cells.append(_summary_cell(v, None, m in best, m in second,
                                               show_std=False, decimals=decimals))
        lines.append(BACKBONE_DISPLAY_LATEX_TABLES[bb] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
    return "\n".join(lines)


def save_cif_ce_absolute_ablation_table(
    results_dir="../results",
    json_path="../results/explanation_metrics/explanation_metrics.json",
    output_path="../results/results_tables/table_cif_ce_absolute_ablation.txt",
    decimals=2,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = (r"Ablation study: per-backbone mean performance of CIF (full model), CIF$_{\leq}$ "
           r"(the No-Jensen ablation, CIF\_NoJ\_MC) and CE (the bare cross-entropy-trained "
           r"backbone). Classification: mean AUROC and mean AUNPRC (Boyd-normalized AUPRC) over "
           r"all datasets. Invariant graph rationale discovery: mean Precision@$k$ and mean "
           r"node-attention AUROC over the synthetic datasets (CE has no rationale, shown as "
           r"$-$). Best per backbone and metric in \textbf{bold}, second-best "
           r"\underline{underlined}.")
    label = r"\label{table:cif_ce_absolute_ablation}"
    table = generate_cif_ce_absolute_ablation_table(results_dir, json_path, cap, decimals=decimals)
    table = table.replace(r"\caption{" + cap + "}", r"\caption{" + cap + r"}" + label, 1)
    with open(output_path, "w") as f:
        f.write(table)
    print(f"CIF/CIF<=/CE absolute ablation table saved to {output_path}")
    return table


def _task_split_tabular(metrics, systems, backbones, decimals):
    n_met = len(metrics)
    grp, cmids = [], []
    for bi, bb in enumerate(backbones):
        sep = "|" if bi < len(backbones) - 1 else ""
        grp.append(r"\multicolumn{" + str(n_met) + r"}{c" + sep + r"}{"
                   + BACKBONE_DISPLAY_LATEX_TABLES[bb] + r"}")
        a, b = 2 + bi * n_met, 1 + (bi + 1) * n_met
        cmids.append(r"\cmidrule(lr){" + f"{a}-{b}" + r"}")
    subhdr = [mlabel for _ in backbones for (mlabel, _vf) in metrics]

    col_spec = "l|" + "|".join("c" * n_met for _ in backbones)
    lines = [
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{" + col_spec + r"}", r"\toprule",
        r"\multirow{2}{*}{\textbf{Method}} & " + " & ".join(grp) + r" \\",
        "".join(cmids),
        r" & " + " & ".join(subhdr) + r" \\",
        r"\midrule",
    ]
    col_marks = {(bb, mi): _best_second_means(
                    [(sp, metrics[mi][1](bb, sp)) for sp, _, _ in systems], decimals=decimals)
                 for bb in backbones for mi in range(n_met)}
    for sp, disp, is_ours in systems:
        if is_ours:
            lines.append(r"\midrule")
        method_lbl = (r"\textbf{" + disp + r"}") if is_ours else disp
        cells = []
        for bb in backbones:
            for mi, (mlabel, vf) in enumerate(metrics):
                best, second = col_marks[(bb, mi)]
                cells.append(_summary_cell(vf(bb, sp), None, sp in best, sp in second,
                                           show_std=False, decimals=decimals))
        lines.append(method_lbl + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}"]
    return "\n".join(lines)


def generate_task_split_summary_subtables(results_dir, json_path, caption,
                                          systems=None, decimals=2):
    if systems is None:
        systems = MAIN_SYSTEM_ROWS
    backbones = BACKBONES_LATEX_TABLES
    class_ds = ALL_CLASSIFICATION_DATASETS
    expl_ds = EXPLANATION_ALL_DATASETS
    agg_c = collect_aggregated_classification(results_dir, class_ds, systems)
    expl = collect_explanation_results(json_path, expl_ds, EXPLANATION_METRICS, systems)

    def rat(bb, sp, key):
        return _agg_mean_std([expl[bb][sp][d][key][0] for d in expl_ds])[0]

    tasks = [
        ("Classification", "class", [
            ("AUROC",  lambda bb, sp: agg_c[bb][sp][0]),
            ("AUCNPR", lambda bb, sp: agg_c[bb][sp][2]),
        ]),
        ("Invariant Graph Rationale Discovery", "rationale", [
            (r"P@$k$", lambda bb, sp: rat(bb, sp, "P@k")),
            ("AUROC",  lambda bb, sp: rat(bb, sp, "AUROC")),
        ]),
    ]
    lines = [r"\begin{table*}[t!]", r"\centering", r"\caption{" + caption + r"}"]
    for i, (tlabel, key, metrics) in enumerate(tasks):
        inner = _task_split_tabular(metrics, systems, backbones, decimals)
        lines += [
            r"\begin{subtable}{\textwidth}", r"\centering",
            r"\caption{" + tlabel + r"}\label{table:summary_task_" + key + r"}",
            inner, r"\end{subtable}",
        ]
        if i < len(tasks) - 1:
            lines += ["", r"\vspace{1em}", ""]
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def save_task_split_summary_subtables(
    results_dir="../results",
    json_path="../results/explanation_metrics/explanation_metrics.json",
    output_path="../results/results_tables/table_summary_task_split.txt",
    decimals=2,
):
    r"""Write the compact task-split summary: two \subtable blocks (Classification,
    Rationale) with backbones as column groups (a sub-column per metric) and the systems as
    rows. Requires \usepackage{subcaption}."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = (r"Summary of experimental results, aggregated across datasets per backbone; all "
           r"reported values are means. Each backbone spans two columns, one per metric. "
           r"(a) Classification: AUROC and normalized AUPRC (AUCNPR; Boyd et al.\ 2012) over "
           r"all datasets. (b) Invariant graph rationale discovery: Precision@$k$ (P@$k$) and "
           r"node-attention AUROC (AUROC) on the synthetic datasets. CIF is our method. Best "
           r"method per column in \textbf{bold}, second-best \underline{underlined}.")
    label = r"\label{table:summary_task_split}"
    table = generate_task_split_summary_subtables(results_dir, json_path, cap, decimals=decimals)
    table = table.replace(r"\caption{" + cap + "}", r"\caption{" + cap + r"}" + label, 1)
    with open(output_path, "w") as f:
        f.write(table)
    print(f"Task-split summary table saved to {output_path}")
    return table


PAY_RE = re.compile(r"-?\d*\.?\d+\\pm\s*\{\\scriptstyle\s*[^}]*\}")
MEAN_RE = re.compile(r"-?\d*\.?\d+")
TABLE_RE = re.compile(r"\\begin\{table\*\}.*?\\end\{table\*\}", re.DOTALL)
WS_RE = re.compile(r"(?s)^(\s*)(.*?)(\s*)$")  
REBOLD_EPS = 1e-9


def payload_mean(payload):
    return float(MEAN_RE.match(payload).group())


def wrap_payload(payload, kind):
    if kind == "bold":
        return r"{$\mathbf{" + payload + r"}$}"
    if kind == "under":
        return r"{$\underline{" + payload + r"}$}"
    return r"{$" + payload + r"$}"


def process_table(block, per_backbone=False):
    rows = block.split(r"\\")
    group_ids = []
    g = -1
    for row in rows:
        if per_backbone and re.search(r"\\midrule", row):
            g += 1
        group_ids.append(max(g, 0))

    col_means = {}
    field_counts = []
    for row, gid in zip(rows, group_ids):
        has_payload = False
        fields = row.split("&")
        for fi, field in enumerate(fields):
            m = PAY_RE.search(field)
            if not m:
                continue
            has_payload = True
            col_means.setdefault((gid, fi), []).append(round(payload_mean(m.group()), 2))
        if has_payload:
            field_counts.append(len(fields))

    if len(set(field_counts)) > 1:
        sys.stderr.write(
            "WARNING: value rows have differing column counts "
            f"{sorted(set(field_counts))}; column alignment may be off.\n"
        )

    best_val, second_val = {}, {}
    for key, means in col_means.items():
        uniq = sorted(set(means), reverse=True)
        best_val[key] = uniq[0]
        second_val[key] = uniq[1] if len(uniq) > 1 else None

    def rewrite_row(row, gid):
        out_fields = []
        for fi, field in enumerate(row.split("&")):
            m = PAY_RE.search(field)
            if not m:
                out_fields.append(field)
                continue
            payload = m.group()
            mean = round(payload_mean(payload), 2)
            key = (gid, fi)
            if abs(mean - best_val.get(key, float("inf"))) < REBOLD_EPS:
                kind = "bold"
            elif second_val.get(key) is not None and abs(mean - second_val[key]) < REBOLD_EPS:
                kind = "under"
            else:
                kind = "plain"
            lead, _, trail = WS_RE.match(field).groups()
            out_fields.append(lead + wrap_payload(payload, kind) + trail)
        return "&".join(out_fields)

    return r"\\".join(rewrite_row(r, gid) for r, gid in zip(rows, group_ids))


def rebold(text, per_backbone=False):
    return TABLE_RE.sub(lambda m: process_table(m.group(), per_backbone), text)
