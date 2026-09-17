# Imports
import argparse
import os
import sys
import gc
import time
import torch
import json
from pathlib import Path
from utils import integrate_results
# Data imports
# Add paths so data_scripts files are findable
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent)+"/data_scripts")
from data_scripts.dataset_names import GET_DATASET, DATASET_N_CLASSES
# Hyperparameter imports
from hyper_parameters import trainer_hparams, model_class_hparams, N_SAMPLES
# Backbones
# GNN backbones
from backbones.MPNNs.gcn import GCN_encoder
from backbones.MPNNs.gat import GAT_encoder
from backbones.MPNNs.gin import GIN_encoder
# GT backbones
from backbones.GTs.graph_GPS import GraphGPS_encoder
from backbones.GTs.GrokFormer import GrokFormer_encoder
from backbones.GTs.DualFormer import DualFormer_encoder
# Baseline and model imports 
#from X2GNNs.x2gnn import X2GNN
from baselines.gnn import GNN
from baselines.cgnn.cal import CAL # Subclass of CGNN
from baselines.cgnn.icl import ICL # Subclass of CGNN
from baselines.cgnn.ace import ACE # Subclass of CGNN 
from baselines.cgnn.dir import DIR # Subclass of CGNN 
# Our proposed models
from cif.cif import CIF # Subclass of CGNN
# Ablation models:
from cif.cif_NoJ_MC import CIF_NoJ_MC
from cif.cif_J_NoMC import CIF_J_NoMC
# Pytorch lightning imports 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
# Optuna imports
import optuna
import warnings
warnings.filterwarnings("ignore")

# Experimental setup
MACHINE_PRECISION = 'bf16-mixed' 
MODEL_CLASSES = [
    GNN,  
    CIF,
    CIF_NoJ_MC,
    CIF_J_NoMC,
    DIR,  
    CAL,  
    ICL,  
    ACE 
]
# MPNN and GT backbones
BACKBONES = [
    GCN_encoder, 
    GAT_encoder, 
    GIN_encoder, 
    GraphGPS_encoder, 
    GrokFormer_encoder, 
    DualFormer_encoder
]
BACKBONE_MAP = {
    'GCN': GCN_encoder,
    'GAT': GAT_encoder,
    'GIN': GIN_encoder,
    'GraphGPS': GraphGPS_encoder,
    'GrokFormer': GrokFormer_encoder,
    'DualFormer': DualFormer_encoder,
}
# Seeds for reproducibility
SEEDS = [28, 1999, 1130, 5898, 820]
_N_SAMPLES_ENV = os.environ.get("N_SAMPLES")
NS_SUFFIX = f"_nsamples{_N_SAMPLES_ENV}" if _N_SAMPLES_ENV else ""
_RUN_TAG_ENV = os.environ.get("RUN_TAG", "")
if _RUN_TAG_ENV and not _RUN_TAG_ENV.startswith("_"):
    _RUN_TAG_ENV = "_" + _RUN_TAG_ENV
ARTIFACT_SUFFIX = f"{NS_SUFFIX}{_RUN_TAG_ENV}"

def artifact_tag(model_class, gnn_backbone):
    """File/dir identity for a (model, backbone) run, suffixed in sensitivity/tagged mode."""
    return f"{model_class.__name__}_{gnn_backbone.__name__}{ARTIFACT_SUFFIX}"

# Set up accelerator 
if torch.cuda.is_available():
    accelerator = "gpu"
else:
    accelerator = "cpu"

print(f"ACCELERATOR is {accelerator}")

# Objective function for optuna hyperparameter tuning
def objective_wrapper(model_class, gnn_backbone, input_channels, num_classes, loader, directory_name):
    def objective(trial):
        two_speed = model_class.__name__ == "CIF_NoJ_MC"
        if two_speed:
            lr_head = trial.suggest_categorical("lr_head", [5e-5, 1e-4, 5e-4, 1e-3])
            lr_base = trial.suggest_categorical("lr_base", [5e-5, 1e-4, 5e-4, 1e-3])
            if lr_base >= lr_head: 
                raise optuna.TrialPruned()          
            wd = trial.suggest_categorical("wd", [1e-3])
            optimizer_hparams = {"lr_head": lr_head, "lr_base": lr_base, "wd": wd}
            run_tag = f"lrh_{lr_head}_lrb_{lr_base}_wd_{wd}"
        else: 
            lr = trial.suggest_categorical("lr", [5e-5, 1e-4, 5e-4, 1e-3])
            wd = trial.suggest_categorical("wd", [1e-3])
            optimizer_hparams = {"lr": lr, "wd": wd}
            run_tag = f"lr_{lr}_wd_{wd}"

        # Build model
        model = model_class(
            gnn_backbone=gnn_backbone,
            in_channels=input_channels['in_channels'],
            in_channels_e=input_channels['in_channels_e'],
            num_classes=num_classes,
            model_hparams=model_class_hparams[f'{model_class.__name__}'],
            optimizer_hparams=optimizer_hparams
        )

        early_stop_callback = EarlyStopping(
            monitor="val_auroc",
            mode="max",
            patience=trainer_hparams["patience_hparam_tun"]
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_auroc",
            mode="max",
            save_top_k=1,
            dirpath=f"{directory_name}/hparam_tuning/checkpoints/{model_class.__name__}_{gnn_backbone.__name__}",
            filename=run_tag
        )

        # Save optuna logs
        log_dir = (
            f"{directory_name}/hparam_tuning/optuna_logs/"
            f"{model_class.__name__}_{gnn_backbone.__name__}"
        )

        logger = TensorBoardLogger(
            save_dir=log_dir,
            name=run_tag
        )

        # Trainer
        trainer = pl.Trainer(
            accelerator=accelerator, 
            devices=1,
            max_epochs=trainer_hparams['epochs_hparam_tun'],
            gradient_clip_val=trainer_hparams['max_norm'] if model.automatic_optimization else None,
            callbacks=[checkpoint_callback, early_stop_callback],  
            logger=logger,
            enable_model_summary=False,
            deterministic=False,
            precision=MACHINE_PRECISION
        )

        # Train
        trainer.fit(
            model=model,
            train_dataloaders=loader['train'],
            val_dataloaders=loader['val']
        )

        # Return the best AUROC returned
        best_val_auroc = float(checkpoint_callback.best_model_score.item())
        return best_val_auroc

    return objective


# Run hyperparameter tuning
def run_hparam_tuning(model_class, gnn_backbone, input_channels, num_classes, loader, directory_name):
    if model_class.__name__ == "CIF_NoJ_MC":
        search_space = {
            "lr_head": [5e-5, 1e-4, 5e-4, 1e-3],
            "lr_base": [5e-5, 1e-4, 5e-4, 1e-3],
            "wd": [1e-3],
        }
    else: 
        search_space = {
            "lr": [5e-5, 1e-4, 5e-4, 1e-3],
            "wd": [1e-3],
        }
    sampler = optuna.samplers.GridSampler(search_space, seed=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler
    )

    objective = objective_wrapper(
        model_class, 
        gnn_backbone, 
        input_channels, 
        num_classes, 
        loader, 
        directory_name
    )

    study.optimize(objective)

    # Save best hyperparameters
    best_trial = study.best_trial

    save_dir = f"{directory_name}/hparam_tuning/hparams_best"
    os.makedirs(save_dir, exist_ok=True)

    best_params_path = (
        f"{save_dir}/{model_class.__name__}_{gnn_backbone.__name__}.json"
    )

    with open(best_params_path, "w") as f:
        json.dump(best_trial.params, f, indent=4)

    return best_trial

 
# Train model 
def train_model(model_class, gnn_backbone, input_channels, num_classes, best_params_file_name, directory_name, run_index, loader):
    # Getting the best hyperparameters
    with open(best_params_file_name, "r") as f:
        best_params = json.load(f)

    # Creating the model with the best hyperparameters
    model = model_class(
        gnn_backbone=gnn_backbone,
        in_channels=input_channels['in_channels'],
        in_channels_e=input_channels['in_channels_e'],
        num_classes=num_classes,
        model_hparams=model_class_hparams[f'{model_class.__name__}'],
        optimizer_hparams=best_params
    )

    # Creating trainer instance
    # Add checkpoint to save the best model
    early_stop_callback = EarlyStopping(
            monitor="val_auroc",
            mode="max",
            patience=trainer_hparams["patience"]
        )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        dirpath=directory_name + f"/run_{run_index + 1}/checkpoints",
        filename=artifact_tag(model_class, gnn_backbone)
    )

    # Save the lightning logs
    logger = TensorBoardLogger(
        save_dir=f"{directory_name}/lightning_logs/{artifact_tag(model_class, gnn_backbone)}",
        name=f"run_{run_index + 1}"
    )
    
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=trainer_hparams['epochs'],
        gradient_clip_val=trainer_hparams['max_norm'] if model.automatic_optimization else None,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        deterministic=False,
        precision=MACHINE_PRECISION
    )
    
    # Training the model with the best hyperparameters.
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    trainer.fit(model=model, train_dataloaders=loader['train'], val_dataloaders=loader['val'])
    train_time_sec = time.perf_counter() - t0
    epochs_trained = int(trainer.current_epoch)
    profile = {
        "n_samples": N_SAMPLES,
        "train_time_sec": train_time_sec,
        "epochs_trained": epochs_trained,
        "train_time_per_epoch_sec": train_time_sec / max(epochs_trained, 1),
    }
    if use_cuda:
        profile["peak_gpu_mem_mib"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        profile["peak_gpu_mem_reserved_mib"] = torch.cuda.max_memory_reserved() / (1024 ** 2)

    mem_str = f"peak_gpu={profile['peak_gpu_mem_mib']:.0f}MiB" if use_cuda else "peak_gpu=n/a(cpu)"
    print(f"[profile] N_SAMPLES={N_SAMPLES} run_{run_index + 1} "
          f"{artifact_tag(model_class, gnn_backbone)} time={train_time_sec:.1f}s "
          f"epochs={epochs_trained} {mem_str}")
    return profile


# Test model
def test_model(model_class, gnn_backbone, input_channels, num_classes, best_params_file_name, checkpoint_path, metrics_file_name, loader, profile=None):
    # Load best hyperparameters
    with open(best_params_file_name, "r") as f:
        best_params = json.load(f)

    # Load model from checkpoint
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        gnn_backbone=gnn_backbone,
        in_channels=input_channels['in_channels'],
        in_channels_e=input_channels['in_channels_e'],
        num_classes=num_classes,
        model_hparams=model_class_hparams[f'{model_class.__name__}'],
        optimizer_hparams=best_params
    )

    trainer = pl.Trainer(
        accelerator=accelerator, 
        deterministic=False, 
        logger=False, 
        enable_checkpointing=False,
        precision=MACHINE_PRECISION
    )

    # Test with loaded model
    results = trainer.test(model, dataloaders=loader["test"])
    if profile is not None:
        for row in results:
            row.update(profile)

    # Save results into JSON file
    with open(metrics_file_name, "w") as f:
        json.dump(results, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument('--dataset',  type=str, required=True, choices=list(GET_DATASET.keys()))
    parser.add_argument('--backbone', type=str, required=True, choices=list(BACKBONE_MAP.keys()))
    parser.add_argument('--model',    type=str, required=True, choices=[m.__name__ for m in MODEL_CLASSES])
    return parser.parse_args()

MODEL_MAP = {m.__name__: m for m in MODEL_CLASSES}

# Run the experiments
def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')

    dataset_name = args.dataset
    gnn_backbone = BACKBONE_MAP[args.backbone]
    model_class  = MODEL_MAP[args.model]

    # Create directories to save best hyperparameters
    directory_name = f"../results/{dataset_name}"
    os.makedirs(directory_name, exist_ok=True)
    
    # Set seed
    pl.seed_everything(0, workers=True)

    # Fetch dataset
    entry = GET_DATASET[dataset_name]
    input_channels, loader = entry() if callable(entry) else entry
    num_classes = DATASET_N_CLASSES[dataset_name]
    
    # Best hyperparameters 
    best_params_file_name = directory_name + f"/hparam_tuning/hparams_best/{model_class.__name__}_{gnn_backbone.__name__}.json"

    # Run hyperparameter tunning
    if ARTIFACT_SUFFIX:
        if not os.path.exists(best_params_file_name):
            raise FileNotFoundError(
                f"Tagged run (suffix '{ARTIFACT_SUFFIX}') expected existing best hyperparameters "
                f"at {best_params_file_name}, but none were found. Run the default (untagged, "
                f"N_SAMPLES=16) experiment for this model/backbone first to produce them."
            )
        print(f"[suffix={ARTIFACT_SUFFIX}] Skipping hparam tuning; reusing {best_params_file_name}")
    else:
        run_hparam_tuning(model_class, gnn_backbone, input_channels, num_classes, loader, directory_name)

    # Train and test the model 5 times with the best found hyperparameter configuration
    for run_index, seed in enumerate(SEEDS):
        # Set the seed
        pl.seed_everything(seed, workers=True)

        # Train
        profile = train_model(model_class, gnn_backbone, input_channels, num_classes, best_params_file_name, directory_name, run_index, loader)

        # Test
        checkpoint_path = directory_name + f"/run_{run_index + 1}/checkpoints/{artifact_tag(model_class, gnn_backbone)}.ckpt"
        os.makedirs(directory_name + f"/run_{run_index + 1}/test_metrics", exist_ok=True)
        metrics_file_name = directory_name + f"/run_{run_index + 1}/test_metrics/{artifact_tag(model_class, gnn_backbone)}.json"
        test_model(model_class, gnn_backbone, input_channels, num_classes, best_params_file_name, checkpoint_path, metrics_file_name, loader, profile=profile)
    # Integrate the results into a single file
    os.makedirs(directory_name + "/integrated", exist_ok=True)
    integrate_results(base_dir=directory_name, integrated_dir=directory_name + "/integrated", name_filter=ARTIFACT_SUFFIX)

if __name__ == "__main__":  
    main()