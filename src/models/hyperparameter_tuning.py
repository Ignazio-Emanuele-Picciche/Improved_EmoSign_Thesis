# =================================================================================================
# COMANDI UTILI PER L'ESECUZIONE
# =================================================================================================
#
# 1. AVVIARE IL SERVER MLFLOW
#    Per monitorare gli esperimenti, prima di eseguire lo script di tuning,
#    aprire un terminale e lanciare il server MLflow:
#
#    mlflow server --host 127.0.0.1 --port 8080
#
# 2. ESEGUIRE L'OTTIMIZZAZIONE DEGLI IPERPARAMETRI
#    Questo script utilizza Optuna per trovare i migliori iperparametri per un dato modello.
#    È possibile specificare il tipo di modello (lstm o stgcn), il numero di "trial" (tentativi)
#    e il numero di epoche per ogni trial.
#
#    ESEMPIO DI COMANDO PER LSTM:
#    Esegue 40 tentativi di ottimizzazione per il modello LSTM, addestrando per 40 epoche ogni volta.
#
#    python3 src/models/hyperparameter_tuning.py --model_type lstm --n_trials 40 --num_epochs 40
#
#    ESEMPIO DI COMANDO PER ST-GCN:
#    Esegue 20 tentativi di ottimizzazione per il modello ST-GCN, addestrando per 25 epoche ogni volta.
#
#    python3 src/models/hyperparameter_tuning.py --model_type stgcn --n_trials 20 --num_epochs 25
#
# =================================================================================================

import os
import sys
import argparse
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
from optuna.pruners import MedianPruner
import random
from torch.backends import cudnn
from optuna.samplers import TPESampler

# Ignite imports
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
from ignite.engine import Events, create_supervised_trainer
from ignite.handlers import EarlyStopping as IgniteEarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optuna.integration import PyTorchIgnitePruningHandler
from optuna.exceptions import TrialPruned

# --- Sezione 1: Setup del Percorso di Base e Import delle Utilità ---
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
)
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

from src.utils.training_utils import (
    get_data_paths,
    get_dataloaders,
    get_class_weights,
    create_model,
    prepare_batch,
    setup_ignite_evaluator,
)

# --- Sezione 2: Setup di MLflow ---
mlflow.set_tracking_uri("http://127.0.0.1:8080")
experiment_name = "VADER 0.65 - Emotion Recognition Experiment"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name)
else:
    mlflow.set_experiment(experiment_name)

# --- Sezione 3: Definizione dei Percorsi Dati ---
(
    TRAIN_LANDMARKS_DIR,
    TRAIN_PROCESSED_FILE,
    VAL_LANDMARKS_DIR,
    VAL_PROCESSED_FILE,
) = get_data_paths(BASE_DIR)

# --- Sezione 4: Variabili Globali ---
NUM_EPOCHS = 5
MODEL_TYPE = "lstm"
PATIENCE = 10


def objective(trial):
    """
    Funzione "obiettivo" di Optuna per un singolo trial di ottimizzazione.
    """
    # --- Sezione 5: Definizione dello Spazio di Ricerca degli Iperparametri ---
    if MODEL_TYPE == "lstm":
        hidden_size = trial.suggest_int("hidden_size", 768, 1024, step=32)
        num_layers = trial.suggest_int("num_layers", 2, 5)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
        batch_size = trial.suggest_categorical("batch_size", [64, 96, 128, 160])
        model_params = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
        }
    else:  # stgcn
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        dropout = trial.suggest_float("dropout", 0, 0.2)
        model_params = {"dropout": dropout}

    # --- Sezione 6: Caricamento Dati ---
    train_loader, val_loader, train_dataset = get_dataloaders(
        TRAIN_LANDMARKS_DIR,
        TRAIN_PROCESSED_FILE,
        VAL_LANDMARKS_DIR,
        VAL_PROCESSED_FILE,
        batch_size,
    )

    # --- Sezione 7: Setup del Modello e dell'Addestramento ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    class_weights = get_class_weights(train_dataset, device)
    input_size = train_dataset[0][0].shape[1]
    num_classes = len(train_dataset.labels)

    model = create_model(MODEL_TYPE, input_size, num_classes, device, model_params)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=3)

    prepare_batch_fn = lambda batch, device, non_blocking: prepare_batch(
        batch, device, MODEL_TYPE, non_blocking
    )

    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device, prepare_batch=prepare_batch_fn
    )
    evaluator = setup_ignite_evaluator(model, criterion, device, MODEL_TYPE, val_loader)

    # --- Sezione 8: Esecuzione del Trial e Logging ---
    with mlflow.start_run(nested=True):
        base_params = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "model_type": MODEL_TYPE,
        }
        mlflow.log_params({**base_params, **model_params})
        class_map = {i: label for i, label in enumerate(train_dataset.labels)}
        mlflow.log_params({f"class_{i}": label for i, label in class_map.items()})

        best_metrics = {"val_loss": float("inf"), "val_f1": 0.0, "val_f1_macro": 0.0}

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            model.eval()
            train_loss_sum = 0.0
            with torch.no_grad():
                for xb, yb in train_loader:
                    xb, yb = prepare_batch_fn((xb, yb), device, non_blocking=False)
                    outputs = model(xb.float())
                    loss = criterion(outputs, yb)
                    train_loss_sum += loss.item() * xb.size(0)
            train_loss = train_loss_sum / len(train_loader.dataset)

            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            val_loss = metrics["val_loss"]
            val_f1 = metrics["val_f1"]
            val_f1_macro = metrics["val_f1_macro"]

            if val_f1_macro > best_metrics["val_f1_macro"]:
                best_metrics["val_f1_macro"] = val_f1_macro
            if val_loss < best_metrics["val_loss"]:
                best_metrics["val_loss"] = val_loss
            if val_f1 > best_metrics["val_f1"]:
                best_metrics["val_f1"] = val_f1

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": metrics["val_acc"],
                    "val_f1": val_f1,
                    "val_f1_macro": val_f1_macro,
                    "val_f1_class_0": metrics["val_f1_class_0"],
                    "val_f1_class_1": metrics["val_f1_class_1"],
                },
                step=engine.state.epoch,
            )
            scheduler.step(val_f1_macro)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

        # --- Sezione 9: Pruning e Early Stopping ---
        pruning_handler = PyTorchIgnitePruningHandler(trial, "val_f1_macro", trainer)
        evaluator.add_event_handler(Events.COMPLETED, pruning_handler)

        early_stopper = IgniteEarlyStopping(
            patience=PATIENCE,
            score_function=lambda eng: eng.state.metrics["val_f1_macro"],
            trainer=trainer,
        )
        evaluator.add_event_handler(Events.COMPLETED, early_stopper)

        try:
            trainer.run(train_loader, max_epochs=NUM_EPOCHS)
        except TrialPruned:
            mlflow.set_tag("status", "pruned")
            raise

        mlflow.log_metrics(
            {
                "best_val_loss": best_metrics["val_loss"],
                "best_val_f1": best_metrics["val_f1"],
                "best_val_f1_macro": best_metrics["val_f1_macro"],
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        return best_metrics["val_f1_macro"]


def main():
    """
    Funzione principale che orchestra l'intero processo di ottimizzazione.
    """
    # --- Sezione 10: Parsing degli Argomenti e Setup dello Studio ---
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning with Optuna and MLflow"
    )
    parser.add_argument(
        "--model_type", type=str, choices=["lstm", "stgcn"], default="lstm"
    )
    parser.add_argument(
        "--n_trials", type=int, default=20, help="Number of Optuna trials"
    )
    parser.add_argument("--num_epochs", type=int, default=5, help="Epochs per trial")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    global NUM_EPOCHS, MODEL_TYPE
    NUM_EPOCHS = args.num_epochs
    MODEL_TYPE = args.model_type
    run_name = f"Optuna_{MODEL_TYPE}_tuning"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("seed", seed)
        study = optuna.create_study(
            direction="maximize",
            pruner=MedianPruner(n_startup_trials=PATIENCE, n_warmup_steps=1),
            sampler=TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=args.n_trials)
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_val_f1_macro_study", study.best_value)
        mlflow.set_tags(
            {
                "project": "EmoSign",
                "optimizer_engine": "optuna",
                "model_family": MODEL_TYPE,
            }
        )
    print("Best trial:", study.best_trial.params)


if __name__ == "__main__":
    main()
