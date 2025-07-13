# python3 src/models/hyperparameter_tuning.py --model_type lstm --n_trials 15 --num_epochs 20
# python3 src/models/hyperparameter_tuning.py --model_type lstm --n_trials 40 --num_epochs 40
# mlflow server --host 127.0.0.1 --port 8080

import os
import sys
import argparse
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from optuna.pruners import MedianPruner
import random
from torch.backends import cudnn
from optuna.samplers import TPESampler

# Ignite imports
# Workaround for MPS OOM: disable MPS upper memory limit (may risk system stability)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
from ignite.handlers import EarlyStopping as IgniteEarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optuna.integration import PyTorchIgnitePruningHandler
from optuna.exceptions import TrialPruned


# Add project src to PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

from data_pipeline.landmark_dataset import LandmarkDataset
from models.lstm_model import EmotionLSTM
from models.stgcn_model import STGCN

# MLflow tracking setup
mlflow.set_tracking_uri("http://127.0.0.1:8080")
experiment_name = "Emotion Recognition Experiment"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name)
else:
    mlflow.set_experiment(experiment_name)

# Data paths
TRAIN_LANDMARKS_DIR = os.path.join(
    BASE_DIR, "data", "raw", "train", "openpose_output_train", "json"
)
TRAIN_PROCESSED_FILE = os.path.join(
    BASE_DIR, "data", "processed", "train", "video_sentiment_data_0.65.csv"
)
VAL_LANDMARKS_DIR = os.path.join(
    BASE_DIR, "data", "raw", "val", "openpose_output_val", "json"
)
VAL_PROCESSED_FILE = os.path.join(
    BASE_DIR, "data", "processed", "val", "video_sentiment_data_0.65.csv"
)

# Globals for closure
NUM_EPOCHS = 5
MODEL_TYPE = "lstm"
PATIENCE = 5  # early stopping patience


def objective(trial):
    # Suggest hyperparameters (range affinati in base ai risultati precedenti)
    # hidden_size = trial.suggest_int("hidden_size", 128, 2048, step=128)
    # num_layers = trial.suggest_int("num_layers", 1, 10)
    # dropout = trial.suggest_float("dropout", 0.0, 0.5)
    # learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    # batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])

    hidden_size = trial.suggest_int("hidden_size", 768, 1024, step=32)
    num_layers = trial.suggest_int("num_layers", 2, 5)
    dropout = trial.suggest_float("dropout", 0.2, 0.4)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
    batch_size = trial.suggest_categorical("batch_size", [64, 96, 128, 160])

    # Load data
    train_dataset = LandmarkDataset(
        landmarks_dir=TRAIN_LANDMARKS_DIR, processed_file=TRAIN_PROCESSED_FILE
    )
    val_dataset = LandmarkDataset(
        landmarks_dir=VAL_LANDMARKS_DIR, processed_file=VAL_PROCESSED_FILE
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Class weights
    labels_arr = train_dataset.processed["emotion"].map(train_dataset.label_map).values
    counts = np.bincount(labels_arr)
    weights = [len(labels_arr) / (len(counts) * c) for c in counts]
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    class_weights = class_weights.to(device)

    # Model instantiation
    input_size = train_dataset[0][0].shape[1]
    num_classes = len(train_dataset.labels)
    if MODEL_TYPE == "lstm":
        model = EmotionLSTM(
            input_size, hidden_size, num_layers, num_classes, dropout
        ).to(device)
    else:
        coords_per_point = 2
        num_point = input_size // coords_per_point
        model = STGCN(
            num_classes, num_point, num_person=1, in_channels=coords_per_point
        ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(
        model,
        metrics={"val_loss": Loss(criterion), "val_acc": Accuracy()},
        device=device,
    )

    # Funzione per calcolare e aggiungere F1 score alle metriche
    @evaluator.on(Events.COMPLETED)
    def compute_f1_scores(engine):
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                if MODEL_TYPE == "stgcn":
                    b, t, f = xb.shape
                    xb = xb.view(b, t, -1, 2)
                xb = xb.to(device)
                outputs = model(xb.float())
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        f1_per_class = f1_score(y_true, y_pred, average=None)
        engine.state.metrics["val_f1_macro"] = f1_macro
        engine.state.metrics["val_f1"] = f1_weighted
        engine.state.metrics["val_f1_class_0"] = float(f1_per_class[0])
        engine.state.metrics["val_f1_class_1"] = (
            float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0
        )

    # nested MLflow run per trial
    with mlflow.start_run(nested=True):
        params = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "model_type": MODEL_TYPE,
        }
        mlflow.log_params(params)
        # Log mapping from class indices to emotion labels
        class_map = {i: label for i, label in enumerate(train_dataset.labels)}
        mlflow.log_params({f"class_{i}": label for i, label in class_map.items()})

        best_metrics = {"val_loss": float("inf"), "val_f1": 0.0, "val_f1_macro": 0.0}

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            # Compute average training loss for this epoch
            model.eval()
            train_loss_sum = 0.0
            with torch.no_grad():
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    outputs = model(xb.float())
                    loss = criterion(outputs, yb)
                    train_loss_sum += loss.item() * xb.size(0)
            train_loss = train_loss_sum / len(train_loader.dataset)

            # Esegui evaluator per calcolare tutte le metriche (loss, acc, e F1)
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            val_loss = metrics["val_loss"]
            val_acc = metrics["val_acc"]
            val_f1 = metrics["val_f1"]
            val_f1_macro = metrics["val_f1_macro"]

            # Update best macro F1
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
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "val_f1_macro": val_f1_macro,
                    "val_f1_class_0": metrics["val_f1_class_0"],
                    "val_f1_class_1": metrics["val_f1_class_1"],
                },
                step=engine.state.epoch,
            )
            scheduler.step(val_loss)
            # Clear GPU/MPS cache to prevent OOM
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

        # Optuna pruner: ora monitora val_f1_macro
        pruning_handler = PyTorchIgnitePruningHandler(trial, "val_f1_macro", trainer)
        evaluator.add_event_handler(Events.COMPLETED, pruning_handler)

        # Early stopping: ora monitora val_f1_macro
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

        # Log delle metriche finali di fine trial
        mlflow.log_metrics(
            {
                "best_val_loss": best_metrics["val_loss"],
                "best_val_f1": best_metrics["val_f1"],
                "best_val_f1_macro": best_metrics["val_f1_macro"],
                # "best_val_f1_class_0": float(f1_per_class[0]),
                # "best_val_f1_class_1": (
                #     float(f1_per_class[1]) if len(f1_per_class) > 1 else None
                # ),
            }
        )
        # Clear cache after trial
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        # Restituisce la metrica da ottimizzare (massimizzare)
        return best_metrics["val_f1_macro"]


def main():
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
    # Set random seeds for reproducibility
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
    # Parent MLflow run per studio di ottimizzazione
    run_name = f"Optuna_{MODEL_TYPE}_tuning"
    with mlflow.start_run(run_name=run_name):
        # Log seed used for reproducibility
        mlflow.log_param("seed", seed)
        study = optuna.create_study(
            direction="maximize",  # Cambiato a "maximize" per f1_macro
            pruner=MedianPruner(n_startup_trials=PATIENCE, n_warmup_steps=1),
            sampler=TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=args.n_trials)
        # Log global tuning info
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
