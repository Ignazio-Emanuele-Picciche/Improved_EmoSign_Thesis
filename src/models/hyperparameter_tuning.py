# python src/models/hyperparameter_tuning.py --model_type lstm --n_trials 15 --num_epochs 20
# python src/models/hyperparameter_tuning.py --model_type lstm --n_trials 40 --num_epochs 40
# mlflow server --host 127.0.0.1 --port 8080

import os
import sys
import argparse
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from optuna.pruners import MedianPruner

# Add project src to PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

from data_pipeline.landmark_dataset import LandmarkDataset
from models.lstm_model import EmotionLSTM
from models.stgcn_model import STGCN
from optuna.exceptions import TrialPruned

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
    hidden_size = trial.suggest_int("hidden_size", 256, 1024, step=128)
    num_layers = trial.suggest_int("num_layers", 1, 6)
    dropout = trial.suggest_float("dropout", 0.0, 0.2)
    learning_rate = trial.suggest_loguniform("learning_rate", 5e-4, 5e-3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

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
        best_val_loss = float("inf")
        best_val_f1 = 0.0
        early_stopper = EarlyStopping(
            patience=PATIENCE, verbose=False, path=f"trial_{trial.number}_best.pth"
        )
        final_val_loss = 0.0  # inizializza variabile di fallback
        final_val_f1 = 0.0
        for epoch in range(NUM_EPOCHS):
            # Training
            model.train()
            train_loss = 0.0
            for seq, lbl in train_loader:
                if MODEL_TYPE == "stgcn":
                    b, t, f = seq.shape
                    seq = seq.view(b, t, -1, 2)
                seq = seq.to(device).float()
                lbl = lbl.to(device)
                out = model(seq)
                loss = criterion(out, lbl)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * lbl.size(0)
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for seq, lbl in val_loader:
                    if MODEL_TYPE == "stgcn":
                        b, t, f = seq.shape
                        seq = seq.view(b, t, -1, 2)
                    seq = seq.to(device).float()
                    lbl = lbl.to(device)
                    out = model(seq)
                    loss = criterion(out, lbl)
                    val_loss += loss.item() * lbl.size(0)
                val_loss /= len(val_loader.dataset)
                # calcolo accuracy e F1-score per monitoraggio
                correct = 0
                total = 0
                all_preds = []
                all_labels = []
                for seq, lbl in val_loader:
                    if MODEL_TYPE == "stgcn":
                        b, t, f = seq.shape
                        seq = seq.view(b, t, -1, 2)
                    seq = seq.to(device).float()
                    lbl = lbl.to(device)
                    out = model(seq)
                    _, preds = torch.max(out, 1)
                    correct += (preds == lbl).sum().item()
                    total += lbl.size(0)
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_labels.extend(lbl.cpu().numpy().tolist())
                val_acc = correct / total if total else 0.0
                val_f1 = f1_score(all_labels, all_preds, average="macro")
            # memorizza valore di validazione per fallback
            final_val_loss = val_loss
            final_val_f1 = val_f1
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0  # resetta il contatore se migliora
            else:
                epochs_no_improve += 1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
            # Log metrics per epoch
            mlflow.log_metrics(
                {
                    "train_loss": train_loss / len(train_loader.dataset),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                },
                step=epoch,
            )
            # Early stopping check
            early_stopper(val_loss, model)
            if early_stopper.early_stop:
                print(f"Trial {trial.number} early stopping at epoch {epoch+1}")
                break
        # Log delle metriche finali di fine trial
        # fallback se non abbiamo trovato miglioramento
        if not np.isfinite(best_val_loss):
            best_val_loss = final_val_loss
        if best_val_f1 == 0.0:
            best_val_f1 = final_val_f1
        # Log delle metriche finali di fine trial
        mlflow.log_metrics({"best_val_loss": best_val_loss, "best_val_f1": best_val_f1})
        return best_val_loss


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
    args = parser.parse_args()

    global NUM_EPOCHS, MODEL_TYPE
    NUM_EPOCHS = args.num_epochs
    MODEL_TYPE = args.model_type
    # Parent MLflow run per studio di ottimizzazione
    run_name = f"Optuna_{MODEL_TYPE}_tuning"
    with mlflow.start_run(run_name=run_name):
        study = optuna.create_study(
            direction="minimize",
            pruner=MedianPruner(n_startup_trials=PATIENCE, n_warmup_steps=1),
        )
        study.optimize(objective, n_trials=args.n_trials)
        # Log global tuning info
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_val_loss", study.best_value)
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
