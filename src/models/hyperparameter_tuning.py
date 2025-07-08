# python src/models/hyperparameter_tuning.py --model_type lstm --n_trials 30 --num_epochs 10

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


def objective(trial):
    # Suggest hyperparameters
    hidden_size = trial.suggest_int("hidden_size", 64, 512, step=64)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

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
        final_val_loss = 0.0  # inizializza variabile di fallback
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
        # Log delle metriche finali di fine trial
        # Assicuriamoci che best_val_loss non sia inf
        if not np.isfinite(best_val_loss):
            best_val_loss = final_val_loss
        # Log delle metriche finali di fine trial
        mlflow.log_metrics({"best_val_loss": best_val_loss})
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
        study = optuna.create_study(direction="minimize")
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
