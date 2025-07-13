# mlflow server --host 127.0.0.1 --port 8080


# python src/models/run_train.py \
#   --batch_size 128 \
#   --hidden_size 896 \
#   --num_layers 5 \
#   --learning_rate 1.0677482709481361e-05 \
#   --dropout 0.23993475643167195 \
#   --num_epochs 100 \
#   --seed 44

# BEST
# python src/models/run_train.py \
#   --batch_size 128 \
#   --hidden_size 896 \
#   --num_layers 3 \
#   --learning_rate 2.621087878265438e-05 \
#   --dropout 0.23993475643167195 \
#   --num_epochs 100 \
#   --seed 44


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
import os, sys
import argparse
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import requests  # for pinging MLflow server

# Aggiungi il percorso base del progetto al PYTHONPATH
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
sys.path.insert(0, BASE_DIR)

# Ignite imports
# Workaround for MPS OOM: disable MPS upper memory limit (may risk system stability)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
from ignite.handlers import EarlyStopping as IgniteEarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data_pipeline.landmark_dataset import (
    LandmarkDataset,
)  # Importa la classe per gestire i dati
from src.models.lstm_model import EmotionLSTM  # Importa l'architettura del modello LSTM
from src.models.stgcn_model import STGCN  # Aggiunto per ST-GCN support
from torch.backends import cudnn
import random


def main(args):
    """Main training and evaluation function."""
    # --- Sezione 2: Setup dell'Ambiente ---
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # Seleziona il dispositivo su cui eseguire i calcoli
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training su dispositivo: {device}")

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment("Emotion Recognition Experiment")

    # --- Sezione 3: Caricamento Dati e Creazione del Modello ---
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

    train_dataset = LandmarkDataset(
        landmarks_dir=TRAIN_LANDMARKS_DIR, processed_file=TRAIN_PROCESSED_FILE
    )
    val_dataset = LandmarkDataset(
        landmarks_dir=VAL_LANDMARKS_DIR, processed_file=VAL_PROCESSED_FILE
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Bilanciamento classi
    labels_array = (
        train_dataset.processed["emotion"].map(train_dataset.label_map).values
    )
    class_counts = np.bincount(labels_array)
    class_weights = [len(labels_array) / (len(class_counts) * c) for c in class_counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Determino dimensione input e creo il modello
    input_size = train_dataset[0][0].shape[1]
    num_classes = len(train_dataset.labels)
    if args.model_type == "lstm":
        model = EmotionLSTM(
            input_size,
            args.hidden_size,
            args.num_layers,
            num_classes,
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == "stgcn":
        coords_per_point = 2
        num_point = input_size // coords_per_point
        model = STGCN(
            num_classes, num_point, num_person=1, in_channels=coords_per_point
        ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    # --- Sezione 4: Definizione delle Metriche e Handlers ---
    # Definisci le metriche per l'evaluator
    val_metrics = {
        "val_loss": Loss(criterion, device="cpu"),
        "val_acc": Accuracy(device="cpu"),
    }

    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    # Funzione per calcolare e aggiungere F1 score alle metriche
    @evaluator.on(Events.COMPLETED)
    def compute_f1_scores(engine):
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                outputs = model(xb.float())
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # Calcola le metriche F1
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Aggiungi le metriche allo stato dell'evaluator
        engine.state.metrics["val_f1_macro"] = f1_macro
        engine.state.metrics["val_f1"] = f1_weighted
        engine.state.metrics["val_f1_class_0"] = float(f1_per_class[0])
        engine.state.metrics["val_f1_class_1"] = (
            float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0
        )

    # Handlers
    # ModelCheckpoint ora usa val_f1_macro per salvare il modello migliore
    checkpoint_handler = ModelCheckpoint(
        dirname=os.path.join(BASE_DIR, "models"),
        filename_prefix=f"emotion_{args.model_type}",
        n_saved=1,
        create_dir=True,
        require_empty=False,
        score_function=lambda eng: eng.state.metrics["val_f1_macro"],
        score_name="val_f1_macro",
    )
    evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, {"model": model})

    # EarlyStopping ora usa val_f1_macro per decidere quando fermarsi
    earlystop_handler = IgniteEarlyStopping(
        patience=args.patience,
        score_function=lambda eng: eng.state.metrics["val_f1_macro"],
        trainer=trainer,
    )
    evaluator.add_event_handler(Events.COMPLETED, earlystop_handler)

    # --- Sezione 5: Ciclo di Addestramento ---
    print("Inizio training con validazione (Ignite)...")
    run_name = f"EmotionRecognition_{args.model_type}_train"

    with mlflow.start_run(run_name=run_name) as run:
        params = {
            "model_type": args.model_type,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "dropout": args.dropout,
            "seed": args.seed,
        }
        mlflow.log_params(params)
        best_metrics = {"val_loss": float("inf"), "val_f1": 0.0, "val_f1_macro": 0.0}

        @trainer.on(Events.EPOCH_COMPLETED)
        def validate_and_log(engine):
            # Compute average training loss on train_loader
            model.eval()
            train_loss_sum = 0.0
            with torch.no_grad():
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    outputs = model(xb.float())
                    loss = criterion(outputs, yb)
                    train_loss_sum += loss.item() * xb.size(0)
            train_loss = train_loss_sum / len(train_loader.dataset)

            # Esegui l'evaluator per calcolare tutte le metriche (inclusi F1 scores)
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            val_loss = metrics["val_loss"]
            val_acc = metrics["val_acc"]
            val_f1 = metrics["val_f1"]
            val_f1_macro = metrics["val_f1_macro"]

            # Update best metrics
            if val_loss < best_metrics["val_loss"]:
                best_metrics["val_loss"] = val_loss
            if val_f1 > best_metrics["val_f1"]:
                best_metrics["val_f1"] = val_f1
            if val_f1_macro > best_metrics["val_f1_macro"]:
                best_metrics["val_f1_macro"] = val_f1_macro

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
            print(
                f"Epoch {engine.state.epoch}/{args.num_epochs}, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val F1 Macro: {val_f1_macro:.4f}"
            )
            scheduler.step(val_loss)

            # Clear cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

        trainer.run(train_loader, max_epochs=args.num_epochs)

        print(
            f"Training completato. Best model saved in {checkpoint_handler.last_checkpoint}"
        )

        # After training: log final metrics (best)
        mlflow.log_metric("best_val_loss", best_metrics["val_loss"])
        mlflow.log_metric("best_val_f1", best_metrics["val_f1"])
        mlflow.log_metric("best_val_f1_macro", best_metrics["val_f1_macro"])

        # Infer model signature and log the model
        # Get a single batch from the train_loader to infer signature
        data_sample, _ = next(iter(train_loader))
        signature = infer_signature(
            data_sample.numpy(),
            model(data_sample.to(device).float()).cpu().detach().numpy(),
        )
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="EmoSign_pytorch",
        )
        mlflow.set_tag("experiment_purpose", "PyTorch emotion recognition")


# parsing iperparametri da linea di comando
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Emotion Recognition Model")
    parser.add_argument(
        "--model_type", type=str, default="lstm", choices=["lstm", "stgcn"]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    main(args)

# python train_model.py --batch_size 64 --hidden_size 512 --num_layers 3 --learning_rate 0.0005 --dropout 0.3
