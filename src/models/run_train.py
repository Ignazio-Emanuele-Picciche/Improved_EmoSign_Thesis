# =================================================================================================
# COMANDI UTILI PER L'ESECUZIONE
# =================================================================================================
#
# 1. AVVIARE IL SERVER MLFLOW
#    Per monitorare gli esperimenti, prima di eseguire lo script di training,
#    aprire un terminale e lanciare il server MLflow:
#
#    mlflow server --host 127.0.0.1 --port 8080
#
# 2. ESEGUIRE L'ADDESTRAMENTO DEL MODELLO
#    Questo script permette di addestrare un modello per il riconoscimento delle emozioni.
#    È possibile specificare il tipo di modello (lstm o stgcn) e i relativi iperparametri.
#
#    ESEMPIO DI COMANDO PER LSTM (con iperparametri ottimizzati):
#    Questo comando avvia l'addestramento di un modello LSTM con i migliori iperparametri
#    trovati durante la fase di tuning.
#
#    python src/models/run_train.py \
#      --model_type lstm \
#      --batch_size 128 \
#      --hidden_size 896 \
#      --num_layers 3 \
#      --learning_rate 2.621087878265438e-05 \
#      --dropout 0.23993475643167195 \
#      --num_epochs 100 \
#      --seed 44
#
#    ESEMPIO DI COMANDO PER ST-GCN (da personalizzare):
#    Questo comando avvia l'addestramento di un modello ST-GCN.
#    Gli iperparametri come learning_rate e dropout andrebbero ottimizzati.
#
#    python src/models/run_train.py \
#      --model_type stgcn \
#      --batch_size 64 \
#      --learning_rate 1e-4 \
#      --dropout 0.1 \
#      --num_epochs 100 \
#      --seed 44
#
# =================================================================================================


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os, sys
import argparse
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import requests  # for pinging MLflow server

# --- Sezione 1: Setup del Percorso di Base e Import delle Utilità ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
sys.path.insert(0, BASE_DIR)

# Ignite imports
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
from ignite.engine import Events, create_supervised_trainer
from ignite.handlers import EarlyStopping as IgniteEarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.backends import cudnn
import random

from src.utils.training_utils import (
    get_data_paths,
    get_dataloaders,
    get_class_weights,
    create_model,
    prepare_batch,
    setup_ignite_evaluator,
)


def main(args):
    """Funzione principale che orchestra l'addestramento e la valutazione del modello."""
    # --- Sezione 2: Setup dell'Ambiente ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training su dispositivo: {device}")

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment("VADER 0.65 - Emotion Recognition Experiment")

    # --- Sezione 3: Caricamento Dati e Creazione del Modello ---
    (
        train_landmarks_dir,
        train_processed_file,
        val_landmarks_dir,
        val_processed_file,
    ) = get_data_paths(BASE_DIR)

    train_loader, val_loader, train_dataset = get_dataloaders(
        train_landmarks_dir,
        train_processed_file,
        val_landmarks_dir,
        val_processed_file,
        args.batch_size,
    )

    class_weights = get_class_weights(train_dataset, device)

    input_size = train_dataset[0][0].shape[1]
    num_classes = len(train_dataset.labels)

    model_params = {
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }

    model = create_model(
        args.model_type, input_size, num_classes, device, model_params
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=3)

    prepare_batch_fn = lambda batch, device, non_blocking: prepare_batch(
        batch, device, args.model_type, non_blocking
    )

    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device, prepare_batch=prepare_batch_fn
    )

    # --- Sezione 4: Definizione delle Metriche e Handlers ---
    evaluator = setup_ignite_evaluator(
        model, criterion, device, args.model_type, val_loader
    )

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
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "seed": args.seed,
        }
        if args.model_type == "lstm":
            params.update(model_params)
        else:  # stgcn
            params["stgcn_dropout"] = args.dropout

        mlflow.log_params(params)
        best_metrics = {"val_loss": float("inf"), "val_f1": 0.0, "val_f1_macro": 0.0}

        @trainer.on(Events.EPOCH_COMPLETED)
        def validate_and_log(engine):
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
            val_acc = metrics["val_acc"]
            val_f1 = metrics["val_f1"]
            val_f1_macro = metrics["val_f1_macro"]

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
            scheduler.step(val_f1_macro)

            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

        trainer.run(train_loader, max_epochs=args.num_epochs)

        print(
            f"Training completato. Best model saved in {checkpoint_handler.last_checkpoint}"
        )

        mlflow.log_metric("best_val_loss", best_metrics["val_loss"])
        mlflow.log_metric("best_val_f1", best_metrics["val_f1"])
        mlflow.log_metric("best_val_f1_macro", best_metrics["val_f1_macro"])

        data_sample, _ = next(iter(train_loader))
        data_sample, _ = prepare_batch_fn((data_sample, _), device, non_blocking=False)

        signature = infer_signature(
            data_sample.cpu().numpy(),
            model(data_sample.float()).cpu().detach().numpy(),
        )
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="EmoSign_pytorch",
        )
        mlflow.set_tag("experiment_purpose", "PyTorch emotion recognition")


# --- Sezione 6: Parsing degli Argomenti da Linea di Comando ---
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
