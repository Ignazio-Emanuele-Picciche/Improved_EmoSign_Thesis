# =================================================================================================
# COMANDI UTILI PER L'ESECUZIONE
# =================================================================================================
#
# 1. AVVIARE IL SERVER MLFLOW
#    mlflow server --host 127.0.0.1 --port 8080
#
# 2. ESEGUIRE L'ADDESTRAMENTO
#    Questo script addestra un modello ViViT per la classificazione delle emozioni.
#    Assicurarsi di aver creato i file CSV di annotazioni in data/processed/.
#
#    ESEMPIO DI COMANDO:
#    python src/models/vivit/run_train_vivit.py --num_epochs 10 --batch_size 4 --learning_rate 5e-5
#
# =================================================================================================

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import mlflow
import mlflow.pytorch
from collections import Counter
import logging

# --- Sezione 1: Setup del Percorso di Base e Import delle Utilità ---
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
    ),
)
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)

from src.models.vivit.vivit_model import create_vivit_model
from src.models.vivit.video_dataset import VideoDataset
from src.utils.training_utils import setup_ignite_evaluator  # Riutilizziamo l'evaluator

# Ignite imports
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import Loss, Accuracy, Precision, Recall, Fbeta

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Sezione 2: Definizione dei Percorsi Dati ---
TRAIN_VIDEO_DIR = os.path.join(BASE_DIR, "data", "raw", "train")
VAL_VIDEO_DIR = os.path.join(BASE_DIR, "data", "raw", "val")
# Aggiorniamo i percorsi per puntare ai file CSV corretti.
# Assumiamo che esista un file analogo per la validazione.
# Se non esiste, andrebbe creato o il dataset di training andrebbe splittato.
TRAIN_ANNOTATIONS_FILE = os.path.join(
    BASE_DIR, "data", "processed", "train", "video_sentiment_data_0.65.csv"
)
VAL_ANNOTATIONS_FILE = os.path.join(
    BASE_DIR, "data", "processed", "val", "video_sentiment_data_0.65.csv"
)
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "vivit_emotion.pth")


def get_class_weights(dataset):
    """Calcola i pesi per ogni classe per gestire lo sbilanciamento."""
    # La colonna delle etichette nel CSV si chiama 'emotion'
    class_counts = Counter(dataset.video_info["emotion"])
    class_weights = {
        dataset.label2id[cls]: 1.0 / count for cls, count in class_counts.items()
    }
    # Assicura che l'ordine sia corretto
    weights = [class_weights[i] for i in sorted(class_weights.keys())]
    return torch.tensor(weights, dtype=torch.float)


def get_sampler(dataset):
    """Crea un WeightedRandomSampler per il dataloader di training."""
    class_weights = get_class_weights(dataset)
    # La colonna delle etichette nel CSV si chiama 'emotion'
    sample_weights = [
        class_weights[dataset.label2id[label]]
        for label in dataset.video_info["emotion"]
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler


def prepare_batch(batch, device=None, non_blocking=False):
    """Sposta il batch di dati sul dispositivo corretto."""
    pixel_values = batch["pixel_values"].to(device, non_blocking=non_blocking)
    labels = batch["labels"].to(device, non_blocking=non_blocking)
    return pixel_values, labels


def main(args):
    """Funzione principale per l'addestramento."""
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    experiment_name = "ViViT - Emotion Recognition"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"vivit_train_{args.num_epochs}epochs"):
        mlflow.log_params(vars(args))

        # --- Setup del Modello e Device ---
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        logger.info(f"Using device: {device}")

        # --- Caricamento Dati ---
        # Creiamo un'istanza preliminare del dataset solo per ottenere l'image_processor e num_classes
        _, temp_image_processor = create_vivit_model(
            num_classes=2
        )  # num_classes temporaneo
        train_dataset = VideoDataset(
            TRAIN_ANNOTATIONS_FILE, TRAIN_VIDEO_DIR, temp_image_processor
        )
        val_dataset = VideoDataset(
            VAL_ANNOTATIONS_FILE, VAL_VIDEO_DIR, temp_image_processor
        )

        # Ora otteniamo il numero corretto di classi dal dataset
        num_classes = len(train_dataset.labels)
        logger.info(f"Numero di classi rilevato: {num_classes}")

        # Creiamo il modello con il numero corretto di classi
        model, image_processor = create_vivit_model(num_classes)
        model.to(device)

        # I dataset devono essere ricreati con l'image_processor finale, sebbene in questo
        # caso specifico l'istanza non cambi. È una buona pratica per coerenza.
        train_dataset = VideoDataset(
            TRAIN_ANNOTATIONS_FILE, TRAIN_VIDEO_DIR, image_processor
        )
        val_dataset = VideoDataset(VAL_ANNOTATIONS_FILE, VAL_VIDEO_DIR, image_processor)

        train_sampler = get_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,  # Aggiunto per parallelizzare il caricamento
            pin_memory=True,  # Aggiunto per velocizzare il trasferimento dei dati a GPU
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # --- Setup Addestramento ---
        optimizer = torch.optim.AdamW(  # Usiamo AdamW che è spesso migliore
            model.classifier.parameters(), lr=args.learning_rate, weight_decay=1e-2
        )
        # Calcoliamo i pesi per la loss in base al dataset di training
        class_weights_tensor = get_class_weights(train_dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        # --- Ignite Trainer ed Evaluator ---
        def train_step(engine, batch):
            model.train()
            optimizer.zero_grad()
            pixel_values, labels = prepare_batch(batch, device=device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            return loss.item()

        trainer = Engine(train_step)

        # Per l'evaluator, dobbiamo adattare l'output_transform
        def output_transform(output):
            y_pred, y = output
            # L'output del modello HuggingFace è un oggetto, estraiamo i logits
            return y_pred.logits, y

        val_metrics = {
            "accuracy": Accuracy(output_transform=output_transform),
            "loss": Loss(criterion, output_transform=output_transform),
            "f1": Fbeta(1.0, average="macro", output_transform=output_transform),
            "precision": Precision(average="macro", output_transform=output_transform),
            "recall": Recall(average="macro", output_transform=output_transform),
        }

        def eval_step(engine, batch):
            model.eval()
            with torch.no_grad():
                pixel_values, labels = prepare_batch(batch, device=device)
                outputs = model(pixel_values=pixel_values)
                return outputs, labels

        evaluator = Engine(eval_step)
        for name, metric in val_metrics.items():
            metric.attach(evaluator, name)

        # --- Handlers e Callbacks ---
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            logger.info(
                f"Validation Results - Epoch: {engine.state.epoch} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Accuracy: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1']:.4f} | "
                f"Precision: {metrics['precision']:.4f} | "
                f"Recall: {metrics['recall']:.4f}"
            )
            mlflow.log_metrics(
                {
                    "val_loss": metrics["loss"],
                    "val_accuracy": metrics["accuracy"],
                    "val_f1_macro": metrics["f1"],
                    "val_precision_macro": metrics["precision"],
                    "val_recall_macro": metrics["recall"],
                },
                step=engine.state.epoch,
            )

        # Early stopping
        handler = EarlyStopping(
            patience=args.patience,
            score_function=lambda e: e.state.metrics["f1"],
            trainer=trainer,
        )
        evaluator.add_event_handler(Events.COMPLETED, handler)

        # Checkpoint
        checkpointer = ModelCheckpoint(
            os.path.join(BASE_DIR, "models"),
            "vivit_emotion",
            n_saved=1,
            create_dir=True,
            save_as_state_dict=True,
            require_empty=False,
            score_function=lambda e: e.state.metrics["f1"],
            score_name="val_f1",
        )
        evaluator.add_event_handler(Events.COMPLETED, checkpointer, {"model": model})

        # --- Avvio Addestramento ---
        logger.info("Starting training...")
        trainer.run(train_loader, max_epochs=args.num_epochs)
        logger.info("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViViT for emotion recognition.")
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Training batch size."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate."
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for early stopping."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of worker processes for data loading.",
    )
    args = parser.parse_args()
    main(args)
