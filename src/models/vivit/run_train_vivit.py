# run_train_vivit.py: script principale per l'addestramento di un modello ViViT
# per la classificazione delle emozioni.
# Gestisce setup di MLflow, caricamento dataset, istanziazione del modello,
# loop di training con Ignite, checkpoint ed EarlyStopping.
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
#  --model_name: Specifica il modello ViViT da utilizzare.
#    Il default è "google/vivit-b-16x2-kinetics400",
#    skywalker290/videomae-vivit-d1
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

from ignite.contrib.handlers import ProgressBar


# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Sezione 2: Definizione dei Percorsi Dati ---
TRAIN_VIDEO_DIR = os.path.join(
    BASE_DIR, "data", "raw", "train", "raw_videos_front_train"
)
VAL_VIDEO_DIR = os.path.join(BASE_DIR, "data", "raw", "val", "raw_videos_front_val")
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

    # Estrai un nome breve del modello per il run_name
    model_short_name = args.model_name.split("/")[-1]
    run_name = f"{model_short_name}_train_{args.num_epochs}epochs"

    with mlflow.start_run(run_name=run_name):
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
            num_classes=2, model_name=args.model_name
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
        model, image_processor = create_vivit_model(
            num_classes, model_name=args.model_name
        )
        model.to(device)
        # Calcolo e log dei parametri del modello
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Usa il numero di frame del modello per campionare correttamente
        num_frames = model.config.num_frames  # es. 32
        # Ricrea i dataset con l'image_processor finale e il numero corretto di frame
        train_dataset = VideoDataset(
            TRAIN_ANNOTATIONS_FILE,
            TRAIN_VIDEO_DIR,
            image_processor,
            num_frames=num_frames,
        )
        val_dataset = VideoDataset(
            VAL_ANNOTATIONS_FILE, VAL_VIDEO_DIR, image_processor, num_frames=num_frames
        )

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
            try:
                logger.info(
                    f"\nBatch keys: {batch.keys()}"
                )  # Log delle chiavi del batch
                pixel_values, labels = prepare_batch(batch, device=device)
                logger.info(
                    f"\nShape of pixel_values: {pixel_values.shape}"
                )  # Log della forma del tensore
                logger.info(
                    f"\nActual shape of pixel_values: {pixel_values.shape}"
                )  # Log dettagliato della forma del tensore
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                return loss.item()
            except Exception as e:
                logger.error(f"\nError during train_step: {e}")
                raise

        trainer = Engine(train_step)

        # Logging training progress
        @trainer.on(Events.EPOCH_STARTED)
        def log_epoch_start(engine):
            logger.info(f"Starting Epoch {engine.state.epoch}.")

        @trainer.on(Events.ITERATION_COMPLETED(every=10))
        def log_iteration(engine):
            loss = engine.state.output
            logger.info(
                f"Epoch[{engine.state.epoch}] Iteration[{engine.state.iteration}] Loss: {loss:.4f}"
            )

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
        "--model_name",
        type=str,
        default="google/vivit-b-16x2-kinetics400",
        help="Name of the Hugging Face model to use.",
    )
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
