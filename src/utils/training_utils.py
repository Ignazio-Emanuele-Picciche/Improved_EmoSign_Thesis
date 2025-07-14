# -*- coding: utf-8 -*-
"""
Modulo di utilità per l'addestramento dei modelli.

Questo file centralizza le funzioni comuni utilizzate sia per l'addestramento standard
(`run_train.py`) sia per l'ottimizzazione degli iperparametri (`hyperparameter_tuning.py`).
L'obiettivo è ridurre la duplicazione del codice e migliorare la manutenibilità.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score

from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy

from src.data_pipeline.landmark_dataset import LandmarkDataset
from src.models.lstm_model import EmotionLSTM
from src.models.stgcn_model import STGCN


def get_data_paths(base_dir):
    """Restituisce i percorsi standard per i dati di training e validazione."""
    train_landmarks_dir = os.path.join(
        base_dir, "data", "raw", "train", "openpose_output_train", "json"
    )
    train_processed_file = os.path.join(
        base_dir, "data", "processed", "train", "video_sentiment_data_0.65.csv"
    )
    val_landmarks_dir = os.path.join(
        base_dir, "data", "raw", "val", "openpose_output_val", "json"
    )
    val_processed_file = os.path.join(
        base_dir, "data", "processed", "val", "video_sentiment_data_0.65.csv"
    )
    return (
        train_landmarks_dir,
        train_processed_file,
        val_landmarks_dir,
        val_processed_file,
    )


def get_dataloaders(
    train_landmarks_dir,
    train_processed_file,
    val_landmarks_dir,
    val_processed_file,
    batch_size,
):
    """
    Crea e restituisce i DataLoader per training e validazione, gestendo lo sbilanciamento delle classi.
    """
    train_dataset = LandmarkDataset(
        landmarks_dir=train_landmarks_dir, processed_file=train_processed_file
    )
    val_dataset = LandmarkDataset(
        landmarks_dir=val_landmarks_dir, processed_file=val_processed_file
    )

    # Weighted Sampler per bilanciare i batch di training
    labels_arr = train_dataset.processed["emotion"].map(train_dataset.label_map).values
    class_counts = np.bincount(labels_arr)
    class_weights_for_sampler = 1.0 / class_counts
    sample_weights = np.array(
        [class_weights_for_sampler[label] for label in labels_arr]
    )
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset


def get_class_weights(train_dataset, device):
    """Calcola i pesi per la funzione di loss per contrastare lo sbilanciamento delle classi."""
    labels_arr = train_dataset.processed["emotion"].map(train_dataset.label_map).values
    class_counts = np.bincount(labels_arr)
    num_classes = len(class_counts)
    total_samples = len(labels_arr)
    class_weights = torch.tensor(
        [total_samples / (num_classes * count) for count in class_counts],
        dtype=torch.float32,
    )
    return class_weights.to(device)


def create_model(model_type, input_size, num_classes, device, params):
    """Crea e restituisce il modello specificato (LSTM o ST-GCN) con i parametri dati."""
    if model_type == "lstm":
        model = EmotionLSTM(
            input_size,
            params["hidden_size"],
            params["num_layers"],
            num_classes,
            dropout=params["dropout"],
        ).to(device)
    elif model_type == "stgcn":
        coords_per_point = 2
        num_point = input_size // coords_per_point
        model = STGCN(
            num_classes,
            num_point,
            num_person=1,
            in_channels=coords_per_point,
            dropout_rate=params["dropout"],
        ).to(device)
    else:
        raise ValueError(f"Tipo di modello non supportato: {model_type}")
    return model


def prepare_batch(batch, device, model_type, non_blocking=False):
    """Prepara un batch di dati, rimodellandolo se necessario per ST-GCN."""
    x, y = batch
    if model_type == "stgcn":
        b, t, f = x.shape
        coords_per_point = 2
        num_point = f // coords_per_point
        x = x.view(b, t, num_point, coords_per_point)
    return (
        x.to(device, non_blocking=non_blocking),
        y.to(device, non_blocking=non_blocking),
    )


def setup_ignite_evaluator(model, criterion, device, model_type, val_loader):
    """
    Configura e restituisce un evaluator di Ignite con le metriche necessarie (Loss, Accuracy, F1).
    """
    prepare_batch_fn = lambda batch, device, non_blocking: prepare_batch(
        batch, device, model_type, non_blocking
    )

    val_metrics = {
        "val_loss": Loss(criterion),
        "val_acc": Accuracy(),
    }
    evaluator = create_supervised_evaluator(
        model, metrics=val_metrics, device=device, prepare_batch=prepare_batch_fn
    )

    @evaluator.on(Events.COMPLETED)
    def compute_f1_scores(engine):
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = prepare_batch_fn((xb, yb), device, non_blocking=False)
                outputs = model(xb.float())
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        engine.state.metrics["val_f1_macro"] = f1_macro
        engine.state.metrics["val_f1"] = f1_weighted
        engine.state.metrics["val_f1_class_0"] = float(f1_per_class[0])
        engine.state.metrics["val_f1_class_1"] = (
            float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0
        )

    return evaluator
