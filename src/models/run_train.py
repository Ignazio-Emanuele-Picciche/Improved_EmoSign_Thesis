# mlflow server --host 127.0.0.1 --port 8080

# PANORAMICA DEL FLUSSO:

# python src/models/run_train.py \
#   --model_type lstm \
#   --batch_size 64 \
#   --hidden_size 512 \
#   --num_layers 2 \
#   --learning_rate 0.0012830838516502473 \
#   --dropout 0.0005204106154637622 \
#   --num_epochs 50

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

# Aggiunge la cartella src al PYTHONPATH per consentire import data e models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
# Definisce la root del progetto per percorsi assoluti
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

from data_pipeline.landmark_dataset import (
    LandmarkDataset,
)  # Importa la classe per gestire i dati
from models.lstm_model import EmotionLSTM  # Importa l'architettura del modello LSTM
from models.stgcn_model import STGCN  # Aggiunto per ST-GCN support

# Removed custom EarlyStopping; using Ignite handlers instead

## Ignite imports for callback-based training
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy, F1
from ignite.handlers import EarlyStopping as IgniteEarlyStopping, ModelCheckpoint

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("Emotion Recognition Experiment")

# --- Sezione 1: Definizione dei Parametri e Iperparametri ---
# Percorsi dei file e delle cartelle
LANDMARKS_DIR = os.path.join(
    BASE_DIR, "data", "raw", "train", "openpose_output_train", "json"
)
PROCESSED_FILE = os.path.join(
    BASE_DIR, "data", "processed", "train", "video_sentiment_data_0.65.csv"
)

# Percorsi specifici per train e validation
TRAIN_LANDMARKS_DIR = LANDMARKS_DIR
TRAIN_PROCESSED_FILE = PROCESSED_FILE
VAL_LANDMARKS_DIR = os.path.join(
    BASE_DIR, "data", "raw", "val", "openpose_output_val", "json"
)
VAL_PROCESSED_FILE = os.path.join(
    BASE_DIR, "data", "processed", "val", "video_sentiment_data_0.65.csv"
)

# Iperparametri
# INPUT_SIZE = 468 * 3  # Calcolato dinamicamente dal dataset, non è più hard-coded
HIDDEN_SIZE = 256  # Complessità del modello LSTM
NUM_LAYERS = 2  # Profondità del modello LSTM
# NUM_CLASSES = 7  # Numero di emozioni da predire (ora calcolato dinamicamente)
BATCH_SIZE = 32  # Quanti video processare in parallelo prima di aggiornare il modello
NUM_EPOCHS = 50  # Quante volte ripetere l'addestramento sull'intero dataset
PATIENCE = 10  # Early stopping patience
LEARNING_RATE = 0.001  # "Velocità" con cui il modello impara e si corregge

# --- Sezione 1.5: Selezione del modello ---
# Opzioni: 'lstm' (default) o 'stgcn'
MODEL_TYPE = "lstm"  # Sostituisci con 'lstm' per LSTM basato su sentimen

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", f"emotion_{MODEL_TYPE}.pth")

# Define hyperparameters dict for MLflow logging
params = {
    "model_type": MODEL_TYPE,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
}


# --- Sezione 2: Setup dell'Ambiente ---
# Seleziona il dispositivo su cui eseguire i calcoli: Apple Silicon (mps) se disponibile, altrimenti GPU (cuda) o CPU.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Training su dispositivo: {device}")

# --- Sezione 3: Caricamento Dati e Creazione del Modello ---
# 1. Crea un'istanza dei nostri Dataset per train e validation
train_dataset = LandmarkDataset(
    landmarks_dir=TRAIN_LANDMARKS_DIR, processed_file=TRAIN_PROCESSED_FILE
)
val_dataset = LandmarkDataset(
    landmarks_dir=VAL_LANDMARKS_DIR, processed_file=VAL_PROCESSED_FILE
)

# Rimuoviamo lo split e usiamo DataLoader separati
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Bilanciamento classi: calcoliamo il peso inverso della frequenza di ogni classe
labels_array = train_dataset.processed["emotion"].map(train_dataset.label_map).values
class_counts = np.bincount(labels_array)
class_weights = [
    len(labels_array) / (len(class_counts) * c) for c in class_counts
]  # Calcola il peso inverso della frequenza
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
    device
)  # Sposta i pesi sul device

# 3. Determino dimensione input e creo il modello in base a MODEL_TYPE
input_size = train_dataset[0][0].shape[1]  # numero di feature per frame
num_classes = len(train_dataset.labels)
if MODEL_TYPE == "lstm":
    model = EmotionLSTM(
        input_size, HIDDEN_SIZE, NUM_LAYERS, num_classes, dropout=0.5
    ).to(device)
elif MODEL_TYPE == "stgcn":
    # ricava numero di punti e crea ST-GCN
    coords_per_point = 2  # x,y per landmark
    num_point = input_size // coords_per_point  # number of landmark nodes
    model = STGCN(
        num_classes, num_point, num_person=1, in_channels=coords_per_point
    ).to(device)

# 4. Definiamo la loss function con pesi di classe per penalizzare le classi minoritarie.
# NOTE: Significa che gli errori sulle classi minoritarie avranno un peso maggiore nella loss.
criterion = nn.CrossEntropyLoss(
    weight=class_weights
)  # Penalizza di più la classe minoritaria
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)  # Algoritmo che aggiorna i pesi del modello per minimizzare la loss.
# Setup Ignite trainer and evaluator
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
evaluator = create_supervised_evaluator(
    model,
    metrics={
        "val_loss": Loss(criterion),
        "val_acc": Accuracy(),
        "val_f1": F1(average=True),
    },
    device=device,
)
# Checkpoint handler: save best model by val_loss
checkpoint_handler = ModelCheckpoint(
    dirname=os.path.join(BASE_DIR, "models"),
    filename_prefix=f"emotion_{MODEL_TYPE}",
    n_saved=1,
    create_dir=True,
    score_function=lambda eng: -eng.state.metrics["val_loss"],
    score_name="val_loss",
)
evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, {"model": model})
# EarlyStopping handler
earlystop_handler = IgniteEarlyStopping(
    patience=PATIENCE,
    score_function=lambda eng: -eng.state.metrics["val_loss"],
    trainer=trainer,
)
evaluator.add_event_handler(Events.COMPLETED, earlystop_handler)

# --- Sezione 4: Ciclo di Addestramento (Training Loop) ---
print("Inizio training con validazione (Ignite)...")

run_name = f"EmotionRecognition_{MODEL_TYPE}_train"

with mlflow.start_run(run_name=run_name) as run:
    # Log hyperparameters
    mlflow.log_params(params)

    # On each epoch end: run evaluation and log metrics
    @trainer.on(Events.EPOCH_COMPLETED)
    def validate_and_log(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        mlflow.log_metrics(
            {
                "val_loss": metrics["val_loss"],
                "val_acc": metrics["val_acc"],
                "val_f1": metrics["val_f1"],
            },
            step=engine.state.epoch,
        )
        print(
            f"Epoch {engine.state.epoch}/{NUM_EPOCHS}, "
            f"Val Loss: {metrics['val_loss']:.4f}, "
            f"Val Acc: {metrics['val_acc']:.4f}, "
            f"Val F1: {metrics['val_f1']:.4f}"
        )

    # Run training with Ignite
    trainer.run(train_loader, max_epochs=NUM_EPOCHS)

    print(f"Training completato. Best model saved in {checkpoint_handler._saved[-1]}")

    # After training: log final metrics (best)
    mlflow.log_metric("best_val_loss", best_val_loss)
    mlflow.log_metric("best_val_f1", val_f1)

    # Infer model signature and log the model
    signature = infer_signature(
        train_dataset[0][0].numpy(),
        model(train_dataset[0][0].unsqueeze(0).to(device).float())
        .cpu()
        .detach()
        .numpy(),
    )
    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="EmoSign_pytorch",
    )

    # Tag the run for reference
    mlflow.set_tag("experiment_purpose", "PyTorch emotion recognition")
    mlflow.set_tag("run_id", run.info.run_id)

# parsing iperparametri da linea di comando
parser = argparse.ArgumentParser(description="Train EmotionLSTM with hyperparameters")
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
parser.add_argument("--hidden_size", type=int, default=HIDDEN_SIZE, help="Hidden size")
parser.add_argument(
    "--num_layers", type=int, default=NUM_LAYERS, help="Number of LSTM layers"
)
parser.add_argument(
    "--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate"
)
parser.add_argument(
    "--num_epochs", type=int, default=NUM_EPOCHS, help="Number of epochs"
)
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
args = parser.parse_args()

# override default iperparametri con valori da linea di comando
BATCH_SIZE = args.batch_size
HIDDEN_SIZE = args.hidden_size
NUM_LAYERS = args.num_layers
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.num_epochs


# python train_model.py --batch_size 64 --hidden_size 512 --num_layers 3 --learning_rate 0.0005 --dropout 0.3
