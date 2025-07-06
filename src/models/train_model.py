# PANORAMICA DEL FLUSSO:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
import os, sys
import argparse

# Aggiunge la cartella src al PYTHONPATH per consentire import data e models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
# Definisce la root del progetto per percorsi assoluti
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

from data.emosign_landmark_dataset import (
    LandmarkDataset,
)  # Importa la classe per gestire i dati
from models.landmark_model import EmotionLSTM  # Importa l'architettura del modello

# --- Sezione 1: Definizione dei Parametri e Iperparametri ---
# Percorsi dei file e delle cartelle
LANDMARKS_DIR = os.path.join(
    BASE_DIR, "data", "raw", "train", "openpose_output_train", "json"
)
PROCESSED_FILE = os.path.join(
    BASE_DIR, "data", "processed", "train", "video_sentiment_data_0.65.csv"
)
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "emotion_lstm.pth")

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
LEARNING_RATE = 0.001  # "Velocità" con cui il modello impara e si corregge

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

# 3. Determino dinamicamente la dimensione dell'input (numero di feature per frame) dal primo sample
input_size = train_dataset[0][0].shape[1]  # lunghezza del vettore di feature
model = EmotionLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, num_classes, dropout=0.5).to(
    device
)

# 4. Definiamo la loss function con pesi di classe per penalizzare le classi minoritarie.
# NOTE: Significa che gli errori sulle classi minoritarie avranno un peso maggiore nella loss.
criterion = nn.CrossEntropyLoss(
    weight=class_weights
)  # Penalizza di più la classe minoritaria
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)  # Algoritmo che aggiorna i pesi del modello per minimizzare la loss.

# --- Sezione 4: Ciclo di Addestramento (Training Loop) ---
# --- Sezione 4: Training Loop con Validazione ---
print("Inizio training con validazione...")
best_val_loss = float("inf")
for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    for sequences, labels in train_loader:
        # Sposta dati su device e assicura float32 (MPS supporta solo float32)
        sequences = sequences.to(device).float()
        labels = labels.to(device)
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * labels.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device).float()
            labels_device = labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels_device)
            val_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels_device).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
        val_loss /= total
        val_acc = correct / total
        # Calcolo F1 macro per valutare performance di classificazione bilanciata
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
        )
    # Salvataggio del modello migliore
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Salvato modello migliorato (Val Loss: {best_val_loss:.4f})")
print(f"Training completato. Miglior modello salvato in {MODEL_SAVE_PATH}")

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
