# PANORAMICA DEL FLUSSO:
# Questo è lo script principale che esegue l'addestramento del modello.
# Agisce come un direttore d'orchestra, mettendo insieme i pezzi:
# 1. Imposta tutti i parametri necessari per l'addestramento (iperparametri).
# 2. Usa `LandmarkDataset` per creare un `DataLoader`, che fornisce i dati al modello in batch.
# 3. Inizializza il modello `EmotionLSTM`.
# 4. Esegue il ciclo di addestramento (training loop) per un numero definito di epoche.
#    In ogni epoca, calcola l'errore (loss), e aggiorna i pesi del modello per ridurlo (ottimizzazione).
# 5. Al termine, salva il modello addestrato su disco per poterlo riutilizzare in futuro.

import torch
import torch.nn as nn
import os, sys

# Aggiunge la cartella src al PYTHONPATH per consentire import data e models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from data.emosign_landmark_dataset import (
    LandmarkDataset,
)  # Importa la classe per gestire i dati
from models.landmark_model import EmotionLSTM  # Importa l'architettura del modello

# --- Sezione 1: Definizione dei Parametri e Iperparametri ---
# Percorsi dei file e delle cartelle
LANDMARKS_DIR = "../../data/raw/openpose_output/json/"  # Cartella con sottocartelle per ogni video contenenti i JSON OpenPose
METADATA_FILE = (
    "../../data/processed/metadata.csv"  # File CSV con nomi video e etichette
)
MODEL_SAVE_PATH = "../../models/emotion_lstm.pth"  # Dove salvare il modello finale

# Iperparametri
INPUT_SIZE = 468 * 3  # Dimensione dell'input, deve corrispondere a quella del dataset
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
# 1. Crea un'istanza del nostro Dataset
train_dataset = LandmarkDataset(
    landmarks_dir=LANDMARKS_DIR, metadata_file=METADATA_FILE
)
# 2. Split train/validation (80/20) e DataLoader
num_classes = len(train_dataset.labels)
val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

# 3. Inizializza il modello e lo sposta sul dispositivo scelto (CPU o GPU)
model = EmotionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes).to(device)

# 4. Definisce la funzione di perdita (Loss Function) e l'ottimizzatore
criterion = (
    nn.CrossEntropyLoss()
)  # Misura quanto le previsioni del modello sono sbagliate.
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
        sequences, labels = sequences.to(device), labels.to(device)
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
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_loss /= total
    val_acc = correct / total
    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )
    # Salvataggio del modello migliore
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Salvato modello migliorato (Val Loss: {best_val_loss:.4f})")
print(f"Training completato. Miglior modello salvato in {MODEL_SAVE_PATH}")
