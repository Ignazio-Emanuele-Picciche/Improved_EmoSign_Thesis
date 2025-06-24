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
from torch.utils.data import DataLoader
from src.data.landmark_dataset import (
    LandmarkDataset,
)  # Importa la classe per gestire i dati
from src.models.landmark_model import EmotionLSTM  # Importa l'architettura del modello

# --- Sezione 1: Definizione dei Parametri e Iperparametri ---
# Percorsi dei file e delle cartelle
LANDMARKS_DIR = "../../data/interim/landmarks/"
METADATA_FILE = (
    "../../data/processed/metadata.csv"  # File CSV con nomi video e etichette
)
MODEL_SAVE_PATH = "../../models/emotion_lstm.pth"  # Dove salvare il modello finale

# Iperparametri: "manopole" che regolano il processo di addestramento
INPUT_SIZE = 468 * 3  # Dimensione dell'input, deve corrispondere a quella del dataset
HIDDEN_SIZE = 256  # Complessità del modello LSTM
NUM_LAYERS = 2  # Profondità del modello LSTM
NUM_CLASSES = 7  # Numero di emozioni da predire (deve corrispondere ai dati)
BATCH_SIZE = 32  # Quanti video processare in parallelo prima di aggiornare il modello
NUM_EPOCHS = 50  # Quante volte ripetere l'addestramento sull'intero dataset
LEARNING_RATE = 0.001  # "Velocità" con cui il modello impara e si corregge

# --- Sezione 2: Setup dell'Ambiente ---
# Seleziona il dispositivo su cui eseguire i calcoli: GPU (cuda) se disponibile, altrimenti CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training su dispositivo: {device}")

# --- Sezione 3: Caricamento Dati e Creazione del Modello ---
# 1. Crea un'istanza del nostro Dataset
train_dataset = LandmarkDataset(
    landmarks_dir=LANDMARKS_DIR, metadata_file=METADATA_FILE
)
# 2. Crea un DataLoader, che gestisce il caricamento dei dati in batch e il mescolamento (shuffle)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# NOTA: In un progetto reale, si creerebbe anche un set di validazione/test per monitorare le performance.

# 3. Inizializza il modello e lo sposta sul dispositivo scelto (CPU o GPU)
model = EmotionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)

# 4. Definisce la funzione di perdita (Loss Function) e l'ottimizzatore
criterion = (
    nn.CrossEntropyLoss()
)  # Misura quanto le previsioni del modello sono sbagliate.
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)  # Algoritmo che aggiorna i pesi del modello per minimizzare la loss.

# --- Sezione 4: Ciclo di Addestramento (Training Loop) ---
print("Inizio addestramento...")
# Ciclo esterno: itera per il numero di epoche definito
for epoch in range(NUM_EPOCHS):
    # Ciclo interno: itera su tutti i batch di dati forniti dal DataLoader
    for i, (sequences, labels) in enumerate(train_loader):
        # Sposta i dati del batch corrente sul dispositivo (CPU/GPU)
        sequences = sequences.to(device)
        labels = labels.to(device)

        # 1. Forward pass: passa i dati attraverso il modello per ottenere le previsioni
        outputs = model(sequences)
        loss = criterion(outputs, labels)  # Calcola l'errore

        # 2. Backward pass e ottimizzazione:
        optimizer.zero_grad()  # Azzera i gradienti calcolati nel passo precedente
        loss.backward()  # Calcola i gradienti (quanto ogni peso ha contribuito all'errore)
        optimizer.step()  # Aggiorna i pesi del modello nella direzione che riduce l'errore

    # Stampa la loss alla fine di ogni epoca per monitorare l'andamento
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

# --- Sezione 5: Salvataggio del Modello Addestrato ---
# Salva solo i pesi del modello (lo "stato" imparato), che è la pratica consigliata.
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Modello addestrato salvato in {MODEL_SAVE_PATH}")
