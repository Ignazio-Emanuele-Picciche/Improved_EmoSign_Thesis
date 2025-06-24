# PANORAMICA DEL FLUSSO:
# Questo file è il primo passo fondamentale: funge da ponte tra i dati grezzi su disco
# (i nostri file JSON con i landmark) e il modello di machine learning.
# La classe `LandmarkDataset` ha il compito di:
# 1. Leggere i metadati (nomi dei video ed etichette delle emozioni).
# 2. Caricare il file JSON corrispondente a un video.
# 3. Trasformare i dati dei landmark in un formato numerico standard (tensore).
# 4. Garantire che tutte le sequenze video abbiano la stessa lunghezza (padding/troncamento).
# In sintesi, prepara i dati per essere "digeriti" dal modello durante l'addestramento.

import torch
from torch.utils.data import (
    Dataset,
)  # Classe base per creare dataset personalizzati in PyTorch
import json  # Per leggere i file di dati in formato JSON
import os  # Per interagire con il sistema operativo, come costruire percorsi di file
import numpy as np  # Libreria per calcoli numerici, specialmente per la gestione di array (matrici)
import pandas as pd  # Libreria per la manipolazione e l'analisi di dati, qui usata per leggere il file CSV dei metadati


# Definiamo una classe Dataset personalizzata. PyTorch richiede che erediti da `Dataset`.
class LandmarkDataset(Dataset):
    # Il metodo __init__ viene eseguito una sola volta, quando si crea un'istanza della classe.
    # Prepara le strutture dati iniziali.
    def __init__(self, landmarks_dir, metadata_file, max_seq_length=100):
        """
        Args:
            landmarks_dir (string): Directory che contiene tutti i file JSON con i landmark estratti.
            metadata_file (string): Percorso del file CSV che contiene i metadati (es. nome del video e emozione associata).
            max_seq_length (int): Lunghezza massima a cui tutte le sequenze di landmark verranno standardizzate (tramite padding o troncamento).
        """
        # Memorizza i percorsi e i parametri passati
        self.landmarks_dir = landmarks_dir
        self.metadata = pd.read_csv(
            metadata_file
        )  # Carica il file CSV in un DataFrame di pandas
        self.max_seq_length = max_seq_length

        # Crea una mappatura da etichetta testuale (es. "felice") a un indice numerico (es. 0, 1, 2...).
        # I modelli di machine learning lavorano con numeri, non con stringhe.
        self.labels = self.metadata[
            "emotion"
        ].unique()  # Trova tutte le emozioni uniche (es. ['felice', 'triste', 'neutro'])
        self.label_map = {
            label: i for i, label in enumerate(self.labels)
        }  # Crea il dizionario di mappatura: {'felice': 0, 'triste': 1, ...}

    # Il metodo __len__ deve restituire la dimensione totale del dataset.
    # Viene usato dal DataLoader di PyTorch per sapere quanti campioni ci sono.
    def __len__(self):
        return len(
            self.metadata
        )  # La dimensione è semplicemente il numero di righe nel nostro file di metadati.

    # Il metodo __getitem__ è il cuore del Dataset.
    # Ha il compito di caricare e restituire un singolo campione di dati dato un indice `idx`.
    def __getitem__(self, idx):
        # 1. Ottiene le informazioni per il campione richiesto (idx) dal DataFrame
        video_info = self.metadata.iloc[idx]
        video_name = video_info["video_name"]
        label_str = video_info["emotion"]
        # 2. Converte l'etichetta testuale nel suo corrispondente numero usando la mappa creata prima
        label = self.label_map[label_str]

        # 3. Costruisce il percorso completo al file JSON dei landmark per questo video
        json_path = os.path.join(self.landmarks_dir, f"{video_name}_landmarks.json")

        # 4. Apre e legge il file JSON
        with open(json_path, "r") as f:
            data = json.load(f)

        # 5. Estrae e trasforma i dati dei landmark in un formato adatto per il modello
        sequence = []
        for frame_data in data:
            # Per ogni frame, appiattiamo i 468 landmark (ognuno con x, y, z) in un unico grande vettore.
            # Il risultato sarà un vettore di 468 * 3 = 1404 features per ogni frame.
            flat_landmarks = [
                coord
                for lm in frame_data["landmarks"]
                for coord in (lm["x"], lm["y"], lm["z"])
            ]
            sequence.append(flat_landmarks)

        # Converte la lista di liste Python in un array NumPy per efficienza
        sequence = np.array(sequence, dtype=np.float32)

        # 6. Standardizza la lunghezza della sequenza (Padding o Troncamento)
        # I modelli come gli LSTM richiedono che tutte le sequenze in un batch abbiano la stessa lunghezza.
        if len(sequence) > self.max_seq_length:
            # Se la sequenza è più lunga del massimo, la tagliamo (tronchiamo).
            sequence = sequence[: self.max_seq_length]
        else:
            # Se la sequenza è più corta, aggiungiamo righe di zeri (padding) fino a raggiungere la lunghezza massima.
            # `sequence.shape[1]` è il numero di feature (1404), per creare zeri della giusta dimensione.
            padding = np.zeros((self.max_seq_length - len(sequence), sequence.shape[1]))
            sequence = np.vstack((sequence, padding))

        # 7. Restituisce i dati come tensori PyTorch.
        # Il DataLoader si aspetta di ricevere tensori.
        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long)
