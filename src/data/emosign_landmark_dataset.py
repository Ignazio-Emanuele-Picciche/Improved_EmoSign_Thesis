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
    def __init__(self, landmarks_dir, processed_file, max_seq_length=100):
        """
        Args:
            landmarks_dir (string): Directory che contiene tutti i file JSON con i landmark estratti.
            processed_file (string): Percorso del file CSV che contiene i metadati (es. nome del video e emozione associata).
            max_seq_length (int): Lunghezza massima a cui tutte le sequenze di landmark verranno standardizzate (tramite padding o troncamento).
        """
        # Memorizza i percorsi e i parametri passati
        self.landmarks_dir = landmarks_dir
        self.processed = pd.read_csv(
            processed_file
        )  # Carica il file CSV in un DataFrame di pandas
        self.max_seq_length = max_seq_length

        # Crea una mappatura da etichetta testuale (es. "felice") a un indice numerico (es. 0, 1, 2...).
        # I modelli di machine learning lavorano con numeri, non con stringhe.
        self.labels = self.processed[
            "emotion"
        ].unique()  # Trova tutte le emozioni uniche (es. ['felice', 'triste', 'neutro'])
        self.label_map = {
            label: i for i, label in enumerate(self.labels)
        }  # Crea il dizionario di mappatura: {'felice': 0, 'triste': 1, ...}

    # Il metodo __len__ deve restituire la dimensione totale del dataset.
    # Viene usato dal DataLoader di PyTorch per sapere quanti campioni ci sono.
    def __len__(self):
        return len(
            self.processed
        )  # La dimensione è semplicemente il numero di righe nel nostro file di metadati.

    # Il metodo __getitem__ è il cuore del Dataset.
    # Ha il compito di caricare e restituire un singolo campione di dati dato un indice `idx`.
    def __getitem__(self, idx):
        # 1. Ottiene le informazioni per il campione richiesto (idx) dal DataFrame
        video_info = self.processed.iloc[idx]
        video_name = video_info["video_name"]
        label_str = video_info["emotion"]
        # 2. Converte l'etichetta testuale nel suo corrispondente numero usando la mappa creata prima
        label = self.label_map[label_str]

        # 3. Costruisce il percorso alla cartella dei file JSON OpenPose per questo video
        video_dir = os.path.join(self.landmarks_dir, video_name)
        # Lista dei file JSON ordinati per frame
        json_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".json")])
        # 4. Estrae e trasforma i keypoints da ogni JSON in un vettore di feature
        sequence = []
        for jf in json_files:
            frame_path = os.path.join(video_dir, jf)
            with open(frame_path, "r") as f:
                frame = json.load(f)
            people = frame.get("people", [])
            if not people:
                continue
            # Prendiamo solo i pose_keypoints_2d, ignorando la confidenza (ogni terzo valore)
            keypoints = people[0].get("pose_keypoints_2d", [])
            flat_landmarks = [coord for i, coord in enumerate(keypoints) if i % 3 != 2]
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
            # Usa dtype float32 per evitare cast a float64 su MPS.
            padding = np.zeros(
                (self.max_seq_length - len(sequence), sequence.shape[1]),
                dtype=np.float32,
            )
            sequence = np.vstack((sequence, padding))

        # 7. Restituisce i dati come tensori PyTorch.
        # Il DataLoader si aspetta di ricevere tensori.
        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long)
