# PANORAMICA DEL FLUSSO:
# Questo file è il primo passo fondamentale: funge da ponte tra i dati grezzi su disco
# (i nostri file JSON con i landmark) e il modello di machine learning.
# La classe `LandmarkDataset` ha il compito di:
# 1. Leggere i metadati (nomi dei video ed etichette delle emozioni).
# 2. Caricare i file JSON corrispondenti a un video.
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
import logging

# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Definiamo una classe Dataset personalizzata. PyTorch richiede che erediti da `Dataset`.
class LandmarkDataset(Dataset):
    # Il metodo __init__ viene eseguito una sola volta, quando si crea un'istanza della classe.
    # Prepara le strutture dati iniziali.
    def __init__(
        self,
        landmarks_dir,
        processed_file,
        max_seq_length=100,
        keypoint_type="pose_keypoints_2d",
    ):
        """
        Args:
            landmarks_dir (string): Directory che contiene le cartelle dei video con i file JSON.
            processed_file (string): Percorso del file CSV che contiene i metadati.
            max_seq_length (int): Lunghezza massima a cui standardizzare le sequenze.
            keypoint_type (string): Tipo di keypoint da estrarre ('pose_keypoints_2d' per OpenPose,
                                    'face_keypoints_2d' per MediaPipe, ecc.).
        """
        # Memorizza i percorsi e i parametri passati
        self.landmarks_dir = landmarks_dir
        self.processed = pd.read_csv(processed_file)
        self.max_seq_length = max_seq_length
        self.keypoint_type = keypoint_type

        # Crea una mappatura da etichetta testuale a un indice numerico.
        self.labels = self.processed["emotion"].unique()
        self.label_map = {label: i for i, label in enumerate(self.labels)}

    # Il metodo __len__ deve restituire la dimensione totale del dataset.
    def __len__(self):
        return len(self.processed)

    # Il metodo __getitem__ è il cuore del Dataset.
    # Carica e restituisce un singolo campione di dati dato un indice `idx`.
    def __getitem__(self, idx):
        # 1. Ottiene le informazioni per il campione richiesto
        video_info = self.processed.iloc[idx]
        video_name = video_info["video_name"]
        label_str = video_info["emotion"]
        label = self.label_map[label_str]

        # 2. Costruisce il percorso alla cartella dei file JSON per questo video
        video_dir = os.path.join(self.landmarks_dir, video_name)
        if not os.path.isdir(video_dir):
            logging.warning(
                f"Directory non trovata per il video {video_name} in {self.landmarks_dir}. Salto il campione."
            )
            # Restituisce un campione vuoto o gestisce l'errore come preferito
            # Qui, per semplicità, restituiamo il primo campione valido del dataset.
            # Una gestione più robusta potrebbe sollevare un'eccezione o filtrare questi campioni a monte.
            return self.__getitem__(0)

        # Lista dei file JSON ordinati per frame
        try:
            json_files = sorted(
                [f for f in os.listdir(video_dir) if f.endswith(".json")]
            )
        except FileNotFoundError:
            logging.warning(
                f"Errore nel leggere la directory: {video_dir}. Salto il campione."
            )
            return self.__getitem__(0)

        # 3. Estrae e trasforma i keypoints da ogni JSON in un vettore di feature
        sequence = []
        for jf in json_files:
            frame_path = os.path.join(video_dir, jf)
            try:
                with open(frame_path, "r") as f:
                    frame_data = json.load(f)

                people = frame_data.get("people", [])
                if not people:
                    continue

                # Estrae i keypoint specificati da `keypoint_type`
                keypoints = people[0].get(self.keypoint_type, [])
                if not keypoints:
                    continue

                # Per OpenPose, i dati sono in formato [x1, y1, c1, x2, y2, c2, ...].
                # Rimuoviamo i punteggi di confidenza (c).
                # Per MediaPipe (come da extract_landmarks.py modificato), i dati sono già [x1, y1, z1, ...].
                # La logica di rimozione del terzo elemento funziona per entrambi i casi se z non è desiderato,
                # altrimenti va adattata. Qui assumiamo di volere solo x, y.
                if "pose" in self.keypoint_type:  # Tipico di OpenPose
                    flat_landmarks = [
                        coord for i, coord in enumerate(keypoints) if i % 3 != 2
                    ]
                else:  # Assume che non ci siano punteggi di confidenza (es. MediaPipe)
                    flat_landmarks = keypoints

                sequence.append(flat_landmarks)

            except (json.JSONDecodeError, IndexError) as e:
                logging.warning(
                    f"Errore nel processare il file {frame_path}: {e}. Sarà saltato."
                )
                continue

        if not sequence:
            logging.warning(
                f"Nessun landmark valido trovato per il video {video_name}. Salto il campione."
            )
            return self.__getitem__(0)

        # Converte la lista di liste Python in un array NumPy per efficienza
        sequence = np.array(sequence, dtype=np.float32)

        # 4. Standardizza la lunghezza della sequenza (Padding o Troncamento)
        if len(sequence) > self.max_seq_length:
            sequence = sequence[: self.max_seq_length]
        else:
            padding = np.zeros(
                (self.max_seq_length - len(sequence), sequence.shape[1]),
                dtype=np.float32,
            )
            sequence = np.vstack((sequence, padding))

        # 5. Restituisce i dati come tensori PyTorch.
        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long)
