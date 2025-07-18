import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import imageio.v3 as iio
import numpy as np
import logging

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoDataset(Dataset):
    """
    Dataset per caricare, campionare e pre-processare file video per modelli come ViViT,
    utilizzando un file CSV di annotazioni e una directory video "piatta".

    Args:
        annotations_file (str): Percorso al file CSV con le annotazioni (es. 'video_sentiment_data_0.65.csv').
                                La colonna dei nomi video deve chiamarsi 'video_name' e quella delle etichette 'emotion'.
        video_root_dir (str): La directory radice dove si trovano TUTTI i file video.
        image_processor (object): L'image processor di Hugging Face per pre-processare i fotogrammi.
        num_frames (int): Il numero di fotogrammi da campionare uniformemente da ogni video.
    """

    def __init__(
        self, annotations_file, video_root_dir, image_processor, num_frames=16
    ):
        self.video_info = pd.read_csv(annotations_file)
        self.video_root_dir = video_root_dir
        self.image_processor = image_processor
        self.num_frames = num_frames

        # Crea una mappa da etichetta stringa a intero
        self.labels = sorted(self.video_info["emotion"].unique())
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        logger.info(f"Trovate {len(self.labels)} classi: {self.label2id}")

    def __len__(self):
        return len(self.video_info)

    def _sample_frames(self, num_total_frames):
        """Campiona indici di fotogrammi in modo uniforme."""
        indices = np.linspace(0, num_total_frames - 1, self.num_frames, dtype=int)
        return indices

    def __getitem__(self, idx):
        video_name = self.video_info.iloc[idx]["video_name"]
        video_filename = f"{video_name}.mp4"  # Aggiunge l'estensione .mp4
        video_path_full = os.path.join(self.video_root_dir, video_filename)

        label_str = self.video_info.iloc[idx]["emotion"]
        label_id = self.label2id[label_str]

        try:
            # Usa imageio per leggere il video
            video = iio.imread(video_path_full, plugin="pyav")
            num_total_frames = video.shape[0]

            if num_total_frames == 0:
                raise ValueError("Il video è vuoto o non può essere letto.")

            frame_indices = self._sample_frames(num_total_frames)

            # Estrai i fotogrammi campionati
            frames = [video[i] for i in frame_indices]

            # Pre-processa i fotogrammi usando l'image processor di ViViT
            pixel_values = self.image_processor(
                frames, return_tensors="pt"
            ).pixel_values

            return {
                "pixel_values": pixel_values,
                "labels": torch.tensor(label_id, dtype=torch.long),
            }

        except Exception as e:
            logger.error(
                f"Errore nel caricare o processare il video {video_path_full}: {e}"
            )
            # Ritorna un tensore vuoto o gestisci l'errore come preferisci
            raise IOError(f"Impossibile processare il video: {video_path_full}") from e
