import os
import requests
from datasets import load_from_disk
from tqdm import tqdm
import logging

# Configura il logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Percorsi
dataset_path = "../../data/raw/emosign_dataset"
video_output_dir = "../../data/raw/videos"
base_url = "https://aslsignbank.hls.harvard.edu/dictionary/signs/"

# Carica il dataset per ottenere i metadati
logging.info(f"Caricamento metadati del dataset da {dataset_path}...")
try:
    dataset = load_from_disk(dataset_path)
except FileNotFoundError:
    logging.error(
        f"Dataset non trovato in {dataset_path}. Esegui prima 'src/data/make_dataset.py'."
    )
    exit()

# Itera su ogni split del dataset (train, test, validation)
for split in dataset.keys():
    split_dir = os.path.join(video_output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    logging.info(f"Inizio download dei video per lo split: {split}")

    # Itera su ogni esempio nello split
    for example in tqdm(dataset[split], desc=f"Downloading {split} videos"):
        video_name = example["video_name"]
        # L'URL richiede il nome del video in minuscolo
        video_url = f"{base_url}{video_name.lower()}.mp4"
        output_filepath = os.path.join(split_dir, f"{video_name}.mp4")

        # Controlla se il file esiste già per evitare di scaricarlo di nuovo
        if os.path.exists(output_filepath):
            continue

        try:
            # Scarica il video
            response = requests.get(video_url, timeout=10)
            response.raise_for_status()  # Lancia un errore per status code non validi (es. 404)

            # Salva il video su disco
            with open(output_filepath, "wb") as f:
                f.write(response.content)

        except requests.exceptions.HTTPError as e:
            logging.warning(f"Errore HTTP per {video_url}: {e}. Video saltato.")
        except requests.exceptions.RequestException as e:
            logging.error(
                f"Errore durante il download di {video_url}: {e}. Video saltato."
            )

logging.info("Download di tutti i video completato.")
