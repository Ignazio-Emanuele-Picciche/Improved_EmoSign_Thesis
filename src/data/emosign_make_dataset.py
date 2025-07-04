from datasets import load_dataset
import logging
import os

# Configura il logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Percorso per salvare il dataset
output_path = "./data/raw/emosign_dataset"

# Carica il dataset EmoSign
logging.info("Caricamento del dataset 'catfang/emosign' da Hugging Face...")
dataset = load_dataset("catfang/emosign", trust_remote_code=True)
logging.info("Dataset caricato con successo.")

# Salva il dataset su disco
logging.info(f"Salvataggio del dataset in '{output_path}'...")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
dataset.save_to_disk(output_path)
logging.info("Salvataggio del dataset completato.")
