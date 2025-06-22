from datasets import load_dataset

# Carica il dataset EmoSign
dataset = load_dataset("catfang/emosign")

# Salva il dataset
dataset.save_to_disk("./data/raw/emosign_dataset")
