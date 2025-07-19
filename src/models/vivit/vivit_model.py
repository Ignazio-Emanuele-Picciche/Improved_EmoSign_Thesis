# vivit_model.py: definisce la funzione create_vivit_model per caricare e configurare
# un modello ViViT pre-allenato da Hugging Face e il relativo image processor.
# Congela il backbone e prepara la testa di classificazione per il fine-tuning.

import torch
import torch.nn as nn
from transformers import VivitForVideoClassification, VivitImageProcessor


def create_vivit_model(num_classes, model_name="google/vivit-b-16x2-kinetics400"):
    """
    Crea e configura un modello ViViT per il fine-tuning sul nostro task di classificazione delle emozioni.

    Args:
        num_classes (int): Il numero di classi di output (es. 2 per 'positive' e 'negative').
        model_name (str): Il nome del modello pre-allenato da scaricare da Hugging Face.
                          Analizziamo il nome 'google/vivit-b-16x2-kinetics400':
                          - 'vivit': Sta per Video Vision Transformer, un'architettura basata su Transformer per l'analisi video.
                          - '-b': Indica la versione "Base" del modello, un buon compromesso tra dimensioni e performance.
                          - '16x2': Specifica la tokenizzazione del video. Il video è diviso in "tubelets" (cubi spazio-temporali)
                            di 16x16 pixel nello spazio e 2 frame nel tempo.
                          - 'kinetics400': È il dataset su cui il modello è stato pre-allenato, contenente 400 classi di azioni umane.

    Returns:
        tuple: Una tupla contenente il modello ViViT configurato e il suo image processor.
    """
    # 1. Carica il modello ViViT pre-allenato da Hugging Face.
    #    `ignore_mismatched_sizes=True` è fondamentale perché stiamo per sostituire la "testa"
    #    di classificazione, che ha una dimensione diversa (num_classes) rispetto a quella originale (400).
    model = VivitForVideoClassification.from_pretrained(
        model_name, num_labels=num_classes, ignore_mismatched_sizes=True
    )

    # 2. Carica l'Image Processor associato al modello.
    #    Questo oggetto è responsabile della preparazione dei fotogrammi video (ridimensionamento,
    #    normalizzazione, ecc.) nel formato esatto che il modello si aspetta.
    image_processor = VivitImageProcessor.from_pretrained(model_name)

    # 3. Congela i pesi del "backbone" (il corpo principale del modello).
    #    Questa è la strategia di fine-tuning più comune: non modifichiamo i pesi
    #    che hanno già imparato a riconoscere feature generali dai video (su Kinetics-400).
    #    Addestreremo solo il nuovo layer di classificazione.
    for param in model.vivit.parameters():
        param.requires_grad = False

    # 4. Assicurati che i pesi della nuova "testa" di classificazione siano addestrabili.
    #    La testa (model.classifier) è stata creata ex-novo durante il caricamento
    #    con `num_labels=num_classes`, quindi i suoi parametri sono già addestrabili
    #    di default, ma è una buona pratica verificarlo.
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model, image_processor
